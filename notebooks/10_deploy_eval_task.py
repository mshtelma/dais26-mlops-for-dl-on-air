# Databricks notebook source
# MAGIC %md
# MAGIC # 10 — Deployment job: Evaluation task
# MAGIC
# MAGIC First task of the `deploy_job_detector` MLflow 3 deployment job. A new
# MAGIC `@challenger` version on a dev detector model auto-triggers the job with
# MAGIC job parameters `model_name` (full dev UC name) + `model_version`; this task:
# MAGIC
# MAGIC 1. loads `models:/{model_name}/{model_version}` and re-scores it on the
# MAGIC    DENTEX **test** split (250 imgs) through the serving pyfunc + COCO eval
# MAGIC    (shared `eval.runner.score_model_on_split`, same code as 09).
# MAGIC 2. logs the test metrics to the model version (MLflow 3 LoggedModel) so they
# MAGIC    surface on the version page + feed the best-in-experiment check.
# MAGIC 3. **gates** on `mAP_50 >= 0.58 AND Caries AP@50 >= 0.30`
# MAGIC    ([docs/BENCHMARKS.md](../docs/BENCHMARKS.md)) AND best-in-experiment:
# MAGIC    the candidate's test `mAP_50` must be `>=` the current prod `@champion`'s
# MAGIC    test `mAP_50` and every prior evaluated version's. Raises (fails the task)
# MAGIC    otherwise, so a worse challenger never reaches approval/promotion.
# MAGIC
# MAGIC Requires a **GPU** (the ViT backbone loads onto CUDA); the job runs this task
# MAGIC on `GPU_1xA10` + `base_environment databricks_ai_v5` (mirrors 09).

# COMMAND ----------
# MAGIC %pip install --quiet ..

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %run ./00_config

# COMMAND ----------
import mlflow
from mlflow.tracking import MlflowClient

from dais26_dentex.eval.runner import score_model_on_split

mlflow.set_registry_uri("databricks-uc")

# ---- Gate thresholds (docs/BENCHMARKS.md per-backbone 0.60 campaign gate) ----
MAP50_THRESHOLD = 0.58
CARIES_AP50_THRESHOLD = 0.30
CARIES_CLASS = "Caries"
EVAL_SPLIT = "test"

# ---- Job params: full dev model name + numeric version (deployment-job inputs) ----
dbutils.widgets.text("model_name", "")
dbutils.widgets.text("model_version", "")
MODEL_NAME = dbutils.widgets.get("model_name").strip()
MODEL_VERSION = dbutils.widgets.get("model_version").strip()
if not MODEL_NAME or not MODEL_VERSION:
    raise ValueError(
        "model_name and model_version job parameters are required "
        f"(got model_name={MODEL_NAME!r}, model_version={MODEL_VERSION!r})."
    )
print(f"Evaluating {MODEL_NAME} v{MODEL_VERSION} on '{EVAL_SPLIT}'")

client = MlflowClient(registry_uri="databricks-uc")

# Prod champion model = same short (last) name in the prod/champion schema.
_short = MODEL_NAME.split(".")[-1]
CHAMPION_FULL = f"{CHAMPION_CATALOG}.{CHAMPION_SCHEMA}.{_short}"

# COMMAND ----------
# ---- Score the candidate version on the test split ----
candidate = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_VERSION}")
metrics = score_model_on_split(candidate, VOLUME_PATH, EVAL_SPLIT)
del candidate

cand_map50 = float(metrics["mAP_50"])
cand_caries = float(metrics["per_class_AP50"].get(CARIES_CLASS, 0.0))
print(f"Candidate test metrics: mAP_50={cand_map50:.4f} Caries AP@50={cand_caries:.4f}")
print(f"  per-class AP@50: {metrics['per_class_AP50']}")

# Flat, MLflow-metric-safe view (drop the nested per_class dict; flatten it).
flat_metrics = {f"{EVAL_SPLIT}/{k}": float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
for cls_name, ap in metrics["per_class_AP50"].items():
    flat_metrics[f"{EVAL_SPLIT}/AP50_{cls_name.replace(' ', '_')}"] = float(ap)

# COMMAND ----------
# ---- Log metrics to the model version (MLflow 3 LoggedModel) ----
# The version page surfaces metrics logged against the LoggedModel id. We resolve
# it from the ModelVersion and log there when MLflow 3 supports `model_id=`; we
# also log to an eval run (tagged with the model+version) so the best-in-experiment
# search below has a uniform place to read prior versions' test scores from.
mlflow.set_experiment(EXPERIMENT_NAME)

_logged_model_id = None
try:
    mv = client.get_model_version(name=MODEL_NAME, version=MODEL_VERSION)
    _logged_model_id = getattr(mv, "model_id", None)
except Exception as e:
    print(f"Could not resolve LoggedModel id for {MODEL_NAME} v{MODEL_VERSION}: {type(e).__name__}: {e}")

with mlflow.start_run(run_name=f"deploy-eval-{_short}-v{MODEL_VERSION}"):
    mlflow.set_tags(
        {
            "deploy_eval": "true",
            "eval_model_name": MODEL_NAME,
            "eval_model_version": MODEL_VERSION,
            "eval_split": EVAL_SPLIT,
        }
    )
    mlflow.log_metrics(flat_metrics)
    if _logged_model_id is not None:
        try:
            mlflow.log_metrics(flat_metrics, model_id=_logged_model_id)
            print(f"Logged test metrics to LoggedModel {_logged_model_id}")
        except TypeError:
            # Older client without `model_id=`; run-level metrics already logged.
            print("mlflow.log_metrics has no model_id kwarg; logged to the eval run only.")

# COMMAND ----------
# ---- Gate 1: absolute thresholds ----
threshold_failures = []
if cand_map50 < MAP50_THRESHOLD:
    threshold_failures.append(f"mAP_50 {cand_map50:.4f} < {MAP50_THRESHOLD}")
if cand_caries < CARIES_AP50_THRESHOLD:
    threshold_failures.append(f"Caries AP@50 {cand_caries:.4f} < {CARIES_AP50_THRESHOLD}")

# COMMAND ----------
# ---- Gate 2: best-in-experiment (>= current champion AND prior evaluated versions) ----
# Compare against (a) the current prod @champion re-scored on test (head-to-head,
# authoritative) and (b) any prior version's logged test/mAP_50 in this experiment.
# Missing comparators are treated as "no prior bar" — the candidate is best by
# default (e.g. very first version, or no champion yet).
prior_best = -1.0
prior_best_source = "none"


def _consider(source: str, value: float | None) -> None:
    global prior_best, prior_best_source
    if value is not None and value > prior_best:
        prior_best = value
        prior_best_source = source


# (a) Current prod champion, re-scored on test for an apples-to-apples number.
try:
    champ_mv = client.get_model_version_by_alias(name=CHAMPION_FULL, alias="champion")
    print(f"Current @champion: {CHAMPION_FULL} v{champ_mv.version}; re-scoring on '{EVAL_SPLIT}'")
    champ_model = mlflow.pyfunc.load_model(f"models:/{CHAMPION_FULL}@champion")
    champ_metrics = score_model_on_split(champ_model, VOLUME_PATH, EVAL_SPLIT)
    del champ_model
    _consider(f"champion v{champ_mv.version}", float(champ_metrics["mAP_50"]))
except Exception as e:
    print(f"No comparable @champion ({type(e).__name__}: {e}); skipping champion bar.")

# (b) Prior evaluated versions of THIS dev model in the experiment.
try:
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        filter_string=(
            f"tags.deploy_eval = 'true' and tags.eval_model_name = '{MODEL_NAME}' "
            f"and tags.eval_split = '{EVAL_SPLIT}'"
        ),
        output_format="list",
    )
    for r in runs:
        if r.data.tags.get("eval_model_version") == MODEL_VERSION:
            continue  # don't compare the candidate against itself
        _consider(
            f"version {r.data.tags.get('eval_model_version')}",
            r.data.metrics.get(f"{EVAL_SPLIT}/mAP_50"),
        )
except Exception as e:
    print(f"Prior-version search failed ({type(e).__name__}: {e}); skipping that bar.")

print(f"Best prior test mAP_50 = {prior_best:.4f} (source: {prior_best_source})")

best_in_experiment = cand_map50 >= prior_best

# COMMAND ----------
# ---- Decide ----
if threshold_failures or not best_in_experiment:
    msg = ["Evaluation gate FAILED."]
    if threshold_failures:
        msg.append("Thresholds: " + "; ".join(threshold_failures))
    if not best_in_experiment:
        msg.append(
            f"Not best-in-experiment: candidate mAP_50 {cand_map50:.4f} < "
            f"prior best {prior_best:.4f} ({prior_best_source})."
        )
    raise RuntimeError(" ".join(msg))

print(
    f"Evaluation gate PASSED: mAP_50={cand_map50:.4f} (>= {MAP50_THRESHOLD} and >= prior "
    f"best {prior_best:.4f}), Caries AP@50={cand_caries:.4f} (>= {CARIES_AP50_THRESHOLD})."
)
dbutils.jobs.taskValues.set(key="test_mAP_50", value=cand_map50)
dbutils.jobs.taskValues.set(key="test_Caries_AP50", value=cand_caries)
dbutils.notebook.exit("ok")
