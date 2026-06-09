# Databricks notebook source
# MAGIC %md
# MAGIC # 10 — Deployment job: Evaluation task (champion-relative validation gate)
# MAGIC
# MAGIC First task of the `deploy_job_detector` MLflow 3 deployment job. A new
# MAGIC `@challenger` version on a dev detector model auto-triggers the job with
# MAGIC job parameters `model_name` (full dev UC name) + `model_version`; this task:
# MAGIC
# MAGIC 1. loads `models:/{model_name}/{model_version}` and scores it on the labeled
# MAGIC    DENTEX **val** split (50 imgs) through the serving pyfunc + COCO eval
# MAGIC    (shared `eval.runner.score_model_on_split`). We gate on **val**, NOT test:
# MAGIC    the 250-image DENTEX test split ships with **no public ground-truth
# MAGIC    annotations**, so COCO mAP on test is meaningless (COCOeval returns -1).
# MAGIC 2. logs the val metrics to the model version (MLflow 3 LoggedModel) + an eval
# MAGIC    run so they surface on the version page.
# MAGIC 3. **gates champion-relative**: the challenger must beat the current prod
# MAGIC    `@champion` (re-scored on the SAME val split, head-to-head) on **>= 2 of 3**
# MAGIC    metrics — `mAP_50`, `mAP_75`, `mAP_50_95` (strictly greater). If there is
# MAGIC    **no current champion**, the challenger auto-passes (first promotion). A
# MAGIC    challenger that wins on < 2 of 3 raises and fails the task, so it never
# MAGIC    reaches Approval / RegisterChampion.
# MAGIC
# MAGIC Requires a **GPU** (the ViT backbone loads onto CUDA); the job runs this task
# MAGIC on `GPU_1xA10` + `base_environment databricks_ai_v5`.

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

# ---- Champion-relative gate config ----
# Gate the challenger ONLY against the reigning @champion on the labeled val split.
# No absolute thresholds and no best-in-experiment bar: the single question is
# "is this challenger better than what's live?" measured head-to-head on val.
EVAL_SPLIT = "val"  # val has GT (50 imgs); test (250) has no public annotations.
GATE_METRICS = ["mAP_50", "mAP_75", "mAP_50_95"]
WINS_REQUIRED = 2  # challenger must STRICTLY beat champion on >= 2 of the 3 metrics.

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
MODEL_SHORT = MODEL_NAME.split(".")[-1]
print(f"Evaluating {MODEL_NAME} v{MODEL_VERSION} on '{EVAL_SPLIT}'")

client = MlflowClient(registry_uri="databricks-uc")

# SINGLE backbone-agnostic prod champion model (CHAMPION_MODEL_NAME from 00_config).
# The challenger competes against the one reigning champion regardless of architecture,
# so prod never carries two competing architecture-named champions.
CHAMPION_FULL = CHAMPION_MODEL_NAME

# COMMAND ----------
# ---- Score the candidate version on the val split ----
candidate = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_VERSION}")
cand_metrics = score_model_on_split(candidate, VOLUME_PATH, EVAL_SPLIT)
del candidate

cand_scores = {m: float(cand_metrics[m]) for m in GATE_METRICS}
print(f"Candidate {EVAL_SPLIT} metrics: " + ", ".join(f"{m}={cand_scores[m]:.4f}" for m in GATE_METRICS))
print(f"  per-class AP@50: {cand_metrics['per_class_AP50']}")

# Flat, MLflow-metric-safe view (drop the nested per_class dict; flatten it).
flat_metrics = {f"{EVAL_SPLIT}/{k}": float(v) for k, v in cand_metrics.items() if isinstance(v, int | float)}
for cls_name, ap in cand_metrics["per_class_AP50"].items():
    flat_metrics[f"{EVAL_SPLIT}/AP50_{cls_name.replace(' ', '_')}"] = float(ap)

# COMMAND ----------
# ---- Log metrics to the model version (MLflow 3 LoggedModel) + an eval run ----
# The version page surfaces metrics logged against the LoggedModel id. We resolve
# it from the ModelVersion and log there when MLflow 3 supports `model_id=`; we
# also log to an eval run (tagged with the model+version) for traceability.
mlflow.set_experiment(EXPERIMENT_NAME)

_logged_model_id = None
try:
    mv = client.get_model_version(name=MODEL_NAME, version=MODEL_VERSION)
    _logged_model_id = getattr(mv, "model_id", None)
except Exception as e:
    print(f"Could not resolve LoggedModel id for {MODEL_NAME} v{MODEL_VERSION}: {type(e).__name__}: {e}")

with mlflow.start_run(run_name=f"deploy-eval-{MODEL_SHORT}-v{MODEL_VERSION}"):
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
            print(f"Logged {EVAL_SPLIT} metrics to LoggedModel {_logged_model_id}")
        except TypeError:
            # Older client without `model_id=`; run-level metrics already logged.
            print("mlflow.log_metrics has no model_id kwarg; logged to the eval run only.")

# COMMAND ----------
# ---- Resolve the current @champion and re-score it head-to-head on val ----
# "No current champion" (alias not set / model absent) => challenger auto-passes
# (this is the first promotion). If a champion DOES exist but cannot be scored, we
# let the error propagate (fail loudly) rather than silently auto-passing a possible
# regression.
champ_mv = None
try:
    champ_mv = client.get_model_version_by_alias(name=CHAMPION_FULL, alias="champion")
except Exception as e:
    print(f"No current @champion on {CHAMPION_FULL} ({type(e).__name__}: {e}); challenger auto-passes.")

champ_scores: dict[str, float] = {}
if champ_mv is not None:
    print(f"Current @champion: {CHAMPION_FULL} v{champ_mv.version}; re-scoring on '{EVAL_SPLIT}' head-to-head")
    champ_model = mlflow.pyfunc.load_model(f"models:/{CHAMPION_FULL}@champion")
    champ_metrics = score_model_on_split(champ_model, VOLUME_PATH, EVAL_SPLIT)
    del champ_model
    champ_scores = {m: float(champ_metrics[m]) for m in GATE_METRICS}
    print("Champion " + EVAL_SPLIT + " metrics: " + ", ".join(f"{m}={champ_scores[m]:.4f}" for m in GATE_METRICS))

# COMMAND ----------
# ---- Decide: champion-relative, >= 2 of 3 strict wins (or auto-pass if no champion) ----
if champ_mv is None:
    print(
        f"Evaluation gate PASSED (no champion): challenger {EVAL_SPLIT} "
        + ", ".join(f"{m}={cand_scores[m]:.4f}" for m in GATE_METRICS)
    )
else:
    wins = [m for m in GATE_METRICS if cand_scores[m] > champ_scores[m]]
    print("Head-to-head (challenger vs champion):")
    for m in GATE_METRICS:
        verdict = "WIN " if cand_scores[m] > champ_scores[m] else "lose"
        print(f"  {m:>9}: {cand_scores[m]:.4f} vs {champ_scores[m]:.4f}  -> {verdict}")
    print(f"Challenger wins {len(wins)}/{len(GATE_METRICS)} (need >= {WINS_REQUIRED}): {wins}")

    if len(wins) < WINS_REQUIRED:
        raise RuntimeError(
            f"Evaluation gate FAILED: challenger beat @champion v{champ_mv.version} on only "
            f"{len(wins)}/{len(GATE_METRICS)} metrics ({wins}); need >= {WINS_REQUIRED}. "
            "Challenger discarded (no promotion)."
        )
    print(
        f"Evaluation gate PASSED: challenger beat @champion v{champ_mv.version} on "
        f"{len(wins)}/{len(GATE_METRICS)} metrics ({wins})."
    )

# COMMAND ----------
# ---- Hand validated metrics to downstream tasks ----
for m in GATE_METRICS:
    dbutils.jobs.taskValues.set(key=f"{EVAL_SPLIT}_{m}", value=cand_scores[m])
dbutils.notebook.exit("ok")
