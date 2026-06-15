# Named configuration

The two lanes stay in lockstep because neither restates values — both select config **by name**.
There are three named axes, all living *inside the wheel* so an air pod and a notebook resolve
byte-identical dicts.

| Axis | Answers | Source | Selected by |
|------|---------|--------|-------------|
| **Recipe** | "which *hyperparameters*?" | [`config/recipes.py`](https://github.com/mshtelma/dais26-mlops-for-dl-on-air/blob/main/src/dais26_dentex/config/recipes.py) | `recipe: <backbone>` / `build_trainer_config(BACKBONE, …)` |
| **Environment** | "which *catalog / schema / experiment*?" | [`config/environments.py`](https://github.com/mshtelma/dais26-mlops-for-dl-on-air/blob/main/src/dais26_dentex/config/environments.py) | `env: <name>` / `ENV` in `00_config.py` |
| **Campaign stage** | "which *HPO search space*?" | [`config/campaigns.py`](https://github.com/mshtelma/dais26-mlops-for-dl-on-air/blob/main/src/dais26_dentex/config/campaigns.py) | `stage: <name>` / `SWEEP_STAGE` |

The schema for *all* tunable knobs is one frozen dataclass,
[`TrainerConfig`](https://github.com/mshtelma/dais26-mlops-for-dl-on-air/blob/main/src/dais26_dentex/config/trainer_config.py)
— see the [Configuration reference](../reference/configuration.md) for every field.

## Recipes — hyperparameters

A *recipe* is the best-known set of `TrainerConfig` overrides for one backbone, proven by the
"push to 0.60" campaigns (see [HPO campaign log](../HPO.md)). `TrainerConfig` field **defaults**
stay frozen at *legacy* values (absolute anchors, class-agnostic NMS) so historical runs stay
byte-identical; the winning recipe lives in `recipes.py`, not in the dataclass defaults.

```python title="config/recipes.py — the C-RADIOv4 recipe (dazzling-mole-850, val mAP@50 0.5931)"
"cradio_v4_so400m": {
    "backbone_mode": "full", "backbone_lr": 1e-5, "weight_decay": 1e-2,
    "anchor_layout": "per_level", "anchor_base_scale": 3.0, "nms_per_class": True,
    "amp_dtype": "auto",       # -> fp16 (C-RADIO's stable path)
    "batch_size": 4, "grad_accum_steps": 2,   # effective 4*2*8 = 64 on one 8xH100
    "img_size": 1024, "lr": 2e-4, "onecycle_pct_start": 0.2,
    "focal_gamma": 2.5, "box_loss_type": "smooth_l1",
    "aug_multiscale_range": [0.8, 1.0], "aug_rotation_deg": 5.0,
    "epochs": 150,
}
```

`build_trainer_config(backbone, *, catalog, schema, …, **overrides)` merges, last-wins:
**recipe → environment kwargs → explicit overrides**. Recipes contain *only* science
(hyperparameters); UC locations and model identity arrive as explicit arguments. Recipes are
plain dicts (not a parallel dataclass) so validity is enforced by building a real `TrainerConfig`
from each in unit tests.

`DETECTOR_NAMES_BY_BACKBONE` (same module) maps each backbone literal to its dev model short name
and dev endpoint name (`cradio_v4_so400m → cradio_detector / dais26-cradio-detector-dev`).

## Environments — UC locations

An *environment* is a named `EnvSpec`: `catalog`, `schema`, `experiment_name`, and the derived
`champion_catalog` / `champion_schema` / `volume_path` / `cache_dir`. Two are committed:

| Env | catalog | schema | champion_schema | Notes |
|-----|---------|--------|-----------------|-------|
| `df1` (default) | `main` | `mshtelma` | `mshtelma` | E2E-gate workspace; `main` grants no `CREATE SCHEMA`, so the champion shares `mshtelma` (still a distinct model, `detector_champion`) |
| `prod` | `mlops_pj` | `dais26_vfm` | `dais26_vfm_prod` | the talk's nominal project workspace; real dev/prod schema split |

`champion_schema` defaults to `<schema>_prod`, `volume_path`/`cache_dir` derive from
`catalog`+`schema` — so a new env usually only states catalog/schema/experiment.

### Resolution precedence (highest wins)

1. explicit keyword overrides to `load_environment(...)`
2. `DAIS26_*` environment variables (`DAIS26_CATALOG`, `DAIS26_SCHEMA`, `DAIS26_EXPERIMENT`, …)
3. an optional per-user `environments.local.yaml` overlay
4. the committed named entry in `ENVIRONMENTS`

See [Per-user environment overrides](../scenarios/env-overrides.md) for the overlay mechanics
(it is **deliberately not git-ignored** so it rides air's snapshot and the notebook `%pip` reinstall).

!!! warning "Secrets never live in environments"
    `EnvSpec` holds non-secret *locations* only. The HuggingFace token flows through Databricks
    secret scopes / the air `secrets:` block — never `config/environments.py`.

## Campaign stages — HPO search spaces

A *stage* is a typed `CampaignStage` (pinned params + a search space + trial budget + schedule +
a `register_winner` flag). The chain (`dinov3_s1…s4`, `cradio_s1…`, `*_fusion`, `*_giou`, …) is a
historical record of the push-to-0.60 build-out. One `SweepRunner` executes any stage from either
lane. Details: [HPO sweep](../lifecycle/hpo-sweep.md) and [HPO campaign log](../HPO.md).

## `00_config.py` — the notebook lane's selector

[`notebooks/00_config.py`](https://github.com/mshtelma/dais26-mlops-for-dl-on-air/blob/main/notebooks/00_config.py)
is **environment selection + per-notebook knobs only** — never hyperparameters. `%run ./00_config`
pulls its constants into every other notebook. The one place to switch targets:

```python
ENV = "df1"                 # the SAME named env the air lane resolves
_env = load_environment(ENV)
CATALOG, SCHEMA = _env.catalog, _env.schema
BACKBONE = "cradio_v4_so400m"
```

Per-notebook launch knobs that *are* sanctioned here (not hyperparameters): `TRAIN_EPOCHS` (demo
wall-time override of the recipe's 150), `TRAIN_GPUS`/`TRAIN_GPU_TYPE`, `DEPLOY_ACTION`,
`DRIFT_MODE`, `SWEEP_STAGE`, `EXPLORE_SPLIT`, the latency-benchmark knobs, etc. See the
[Configuration reference](../reference/configuration.md) for the full list.

!!! note "The two sanctioned job-parameter exceptions"
    Everything is read from `00_config.py` except **`sweep_stage`** (`campaign_sweep`) and
    **`deploy_action`** (`confirm_challenger` / break-glass deploy), which ride DAB
    `base_parameters` into notebook widgets so one job definition serves every stage/mode.

## How `$HYPERPARAMETERS_PATH` is shaped (air lane)

`air` writes the workload's `parameters:` block to `$HYPERPARAMETERS_PATH` as a flat JSON mapping.
`train.cli.load_config` resolves the `recipe:` and `env:` names, merges the remaining keys on top,
and hands a flat dict to `TrainerConfig.from_dict` (which coerces types and validates). The
workload YAML's *nested* `parameters:` is **not** the same shape as `$HYPERPARAMETERS_PATH` — air
flattens it. See [Architecture → AIR runtime contracts](../ARCHITECTURE.md) and the
[air lane reference](air.md).
