# Overview & mental model

Five ideas underpin this repo. Once they click, every page that follows is obvious. Read this
once; the rest of the docs assume it.

## 1. Frozen backbone, tiny head

A Vision Foundation Model (VFM) is already trained on hundreds of millions of images. We don't
need its gradients — we need its features. So the **412M-parameter backbone is frozen** (by
default) and only a **~5M-parameter detection head** (FPN adapter + RetinaNet) trains on top.
The single architectural choice that makes detection on a 705-image medical dataset tractable on
one GPU.

The backbone emits **two distinct tensors** with **distinct dimensions** that must never be
interchanged:

- `summary` — a pooled global feature (C-RADIOv4: **2304** = 2×1152; DINOv3: 1024) → embeddings, similarity, drift.
- `spatial_features` — per-patch features (C-RADIOv4: **1152**; DINOv3: 1024) → FPN → detection head.

Every dimension-dependent line in the codebase parameterizes on `BackboneInfo.summary_dim` /
`spatial_dim`. Hardcoding `2304` or `1152` anywhere outside `models/backbones.py` is a bug. See
**[Architecture → BackboneInfo contract](../ARCHITECTURE.md)**.

## 2. Two launch lanes, one core

Training launches from either lane, and both execute the *same*
[`Trainer`](https://github.com/mshtelma/dais26-mlops-for-dl-on-air/blob/main/src/dais26_dentex/train/trainer.py)
class:

| | DAB (Asset Bundle) lane | air CLI lane |
|---|---|---|
| Launch | `databricks bundle run train_detector -t dev` | `air run -f air/workload_train_detector.yaml -p df1` |
| Notebook / entrypoint | `notebooks/02_train_detector_air.py` | `dais26_dentex.train.cli` |
| Distribution | `serverless_gpu.@distributed` (no `torchrun`) | `torchrun` on the snapshotted repo |
| Compute | one `GPU_8xH100` notebook task | one 8×H100 Serverless GPU pod (scalable to multi-node) |

Both are **AIR-only** — no traditional ML clusters exist anywhere in this repo. Details:
**[The Two Lanes](../lanes/overview.md)**.

## 3. Named configuration (why the lanes can't drift)

Neither lane restates hyperparameters or UC paths. Each selects them **by name**:

- **Recipe** — best-known hyperparameters per backbone, in [`config/recipes.py`](https://github.com/mshtelma/dais26-mlops-for-dl-on-air/blob/main/src/dais26_dentex/config/recipes.py). The air workload says `recipe: cradio_v4_so400m`; the notebook calls `build_trainer_config(BACKBONE, …)`. Same dict.
- **Environment** — UC catalog/schema/volumes/experiment, in [`config/environments.py`](https://github.com/mshtelma/dais26-mlops-for-dl-on-air/blob/main/src/dais26_dentex/config/environments.py). The air workload says `env: df1`; the notebook sets `ENV = "df1"` in `00_config.py`. Same `EnvSpec`.
- **Campaign stage** — HPO search space, in [`config/campaigns.py`](https://github.com/mshtelma/dais26-mlops-for-dl-on-air/blob/main/src/dais26_dentex/config/campaigns.py), driven by one `SweepRunner`.

All three live **inside the wheel**, so an air pod and a notebook resolve byte-identical config.
A target switch is one token (`env: df1` → `env: prod`), not five hand-mirrored keys. Full
detail: **[Named configuration](../lanes/configuration.md)**.

## 4. Two schemas: dev `@challenger`, prod `@champion`

Following the Databricks "Big Book of MLOps" *deploy-code* pattern, dev and prod registered
models live in **separate UC schemas**:

- **Dev** — backbone-keyed models (`cradio_detector`, `dinov3_detector`) in `CATALOG.SCHEMA`,
  carrying the `@challenger` alias. Architectures compete here on the eval gate.
- **Prod** — a **single, backbone-agnostic** `detector_champion` in
  `CHAMPION_CATALOG.CHAMPION_SCHEMA`, carrying `@champion`. The approved dev winner of *any*
  architecture is copied here (lineage preserved), served from one endpoint.

Promotion is a lineage-preserving `copy_model_version`, never a same-model alias flip. The alias
chain is `@challenger → @champion_candidate → @champion`. See
**[Evaluate → approve → promote](../lifecycle/evaluate-approve-promote.md)**.

## 5. Two-phase deployment

`databricks bundle deploy` **cannot** reference `@champion` before any version exists — a
declarative endpoint YAML fails on first deploy with "alias not found". So:

- **Phase 1 — `bundle deploy`** ships UC objects + job definitions + the MLflow experiment.
  **Never endpoints.**
- **Phase 2 — SDK-driven** endpoint creation, gated on a registered version *and* a passing
  smoke test, inside a deployment job.

The endpoint therefore never exists in a broken state. See
**[Architecture → two-phase deploy](../ARCHITECTURE.md#two-phase-deployment)** and
**[Serve & AI Gateway](../lifecycle/serve.md)**.

---

## Where to go next

| You want to… | Go to |
|---|---|
| Check you can run this | [Prerequisites](prerequisites.md) |
| Install + authenticate | [Install & authenticate](installation.md) |
| Train your first model (DAB) | [Quickstart — DAB lane](quickstart-dab.md) |
| Train your first model (terminal) | [Quickstart — air CLI lane](quickstart-air.md) |
| Understand the full pipeline | [MLOps Lifecycle](../lifecycle/overview.md) |
