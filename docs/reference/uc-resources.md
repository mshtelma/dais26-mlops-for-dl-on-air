# Unity Catalog resource map

Every name is config-driven: UC locations come from the named [environment](configuration.md)
(`ENV` in `00_config.py` / `env:` in air), with `TABLE_PREFIX = dais26_dentex_` and the
backbone-keyed model/endpoint names. The default `df1` env resolves to catalog `main`, schema
`mshtelma`; `prod` → `mlops_pj` / `dais26_vfm` (+ champion schema `dais26_vfm_prod`).

| Resource | Example (df1) | Notes |
|----------|---------------|-------|
| **Schema** (dev) | `main.mshtelma` | dev models + data; created by `00_setup.py` (**not** bundle-managed) |
| **Schema** (champion) | `main.mshtelma` (df1) / `mlops_pj.dais26_vfm_prod` (prod) | champion model + prod tables; Terraform-managed on `prod` |
| **Volume** | `…/dentex_raw` | raw DENTEX images + COCO JSON |
| **Volume** | `…/model_cache` | pinned backbone weights + DINOv2 fallback head |
| **Delta table** | `…/dais26_dentex_train_embeddings` | `ARRAY<FLOAT>` dim = `summary_dim`, CDF on (champion schema) |
| **Delta table** | `…/dais26_dentex_drift_scores` | hourly drift output (champion schema) |
| **Delta table** | `…/dais26_dentex_detector_inference_payload` | auto-created by AI Gateway on first request |
| **Registered model** (dev) | `main.mshtelma.cradio_detector` (or `dinov3_detector`) | backbone-keyed; alias `@challenger`; runtime-registered |
| **Registered model** (champion) | `…detector_champion` | **single, backbone-agnostic**; `@champion_candidate` → `@champion`; bundle-managed (prod) |
| **Serving endpoint** (dev) | `dais26-cradio-detector-dev` | backbone-keyed; `scale_to_zero=true` |
| **Serving endpoint** (champion) | `dais26-detector-champion` | single; `scale_to_zero=false`; serves whatever holds `@champion` |
| **VS endpoint** | `dais26-vfm-vs` | Vector Search endpoint |
| **VS index** | `…/dais26_dentex_embeddings_index` | `DELTA_SYNC`, HNSW+L2, dim derived from the table |
| **MLflow experiment** | `/Users/<you>/dais26_vfm_experiment` | shared by both lanes + the gates (the env's `experiment_name`) |
| **Secret scope** | `dais26-secrets` | key `hf-token` (DINOv3 only) |

## Naming derivation (from `00_config.py`)

```python
TRAIN_EMBEDDINGS_TABLE = f"{CHAMPION_CATALOG}.{CHAMPION_SCHEMA}.{TABLE_PREFIX}train_embeddings"
DRIFT_SCORES_TABLE     = f"{CHAMPION_CATALOG}.{CHAMPION_SCHEMA}.{TABLE_PREFIX}drift_scores"
DETECTOR_MODEL_NAME    = f"{CATALOG}.{SCHEMA}.{DETECTOR_MODEL_SHORT}"          # backbone-keyed
CHAMPION_MODEL_NAME    = f"{CHAMPION_CATALOG}.{CHAMPION_SCHEMA}.detector_champion"  # single
VS_INDEX_NAME          = f"{CHAMPION_CATALOG}.{CHAMPION_SCHEMA}.{TABLE_PREFIX}embeddings_index"
```

UC identifiers are built via `platform.uc.UCName` / `VolumePath` (regex-validated) — never
hand-rolled `f"{cat}.{sch}.{name}"` — so dotted-catalog typos fail fast.

## Why two schemas / two endpoints

Dev models (backbone-keyed, `@challenger`) compete on the eval gate. The approved winner of *any*
architecture is copied — lineage preserved — into the **single** prod `detector_champion` and
served from the **single** champion endpoint. Broad deployment therefore comes from one
schema/model/endpoint, never two competing architecture-named champions. See
[Architecture → deployment jobs + cross-schema promotion](../ARCHITECTURE.md#deployment-jobs-and-cross-schema-promotion).

## Feature-dimension cascade

The Vector Search index dimension, the embeddings `ARRAY<FLOAT>` length, and the drift reference
all flow from `BackboneInfo.summary_dim` (C-RADIOv4 2304 / DINOv3 1024 / DINOv2 768). The VS index
even derives its dimension from `size(embedding)` on the source table — so switching `BACKBONE`
needs no edits here. See [Architecture → BackboneInfo contract](../ARCHITECTURE.md).
