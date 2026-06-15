# DINOv2 emergency fallback

Use this if C-RADIOv4 has a breaking change or is yanked from HuggingFace before/at the
conference. **DINOv2-base is not a drop-in swap** — it has different dimensions
(`summary_dim = spatial_dim = 768`, vs 2304 / 1152 for C-RADIOv4-SO400M), so every
dimension-dependent artifact (model, embeddings, VS index, drift reference) must be rebuilt.
Budget ≈2 hours.

!!! tip "Pre-baked shortcut"
    `scripts/pin_model_cache.py` generates a DINOv2 fallback head checkpoint into the
    `model_cache` UC Volume. If that checkpoint exists, step 1 loads it directly and skips
    training (~15 min saved). Run `make pin-cache` as part of [setup](../lifecycle/setup-and-data.md).

## Step 1 — Retrain the detection head with DINOv2

=== "DAB"

    ```bash
    databricks bundle run train_detector -t dev --params train_epochs=10,backbone=dinov2_base
    ```

    If the pre-baked checkpoint exists, the training script loads it and skips training. Look for:
    `Found pre-baked DINOv2 fallback checkpoint … Loading … Skipping training.`

=== "air CLI"

    ```bash
    air run -f air/workload_train_detector.yaml \
      --override parameters.recipe=dinov2_base parameters.epochs=10 --watch -p df1
    ```

The `dinov2_base` recipe keeps the cheap **frozen-head** path plus the backbone-agnostic
structural fixes (per-level anchors, per-class NMS).

## Step 2 — Recompute embeddings at dim=768

`precompute_embeddings` **self-selects the backbone from the live champion** (the
`source_dev_model` tag on `@champion`), so once the DINOv2 model is the champion, the embeddings
rebuild at 768 automatically:

```bash
databricks bundle run deploy_champion_job -t prod --only precompute_embeddings
```

This rewrites `train_embeddings` as 768-dim `ARRAY<FLOAT>`.

## Step 3 — Recreate the Vector Search index at dim=768

The dimension is **derived from the source table**, so `create_vector_search` builds a 768-dim
index from the rewritten table — usually no manual edit needed:

```bash
databricks bundle run deploy_champion_job -t prod --only create_vector_search
```

If you must recreate it by hand (existing index at the wrong dim), drop and recreate with
`embedding_dimension=768`:

```python
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import DeltaSyncVectorIndexSpecRequest
w = WorkspaceClient()
idx = "<champion_catalog>.<champion_schema>.dais26_dentex_embeddings_index"
try:
    w.vector_search_indexes.delete_index(index_name=idx)
except Exception as e:
    print(f"Drop failed (may not exist): {e}")
w.vector_search_indexes.create_index(
    name=idx, endpoint_name="dais26-vfm-vs", primary_key="image_id",
    index_type="DELTA_SYNC",
    delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
        source_table="<champion_catalog>.<champion_schema>.dais26_dentex_train_embeddings",
        embedding_vector_column="embedding", embedding_dimension=768,
        pipeline_type="TRIGGERED",
        columns_to_sync=["image_id", "diagnosis", "split"]))
```

## Step 4 — Regenerate the drift reference

The drift monitor reads `backbone_info.summary_dim` automatically, so with the DINOv2 champion the
reference is built from 768-dim embeddings. Re-run the drift baseline (the `drift_baseline` task of
the champion job, or the standalone monitor):

```bash
databricks bundle run deploy_champion_job -t prod --only drift_baseline
```

## Step 5 + 6 — Deploy the DINOv2 version and flip `@champion`

The governed path handles this: the new DINOv2 `@challenger` flows through
[eval → approve → promote](../lifecycle/evaluate-approve-promote.md) and the
[champion deploy](../lifecycle/serve.md) flips `@champion`. To do it manually, resolve the
DINOv2 version and `update_config_and_wait` the endpoint, then
`set_registered_model_alias(..., "champion", <dinov2_version>)` — same shape as
[Rollback](../lifecycle/rollback.md).

## Talk narrative adjustment

Update the "BackboneInfo contract" slide to show `summary_dim = 768` instead of 2304. The
three-jobs story is identical; only the dimensions change. See [The talk](../TALK.md).
