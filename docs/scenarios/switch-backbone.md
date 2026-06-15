# Switch backbone

The backbone is selectable. All dimension-dependent code parameterizes on `BackboneInfo`, so
switching needs no downstream dimension edits — the Vector Search index dimension is even derived
from the embeddings table at index-creation time.

| Backbone | Literal | `summary` / `spatial` dim | Gated? | Role |
|----------|---------|---------------------------|--------|------|
| NVIDIA C-RADIOv4-SO400M | `cradio_v4_so400m` | 2304 / 1152 | No | **default**, commercial-OK |
| Meta DINOv3-ViTL16 | `dinov3_vitl16` | 1024 / 1024 | **Yes** (HF token) | comparison |
| DINOv2-base | `dinov2_base` | 768 / 768 | No | emergency fallback (see [DINOv2 fallback](dinov2-fallback.md)) |

## Switch in the DAB / notebook lane

Edit `notebooks/00_config.py`:

```python
BACKBONE = "dinov3_vitl16"     # was "cradio_v4_so400m"
```

This retargets the recipe, the backbone-keyed dev model/endpoint names (`dinov3_detector`,
`dais26-dinov3-detector-dev`), and the dimensions — all from `config.recipes` +
`config.backbones`. The experiment stays shared so C-RADIO and DINOv3 runs compare side-by-side.

## Switch in the air lane

Name the other recipe (or use the dedicated DINOv3 workload, which already wires the secret):

```bash
# Dedicated DINOv3 workload (secrets block already active):
air run -f air/workload_train_detector_dinov3.yaml --watch -p df1

# Or override the recipe on the C-RADIO workload (also needs the HF token wired):
air run -f air/workload_train_detector.yaml --override parameters.recipe=dinov3_vitl16 --watch -p df1
```

## DINOv3 is gated — wire the HuggingFace token

DINOv3 requires HF approval. `load_backbone` reads `os.environ["HF_TOKEN"]`. Create the scope +
secret once:

```bash
databricks secrets create-scope dais26-secrets
databricks secrets put-secret dais26-secrets hf-token
```

- **air lane**: `workload_train_detector_dinov3.yaml` already activates `secrets: { HF_TOKEN:
  "dais26-secrets/hf-token" }`; for sweeps, uncomment that block in `workload_sweep.yaml`.
- **DAB/notebook lane**: `02_train_detector_air.py` reads the secret on the driver and forwards
  `HF_TOKEN` into the `@distributed` worker.

C-RADIOv4 needs **no** token.

!!! warning "DINOv3 is autocast-unstable — train it in fp32"
    DINOv3's RoPE/LayerScale encoder NaNs under both fp16 and bf16. The recipe sets
    `amp_dtype: auto`, which resolves to **fp32** for DINOv3 (fp16 for C-RADIO). Inputs are also
    normalized with **ImageNet** stats for DINOv3 (CLIP for C-RADIO), carried on
    `BackboneInfo.image_mean/std`. See [HPO campaign log → DINOv3 A/B](../HPO.md) and
    [Configuration reference](../reference/configuration.md).

## Reproducibility — pin the backbone revision

For a reproducible run, pin the 40-char HF commit SHA instead of `main`:

```bash
# air:
air run -f air/workload_train_detector.yaml --override parameters.backbone_revision=<sha> --watch -p df1
# notebook: set BACKBONE_REVISION in 00_config.py
```

## C-RADIOv4 trust_remote_code transitive deps

C-RADIOv4 loads via `trust_remote_code=True`, which needs `timm` / `einops` / `open_clip` at
runtime. These are declared in `pyproject.toml` (`[project] dependencies` **and**
`[tool.dais26.serving-deps].detector`) and guarded by `_assert_cradio_runtime_deps` in
`models/backbones.py`. If you add a backbone with new transitive deps, add them to both places —
CI's `assert_serving_reqs_match_pyproject` guards drift. See
[pip_requirements source of truth](../RUNBOOK.md#pip-requirements-rationale).

For the architectural detail of the two outputs and dimension cascade, see
[Architecture → BackboneInfo contract](../ARCHITECTURE.md).
