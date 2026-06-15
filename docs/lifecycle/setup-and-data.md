# 1 · Setup & data

Before any training, Unity Catalog must hold the schemas, volumes, tables, grants, and the DENTEX
dataset. This is the `setup` task — `notebooks/00_setup.py` — that fronts every job.

## What `bundle deploy` creates vs what `00_setup` creates

| Created by `databricks bundle deploy` | Created by `notebooks/00_setup.py` (the `setup` task) |
|---|---|
| Job definitions, MLflow experiment, secret scope | `CREATE IF NOT EXISTS` for schemas, volumes, tables |
| Prod champion schema + model (`-t prod`, Terraform) | UC grants for the service principal |
| — | `train_embeddings` table with Change Data Feed enabled |
| — | downloads the DENTEX dataset into the `dentex_raw` volume |

Run setup standalone, or let any job run it as its first task:

=== "DAB"

    ```bash
    databricks bundle deploy -t dev                  # Phase 1 (idempotent)
    databricks bundle run train_detector -t dev      # runs setup → train → confirm_challenger
    ```

    The `setup` task runs `00_setup.py` on the standard serverless env.

=== "air CLI"

    The air training/sweep workloads call into the same `dais26_dentex` package, which performs
    UC bootstrap as needed. For first-time UC object + grant creation, run the DAB `setup` task
    once (it is the canonical bootstrap):

    ```bash
    databricks bundle run train_detector -t dev --only setup
    ```

## The DENTEX dataset

DENTEX dental panoramic X-rays, **705 train / 50 val / 250 test**, 4 collapsed diagnosis classes:
**Caries, Deep Caries, Periapical Lesion, Impacted**. License **CC-BY-NC-SA 4.0** — research and
demo only, no commercial use.

`00_setup.py` performs a 3-step download → `extract_all_zips` → `convert_to_coco`, with a
count-match check and a per-image aggregation fallback for the test split, and remaps DENTEX's
`category_id_3` to our `category_id`. Images land in the `dentex_raw` volume; COCO-format
annotation JSONs are written alongside. The loader code is in
[`data/dentex_loader.py`](https://github.com/mshtelma/dais26-mlops-for-dl-on-air/blob/main/src/dais26_dentex/data/dentex_loader.py).

!!! note "test ships images only"
    The DENTEX **test** split has no public annotations, so promotion is gated on the labeled
    **val** (50 imgs) at the challenger registration gate and re-scored on **test** where the
    deployment-job eval task can. See [Benchmarks](../BENCHMARKS.md).

## UC objects created (default env `df1` → `main.mshtelma`)

| Resource | Example name | Notes |
|----------|--------------|-------|
| Schema (dev) | `main.mshtelma` | dev models + data; created here (not bundle-managed) |
| Schema (champion) | `main.mshtelma` (df1) / `mlops_pj.dais26_vfm_prod` (prod) | champion model + prod tables |
| Volume | `…/dentex_raw` | raw DENTEX images + COCO JSON |
| Volume | `…/model_cache` | pinned backbone weights + DINOv2 fallback head |
| Delta table | `…/dais26_dentex_train_embeddings` | `ARRAY<FLOAT>`, CDF enabled (champion schema) |
| Delta table | `…/dais26_dentex_drift_scores` | drift job output (champion schema) |

Full map: [Unity Catalog resource map](../reference/uc-resources.md).

## Cache the backbone weights (recommended)

`scripts/pin_model_cache.py` downloads + pins the C-RADIOv4 weights (and bakes a DINOv2 fallback
head) into the `model_cache` volume, so training/serving load offline and reproducibly by SHA:

```bash
make pin-cache        # python scripts/pin_model_cache.py
```

No trained weights are stored in the repo; everything is fetched at runtime and cached here.

## Grants for the prod service principal

`00_setup.py` applies the UC grants the prod `run_as` SP needs (USE CATALOG/SCHEMA, CREATE
TABLE/MODEL, READ/WRITE VOLUME, APPLY TAG, EXECUTE MODEL, and the champion-schema grants). The
manual SQL equivalent is in
[Operations & runbook → service principal creation](../RUNBOOK.md#service-principal-creation).

Next: **[Train & register @challenger](train-and-register.md)**.
