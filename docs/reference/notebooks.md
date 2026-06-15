# Notebooks catalog

Every notebook lives in `notebooks/` and pulls shared config via `%run ./00_config`. Stages map to
the [MLOps lifecycle](../lifecycle/overview.md). Links go to the source on GitHub.

| Notebook | Stage | Run by job | Key `00_config` knobs | Purpose |
|----------|-------|-----------|------------------------|---------|
| [`00_config.py`](https://github.com/mshtelma/dais26-mlops-for-dl-on-air/blob/main/notebooks/00_config.py) | config | (imported everywhere) | `ENV`, `BACKBONE`, all knobs | Environment selection + per-notebook knobs. `%run ./00_config` pulls these in. Not hyperparameters. |
| [`00_setup.py`](https://github.com/mshtelma/dais26-mlops-for-dl-on-air/blob/main/notebooks/00_setup.py) | setup | `setup` task of every job | `SP_APP_ID`, `HF_TOKEN` | UC bootstrap (schemas/volumes/tables + CDF), SP grants, DENTEX download → COCO. |
| [`01_explore_dentex.py`](https://github.com/mshtelma/dais26-mlops-for-dl-on-air/blob/main/notebooks/01_explore_dentex.py) | explore | — | `EXPLORE_SPLIT` | Visualize DENTEX X-rays + annotations; talk hook. |
| [`02_train_detector_air.py`](https://github.com/mshtelma/dais26-mlops-for-dl-on-air/blob/main/notebooks/02_train_detector_air.py) | train | `train_detector` (`train`) | `TRAIN_EPOCHS`, `TRAIN_GPUS`, `TRAIN_GPU_TYPE`, `TRAIN_USE_LORA` | `serverless_gpu.@distributed` training → register `@challenger`. |
| [`02b_hpo_sweep.py`](https://github.com/mshtelma/dais26-mlops-for-dl-on-air/blob/main/notebooks/02b_hpo_sweep.py) | sweep | `campaign_sweep` (`sweep`) | `SWEEP_STAGE` (+ `sweep_stage` widget) | Run a campaign stage via `SweepRunner`; best-in-experiment `@challenger` gate. |
| [`03_precompute_embeddings.py`](https://github.com/mshtelma/dais26-mlops-for-dl-on-air/blob/main/notebooks/03_precompute_embeddings.py) | embeddings | `deploy_champion_job` (`precompute_embeddings`) | `EMBEDDINGS_BATCH_SIZE`, `EMBEDDINGS_VS_*` | Frozen-backbone `summary` over 1005 images → `train_embeddings` Delta (CDF). Backbone self-selected from champion. |
| [`04_deploy_serving.py`](https://github.com/mshtelma/dais26-mlops-for-dl-on-air/blob/main/notebooks/04_deploy_serving.py) | serve | `train_detector`/`campaign_sweep` (`confirm_challenger`), `deploy_endpoint` | `DEPLOY_ACTION`, `DEPLOY_WORKLOAD_TYPE`, `DEPLOY_SCALE_TO_ZERO`, `DEPLOY_TIMEOUT_SECONDS` | Switches on `DEPLOY_ACTION`: confirm `@challenger` / deploy+smoke+`@champion` / create VS. |
| [`04b_create_vector_search.py`](https://github.com/mshtelma/dais26-mlops-for-dl-on-air/blob/main/notebooks/04b_create_vector_search.py) | embeddings | `deploy_champion_job` (`create_vector_search`) | `VS_ENDPOINT_NAME`, `VS_INDEX_NAME` | Idempotent VS endpoint + `DELTA_SYNC` index (dim derived from table), sync, smoke query. |
| [`05_drift_demo.py`](https://github.com/mshtelma/dais26-mlops-for-dl-on-air/blob/main/notebooks/05_drift_demo.py) | drift | `deploy_champion_job` (`drift_baseline`), `drift_monitor` | `DRIFT_MODE`, `DRIFT_KNN_K`, `DRIFT_ALERT_THRESHOLD` | Demo (clean vs shifted) or scheduled (inference-table) drift → `drift_scores`. |
| [`06_similarity_search_demo.py`](https://github.com/mshtelma/dais26-mlops-for-dl-on-air/blob/main/notebooks/06_similarity_search_demo.py) | similarity | — | `SIMILARITY_QUERY_COUNT` | VS top-10 + same-class recall@10; talk demo. |
| [`07_latency_benchmark.py`](https://github.com/mshtelma/dais26-mlops-for-dl-on-air/blob/main/notebooks/07_latency_benchmark.py) | benchmark | — | `LATENCY_NUM_REQUESTS`, `LATENCY_WARMUP_REQUESTS`, `LATENCY_PIVOT_THRESHOLD_MS` | Endpoint p50/p95/p99 → [Benchmarks](../BENCHMARKS.md). |
| [`09_eval_comparison.py`](https://github.com/mshtelma/dais26-mlops-for-dl-on-air/blob/main/notebooks/09_eval_comparison.py) | eval | `eval_comparison` | (`eval` split/alias knobs) | Re-score every registered detector through the serving pyfunc on val + test. |
| [`09b_eval_threshold_grid.py`](https://github.com/mshtelma/dais26-mlops-for-dl-on-air/blob/main/notebooks/09b_eval_threshold_grid.py) | eval | `eval_threshold_grid` | — | Free, no-retrain decode/NMS grid; banks best thresholds + Caries AP@50. |
| [`10_deploy_eval_task.py`](https://github.com/mshtelma/dais26-mlops-for-dl-on-air/blob/main/notebooks/10_deploy_eval_task.py) | eval | `deploy_job_detector` (`Evaluation`) | `model_name`, `model_version` (widgets) | Score the `@challenger` version; gate vs `@champion` on ≥2/3 metrics. |
| [`11_deploy_approval_task.py`](https://github.com/mshtelma/dais26-mlops-for-dl-on-air/blob/main/notebooks/11_deploy_approval_task.py) | approve | `deploy_job_detector` (`Approval_Check`) | `APPROVAL_TAG="Approval_Check"` | Human-in-the-loop gate: pass only if UC tag `Approval_Check=Approved`. |
| [`12_promote_task.py`](https://github.com/mshtelma/dais26-mlops-for-dl-on-air/blob/main/notebooks/12_promote_task.py) | promote | `deploy_job_detector` (`RegisterChampion`) | `model_name`, `model_version` | `copy_model_version` dev→prod champion; set `@champion_candidate`. |
| [`13_connect_deployment_job.py`](https://github.com/mshtelma/dais26-mlops-for-dl-on-air/blob/main/notebooks/13_connect_deployment_job.py) | wiring | `connect_deployment_job` | `bundle_target`, `*_deployment_job_id` | Target-aware `update_registered_model(deployment_job_id=...)`. |
| [`14_champion_deploy.py`](https://github.com/mshtelma/dais26-mlops-for-dl-on-air/blob/main/notebooks/14_champion_deploy.py) | champion-deploy | `deploy_champion_job` (`deploy_champion`) | (champion endpoint knobs) | Deploy `@champion_candidate` + smoke test → flip `@champion` on success. |
| [`diagnostics/02a_arch_probe.py`](https://github.com/mshtelma/dais26-mlops-for-dl-on-air/blob/main/notebooks/diagnostics/02a_arch_probe.py) | diagnostics | — | — | Anchor/positive-ratio/NMS probe + `KNOWN_ISSUES`; run before a sweep. |

!!! note "The two sanctioned widget params"
    Almost everything is read from `00_config.py`. The only job-parameter exceptions are
    **`sweep_stage`** (`campaign_sweep` → `02b`) and **`deploy_action`** (`confirm_challenger` /
    `deploy_endpoint` → `04`), which ride DAB `base_parameters` into notebook widgets. The
    deployment-job notebooks (`10`/`11`/`12`) additionally read `model_name` / `model_version`
    widgets injected by the MLflow 3 framework. See [Named configuration](../lanes/configuration.md).

Job task graphs: [Jobs catalog](jobs.md). Per-notebook config knobs:
[Configuration reference](configuration.md).
