# CLI cheat-sheet

Every command you need, grouped by tool. Defaults: DAB target `-t dev`; air profile `-p df1`.

## `make` shortcuts

```bash
make install          # uv pip install -e ".[dev]"
make lint             # ruff check src/ tests/ notebooks/ scripts/
make test             # pytest tests/unit/
make test-integration # pytest tests/integration/ -m integration (needs a workspace)
make build            # uv build → dist/*.whl
make bundle-validate  # databricks bundle validate -t dev
make bundle-deploy-dev
make bundle-run-train         # databricks bundle run train_detector -t dev
make bundle-run-embeddings    # deploy_champion_job -t prod --only precompute_embeddings
make bundle-run-drift         # drift_monitor -t prod
make warmup           # python scripts/warmup_endpoints.py
make pin-cache        # python scripts/pin_model_cache.py
make discover         # python scripts/discover_air_runtime.py
make help             # full list
```

## Databricks CLI — bundle

```bash
databricks bundle validate -t dev
databricks bundle deploy   -t dev               # Phase 1: UC + jobs (no endpoints)
databricks bundle deploy   -t prod --var sp_app_id="$SP_APP_ID"

databricks bundle run train_detector       -t dev
databricks bundle run campaign_sweep        -t dev -- --params sweep_stage=cradio_s2
databricks bundle run eval_comparison       -t dev
databricks bundle run eval_threshold_grid   -t dev
databricks bundle run deploy_job_detector   -t dev
databricks bundle run deploy_champion_job   -t prod
databricks bundle run deploy_champion_job   -t prod --only precompute_embeddings
databricks bundle run connect_deployment_job -t dev
databricks bundle run connect_deployment_job -t prod
databricks bundle run deploy_endpoint       -t dev        # break-glass
databricks bundle run drift_monitor         -t prod

databricks bundle run train_detector -t dev --only setup  # single task
./scripts/deploy_bundle.sh -t dev                          # deploy + connect (CI equivalent)
```

## Databricks CLI — serving / models / VS / secrets

```bash
databricks serving-endpoints get dais26-detector-champion | jq .state
databricks jobs get-run <run-id>
databricks vector-search-indexes get <cat>.<sch>.dais26_dentex_embeddings_index | jq .status
databricks tables list --catalog <cat> --schema <sch> | grep detector_inference
databricks secrets create-scope dais26-secrets
databricks secrets put-secret dais26-secrets hf-token
databricks service-principals create --display-name dais26-vfm-sp --output JSON
```

## AIR CLI (`air`)

```bash
air --version
air run -f air/workload_train_detector.yaml --watch -p df1
air run -f air/workload_train_detector_dinov3.yaml --watch -p df1     # gated DINOv3
air run -f air/workload_sweep.yaml --watch -p df1 --override parameters.stage=cradio_s2
air run -f air/workload_train_detector.yaml --override parameters.epochs=150 --watch -p df1
air run -f air/workload_train_detector.yaml --override parameters.env=prod --watch -p df1
air run -f air/workload_train_detector.yaml \
  --override compute.num_accelerators=16 timeout_minutes=720 --watch -p df1

air list runs --limit 10 -p df1
air get run <run-id> -p df1
air logs <run-id> -p df1
air logs <run-id> --node 1 -p df1
air logs <run-id> --download-to ./logs -p df1
air cancel <run-id> -p df1
air cancel --all -p df1

air -h ; air config -h ; air config.<field> -h   # always-current reference
```

## Operator scripts

```bash
python scripts/discover_air_runtime.py
python scripts/pin_model_cache.py
python scripts/warmup_endpoints.py
python scripts/probe_endpoint_gpu.py
python scripts/grant_inference_table_access.py --catalog <cat> --schema <sch> --sp-app-id "$SP_APP_ID"
bash   scripts/latency_probe.sh &
```

## Docs site (this site)

```bash
pip install -r docs-requirements.txt
mkdocs serve            # local preview at http://127.0.0.1:8000
mkdocs build --strict   # CI gate — fails on broken internal links/anchors
```

Catalogs: [Jobs](jobs.md) · [Notebooks](notebooks.md) · [Scripts](scripts.md) · lane references:
[DAB](../lanes/dab.md) · [air](../lanes/air.md).
