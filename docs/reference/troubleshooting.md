# Troubleshooting

The single consolidated symptom → cause → fix table. Deep rationale for the distributed-training
and serving items lives in [Engineering rationale](../RUNBOOK.md#engineering-rationale).

## Build & deploy

| Symptom | Cause | Fix |
|---------|-------|-----|
| `dist/*.whl` not found during `bundle deploy` | build step skipped | `uv build` before deploy |
| `FileNotFoundError: Could not locate pyproject.toml` at log-time | stale wheel built before the `force-include` block | re-`uv build`; verify `python -m zipfile -l dist/*.whl \| grep _pyproject.toml` |
| `bundle validate`/`deploy` 403 from CI | workspace IP access list blocks GitHub-hosted runners | deploy locally / allowlisted runner via `scripts/deploy_bundle.sh`; see [CI/CD](../scenarios/cicd.md) |
| `bundle deploy -t prod` errors that the schema already exists | champion schema exists outside Terraform state | `databricks bundle deployment bind dais26_vfm_prod mlops_pj.dais26_vfm_prod -t prod --auto-approve` |

## Setup / training

| Symptom | Cause | Fix |
|---------|-------|-----|
| `setup` task fails | UC catalog/schema missing | verify UC enabled; check the env's `catalog`/`schema` |
| `confirm_challenger` fails | training didn't register/alias a version | inspect the `train` task logs + MLflow registration |
| HF download fails with `os error 5` / `os error 95` on AIR | `HF_HUB_ENABLE_HF_TRANSFER=1` or `hf-xet` writing to UC Volume FUSE | set `HF_HUB_ENABLE_HF_TRANSFER=0` + `HF_HUB_DISABLE_XET=1` **before** importing `dais26_dentex` (`platform.hf_env.configure_hf_env`); see [`#hf-transfer-fuse-incompat`](../RUNBOOK.md#hf-transfer-fuse-incompat) |
| Cold-cache HF download deadlock on multi-rank | naive `barrier()` doesn't fix the `from_pretrained` race | use `distributed.barrier_dance.rank0_first` (already wired in `models/builder.py`); see [`#hf-cache-race`](../RUNBOOK.md#hf-cache-race) |
| `BarrierTimeoutError` from `safe_barrier` | a rank crashed earlier; NCCL would have hung silently | inspect ranks' logs in order — the bounded wait surfaces the dead rank instead of hanging |
| DINOv3 trains to 0.0 mAP, dead-flat loss | fp16/bf16 NaN the RoPE/LayerScale encoder → GradScaler skips every step | keep `amp_dtype: auto` (→ fp32 for DINOv3); ImageNet norm (not CLIP); see [HPO → DINOv3 A/B](../HPO.md) |
| `trust_remote_code` error loading C-RADIOv4 | transformers version mismatch | pin `transformers>=4.48.0` in your training env (`pyproject.toml`) |
| `ModuleNotFoundError: serverless_gpu` in the air/cli flow | not needed — the CLI is the `torchrun` path | confirm you're running `air`/`torchrun`, not the notebook `@distributed` path |
| `MODEL_URI=` missing from rank-0 stdout (air) | rank 0 crashed in `_save_and_register` | inspect rank-0 logs (`air logs <run-id>`); `MlflowReporter` raises typed `AliasingError` |

## HPO sweep / fine-tuning

| Symptom | Cause | Fix |
|---------|-------|-----|
| Sweep / fine-tune OOMs on the H100 pool | `backbone_mode=full` doubles activations | drop to `partial` with small `backbone_trainable_blocks`, or lower `batch_size`/raise `grad_accum_steps` |
| Loss diverges immediately when fine-tuning the backbone | `backbone_lr` too high → catastrophic forgetting | keep `backbone_lr` ≈ 1e-5 (10–100× below head `lr`) |
| DDP `find_unused_parameters` error | `full` expects every param to get a grad | Trainer sets `find_unused_parameters=False` only for `full`; `True` for frozen/lora/partial |
| Anchor changes have no effect | `anchor_scales`/`aspect_ratios` left unset | set both, or use `anchor_layout=per_level` + `anchor_base_scale` (the sweep's `anchor_mode`) |
| Sweep job times out | multi-trial + winner retrain exceeds the timeout | `campaign_sweep` carries 48h (`172800`); air uses `timeout_minutes: 2880` |

## Serving

| Symptom | Cause | Fix |
|---------|-------|-----|
| Endpoint `DEPLOYMENT_FAILED` / `ModuleNotFoundError: transformers_modules` at load | model logged as a pickled pyfunc instance captured the dynamic `trust_remote_code` class | already fixed — logged via **models-from-code** (`serve/detector_model_script.py`); re-train against current code, don't pickle a `DetectorPyfunc()`; see [`#models-from-code`](../RUNBOOK.md#models-from-code) |
| `ModuleNotFoundError: dais26_dentex` at serving | package source not bundled | `MlflowReporter.log_pyfunc` passes `code_paths=[…]` by default; verify the model's `code/` dir holds the package |
| Endpoint serves on CPU (0% GPU util, slow) on GPU_SMALL | unpinned `torch` resolved to a cu126/cu128 wheel the T4 driver (CUDA 12.4) can't init | keep `torch==2.6.0` / `torchvision==0.21.0` (cu124) pinned; see [`#torch-cu124-pin`](../RUNBOOK.md#torch-cu124-pin) |
| Backbone tries to reach huggingface.co at serving and won't start | online HF load in an egress-less container | serving forces `local_files_only` + offline HF env from the bundled `model_cache` (handled in `detector_pyfunc.load_context`) |
| `ModuleNotFoundError: timm`/`einops`/`open_clip` at serving | runtime dep missing from `[tool.dais26.serving-deps].detector` | add it there; `assert_serving_reqs_match_pyproject` is the CI guard |
| `ai_gateway` config silently ignored | nested under `config` instead of top-level | pass `ai_gateway=` as a top-level arg of `create_and_wait`; see [`#ai_gateway-placement`](../ARCHITECTURE.md#ai_gateway-placement) |
| Endpoint stuck `PENDING`/`PaaS update` >10 min | cold multi-GB GPU deploy or capacity | deploy waits ~90 min (`DEPLOY_TIMEOUT_SECONDS`); check the deploy task logs / GPU_SMALL quota |
| Good train metrics, bad served detections on 2:1 X-rays | train/serve preprocessing skew (letterbox vs squash) | fixed — pyfunc now letterboxes like training (`test_letterbox_decode_and_inverse_roundtrip`); see [HPO → serving re-eval](../HPO.md) |
| Registered model serves garbage (e.g. DINOv3 ~0.027) | registration/serialization break invisible to train metrics | re-register from a known-good run; re-eval through the pyfunc (`eval_comparison`) |

## Promotion / aliases / Vector Search

| Symptom | Cause | Fix |
|---------|-------|-----|
| `@champion` not set after champion deploy | smoke test failed; `@champion_candidate` remains staged | check `deploy_champion` logs; re-run or promote manually; prior champion still serves |
| Auto-`@challenger`/`@candidate` alias on a sub-best run | `register_winner=True` aliased whatever registered last | register the intended run + set the alias explicitly; see [HPO → champion registration](../HPO.md) |
| Vector Search index stuck syncing | CDF not enabled on the source table | `DESCRIBE EXTENDED …train_embeddings` → verify `delta.enableChangeDataFeed = true` |
| `IncompatibleArtifactError: artifact_format_version=1` at load | loading a v1 artifact with the v2 loader | re-train against current code (v1→v2 is not auto-converted) |
| `deploy_champion_job` never triggers after RegisterChampion | `deployment_job_id` not wired on `detector_champion` | run `connect_deployment_job -t prod`; see [Production deployment](../scenarios/production-deploy.md) |

If a symptom isn't here, the deep-dive rationale (with race traces) is in
[Engineering rationale](../RUNBOOK.md#engineering-rationale) and the
operational procedures are in [Operations & runbook](../RUNBOOK.md).
