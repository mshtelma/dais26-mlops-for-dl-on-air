# Talk: One frozen backbone, three jobs — MLOps for DL on AIR

**Event:** Data + AI Summit 2026 (DAIS26), San Francisco, June 15-18
**Duration:** 45 minutes
**Subtitle:** Why deep learning is hard, why distributed DL is harder, why MLOps compounds the
pain — and how Databricks AI Runtime collapses the stack into one frozen backbone serving
detection, embeddings, and drift from a single Unity Catalog artifact.

---

## Narrative arc

The talk is built around a four-step pain ladder followed by a single concrete answer:

1. **DL is hard.** Even on one GPU, with one model, one dataset.
2. **Distributed DL at scale is harder.** NCCL, env propagation, cold-cache races, dead-rank
   deadlocks — every layer adds a new failure mode.
3. **MLOps is hard on its own.** Registry, serving, lineage, drift, governance, promotion.
4. **DL × MLOps compounds.** A 412 M-parameter ViT under MLflow, multi-task heads, GPU serving,
   hourly drift on production traffic — each integration multiplies the surface area.
5. **Then — here's how Databricks AI Runtime collapses it.** Frozen Vision FMs + Mosaic AI
   serving + AI Gateway + Unity Catalog turn the four-layer mess into one bundle command and three
   jobs sharing one backbone artifact.
6. **The example.** Dental X-ray detection on DENTEX with C-RADIOv4-SO400M, where the same frozen
   backbone powers detection, similarity search, and drift — and AIR removes every pain we just
   listed.

---

## Timing overview

| Segment | Time | Type | Notebook / Slide |
|---------|------|------|-----------------|
| Hook: live dental X-ray | 0:00 – 0:03 | Live demo | `01_explore_dentex.py` |
| Pain layer 1: DL is hard | 0:03 – 0:06 | Slides | Slides 3-5 |
| Pain layer 2: Distributed DL at scale is harder | 0:06 – 0:09 | Slides | Slides 6-8 |
| Pain layer 3: MLOps is hard on its own | 0:09 – 0:12 | Slides | Slides 9-10 |
| Pain layer 4: DL × MLOps compounds | 0:12 – 0:14 | Slides | Slide 11 |
| The reframe: how Databricks + AIR helps | 0:14 – 0:18 | Slides | Slides 12-15 |
| Live demo 1: detection fine-tune (frozen backbone, AIR `@distributed`) | 0:18 – 0:25 | Live demo | `02_train_detector_air.py` |
| Eval beyond mAP: similarity search via the same backbone | 0:25 – 0:30 | Live demo | `06_similarity_search_demo.py`, `03_precompute_embeddings.py` |
| Live demo 2: drift sensor on inference traffic | 0:30 – 0:36 | Live demo | `05_drift_demo.py` |
| Serving + runtime: two-phase deploy, AI Gateway, live curl | 0:36 – 0:42 | Slides + terminal | Slides 16-20 |
| The reframe: one backbone, three roles | 0:42 – 0:44 | Slides | Slides 21-22 |
| Q&A teaser + repo link | 0:44 – 0:45 | Slides | Slide 23 |

Each live segment has a pre-recorded backup (`seg{N}.mp4`). The switch-to-video procedure is in
[RUNBOOK.md](RUNBOOK.md#switch-to-video-procedure); two consecutive `latency_probe` failures
trigger it.

---

## Segment detail

### 0:00 – 0:03 | Hook — show DL works before talking about why it's hard

**Goal:** Land a visceral "DL works" moment in three minutes, before the pain arc starts. No
slides. Open directly with a notebook.

**Notebook:** `notebooks/01_explore_dentex.py` (cells have cached outputs committed)

1. Display a panoramic dental X-ray with no annotations.
2. Ask the audience: "How many carious lesions do you see?"
3. Wait 10 seconds — let the room squint.
4. Run the model prediction cell (cached output from the deployed detection endpoint).
5. Overlay bounding boxes: Caries, Deep Caries, Periapical Lesion, Impacted.
6. **Key fact to land:**
   > "The model found 7. One backbone — NVIDIA C-RADIOv4-SO400M, 412 million parameters,
   > **frozen** — plus a 5-million-parameter detection head, trained in under 20 minutes on a
   > single H100 over 705 X-rays. By the end of this talk you'll have a public repo that
   > reproduces this with two `databricks bundle` commands."

**Backup:** Cached output is committed; "Run All" still works if the endpoint is down. If even
that fails, switch to `seg1_hook.mp4` in OBS.

**Transition line:** "That looked easy. It wasn't. Let's count the reasons why."

---

### 0:03 – 0:06 | Pain layer 1: DL is hard (even on one GPU)

**Goal:** Establish that DL alone, at single-GPU scale, is already a tower of failure modes. No
demo. The audience needs to feel the floor before we add ceilings.

**Slide 3 — DL is hard, before we even talk about scale:**
- **Data hunger.** ImageNet was 1.3 M labeled images. DENTEX is 705 train + 50 val + 250 test.
  Off-the-shelf supervised training collapses on this.
- **Hyperparameter brittleness.** Learning rate, batch size, augmentation pipeline, optimizer,
  warmup, weight decay — every knob shifts mAP by 5-15 points. A wrong anchor scale silently
  caps your AP at 0.2.
- **GPU memory wall.** A 412 M-parameter ViT in FP32, plus gradients, plus optimizer state, plus
  activations at 1024×1024 doesn't fit on one H100 if you're full-fine-tuning. OOM is the
  default, not the exception.
- **Silent correctness bugs.** A wrong patch size, a wrong normalization stat, a swapped
  channel order — the model still trains, the loss still goes down, the mAP just stays at zero.

**Slide 4 — The one decision that makes this tractable:**
- **Freeze the backbone.** Vision FMs (DINOv2, DINOv3, C-RADIOv4) are already trained on
  hundreds of millions of images. We don't need their gradients. We need their features.
- Only the **5 M-parameter head** trains. The 412 M-parameter backbone is a fixed feature
  extractor. OOM disappears, training time drops 10×, and accuracy goes UP because the head
  can't overfit a tiny medical dataset.

**Slide 5 — The C-RADIOv4 BackboneInfo contract (load-bearing):**
```
backbone(images) -> (summary, spatial_features)
                     |          |
                     |          +-- (B, T, 1152) per-patch  -> FPN -> RetinaNet head
                     +-- (B, 2304) pooled global -> embeddings, drift, Vector Search
```
- Two **distinct** outputs with distinct dims: `spatial_features` keep the 1152 ViT hidden
  dim; `summary` is RADIO's pooled global feature at 2304 (2×1152).
- Everything downstream parameterizes on `backbone_info.summary_dim` /
  `backbone_info.spatial_dim` — never hardcoded. Hardcoding a dimension anywhere outside
  `models/backbones.py` is a bug.

**Transition line:** "OK, one GPU is hard. Now let's add seven more."

---

### 0:06 – 0:09 | Pain layer 2: Distributed DL at scale is harder

**Goal:** Convey that going from one GPU to eight (and from eight to multi-node) is not a
linear cost increase — it's a step-function in failure modes. No demo; this is the fear part.

**Slide 6 — Things that go wrong at 8 H100s that don't go wrong at 1:**
- **NCCL silent hangs.** A peer rank crashes; the survivors hang on `barrier()` indefinitely.
  No error, no log line, just a stalled job 90 minutes in.
- **Cold-cache HuggingFace race.** Eight ranks call `from_pretrained` simultaneously into a
  shared UC Volume cache. They all try to write the same file. The first writer wins; the rest
  get partial files; the next epoch reads garbage.
- **Environment propagation.** The driver sets `HF_HUB_ENABLE_HF_TRANSFER=0`. The eight
  serverless GPU workers run in fresh processes that **do not inherit it**. The default chunked
  downloader hits FUSE on the UC Volume, gets `os error 95`, the job fails 4 minutes in.
- **DDP gotchas with frozen modules.** A frozen backbone has zero-grad parameters. Default DDP
  expects all parameters to receive grads. You get a deadlock at `loss.backward()` unless you
  pass `find_unused_parameters=True`.
- **Cluster vs. notebook vs. CLI launch.** The same training code has to run from three
  surfaces — interactive notebook, scheduled job, terminal CLI — and they all need identical
  hyperparameters, identical env, identical model artifact at the end.

**Slide 7 — How we fixed each one (named code anchors):**

| Failure mode | Fix in this repo |
|---|---|
| NCCL silent hang on dead rank | `distributed/primitives.py::safe_barrier` — bounded `wait()` over `dist.barrier(async_op=True)`; surfaces `BarrierTimeoutError` instead of hanging. |
| Cold-cache HF race | `distributed/barrier_dance.py::rank0_first` — sequence-matched; non-rank-0 hits its barrier first, rank 0 downloads, then both proceed. |
| Env propagation to workers | `platform/hf_env.py::configure_hf_env` — one canonical site for `HF_HUB_ENABLE_HF_TRANSFER=0` + `HF_HUB_DISABLE_XET=1`, called at the top of every worker entry. |
| DDP frozen-param deadlock | `train/trainer.py` — `find_unused_parameters=True` in the DDP wrap. |
| Three launch surfaces, one truth | `config/trainer_config.py::TrainerConfig` — frozen dataclass; same instance feeds notebook `@distributed` AND air/torchrun. |

**Slide 8 — The launcher you don't have to write:**
- AIR's `serverless_gpu.@distributed` decorator handles cluster bring-up, NCCL bootstrapping,
  rendezvous, and per-rank result collection. Eight H100s, one decorator.
- Compare to: write a Kubernetes Job spec, install NVIDIA device plugin, build a CUDA container,
  expose torchrun rendezvous, wire MASTER_ADDR/MASTER_PORT, plumb logs to S3, hook MLflow into
  every container. Ouch.

**Transition line:** "Cool. Now let's say you actually shipped a model. Welcome to MLOps."

---

### 0:09 – 0:12 | Pain layer 3: MLOps is hard on its own

**Goal:** Make the audience feel that even *without* deep learning, the operational stack —
registry, deployment, lineage, drift, promotion, audit — is non-trivial. No demo.

**Slide 9 — MLOps surface area (model-agnostic):**
- **Registry.** Where does the model live? How is it versioned? Who can read it? When does it
  retire? `@candidate` vs. `@champion` vs. `@archived` — and who flips the alias?
- **Deployment.** What runs the model? CPU? GPU? What size? What auth? What scale-to-zero?
  How do you canary? How do you roll back?
- **Lineage.** Which dataset version trained which model version, served by which endpoint, with
  which preprocessor — and at what code SHA?
- **Drift.** Production traffic shifts. How do you detect it? Where do you store inference
  payloads to score later? Without adding latency to the hot path?
- **Promotion gating.** New version must beat champion on a smoke test before it serves
  production traffic. Where does that smoke test live? Who runs it? What if it half-passes?
- **Governance.** Auditor walks in: "Show me which model is serving, on which data, registered
  by whom, against which experiment, with which artifact hashes." Can you answer in one query?

**Slide 10 — The naïve stack to do all that:**
- MLflow tracking server (you host) + S3 artifacts (you manage) + custom registry (or hand-rolled
  conventions) + Triton container (you build) + Kubernetes (you operate) + Prometheus (drift
  metrics, you wire) + a separate audit pipeline (you write).
- Seven systems. Seven sets of credentials. Seven failure modes. No single source of truth.

**Transition line:** "Now multiply this by deep learning."

---

### 0:12 – 0:14 | Pain layer 4: DL × MLOps compounds

**Goal:** Land the multiplier. DL pain × MLOps pain isn't additive — it compounds.

**Slide 11 — Where the layers compound:**
- **Pyfunc dependencies.** A pyfunc that wraps a 412 M-parameter ViT needs `timm`, `einops`,
  `open_clip`, `transformers>=4.48`, plus `trust_remote_code=True` for C-RADIOv4. If your serving
  env lacks any of them, the endpoint comes up green but every request 500s on import. We hit
  this twice during dev.
- **Artifact contract.** Detector pyfunc loads `manifest.json` to know which backbone, which
  patch size, which spatial dim, which label map. If the manifest format drifts between training
  and serving, you get a silent dimension mismatch — same shape, wrong meaning.
- **Multi-task pyfunc cost.** One backbone artifact, three downstream consumers (detection,
  embeddings, drift). If each rebuilds the backbone from scratch, you've burned 3× the GPU
  warm-up budget and tripled your endpoint count.
- **Drift on a 1152-dim float vector** is not the same as drift on a tabular column. KNN, MMD,
  bootstrap CI — and all of it has to read AI Gateway's `STRING request` JSON column, decode
  base64 PNGs, re-embed via the same frozen backbone. Without adding latency to detection.
- **Endpoint promotion that depends on training.** `databricks bundle deploy` can't reference
  `@champion` before any version exists. So you can't ship endpoints declaratively as YAML on
  first deploy. You need an SDK-driven gate that fires *after* training succeeds and *only* if
  the smoke test passes.

**Transition line:** "All of that. In one repo. With two commands. Watch."

---

### 0:14 – 0:18 | Reframe: how Databricks + AIR collapses the stack

**Goal:** Show the audience the full answer before any demo runs. They should see the diagram
before they see the code.

**Slide 12 — The four pains, the four answers:**

| Pain | Databricks / AIR answer |
|------|-------------------------|
| DL is hard | Frozen VFM + tiny head; H100 single-card fits; mAP ≥ 0.45 in 20 min |
| Distributed DL is harder | `serverless_gpu.@distributed` + `safe_barrier` + `rank0_first` — one decorator, no cluster YAML |
| MLOps is hard | UC registry + Mosaic AI Model Serving + AI Gateway inference tables + Vector Search Delta Sync — one platform, one credential, one lineage graph |
| DL × MLOps compounds | One backbone artifact, three downstream consumers — `[tool.dais26.serving-deps]` is the SoT for pyfunc deps; `BackboneInfo` is the SoT for dimensions |

**Slide 13 — The system diagram (matches `docs/ARCHITECTURE.md`):**
```
                  Unity Catalog: main.mshtelma
                  +-------------------------+
                  | dentex_raw / model_cache (Volumes) |
                  | dais26_dentex_train_embeddings (Delta + CDF) |
                  | dais26_dentex_drift_scores (Delta) |
                  +-----------+-------------+
                              |
        +---------------------+---------------------+
        |                     |                     |
   AIR @distributed     AIR @distributed       AIR scheduled
   train_detector       precompute_embeddings  drift_monitor
   spatial_features     summary (2304)         summary (2304)
        |                     |                     |
        v                     v                     v
   UC model registry    Mosaic AI                Delta:
   cradio_detector      Vector Search            drift_scores
   @candidate->@champion (HNSW+L2, dim=2304)    (alert BOOLEAN)
        |
        v
   Mosaic AI Model Serving GPU endpoint
   AI Gateway -> dais26_dentex_detector_inference_payload (auto Delta)
        ^
        |
   drift_monitor reads inference table -> re-embeds via summary -> writes drift_scores
```

**Slide 14 — Why Databricks-native (not just "we like the brand"):**
- **UC lineage.** Dataset → model version → endpoint → inference table → drift score —
  governed and auditable as a single graph. No external tracking system to keep in sync.
- **AI Gateway inference tables.** Every request automatically logged to a Delta table. Drift
  monitor reads from there — no client-side instrumentation, no Kafka, no shadow logger.
- **Vector Search Delta Sync.** Embeddings table is the source; the index syncs from CDF. Zero
  ops to maintain.
- **Serverless GPU pool.** No clusters to manage, no idle cost, no driver/worker config.
- **One auth surface.** Same PAT/OAuth for the registry, the endpoint, the inference table, the
  Vector Search index, the drift monitor.

**Slide 15 — Two commands, two phases:**
```bash
databricks bundle deploy -t dev               # Phase 1: UC + jobs (no endpoints yet)
databricks bundle run train_detector -t dev   # Phase 2: train -> @candidate ->
                                              #          deploy endpoint (SDK) ->
                                              #          smoke test -> @champion
```
- Phase 1 is YAML-deployable — UC objects + job definitions only.
- Phase 2 is SDK-driven inside the training job — endpoint creation is gated on a registered
  model version *and* a passing smoke test.

**Transition line:** "Talk is cheap. Code:"

---

### 0:18 – 0:25 | Live demo 1: detection fine-tune on AIR

**Goal:** Show training on AIR H100. Demonstrate `@distributed`, MLflow autolog, air-lane parity.
Budget 7 minutes. Run 1-2 epochs live; switch to a pre-baked run for epochs 3-10.

**Notebook:** `notebooks/02_train_detector_air.py`

1. Open `notebooks/00_config.py` first. Call out:
   - `CATALOG = "mlops_pj"`, `SCHEMA = "dais26_vfm"`, `BACKBONE = "cradio_v4_so400m"`.
   - `TRAIN_EPOCHS = 50` (demo override; the recipe's full schedule is 150),
     `TRAIN_GPUS = 8`, `TRAIN_GPU_TYPE = "h100"`.
   - "This file is ENVIRONMENT only — catalog, schema, experiment, demo overrides.
     The hyperparameters live in one place, `config/recipes.py`: the campaign-final
     recipe per backbone. The notebook builds from it; the air workload names it.
     Neither lane can drift from the other."

2. Switch to `02_train_detector_air.py`. Walk through the four cells:
   - `%pip install --quiet ..` + `dbutils.library.restartPython()`.
   - `%run ./00_config` — pulls every constant into the notebook namespace and the
     `@distributed` closure.
   - `from serverless_gpu import distributed` — "**This is the only line that distinguishes a
     single-GPU notebook from an 8-H100 distributed training run.** No cluster YAML."
   - `@distributed(gpus=TRAIN_GPUS, gpu_type=TRAIN_GPU_TYPE)` over `run_train()` — call out
     that the body re-sets `HF_HUB_ENABLE_HF_TRANSFER=0` and `MLFLOW_EXPERIMENT_NAME` because
     **AIR workers don't inherit driver env**.

3. Start training. MLflow autolog shows loss per batch.
4. Talk over the training loop while epochs run:
   - "The recipe fine-tunes the full backbone at a discriminative LR (`backbone_lr=1e-5`
     vs head `2e-4`) — the campaign proved full FT beats frozen/LoRA here. The FPN adapter
     (`in_channels = backbone_info.spatial_dim`) and RetinaNet head train at the head LR."
   - "FPN takes `spatial_features` (B, T, 1152), reshapes to (B, 1152, 64, 64), produces P3-P6
     feature maps."
   - "RetinaNet — focal loss (alpha=0.25, gamma=2.0) for class imbalance, smooth-L1 for box
     regression, NMS at 0.5."
   - "There's one training core: `Trainer` in `src/dais26_dentex/train/trainer.py`. The notebook
     dispatches it via `serverless_gpu.@distributed`. **the AIR CLI runs the same core via
     torchrun** — `air/workload_train_detector.yaml` just says `recipe: cradio_v4_so400m`
     and `env: df1`; the CLI resolves the identical recipe (hyperparameters) and env (UC
     locations) through `$HYPERPARAMETERS_PATH` — the SAME two names `00_config.py` selects.
     Two surfaces, one core, one recipe, one env. Even the HPO sweep runs on both:
     `campaign_sweep` (DAB) and `workload_sweep.yaml` (terminal) drive the same `SweepRunner`."
5. After 2 epochs: switch to the pre-baked MLflow run (epochs 3-10 pre-logged).
6. Show the val/mAP@50 curve. Target: ≥ 0.45 after 10 epochs.
7. Show the UC model registration: `mlops_pj.dais26_vfm.cradio_detector`, `@challenger` alias.
8. Call out the rank-0-only MLflow logic in `Trainer._save_and_register`: only rank 0 logs the
   pyfunc, sets the alias, and returns the run_id. All other ranks return `None`. The
   `serving_pip_requirements` it logs comes from `[tool.dais26.serving-deps]` in
   `pyproject.toml` — which **ships inside the wheel** as `dais26_dentex/_pyproject.toml` so
   AIR's ephemeral env can read it via `importlib.resources`. CI guards drift via
   `assert_serving_reqs_match_pyproject`.

**Key slide to call out (Slide 16, two-phase deploy):** Already shown on Slide 15, but pause
here to mark "we're between phases right now — training has registered `@candidate`; the next
task in this DAB job will deploy the endpoint and gate `@champion` on the smoke test."

**Backup:** Switch to `seg2_detection.mp4` if AIR cluster start exceeds 90 s.

**Pain → answer mapping (call out aloud during the demo):**
- "Cold-cache HF race?" → `rank0_first` already wraps `build_detector`.
- "NCCL hang on dead rank?" → `safe_barrier` would surface `BarrierTimeoutError`.
- "Env propagation?" → `configure_hf_env` plus the in-worker `os.environ` lines.
- "Cluster bring-up?" → there isn't one. `@distributed` did it.

---

### 0:25 – 0:30 | Eval beyond mAP — same backbone, different output

**Goal:** Show that the same frozen backbone powers similarity search via its `summary`
embeddings — without retraining, without a second model artifact. Demonstrate Vector Search
top-10 plus a UMAP scatter.

**Notebooks:**
- `notebooks/06_similarity_search_demo.py` — Vector Search top-10 + same-class recall
- `notebooks/03_precompute_embeddings.py` — UMAP scatter (pre-baked output)

1. Open `06_similarity_search_demo.py`. Show a query X-ray with a Periapical Lesion.
2. Run the VS query loop:
   ```python
   res = w.vector_search_indexes.query_index(
       index_name=VS_INDEX_NAME,                # main.mshtelma.dais26_dentex_embeddings_index
       columns=["image_id", "diagnosis", "split"],
       query_vector=query_vec,                  # summary, dim=2304
       num_results=10,
   )
   ```
3. Display the top-10 images. Most should share the query's diagnosis.
4. "The embedding is `summary` — dim 2304 (RADIO pools two reps, 2×1152), L2-normalized. The
   Vector Search index holds 1005 embeddings, one per DENTEX image, HNSW+L2."
5. **Same-class recall**: target `recall@10 ≥ 0.80` per `tests/integration/E7`. The notebook
   prints the live number.
6. Switch to `03_precompute_embeddings.py` (pre-baked UMAP cell). Show the scatter plot:
   embeddings colored by diagnosis form visible clusters.
7. Land the reframe inline:
   > "Same backbone. Different output tensor. Different downstream artifact. `summary` for
   > semantic similarity. `spatial_features` for pixel-level detection. **One frozen UC artifact
   > funds two products.**"

**Key slide (Slide 17):** Vector Search Delta Sync diagram.
- Source: `dais26_dentex_train_embeddings` Delta table (CDF=on, ARRAY<FLOAT> dim=2304).
- Sync mode: `TRIGGERED` (manual or post-write).
- Index type: `DELTA_SYNC`, embedding dim=2304 (derived from the table, not hardcoded),
  primary_key=`image_id`.
- "Embedding pipeline is just a Delta table write — VS handles indexing, syncing, and serving."

**Backup:** Switch to `seg3_similarity.mp4`.

---

### 0:30 – 0:36 | Live demo 2: drift sensor on production traffic

**Goal:** Show that the same backbone detects distribution shift on detector traffic, **without
adding latency to detection requests**. The drift signal comes from re-embedding the AI Gateway
inference table on a separate hourly job.

**Notebook:** `notebooks/05_drift_demo.py` (set `DRIFT_MODE = "demo"` in `00_config.py`)

1. Run the "demo" mode cell:
   - 25 clean val images vs. 25 synthetically shifted (contrast=0.5, gamma=2.0).
   - Re-embed all 50 via C-RADIOv4 `summary` (dim 2304), L2-normalized.
   - Reference embeddings come from `dais26_dentex_train_embeddings` (Delta).
   - Compute KNN distance (k=50) and bootstrap 95% CI (1000 iterations).
2. Display the drift score bar: clean batch vs. shifted batch.
3. **Acceptance numbers (E5, E6):**
   - Ratio (shifted / clean) ≥ 2.0 → pass.
   - Bootstrap 95% CI lower bound > clean baseline → pass.
4. Switch to `DRIFT_MODE = "scheduled"` to show the production path:
   - `dais26_dentex.drift.monitor.run_drift_monitor` reads the AI Gateway inference table
     (`dais26_dentex_detector_inference_payload`) — STRING `request` column, JSON-encoded.
   - Parses `dataframe_split` / `dataframe_records` shapes; skips NULL rows (>1 MiB cap).
   - Re-embeds via the frozen C-RADIOv4 `summary` head.
   - Writes `knn_distance`, `mmd_score`, `alert` to `dais26_dentex_drift_scores`.
5. Land the architectural punchline:
   > "Drift computation runs on a separate hourly job. **Zero added latency to detection
   > requests.** The detection endpoint serves on GPU_SMALL; the drift sensor is a scheduled
   > AIR notebook task. Same backbone artifact, two consumers, one Unity Catalog."

**Key slide (Slide 18):** Drift pipeline diagram.
- Detection endpoint -> AI Gateway -> Delta inference table.
- Drift monitor (scheduled, AIR) -> read STRING request -> decode b64 -> embed via `summary` ->
  KNN distance vs. reference -> Delta `drift_scores` -> Lakehouse Monitoring (alert layer).

**Backup:** Switch to `seg4_drift.mp4`.

---

### 0:36 – 0:42 | Serving + runtime — two-phase deploy, AI Gateway, live curl

**Goal:** Close the loop on the MLOps-pain answers. Show the live endpoint, the inference table,
and the latency numbers.

**Slide 19 — Two-phase deploy (annotated):**
- Phase 1: `bundle deploy` -> UC catalog/schema/volumes, MLflow experiment, job definitions,
  secret scope. **No endpoints.**
- Phase 2: `bundle run train_detector` -> `setup` -> `train` -> `deploy_endpoint`.
- The `deploy_endpoint` task switches on `DEPLOY_ACTION`:
  - `register_and_set_candidate` → verify `@candidate` exists and exit.
  - `deploy_and_smoke_test` → resolve `@candidate` to numeric version N → `create_and_wait` (or
    `update_config_and_wait`) → poll READY → smoke-test 1 image → `@champion = N` on success.
  - On failure, `@champion` stays on the previous version. Captured `previous_champion` is
    returned for rollback.

**Slide 20 — Why SDK-driven (not declarative YAML):**
- `databricks bundle deploy` cannot reference `@champion` before any model version exists.
- A declarative endpoint resource on first deploy fails with "alias not found."
- SDK-driven means the endpoint is created **after** training, **gated on** a passing smoke
  test. The endpoint never exists in a broken state.
- `ai_gateway` is a **top-level sibling of `config`** in `create_and_wait` — nesting it under
  `config` is a silent no-op. (Real bug we hit. Documented in `endpoint_manager.py`.)

**Slide 21 — Latency table (Phase 4 numbers, populated from `docs/BENCHMARKS.md`):**
- p50 / p95 / p99 at GPU_SMALL, batch=1, 1024×1024.
- Pivot ladder if p99 > 150 ms: 1024→768, GPU_SMALL→GPU_MEDIUM, FP16-only.

**Live terminal — curl against the running endpoint:**
```bash
curl -X POST \
  -H "Authorization: Bearer $DATABRICKS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"dataframe_split": {"columns": ["image"], "data": [["'"$IMG_B64"'"]]}}' \
  "https://$DATABRICKS_HOST/serving-endpoints/dais26-cradio-detector-dev/invocations"
```
Expected response:
```json
{"predictions": [{"boxes": [[...]], "scores": [...], "labels": ["Caries", ...],
                  "num_detections": 7}]}
```

**Then immediately after:**
```sql
SELECT request_time, request, response
FROM main.mshtelma.dais26_dentex_detector_inference_payload
ORDER BY request_time DESC
LIMIT 5;
```
- "That row was written by AI Gateway. We didn't instrument the client. We didn't deploy a side
  car. **It's just a Delta table.** That's the table the drift monitor reads."

**Backup:** Switch to `seg5_serving.mp4`.

---

### 0:42 – 0:44 | The reframe — one backbone, three roles

**Slide 22 — The summary table:**
| Role | Backbone output | Artifact | UC location |
|------|----------------|----------|-------------|
| Detection head | `spatial_features` (B, T, 1152) | Mosaic AI serving endpoint | `dais26-cradio-detector-dev` |
| Embedding service | `summary` (B, 2304) | Vector Search index + Delta table | `…dais26_dentex_embeddings_index` |
| Drift sensor | `summary` (B, 2304) | `drift_scores` Delta table | `…dais26_dentex_drift_scores` |

**Slide 23 — The four pains, the four wins:**
- DL is hard → **frozen VFM + tiny head** beats full fine-tuning at this scale, in 20 min on
  one H100.
- Distributed DL is harder → **`@distributed` + `safe_barrier` + `rank0_first`** turn 8 H100s
  into one decorator.
- MLOps is hard → **UC + Mosaic AI Serving + AI Gateway + Vector Search** is one auth, one
  lineage graph, one bundle command.
- DL × MLOps compounds → **one backbone artifact funds three downstream consumers**, none of
  which retrain or duplicate the backbone.

**Land the close:**
> "`databricks bundle deploy -t dev` and `databricks bundle run train_detector -t dev` is all
> it takes. The repo is public as of today. C-RADIOv4 is ungated. DENTEX is open. There is
> nothing here you can't reproduce by Monday."

---

### 0:44 – 0:45 | Q&A teaser + repo link

**Slide 24:**
- GitHub repo URL (large, readable from the back of the room)
- QR code linking to the repo
- "Questions? I'll be at the sponsor booth this afternoon."

Common expected questions and suggested answers:

| Question | Suggested answer |
|----------|-----------------|
| Why C-RADIOv4 and not DINOv3? | DINOv3 is gated; attendees can't reproduce without HuggingFace approval. C-RADIOv4 is ungated under the NVIDIA Open Model License (commercial OK). DINOv3 is in the repo as a comparison only. |
| Why frozen backbone, not full fine-tune? | OOM on a single H100 at 412 M params + grads + optimizer + 1024² activations. Frozen backbone trains in 20 min, ≥0.45 mAP. LoRA is in the repo as a stretch path (`TRAIN_USE_LORA=True`). |
| Why SDK-driven endpoint instead of YAML? | `bundle deploy` can't reference `@champion` before any version exists. SDK is gated on training success **and** smoke-test pass. Endpoints never exist in broken state. |
| Can this run outside AIR? | Yes — `databricks bundle deploy -t dev_non_air` substitutes standard DBR ML GPU runtime. Node defaults: AWS `g5.12xlarge`, Azure `Standard_NC24ads_A100_v4`, GCP `a2-highgpu-1g`. Endpoints are independent of AIR. |
| Commercial use of DENTEX? | CC-BY-NC-SA — research and demo only. No commercial use. This is a demo dataset, not a production medical dataset. |
| How does drift not add latency? | Detection runs on the GPU_SMALL serving endpoint. Drift is a separate hourly AIR notebook task that reads the AI Gateway inference Delta table. The detection hot path never sees the drift code. |
| What about retraining when drift fires? | Out of scope today, but the hooks exist: `drift_scores.alert` is a `BOOLEAN`; Lakehouse Monitoring can trigger on it; the same `train_detector` job re-runs end-to-end. |

---

## Key slides to not skip under any time pressure

| Slide | Content | Why critical |
|-------|---------|-------------|
| Slide 4 | Frozen-backbone decision | The single architectural choice that makes this tractable. |
| Slide 5 | BackboneInfo contract | summary=2304 (pooled, 2×1152), spatial=1152 (per-patch). Wrong tensor → wrong artifacts. |
| Slide 7 | Failure-mode → fix table | Concrete, named code anchors. Audience leaves with grep targets. |
| Slide 13 | System diagram | The reframe slide before the demos. If they only see one slide, this is it. |
| Slide 19 | Two-phase deploy | Explains why endpoints are SDK-driven, not YAML. |
| Slide 21 | Latency table | Real numbers from Phase 4 — credibility anchor. |
| Slide 22 | One-backbone-three-roles table | The talk's title in one image. |

---

## License & ethics callouts (must land verbally on Slide 4)

- **DENTEX dataset:** CC-BY-NC-SA 4.0. Research and demo only. **No commercial use.**
- **C-RADIOv4-SO400M weights:** NVIDIA Open Model License. Commercial use permitted. Ungated.
- **DINOv3 weights:** custom `dinov3-license`. Gated. Comparison only.
- **No trained weights** are stored in this repo. Weights are pulled from HuggingFace at runtime
  and cached in the `model_cache` UC Volume (pinned by SHA via
  `scripts/pin_model_cache.py`).

---

## Demo environment setup (talk day)

Run these before going on stage:

```bash
# 30 minutes before talk
python scripts/warmup_endpoints.py            # warm up dais26-cradio-detector-dev

# Continuous, in background, kill on stage exit
bash scripts/latency_probe.sh &               # 2 consecutive failures -> switch to video

# Confirm cached cells in each demo notebook show outputs without re-running
# (open in Databricks, scroll, do not Run All)

# Verify endpoint state
databricks serving-endpoints get dais26-cradio-detector-dev | jq .state

# Pre-load 5 backup videos in OBS (seg1_hook .. seg5_serving)
```

If 2 consecutive `latency_probe` failures fire during the talk, follow the switch-to-video
procedure in [RUNBOOK.md](RUNBOOK.md#switch-to-video-procedure).

---

## Pre-talk dry-run checklist (D-1)

- [ ] Full end-to-end deploy on a fresh dev target completes < 60 min from `bundle deploy` to
      `@champion` set.
- [ ] All 8 demo notebooks have committed cached outputs visible without re-running.
- [ ] All 5 backup videos exist, are < 90 s each, and play correctly in OBS.
- [ ] `latency_probe.sh` is green for 30 minutes uninterrupted.
- [ ] `docs/BENCHMARKS.md` is fully populated (no `TBD`).
- [ ] `discover_air_runtime.py` matches the cluster the demo will use (no surprise runtime
      version drift).
- [ ] Slides 4 (license), 13 (system diagram), 22 (reframe table) are reviewed and rehearsed —
      these are the must-land slides if the live demos fail.
- [ ] QR code on Slide 24 resolves to the public repo URL.
