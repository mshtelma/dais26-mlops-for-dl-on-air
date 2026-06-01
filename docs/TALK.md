# Talk outline: One frozen backbone, three jobs

**Event:** Data + AI Summit 2026 (DAIS26), San Francisco, June 15-18
**Duration:** 45 minutes
**Subtitle:** Detection head, drift sensor, embedding service — one C-RADIOv4-SO400M backbone

---

## Timing overview

| Segment | Time | Type | Notebook / Slide |
|---------|------|------|-----------------|
| Hook: live dental X-ray | 0:00 – 0:03 | Live demo | `01_explore_dentex.py` |
| Why VFMs now | 0:03 – 0:08 | Slides only | Slides 3-7 |
| Live demo 1: detection fine-tune | 0:08 – 0:15 | Live demo | `02_train_detector_air.py` |
| Eval beyond mAP | 0:15 – 0:22 | Live demo | `06_similarity_search_demo.py`, `03_precompute_embeddings.py` |
| Live demo 2: drift sensor | 0:22 – 0:30 | Live demo | `05_drift_demo.py` |
| Serving + runtime | 0:30 – 0:38 | Slides + live curl | Slides 14-18 + terminal |
| The reframe | 0:38 – 0:42 | Slides | Slides 19-21 |
| Q&A teaser + repo link | 0:42 – 0:45 | Slides | Slide 22 |

---

## Segment detail

### 0:00 – 0:03 | Hook

**Goal:** Grab attention in 3 minutes. No slides. Open directly with a notebook.

**Notebook:** `notebooks/01_explore_dentex.py` (cells have cached outputs committed)

1. Display a panoramic dental X-ray with no annotations.
2. Ask the audience: "How many carious lesions do you see?"
3. Wait 10 seconds.
4. Run the model prediction cell (uses cached output from the detection endpoint).
5. Overlay bounding boxes: Caries, Deep Caries, Periapical Lesion, Impacted.
6. "The model found 7. One backbone, frozen, trained in 15 minutes on 705 X-rays."

**Backup:** If the notebook cell fails, the cached output image is committed — "Run All" shows the
result. If even that fails, switch to `seg1_hook.mp4` in OBS.

**Key fact to land:** "No backbone gradients. The 412-million-parameter C-RADIOv4 never updates.
Only the 5-million-parameter detection head learns."

---

### 0:03 – 0:08 | Why VFMs now

**Goal:** Give the audience context and motivate the design choices. No live demo.

**Slides 3-7:**

- **Slide 3:** VFM landscape (2024-2026). DINOv2 → DINOv3 → C-RADIOv4. Positioning: C-RADIOv4
  is ungated and commercially OK; DINOv3 is gated and requires HF approval.

- **Slide 4 (CRITICAL — license slide):**
  - DENTEX dataset: CC-BY-NC-SA 4.0. Research and demo only. **No commercial use.**
  - C-RADIOv4 weights: NVIDIA Open Model License. Commercial use permitted.
  - DINOv3 weights: custom `dinov3-license`. Gated. Comparison only.
  - "No trained weights are in the repo. You download them at runtime."

- **Slide 5:** BackboneInfo contract diagram.
  - `summary` → shape `(B, 1152)` — global image feature. Used for embeddings, drift, VS.
  - `spatial_features` → shape `(B, T, 1152)` — patch features. Used for detection FPN.
  - "Same hidden dim (1152), but two distinct outputs — `summary` is pooled, `spatial_features` is
    per-patch. Everything downstream parameterizes on `backbone_info.summary_dim` or
    `backbone_info.spatial_dim` — never hardcoded."

- **Slide 6:** The three-jobs architecture diagram (matches ARCHITECTURE.md system overview).
  One UC Volume. Three consumers. One frozen backbone artifact.

- **Slide 7:** Why Databricks-native?
  - UC lineage from data to model to endpoint.
  - AI Gateway inference tables: every request logged automatically.
  - Mosaic AI Vector Search: Delta Sync, governed, zero-ops.
  - No custom containers, no BYO orchestration.

---

### 0:08 – 0:15 | Live demo 1: detection fine-tune

**Goal:** Show training on AIR H100. Demonstrate MLflow logging, loss curves, sample predictions.
Budget 7 minutes. Run 1-2 epochs live; show pre-baked epochs 3-10.

**Notebook:** `notebooks/02_train_detector_air.py`

1. Open `notebooks/00_config.py` first to call out the params: `BACKBONE_NAME=nvidia/C-RADIOv4-SO400M`,
   `TRAIN_EPOCHS=2` (live override for the demo), then we'll load a pre-baked run for epochs 3-10.
   "No widgets, no DAB `base_parameters` — one config file is the only place params live."
2. Start training. MLflow autolog shows loss per batch.
3. Talk over the training loop while epochs run:
   - "The backbone is frozen. We're only updating the FPN adapter and RetinaNet head."
   - "FPN takes `spatial_features` (dim 1152), produces 4 feature maps: P3-P6."
   - "RetinaNet applies focal loss — designed for class imbalance."
   - "There's one training core — `Trainer` in `src/dais26_dentex/train/trainer.py`. The notebook
     dispatches it via `serverless_gpu.@distributed`; sgcli runs the same core via `torchrun`.
     Same `TrainerConfig` dataclass feeds both surfaces."
4. After 2 epochs: switch to the pre-baked MLflow run (epochs 3-10 pre-logged).
5. Show the val/mAP@50 curve. Target: ≥ 0.45 after 10 epochs.
6. Show the MLflow model registration: `ml.dais26_vfm.cradio_detector`, `@champion` alias.

**Key slide to call out (Slide 10):** Two-phase deploy diagram.
- Phase 1: `bundle deploy` — UC + jobs only.
- Phase 2: `bundle run train_detector` — trains → `@candidate` → deploy endpoint (SDK) →
  smoke test → `@champion`.
- "The endpoint is never created before a model version exists."

**Backup:** Switch to `seg2_detection.mp4` if the AIR cluster fails to start.

---

### 0:15 – 0:22 | Eval beyond mAP

**Goal:** Show that the same backbone powers similarity search via its `summary` embeddings.
Demonstrate Vector Search top-10 and a UMAP of the embedding space.

**Notebooks:**
- `notebooks/06_similarity_search_demo.py` — Vector Search top-10
- `notebooks/03_precompute_embeddings.py` — UMAP scatter (pre-baked output)

1. Open `06_similarity_search_demo.py`. Show a query X-ray with a Periapical Lesion.
2. Run the VS query cell:
   ```python
   results = w.vector_search_indexes.query_index(
       index_name="ml.dais26_vfm.embeddings_index",
       columns=["image_id", "diagnosis"],
       query_vector=query_embedding,   # summary dim=1152
       num_results=10,
   )
   ```
3. Display the top-10 images. Most should be Periapical Lesion (same diagnosis).
4. "The embedding is `summary` — dim 1152. The Vector Search index holds 1005 embeddings, one
   per training image, HNSW+L2."
5. Switch to `03_precompute_embeddings.py` (pre-baked UMAP cell). Show the scatter plot:
   embeddings colored by diagnosis form visible clusters.
6. "Same backbone as the detector. Different output tensor. Different downstream artifact.
   `summary` for semantic similarity; `spatial_features` for pixel-level detection."

**Key slide (Slide 12):** BackboneInfo contract revisited. `summary` (1152, pooled) and `spatial_features` (1152, per-patch) — same hidden dim, distinct outputs.

**Backup:** Switch to `seg3_similarity.mp4`.

---

### 0:22 – 0:30 | Live demo 2: drift sensor

**Goal:** Show that the same backbone detects distribution shift in detector traffic, without adding
any latency to detection requests.

**Notebook:** `notebooks/05_drift_demo.py`

1. Run the "demo" mode cell. This:
   - Takes 50 clean val images and 50 synthetically shifted images (contrast=0.5, gamma=2.0).
   - Re-embeds all 100 images via C-RADIOv4 `summary` (dim 1152).
   - Computes KNN distance (k=50) against the 705-image training reference.
2. Display the KNN distance bar chart: clean batch vs. shifted batch.
3. "The shifted batch has a KNN distance ≥ 2× the clean batch. The 95% bootstrap CI excludes zero."
4. "The drift monitor runs hourly. It reads the AI Gateway inference table — those STRING `request`
   JSON rows — re-embeds the images, and writes a score to a Delta table."
5. Show the architecture: drift monitor → inference table → re-embed → score → `drift_scores` Delta.
6. "Zero added latency to detection. The backbone runs in a separate hourly job."

**Key slide (Slide 15):** Drift pipeline diagram. Emphasize: inference table → re-embed via summary
→ KNN distance. Separate job, not on the detection hot path.

**Backup:** Switch to `seg4_drift.mp4`.

---

### 0:30 – 0:38 | Serving + runtime

**Goal:** Show the two-phase deploy story and demonstrate a live curl to the detection endpoint.
Introduce the Mosaic AI vs. alternatives comparison.

**Slides 14-18 + terminal:**

- **Slide 14:** Two-phase deploy diagram (same as Slide 10, revisited for serving context).
  - `bundle deploy` → UC + jobs (no YAML endpoint resources).
  - `bundle run train_detector` → trains → `@candidate` → SDK endpoint create → smoke test →
    `@champion`.

- **Slide 15:** `@candidate` → `@champion` promotion sequence diagram (matches ARCHITECTURE.md).
  - Resolve alias to numeric version.
  - `create_and_wait` (new) or `update_config_and_wait` (existing).
  - Wait for READY.
  - Smoke test.
  - Set `@champion` only on success.

- **Slide 16 (CRITICAL — latency table):** Final benchmark numbers from `docs/BENCHMARKS.md`.
  - p50 / p95 / p99 at GPU_SMALL, batch=1, 1024×1024.
  - Workload type chosen (GPU_SMALL or GPU_MEDIUM based on Phase 4 results).

- **Slide 17:** Mosaic AI vs. Triton / BentoML comparison table (matches ARCHITECTURE.md).
  Emphasize: UC lineage, AI Gateway inference tables, no custom containers, audience reproducibility.

- **Live terminal:** Run a curl to the detection endpoint.

  ```bash
  curl -X POST \
    -H "Authorization: Bearer $DATABRICKS_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"dataframe_split": {"columns": ["image"], "data": [["'"$IMG_B64"'"]]}}' \
    "https://$DATABRICKS_HOST/serving-endpoints/dais26-cradio-detector-prod/invocations"
  ```

  Expected response:
  ```json
  {"predictions": [{"boxes": [[...]], "scores": [...], "labels": ["Caries", ...]}]}
  ```

- **Slide 18:** AI Gateway inference table schema. Show `detector_inference_*` table in UC.
  "Every request is logged. The drift monitor reads this table."

**Backup:** Switch to `seg5_serving.mp4`.

---

### 0:38 – 0:42 | The reframe

**Goal:** Land the "one backbone, three roles" narrative. Show UC lineage.

**Slides 19-21:**

- **Slide 19:** The reframe slide.
  "One backbone artifact. One UC Volume. Three roles:"
  | Role | Backbone output | Artifact |
  |------|----------------|---------|
  | Detection head | `spatial_features` dim=1152 | Mosaic AI serving endpoint |
  | Embedding service | `summary` dim=1152 | VS index + Delta table |
  | Drift sensor | `summary` dim=1152 | `drift_scores` Delta table |

- **Slide 20:** UC lineage diagram.
  `dentex_raw` Volume → training job → `cradio_detector` model → serving endpoint → inference table
  → drift monitor → `drift_scores` table.
  "Complete lineage. Governed. Auditable. Reproducible."

- **Slide 21 (CRITICAL):** What to take home.
  - `databricks bundle deploy -t dev` then `databricks bundle run train_detector -t dev`
    is all it takes to reproduce this from scratch.
  - The repo is public as of today. Link on the next slide.
  - `BackboneInfo` is the pattern: one dataclass as the single source of truth for all
    dimension-dependent code.

---

### 0:42 – 0:45 | Q&A teaser + repo link

**Slide 22:**
- GitHub repo URL (large, readable from the back of the room)
- QR code linking to the repo
- "Questions? I'll be at the sponsor booth this afternoon."

Common expected questions and suggested answers:

| Question | Suggested answer |
|----------|-----------------|
| Why not DINOv3? | Gated model; attendees can't reproduce without HF approval. C-RADIOv4 is ungated and commercially OK. DINOv3 is in the repo as a comparison only. |
| Why not full fine-tune? | OOM on single H100 (412M params + gradients + optimizer). Frozen backbone with trainable head trains in 15 minutes and achieves ≥0.45 mAP. LoRA is in the repo as a stretch path. |
| Why SDK-driven endpoint? | `bundle deploy` can't reference `@champion` before any model version exists. SDK-driven deploy is gated on training completion, enabling smoke-test-before-promotion. |
| Can this run outside AIR? | Yes. Use `databricks bundle deploy -t dev_non_air`. Standard GPU cluster (g5.12xlarge on AWS, A100 on Azure/GCP). The serving endpoints are independent of AIR. |
| Commercial use of DENTEX? | CC-BY-NC-SA — research and demo only. No commercial use. This is a demo dataset, not a production medical dataset. |

---

## Key slides to not skip

| Slide | Content | Why critical |
|-------|---------|-------------|
| Slide 4 | License | DENTEX is CC-BY-NC-SA. Must be stated explicitly. |
| Slide 5 | BackboneInfo contract | summary=1152 (pooled), spatial=1152 (per-patch). Wrong tensor → wrong artifacts. |
| Slide 10 | Two-phase deploy | Explains why endpoints are SDK-driven, not YAML. |
| Slide 16 | Latency table | Actual benchmark numbers from Phase 4. |
| Slide 21 | What to take home | Reproducibility message + repo link. |

---

## Demo environment setup (talk day)

Run these before going on stage:

```bash
# 1. Warm up the endpoint (30 min before talk)
python scripts/warmup_endpoints.py

# 2. Start the latency probe (keep running in background)
bash scripts/latency_probe.sh &

# 3. Pre-load notebook outputs (verify cached cells are visible)
# Open each demo notebook in Databricks. Confirm cells show outputs without running.

# 4. Verify endpoint state
databricks serving-endpoints get dais26-cradio-detector-prod | jq .state

# 5. Load backup videos in OBS (6 segments)
```

If 2 consecutive probe failures fire during the talk, follow the switch-to-video procedure in
[RUNBOOK.md](RUNBOOK.md#switch-to-video-procedure).
