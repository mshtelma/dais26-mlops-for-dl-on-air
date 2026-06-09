# Deck brief: Deep Learning Breaks MLOps

> **Purpose of this file.** A self-contained slide-by-slide brief a deck-building agent (or a human)
> can turn into a presentation without re-reading the codebase. Every claim here is grounded in this
> repo (`docs/ARCHITECTURE.md`, `docs/HPO.md`, `docs/BENCHMARKS.md`, `databricks.yml`) and in the
> Databricks **Big Book of MLOps, 2nd edition** + the canonical
> [MLOps workflows on Databricks](https://docs.databricks.com/aws/en/machine-learning/mlops/mlops-workflow)
> doc. Where a number is still unverified, it is marked **[TBD]** — do not invent values.

---

## 1. Session metadata

| Field | Value |
|-------|-------|
| Title | **Deep Learning Breaks MLOps** |
| Subtitle | A reference architecture for DL MLOps on Databricks |
| Event | Data + AI Summit 2026 (in person) |
| Track (catalog) | Artificial Intelligence & Agents |
| Industries (catalog) | Energy & Utilities, Enterprise Technology, Financial Services |
| Skill level | Advanced |
| Duration | 45 minutes |
| Speaker | Puneet Jain |
| Repo | `github.com/mshtelma/dais26-mlops-for-dl-on-air` |

> **Catalog-metadata caveat.** The session is tagged *Agents* in the catalog, but the content is
> **vision / detection DL-MLOps** — there are no agents in the repo. Do **not** add agent slides.
> If the committee requires an agents tie-in, add a single bridge line (serving + monitoring +
> registry is the same backbone you'd put under an agent tool), nothing more.

---

## 2. Thesis & narrative spine

**Thesis:** Databricks publishes a canonical MLOps playbook (the Big Book). We follow it where the
physics of deep learning allows, and we **deliberately break it in four places** — each break forced
by a property of deep learning, not by convenience.

**The Big Book's four load-bearing assertions, and how DL bends them:**

1. *Deploy code, not models* (retrain per environment) → **adapts (sanctioned)**: a 412M-param model
   with a ~48h sweep is expensive + non-deterministic; we deploy **models**. This is **not a
   deviation** — the Big Book itself reserves *deploy-models* for "expensive training" and "one-off
   models" (see §8). Frame it as picking the *correct* Big-Book branch, not breaking the book.
2. *Three environments (dev/staging/prod catalogs)* → **bends**: we run 2 schemas in one catalog,
   one workspace. Defensible for a single team; say so.
3. *Champion/Challenger via alias flip; compare offline or online A/B* → **holds but insufficient**:
   offline metric ≠ served metric for DL. Alias flip alone is unsafe.
4. *Monitor inference tables for drift AND accuracy → trigger retraining* → **breaks**: imaging
   labels are delayed/absent; we monitor **representation drift** as a leading indicator; retraining
   is still manual.

**Recurring motif (use as the connective tissue between war stories):**
*"Every place deep learning broke our MLOps was an integration boundary, not the model."*

**The single most important takeaway:** **train-time metric ≠ served metric.** Two real war stories
in this repo prove it; both belong on slides.

---

## 3. Grounded fact sheet (cite these; do not embellish)

**Model / data**
- Backbone: NVIDIA **C-RADIOv4-SO400M**, **412M params**, **frozen** by default; trainable head ≈ **5M params**.
- Backbone outputs two tensors: `summary` (dim **2304** = 2×1152) and `spatial_features` (dim **1152**).
- Dataset: **DENTEX** dental X-rays, **705 / 50 / 250** train/val/test, 4 classes (Caries, Deep Caries, Periapical Lesion, Impacted). License **CC-BY-NC-SA 4.0**, research/demo only.
- Compute: Databricks AI Runtime / **Serverless GPU** only (no traditional ML clusters). HPO sweep on **GPU_8xH100**, 48h timeout.

**Accuracy journey (val/mAP@50 unless noted)** — from `docs/HPO.md`
- Broken baseline: **~0.03** (random-box ceiling).
- After full fine-tune recipe fix: **0.335**.
- After per-level anchor + per-class NMS fix: **0.522** (C-RADIO), **0.518** (DINOv3).
- Push-to-0.60 campaign best: **C-RADIO 0.5931** (`dazzling-mole-850`, 150ep); **DINOv3 0.5738** (`capricious-hound-240`, fusion×150ep).

**War story #1 — registry / serialization** (`docs/HPO.md`)
- Registered DINOv3 served **0.027 mAP@50** vs ~0.53 in training — a serialization break invisible
  to training metrics. The 09b grid / serving re-eval caught it. The auto-`@candidate` aliases also
  landed on **sub-best** runs; the best C-RADIO model was left **unregistered**.

**War story #2 — train/serve skew** (`docs/HPO.md`)
- Training used aspect-preserving **letterbox**; serving pyfunc used anisotropic **squash**.
- Served mAP@50 **0.176** (bug) → **0.519** (fix); per-area split medium AP **0.90** vs large AP **0.05** was the smoking gun. Train-time was 0.522.

**Serving / deploy** (`docs/ARCHITECTURE.md`, `databricks.yml`)
- Two-phase: `bundle deploy` ships UC + jobs (**never** endpoints); endpoint is **SDK-driven** after training.
- Promotion: resolve `@challenger`/`@candidate` → **numeric** version → `create_and_wait` / `update_config_and_wait` on **GPU_SMALL** → poll READY → **smoke test one real image** → flip `@champion` only on success.
- `scale_to_zero=true` (dev) / `false` (prod). Zero-downtime via `update_config_and_wait`.
- Every request flows through **AI Gateway** → inference table `detector_inference_*`.
- Cross-schema champion: dev models backbone-keyed (`cradio_detector`, `dinov3_detector`) in `mlops_pj.dais26_vfm`; single backbone-agnostic prod `detector_champion` in `mlops_pj.dais26_vfm_prod`; promotion is a lineage-preserving `copy_model_version`, alias chain `@challenger → @champion_candidate → @champion`.
- Packaging: **models-from-code** (`serve/detector_model_script.py`) + `code_paths` (avoids `ModuleNotFoundError: transformers_modules` / `dais26_dentex`); backbone loads **offline** (`local_files_only`); `torch.compile` disabled at serving.

**Monitoring** (`docs/ARCHITECTURE.md`)
- Drift monitor re-embeds inference-table images via frozen `summary` (2304), computes **KNN (k=50) + MMD** vs the training reference, writes `drift_scores`, alerts at **2× baseline** via **Lakehouse Monitoring**. Runs as a **separate hourly job** → **zero added latency** to detection.
- No production accuracy monitoring (no prod labels). Retraining is **manual** today.

**Distributed-training hardening** (`docs/ARCHITECTURE.md`, README)
- `serverless_gpu.@distributed` (no clusters); one `Trainer` core shared by notebook `@distributed` and `torchrun`/`sgcli`; `TrainerConfig` single source of truth (no widgets).
- `rank0_first` (cold-cache HF download race), `safe_barrier` (dead-rank → `BarrierTimeoutError`, not a hang), `configure_hf_env` (`HF_HUB_ENABLE_HF_TRANSFER=0`, `HF_HUB_DISABLE_XET=1`; UC Volume FUSE rejects parallel chunked writes).
- MLflow 3 LoggedModel metric linkage; dataset lineage via `mlflow.log_input`.

**Unverified — mark [TBD], do not fabricate**
- Endpoint latency p50/p95/p99 (`docs/BENCHMARKS.md` still TBD).
- Training wall-clock time.
- **Traffic splitting is NOT implemented** in the repo (smoke-gated cutover only). Either build a
  90/10 served-entity split before claiming it, or omit it. Do not show a fake A/B.

---

## 4. Conformance scorecard (this is a hero slide)

🟢 follow · 🟡 adapt · 🔴 deliberately deviate

| Big Book element | Repo reality | Verdict |
|---|---|---|
| Deploy **code** (retrain per env) | Train once in dev; `copy_model_version` promotes the artifact | 🟡 We deploy **models** — the Big-Book-sanctioned branch for expensive training |
| 3 environments (dev/staging/prod) | 2 schemas, 1 catalog, 1 workspace, no staging | 🟡 |
| Champion/Challenger alias flip | `@challenger → @champion_candidate → @champion`, smoke-gated | 🟢 (extended, safer) |
| Validate on data slices | Per-class `Caries AP@50 ≥ 0.30` gate | 🟢 |
| Challenger ≥ champion | Eval on **test** + best-in-experiment + human approval | 🟢 |
| Online A/B traffic split | None (smoke-gated cutover) | 🔴 / gap |
| Zero-downtime update | `update_config_and_wait` | 🟢 |
| Monitor: drift | Embedding KNN/MMD | 🟢🔴 representation, not feature |
| Monitor: accuracy | None (no prod labels) | 🔴 |
| Retraining scheduled→triggered | Manual only | 🟡 gap |
| Lineage / reproducibility | models-from-code, register-from-LoggedModel, log_input, manifest | 🟢➕ exceeds the book |

---

## 5. Slide-by-slide spec

> Format per slide: **#. [slide-type] Title** — *on-screen content* — **Notes:** speaker track.
> `slide-type` values map to common deck primitives: `title, section, callout, icon-grid, agenda,
> timeline, comparison, two-column, cards, card-right, stat-row, three-column, checklist, closing`.

### Act 0 — Hook (0:00–0:03)

**1. [title] Deep Learning Breaks MLOps**
*Subtitle: "A reference architecture for DL MLOps on Databricks." Speaker + June 2026.*
**Notes:** Optional live cold-open: run `notebooks/01_explore_dentex.py`, show X-ray, ask "how many lesions?", reveal 7 boxes.

**2. [callout] "412M parameters frozen. 5M trained. Your MLOps playbook was written for the 5M."**
**Notes:** This is the whole talk in one line. The 412M C-RADIOv4 never updates; only the ~5M head learns.

**3. [icon-grid] Your playbook assumes small models**
*Four items: Multi-GB weights ("pickle.dumps was never the plan"); GPU-heavy training ("no .fit() on a laptop"); Distributed checkpoints ("rank-0, NCCL, barriers"); Fine-tuning loops ("frozen / LoRA / full — a search").*
**Notes:** Classical MLOps quietly assumes small + single-node + `pickle`. DL violates all three; you pay in downtime, drift, missed retraining.

### Act 1 — The canonical playbook (0:03–0:08)

**4. [agenda] Agenda**
*Items: The canonical playbook (Big Book); Pillar 1 training; Pillar 2 registry; Pillar 3 serving; Pillar 4 monitoring; Conformance scorecard.*

**5. [section-description] The Big Book of MLOps**
*Subtitle "Databricks' canonical reference architecture." Bullets: deploy code not models; dev→staging→prod; Champion/Challenger via UC aliases; monitor inference tables → trigger retraining.*
**Notes:** Authors Bradley/Kurlansik/Thomson/Turbitt, 2nd edition. This is the standard we'll measure against.

**6. [timeline] The 7-step production workflow**
*Steps: Train · Validate (slices+compliance → @challenger) · Deploy (compare vs champion, flip alias) · Serve (zero-downtime) · Infer (batch/stream/real-time) · Monitor (Lakehouse Monitoring: drift + accuracy) · Retrain (scheduled → triggered).*
**Notes:** "Here is the canonical answer. Now watch DL break it." Without this slide the hook is a strawman.

**7. [comparison] The fork every DL team faces**
*Left: "Deploy code (retrain per env)." Right: "Deploy models (promote artifact)."*
**Notes:** Big Book recommends deploy-code; reserves deploy-models for expensive/non-reproducible training — i.e. exactly large DL models.

**8. [callout] "DL bends three of the playbook's four core assumptions. Each bend is forced by physics, not laziness."**

### Act 2 — Pillar 1: Distributed training + tracking (0:08–0:17)

**9. [section] Pillar 1 — Distributed training + MLflow tracking**

**10. [two-column] Single-node .fit() doesn't exist here**
*Left "What breaks": 8×H100 DDP; NCCL barriers, dead-rank hangs; HF cold-cache races; UC Volume FUSE parallel writes. Right "What we built": serverless GPU @distributed (no clusters); one Trainer core (notebook + torchrun); TrainerConfig single source of truth; rank0_first + safe_barrier.*

**11. [icon-grid] Every failure was an integration boundary**
*rank0_first ("rank 0 downloads; peers wait on matched barriers"); safe_barrier ("dead rank → timeout error, not a hang"); configure_hf_env ("disable xet/transfer; FUSE rejects parallel writes").*
**Notes:** Each was a hang or silent fallback, not a model bug. Debuggability is an MLOps feature.

**12. [stat-row] Tracked, not guessed: the mAP journey**
*0.03 (broken baseline) · 0.52 (anchor fix) · 0.59 (schedule + fusion) · 8×H100 (HPO sweep).*
**Notes:** MLflow 3 LoggedModel metric linkage + dataset lineage make this a tracked experiment, not a guess. Live demo option: `notebooks/02_train_detector_air.py`, 1–2 epochs live then pre-baked.

### Act 3 — Pillar 2: Registry for large models (0:17–0:25)

**13. [section] Pillar 2 — Registry workflows for large models**

**14. [cards] Registering a model you can't pickle**
*Models-from-code ("log a script, not an instance — dodges trust_remote_code"); Register-from-LoggedModel ("lineage to source run survives promotion"); Cross-schema champion ("dev backbones funnel into one prod model").*
**Notes:** Pickling captured the dynamic `trust_remote_code` backbone class → `ModuleNotFoundError: transformers_modules` at serving. Here is the deploy-models deviation; the Big Book's LLMOps chapter sanctions it.

**15. [card-right] Train-time metric ≠ served metric (War story #1)**
*Bullets: offline val/mAP said 0.53; the registered served pyfunc scored 0.027; a serialization break invisible to training metrics; the gate that matters re-evaluates through serving. Card: "A promotion gate is only real if it scores the artifact you actually serve."*
**Notes:** Be honest — this happened in our own repo (`HPO.md`): auto-aliases landed on sub-best/broken models. Turn the weakness into the lesson.

### Act 4 — Pillar 3: GPU-aware serving (0:25–0:33)

**16. [section] Pillar 3 — GPU-aware serving**

**17. [timeline] Two-phase deploy, smoke-gated promotion**
*bundle deploy ("UC + jobs only — never endpoints") · Train ("register @challenger") · Deploy SDK ("resolve alias → numeric version, GPU_SMALL") · Smoke test ("one real image → 200 + detections") · Promote ("flip @champion only on success").*
**Notes:** YAML can't reference @champion before a version exists → endpoint is SDK-driven, gated on training. Zero-downtime via update_config_and_wait. Every request → AI Gateway inference table. Live option: curl the endpoint.

**18. [stat-row] War story #2 — train/serve skew**
*0.176 (served, squash bug) · 0.519 (served, letterbox fix) · 0.90→0.05 (medium vs large AP).*
**Notes:** Training letterboxed; serving squashed. On 2:1 panoramics the model saw a stretched image and mapped boxes back wrong. The model was fine; the serving preprocessing wasn't. [If/when latency numbers land, add a p50/p95/p99 stat-row here.]

### Act 5 — Pillar 4: Monitoring + retraining (0:33–0:39)

**19. [section] Pillar 4 — Monitoring + retraining**

**20. [card-right] You can't histogram an image**
*Subtitle "Representation drift, zero added latency." Bullets: re-embed inference-table images via frozen summary; KNN (k=50) + MMD vs training reference; Lakehouse Monitoring alerts at 2× baseline; separate hourly job off the detection hot path. Card: "No prod labels in imaging → drift is a leading indicator, not accuracy."*
**Notes:** Classical drift watches tabular features; we have images. Two regressions to catch: pre-deploy (eval gate vs champion) and post-deploy (drift). Live option: `notebooks/05_drift_demo.py` clean vs shifted ≥ 2×.

**21. [timeline] Closing the loop: retraining path**
*Manual (today) ("drift alert → human → retrain") · Scheduled ("periodic train on latest data") · Triggered ("drift_scores → SQL alert → webhook → train job").*
**Notes:** Abstract names "missed retraining cycles" as a pain. Be honest: manual today; show the Big Book's exact path to triggered.

### Act 6 — Scorecard & reframe (0:39–0:43)

**22. [section] Conformance scorecard — where we keep, adapt, and break the Big Book**

**23. [three-column] Keep · Adapt · Deviate**
*🟢 Keep: UC champion/challenger aliases; validate on data slices; zero-downtime updates; lineage + reproducibility. 🟡 Adapt: deploy models not code (Big-Book-sanctioned branch for expensive training); 2 schemas not 3 envs; representation-drift monitoring; CI = unit + contract tests. 🔴 Deviate: re-eval through serving pyfunc as the real gate; cross-schema champion copy; no online traffic-split A/B (platform supports it — we don't use it).*
**Notes:** "Every red/amber is forced by deep learning, not laziness." This slide is what makes it an Advanced talk, not a demo. Deploy-models is amber, not red — we picked the right branch the Big Book already offers.

**24. [cards] One frozen backbone, three roles**
*Detection ("spatial_features 1152 → GPU endpoint"); Search ("summary 2304 → Vector Search"); Drift ("summary 2304 → drift_scores").*
**Notes:** The cost story: one artifact, three consumers, zero backbone gradients. `BackboneInfo` is the single source of truth for every dimension.

### Act 7 — Take-home & close (0:43–0:45)

**25. [checklist] What to take home**
*Anchor your deviations to the Big Book, not vibes; score the artifact you actually serve; monitor representation when labels are absent; reproduce: `bundle deploy` + `bundle run train`.*

**26. [closing] github.com/mshtelma/dais26-mlops-for-dl-on-air**
**Notes:** QR code; "Questions? I'll be at the booth."

---

## 6. Optional slides (add on request)

- **Industry bridge** (after slide 24): same frozen-backbone pattern → utilities equipment/line-defect imagery, FSI document & ID verification, enterprise visual inspection. "Swap the dataset; the architecture holds."
- **LLMOps mapping**: position Vision FMs as "large models" in the Big Book LLMOps sense — deploy-models ✓, self-hosted-over-API ✓, frozen/LoRA/full ✓, models-from-code packaging ✓, Vector Search = the RAG vector DB repurposed for image similarity.
- **Traffic-split A/B** (ONLY if implemented): one endpoint, two served entities (champion vs challenger), 90/10, satisfies both the abstract and the Big Book.

---

## 7. Design guidance for the deck agent

- Audience is **advanced MLOps practitioners** — favor architecture, trade-offs, and the two war
  stories over product marketing.
- Vary rhythm: dark `section`/`callout` slides between dense `two-column`/`timeline` slides.
- Hero moments: the **scorecard** (slide 23) and the two **war-story stat-rows** (15, 18).
- Titles ≤ 8 words; ≤ 5 bullets/slide; one idea per slide.
- **Never fabricate** latency numbers or a traffic-split demo. Use the grounded fact sheet (§3).
- Demote the VFM-licensing and BackboneInfo-dimension detail to single lines — they are correctness,
  not lifecycle.

---

## 8. Research addendum (Glean + docs, re-analyzed June 2026)

> Pulled from internal Databricks sources to harden the talk's claims against the *current* product
> story and the *internal* Big Book guidance. Use these to fix framing, add credibility facts, and
> (optionally) add an industry hook. Cite the repo first; these reinforce, not replace, §3.

### 8.1 Correction that strengthens the talk — deploy-models is *sanctioned*, not a rebellion
The **internal Big Book / MLOps deck** (*"MLOps on Databricks – BBMLOps V2 Update"*, Niall Turbitt,
`go/mlops/deck`) is explicit:

- **Deploy-code** is the default, but **deploy-model** is the recommended branch for **"one-off models"**
  and **"expensive training where read-access to prod data from dev is possible."** A 412M-param VFM
  with a ~48h H100 sweep is the textbook case.
- It also frames code and model lifecycles as **asynchronous**: *fraud models retrain often on stable
  code; LLM fine-tunes are stable models under evolving (RAG) code.* Our frozen-backbone + small-head
  detector sits on the "stable model, code-driven plumbing" end — a clean talking point.

**Action:** stop calling deploy-models a "deviation/break." Call it **picking the correct Big-Book
branch**. This is more accurate and *more* credible to an advanced audience. (Already applied to §2,
§4, slide 23.)

### 8.2 Traffic splitting is a platform capability — own the gap honestly
- Model Serving has supported **multiple served models behind one endpoint with a configurable
  traffic split** (A/B + canary) since GA (Databricks Model Serving GA blog; internal MLOps deck
  "Champion/Challenger → online A/B"). So the abstract's "traffic splitting" is **real on the
  platform**; the repo simply does **smoke-gated 100% cutover** instead.
- **Framing:** "The platform does 90/10 canary out of the box. We deliberately chose a hard
  smoke-gated cutover because our gate re-scores the *served* artifact (War story #1) — for a 4-class
  detector with no live labels, an online split buys little and complicates rollback." That converts
  a perceived missing-feature into a *defended design choice*. (Keep the "do not fake an A/B" rule.)

### 8.3 Distributed-training menu — position DDP against the alternatives (one line)
Databricks DL guidance (Confluence *Deep Learning*, `go/` enablement) lists the sanctioned options:
**TorchDistributor**, **DeepSpeed Distributor**, **Ray on Databricks**, **TensorFlow Distributor**,
plus legacy **Horovod/Petastorm**; **StreamingDataset** (GA, all clouds) and **Composer** for the
data/loop layer; **Mosaic AI Model Training / LLM Foundry** for GenAI fine-tuning.

**Action:** on Pillar-1 (slide 10/11) add one line — *"We use plain DDP on Serverless GPU because the
head is ~5M params; DeepSpeed/FSDP/Ray are the menu when the trainable surface is the whole model."*
Shows you know the landscape and chose deliberately.

### 8.4 Serverless GPU status — a credibility fact for slide 9/10
Serverless GPU availability (internal DL enablement): **Azure GA · AWS GA · GCP Public Preview · PVC
GA**. Safe to say "Serverless GPU is GA on AWS and Azure" on stage. Avoids the "is this even GA?"
heckle.

### 8.5 MLflow 3 deployment jobs — your registry/promotion story matches the GA pattern
Current GA mechanics (docs, updated 2026-03; matches `.cursor/plans/mlflow_deployment_jobs_*.plan.md`):
- A deployment job = **evaluation → approval (human-in-the-loop) → deployment**, orchestrated by
  **Lakeflow Jobs**, **governed by Unity Catalog**, with an **activity log on the model-version page**.
- Required job params: **`model_name`, `model_version`**; recommended **max concurrent run = 1** to
  avoid deployment races. Supports **staged rollout with a metrics-collection step** (the platform's
  built-in A/B path from §8.2).
- Works on **existing UC models** even without full MLflow-3 tracking.

**Action:** on Pillar-2 (slide 14/15) you can credibly say *"this maps 1:1 onto MLflow 3 deployment
jobs — eval, human approval, deploy, all logged on the UC model version."* Optional new slide below.

### 8.6 Industry hook (matches the catalog verticals: Energy & Utilities, FSI, Enterprise Tech)
From **"Computer Vision for Energy | Databricks Sales Play"** (Confluence, `go/`):
- **Use cases:** drone/camera **asset inspection** (corrosion, broken insulators), **vegetation
  management** near power lines, **leak/defect detection**, **PPE/safety compliance**.
- **Why GPU DL:** YOLO/Detectron2/fusion models; thousands of field images/site; streaming video;
  transfer learning on domain defects — all compute-heavy.
- **Production MLOps pain (verbatim-aligned to our four pillars):** *no reproducible retraining as
  conditions/lighting/angles drift; CV siloed on edge boxes with no central deploy/monitor/benchmark;
  labeling/lineage chaos; no systematic precision / false-positive / latency monitoring.*
- **Business impact anchors:** −40% inspection hours; −28% unplanned outages (vegetation); leak
  detection 90 min → 12 min.

**Action:** consider opening slide 2/3 with a 1-line energy framing — *"Same architecture, swap dental
X-rays for substation drone imagery"* — so the verticals in the catalog aren't orphaned. Keep DENTEX
as the worked example; the energy bullets are the "why this matters" wrapper. (This upgrades the
optional Industry-bridge slide in §6.)

### 8.7 Lakehouse Monitoring loop — confirms (doesn't change) our story
Internal deck reiterates: **inference tables → Lakehouse Monitoring (profile + drift + custom SQL
metrics) → auto DBSQL dashboard → drift alert → retraining trigger.** Our representation-drift job is
a DL-specific *substitution* for the tabular-drift step in this exact loop — say it that way on slide
20/21.

### 8.8 Optional new slide enabled by this research
**[two-column] "We didn't break the registry — we used MLflow 3 deployment jobs"** *(insert after
slide 15)*
- Left "The GA pattern": eval → human approval → deploy; UC-governed; activity log on model version;
  `model_name`+`model_version`; max-concurrent=1.
- Right "Our instantiation": re-eval through the **serving pyfunc** (not just `mlflow.evaluate`);
  best-in-experiment + per-class gate; smoke test on a real image; `copy_model_version` cross-schema;
  flip `@champion`.
- **Notes:** ties the war stories to the *governed* GA workflow — proves the deviations live *inside*
  the supported framework, not outside it.

### 8.9 Sources (internal — for the speaker, not the slides)
- `go/mlops/deck` — *MLOps on Databricks, BBMLOps V2 Update* (Turbitt). Deploy-code vs deploy-model;
  champion/challenger A/B; inference tables; Lakehouse Monitoring; retraining loop.
- Confluence *Deep Learning* (FE space, id 2505607721) — distributed-training menu; Serverless GPU
  availability matrix; StreamingDataset/Composer/Mosaic AI Model Training.
- *MLOps with Databricks Best Practices* (GDoc) — GPU vs CPU split; **no autoscaling for ML
  workloads**; Feature Store for train/serve consistency; MLflow reproducibility.
- Confluence *Computer Vision for Energy* (id 5179343676) — industry hook + impact numbers.
- Docs: *MLflow 3 deployment jobs* + *Get started with MLflow 3* (updated 2026-03); *Model Serving GA*
  blog (multi-model endpoint + traffic split).
