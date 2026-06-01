# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Drift Demo (synthetic shift) / Drift Monitor (scheduled)

# COMMAND ----------
# MAGIC %pip install --quiet ..

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %run ./00_config

# COMMAND ----------

if DRIFT_MODE == "demo":
    # Synthetic shift demo: contrast/gamma applied to val images, compare drift scores
    import io
    import numpy as np
    import torch
    from PIL import Image
    from pathlib import Path
    import torchvision.transforms.functional as TF
    from dais26_dentex.drift.embeddings import compute_embeddings
    from dais26_dentex.drift.monitor import score_drift, bootstrap_drift_ci
    from dais26_dentex.drift.reference import fit_reference
    from dais26_dentex.models.backbones import load_backbone

    val_dir = Path(VOLUME_PATH) / "images" / "val"
    print(f"Loading {len(list(val_dir.glob('*.png')))} val images")

    # Load val images as bytes
    def _read_bytes(p: Path) -> bytes:
        return p.read_bytes()

    def _shift_image_to_bytes(p: Path, contrast: float = 0.5, gamma: float = 2.0) -> bytes:
        img = Image.open(p).convert("RGB")
        t = TF.pil_to_tensor(img).float() / 255.0
        t = TF.adjust_contrast(t, contrast)
        t = TF.adjust_gamma(t, gamma)
        out = TF.to_pil_image((t * 255).clamp(0, 255).byte())
        buf = io.BytesIO()
        out.save(buf, format="PNG")
        return buf.getvalue()

    val_files = sorted(val_dir.glob("*.png"))[:25]
    clean_bytes = [_read_bytes(p) for p in val_files]
    shifted_bytes = [_shift_image_to_bytes(p) for p in val_files]

    backbone, info = load_backbone(name=BACKBONE, revision=BACKBONE_REVISION,
                                   cache_dir=CACHE_DIR, device="cuda" if torch.cuda.is_available() else "cpu")

    # Reference from train embeddings (read from Delta)
    train_df = spark.table(TRAIN_EMBEDDINGS_TABLE).select("embedding").toPandas()
    ref_arr = np.stack(train_df["embedding"].apply(np.asarray).to_list()).astype(np.float32)
    ref = fit_reference(ref_arr, method="knn", k=DRIFT_KNN_K)

    clean_emb = compute_embeddings(backbone, clean_bytes)
    shifted_emb = compute_embeddings(backbone, shifted_bytes)

    clean_score = score_drift(clean_emb, ref)
    shifted_score = score_drift(shifted_emb, ref)
    ratio = shifted_score / max(clean_score, 1e-9)
    print(f"Clean drift score:    {clean_score:.4f}")
    print(f"Shifted drift score:  {shifted_score:.4f}")
    print(f"Ratio (>=2.0 passes): {ratio:.2f}")

    ci = bootstrap_drift_ci(shifted_emb, ref, n_iterations=1000)
    print(f"Bootstrap 95% CI for shifted: [{ci['p2_5']:.4f}, {ci['p97_5']:.4f}]; mean={ci['mean']:.4f}")
    print(f"E6 pass (CI lower bound > clean baseline): {ci['p2_5'] > clean_score}")

# COMMAND ----------

if DRIFT_MODE == "demo":
    # Visualization
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=np.linspace(0, clean_score, 25), name="clean ref", opacity=0.6))
    fig.add_bar(x=["clean", "shifted"], y=[clean_score, shifted_score], name="drift score")
    fig.update_layout(title="Drift score: clean val vs synthetic shift", barmode="overlay")
    fig.show()

# COMMAND ----------

if DRIFT_MODE == "scheduled":
    import torch
    from dais26_dentex.drift.monitor import run_drift_monitor
    from dais26_dentex.models.backbones import load_backbone
    backbone, _ = load_backbone(name=BACKBONE, revision=BACKBONE_REVISION,
                                cache_dir=CACHE_DIR, device="cuda" if torch.cuda.is_available() else "cpu")
    result = run_drift_monitor(
        spark=spark,
        backbone=backbone,
        catalog=CATALOG,
        schema=SCHEMA,
        inference_table=DETECTOR_INFERENCE_TABLE,
        k=DRIFT_KNN_K,
        alert_threshold=DRIFT_ALERT_THRESHOLD,
        lookback_hours=1,
    )
    print(result)
