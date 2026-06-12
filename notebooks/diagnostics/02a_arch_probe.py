# Databricks notebook source
# MAGIC %md
# MAGIC # 02a — Architecture-consistency probe
# MAGIC
# MAGIC Read-only diagnostic that runs ONE forward + anchor-match pass on the real
# MAGIC detector (frozen backbone + FPN + RetinaNet head) over a real DENTEX batch,
# MAGIC then prints:
# MAGIC
# MAGIC * per-FPN-level anchor counts + the "all scales at every level" smell,
# MAGIC * how many of the batch's matched positives land on each level,
# MAGIC * the fraction of box-regression targets that exceed the decoder's `exp`
# MAGIC   clamp (unreachable targets),
# MAGIC * token/grid alignment + head-vs-anchor count agreement,
# MAGIC
# MAGIC followed by the curated register of architectural issues to flag **before**
# MAGIC spending GPU hours on the HPO sweep. See `dais26_dentex.models.arch_probe`.
# MAGIC
# MAGIC Requires a **GPU** notebook (the ViT backbone loads onto CUDA).

# COMMAND ----------
# MAGIC %pip install --quiet ../..

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %run ../00_config

# COMMAND ----------
import torch

from dais26_dentex.config.trainer_config import TrainerConfig
from dais26_dentex.data.dataset import DENTEXDetectionDataset, detection_collate
from dais26_dentex.data.transforms import get_val_transforms
from dais26_dentex.models.arch_probe import probe_detection_model, render_report
from dais26_dentex.models.builder import build_detector, resolve_num_classes

PROBE_IMG_SIZE = 1024  # match training; lower (e.g. 512) for a faster smoke probe
PROBE_BATCH = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device = {device}")

# COMMAND ----------
# ---- Build the detector exactly as the trainer would ----
cfg = TrainerConfig(
    catalog=CATALOG,
    schema=SCHEMA,
    backbone_name=BACKBONE,  # type: ignore[arg-type]
    backbone_revision=BACKBONE_REVISION,
    volume_path=VOLUME_PATH,
    cache_dir=CACHE_DIR,
    img_size=PROBE_IMG_SIZE,
)
model, _info = build_detector(cfg, device=device)
model.eval()
num_classes = resolve_num_classes(cfg)
print(f"backbone={cfg.backbone_name} num_classes={num_classes} patch_size={model.patch_size}")

# COMMAND ----------
# ---- Pull one real DENTEX batch (val split) so positives reflect real box sizes ----
ds = DENTEXDetectionDataset(
    volume_path=VOLUME_PATH,
    split="val",
    transforms=get_val_transforms(PROBE_IMG_SIZE),
)
loader = torch.utils.data.DataLoader(
    ds, batch_size=PROBE_BATCH, shuffle=False, collate_fn=detection_collate
)
images, targets = next(iter(loader))
images = images.to(device)
print(f"probe batch: images={tuple(images.shape)}  gts={[int(t['labels'].numel()) for t in targets]}")

# COMMAND ----------
# ---- Run the probe + print the report ----
report = probe_detection_model(model, images, targets, num_classes=num_classes)
print(render_report(report))

# COMMAND ----------
# MAGIC %md
# MAGIC ## How to read this
# MAGIC
# MAGIC * `all-scales-every-level=True` + roughly equal anchor counts skewed toward
# MAGIC   the high-resolution level confirm the **anchor over-generation** issue:
# MAGIC   small anchors are wasted on coarse levels and vice-versa.
# MAGIC * A very low `positive_fraction` (<<1%) with positives concentrated on one
# MAGIC   level means most of the 12-anchors/cell are dead weight — the matcher sees
# MAGIC   almost no foreground, which starves the focal loss and pins mAP near zero.
# MAGIC * A non-trivial `delta_overflow_fraction` means some GT boxes can't be
# MAGIC   regressed because the decode clamp is tighter than the encode range.
# MAGIC
# MAGIC The flagged-issues block lists the fixes wired into the HPO/backbone plan
# MAGIC (per-level anchors, per-class NMS, trainable-backbone gate, discriminative LR).
