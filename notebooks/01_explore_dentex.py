# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Explore DENTEX
# MAGIC Sample images with annotations, class distribution, size stats.

# COMMAND ----------
# MAGIC %pip install --quiet ..

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %run ./00_config

# COMMAND ----------

from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from dais26_dentex.data.dentex_loader import get_label_map, load_canonical_split

label_map = get_label_map()

# COMMAND ----------

coco = load_canonical_split(VOLUME_PATH, EXPLORE_SPLIT)
print(f"Split: {EXPLORE_SPLIT}")
print(f"Images: {len(coco['images'])}, Annotations: {len(coco['annotations'])}, Categories: {len(coco['categories'])}")

# COMMAND ----------

# Class distribution
cls_counts = Counter(label_map[a["category_id"]] for a in coco["annotations"])
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(cls_counts.keys(), cls_counts.values())
ax.set_title(f"DENTEX {EXPLORE_SPLIT} — class distribution")
ax.set_ylabel("bbox count")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# COMMAND ----------

# Image size stats
widths = [img["width"] for img in coco["images"]]
heights = [img["height"] for img in coco["images"]]
print(f"Width:  min={min(widths)} max={max(widths)} median={sorted(widths)[len(widths)//2]}")
print(f"Height: min={min(heights)} max={max(heights)} median={sorted(heights)[len(heights)//2]}")

# COMMAND ----------

# Show a few sample images with bbox overlay
import random
random.seed(0)
sample_ids = random.sample([img["id"] for img in coco["images"]], min(4, len(coco["images"])))
img_by_id = {i["id"]: i for i in coco["images"]}
anns_by_img: dict[int, list] = {}
for a in coco["annotations"]:
    anns_by_img.setdefault(a["image_id"], []).append(a)

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
for ax, img_id in zip(axes.flat, sample_ids, strict=False):
    info = img_by_id[img_id]
    p = Path(VOLUME_PATH) / "images" / EXPLORE_SPLIT / info["file_name"]
    img = Image.open(p).convert("RGB")
    draw = ImageDraw.Draw(img)
    for ann in anns_by_img.get(img_id, []):
        x, y, w, h = ann["bbox"]
        draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
        draw.text((x, max(0, y - 12)), label_map[ann["category_id"]], fill="red")
    ax.imshow(img)
    ax.set_title(f"image_id={img_id}")
    ax.axis("off")
plt.tight_layout()
plt.show()
