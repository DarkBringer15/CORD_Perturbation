import json
import os
from pathlib import Path

import cv2
import numpy as np


def load_annotations(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_image(image_path):
    """Load as RGB."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(image_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_image(image, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def save_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def group_annotations_by_image(annotations):
    """Group annotations by image_id."""
    grouped = {}
    for ann in annotations["annotations"]:
        img_id = ann["image_id"]
        grouped.setdefault(img_id, []).append(ann)
    return grouped


def merge_word_to_line(annotations):
    """
    Merge word-level bboxes into line-level.
    Heuristic: same image + overlapping y-range.
    """
    # Placeholder: implement according to your dataset.
    # For FUNSD/CORD/SROIE, line info may already exist.
    raise NotImplementedError