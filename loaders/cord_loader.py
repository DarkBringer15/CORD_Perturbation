"""
Loader for CORD dataset.
Converts CORD JSON format into (image, bboxes, texts, categories).
"""
import json
from pathlib import Path

import numpy as np
import cv2


def quad_to_bbox(quad):
    """
    CORD quad = {x1,y1,x2,y2,x3,y3,x4,y4}
    Returns [x, y, w, h].
    """
    xs = [quad["x1"], quad["x2"], quad["x3"], quad["x4"]]
    ys = [quad["y1"], quad["y2"], quad["y3"], quad["y4"]]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def bbox_to_quad(bbox):
    """
    Convert [x, y, w, h] back to quad dict.
    """
    x, y, w, h = bbox
    return {
        "x1": int(x),       "y1": int(y),
        "x2": int(x + w),   "y2": int(y),
        "x3": int(x + w),   "y3": int(y + h),
        "x4": int(x),       "y4": int(y + h),
    }


def load_cord_document(image_path, json_path):
    """
    Returns:
        image: np.ndarray (H, W, 3) RGB
        words: list of dict with keys:
            bbox, text, category, group_id, sub_group_id, row_id, is_key
        raw_json: original JSON
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    words = []
    for line in data.get("valid_line", []):
        category = line.get("category", "other")
        group_id = line.get("group_id", -1)
        sub_group_id = line.get("sub_group_id", -1)
        for w in line.get("words", []):
            bbox = quad_to_bbox(w["quad"])
            words.append({
                "bbox": bbox,
                "text": w.get("text", ""),
                "category": category,
                "group_id": group_id,
                "sub_group_id": sub_group_id,
                "row_id": w.get("row_id", -1),
                "is_key": w.get("is_key", "false"),
            })

    return image, words, data


def save_cord_document(image, words, raw_json, image_out, json_out):
    """
    Save perturbed image + updated JSON.
    - bbox di-update dari words
    - text di-update dari words
    """
    # Save image
    Path(image_out).parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(image_out), bgr)

    # Update JSON
    new_json = json.loads(json.dumps(raw_json))  # deep copy
    idx = 0
    for line in new_json.get("valid_line", []):
        for w in line.get("words", []):
            if idx >= len(words):
                break
            item = words[idx]
            w["quad"] = bbox_to_quad(item["bbox"])
            w["text"] = item["text"]
            # update original_* jika perlu
            w["original_quad"] = w.get("original_quad", raw_json["valid_line"][
                new_json["valid_line"].index(line)
            ]["words"][line["words"].index(w)]["quad"])
            w["original_text"] = w.get("original_text", item["text"])
            idx += 1

    Path(json_out).parent.mkdir(parents=True, exist_ok=True)
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(new_json, f, ensure_ascii=False, indent=2)