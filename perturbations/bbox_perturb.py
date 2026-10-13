"""
Layout perturbation (BBox perturbation).
Random shift + PGD-style optimization under IoU budget.
"""
import numpy as np


def compute_iou(box1, box2):
    """box = [x, y, w, h]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def random_shift_bbox(bbox, tau=0.6, max_shift=0.15, max_scale=0.05,
                      max_attempts=100, rng=None):
    """
    Random shift + slight scale.
    IoU(bbox, new_bbox) >= tau.
    """
    rng = rng or np.random.default_rng()
    x, y, w, h = bbox
    for _ in range(max_attempts):
        dx = rng.uniform(-max_shift, max_shift) * w
        dy = rng.uniform(-max_shift, max_shift) * h
        dw = rng.uniform(-max_scale, max_scale) * w
        dh = rng.uniform(-max_scale, max_scale) * h
        new_bbox = [x + dx, y + dy, w + dw, h + dh]
        if new_bbox[2] <= 1 or new_bbox[3] <= 1:
            continue
        if compute_iou(bbox, new_bbox) >= tau:
            return new_bbox, compute_iou(bbox, new_bbox)
    return bbox.copy(), 1.0


def pgd_bbox_attack(bbox, loss_fn, tau=0.6, steps=10, alpha=0.05,
                    rng=None):
    """
    PGD-style BBox attack.
    loss_fn(bbox) -> scalar (higher = worse for model).
    Here we use a simple surrogate: maximize displacement while keeping IoU.
    In real use, replace loss_fn with model gradient.
    """
    rng = rng or np.random.default_rng()
    x, y, w, h = bbox
    current = np.array(bbox, dtype=np.float32)

    for _ in range(steps):
        # Random direction as gradient surrogate
        direction = rng.normal(0, 1, 4)
        direction /= (np.linalg.norm(direction) + 1e-8)

        # Scale by alpha * box size
        step = np.array([alpha * w, alpha * h, alpha * w, alpha * h])
        candidate = current + direction * step

        # Enforce min size
        candidate[2] = max(candidate[2], 2.0)
        candidate[3] = max(candidate[3], 2.0)

        # Project to IoU budget
        if compute_iou(bbox, candidate.tolist()) >= tau:
            current = candidate

    iou_val = compute_iou(bbox, current.tolist())
    if iou_val < tau:
        return bbox.copy(), 1.0
    return current.tolist(), iou_val


def perturb_bboxes(bboxes, method="random", tau=0.6, rng=None):
    """
    Apply perturbation to list of bboxes.
    method: 'random' or 'pgd'
    Returns: list of (new_bbox, iou)
    """
    rng = rng or np.random.default_rng()
    results = []
    for bbox in bboxes:
        if method == "random":
            new_bbox, iou = random_shift_bbox(bbox, tau=tau, rng=rng)
        elif method == "pgd":
            new_bbox, iou = pgd_bbox_attack(bbox, loss_fn=None, tau=tau, rng=rng)
        else:
            raise ValueError(f"Unknown method: {method}")
        results.append((new_bbox, iou))
    return results