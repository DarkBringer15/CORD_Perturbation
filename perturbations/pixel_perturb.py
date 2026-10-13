"""
Pixel perturbation inside BBox (RoDLA-style).
12 document-specific transforms, applied subtly.
"""
import cv2
import numpy as np


TRANSFORMS = [
    "gaussian_noise", "gaussian_blur", "brightness", "contrast",
    "shadow", "occlusion", "jpeg_compression", "rotation",
    "scaling", "translation", "color_jitter", "sharpen"
]


def apply_transform(roi, transform, rng=None):
    """Apply one subtle transform to ROI (H, W, 3)."""
    rng = rng or np.random.default_rng()

    if transform == "gaussian_noise":
        sigma = rng.uniform(0.01, 0.03) * 255
        noise = rng.normal(0, sigma, roi.shape)
        return np.clip(roi.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if transform == "gaussian_blur":
        k = 3
        return cv2.GaussianBlur(roi, (k, k), 0)

    if transform == "brightness":
        delta = rng.uniform(-5, 5)
        return np.clip(roi.astype(np.float32) + delta, 0, 255).astype(np.uint8)

    if transform == "contrast":
        alpha = rng.uniform(0.95, 1.05)
        mean = roi.mean()
        return np.clip((roi.astype(np.float32) - mean) * alpha + mean, 0, 255).astype(np.uint8)

    if transform == "shadow":
        h, w = roi.shape[:2]
        mask = np.ones((h, w), dtype=np.float32)
        # gradient shadow
        grad = np.linspace(1.0, rng.uniform(0.85, 0.95), w)
        mask = mask * grad[None, :]
        roi_f = roi.astype(np.float32) * mask[:, :, None]
        return np.clip(roi_f, 0, 255).astype(np.uint8)

    if transform == "occlusion":
        h, w = roi.shape[:2]
        occ_h = max(1, int(h * rng.uniform(0.02, 0.05)))
        occ_w = max(1, int(w * rng.uniform(0.02, 0.05)))
        y0 = rng.integers(0, max(1, h - occ_h))
        x0 = rng.integers(0, max(1, w - occ_w))
        out = roi.copy()
        out[y0:y0+occ_h, x0:x0+occ_w] = 255
        return out

    if transform == "jpeg_compression":
        quality = int(rng.uniform(85, 95))
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, enc = cv2.imencode('.jpg', roi, encode_param)
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)

    if transform == "rotation":
        angle = rng.uniform(-1.5, 1.5)
        h, w = roi.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        return cv2.warpAffine(roi, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    if transform == "scaling":
        scale = rng.uniform(0.97, 1.03)
        h, w = roi.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(roi, (new_w, new_h))
        # pad or crop back to (h, w)
        out = np.full_like(roi, 255)
        y0 = max(0, (h - new_h) // 2)
        x0 = max(0, (w - new_w) // 2)
        y1 = min(h, y0 + new_h)
        x1 = min(w, x0 + new_w)
        out[y0:y1, x0:x1] = resized[:y1-y0, :x1-x0]
        return out

    if transform == "translation":
        dx = rng.integers(-2, 3)
        dy = rng.integers(-2, 3)
        h, w = roi.shape[:2]
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(roi, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    if transform == "color_jitter":
        roi_f = roi.astype(np.float32)
        for c in range(3):
            delta = rng.uniform(-5, 5)
            roi_f[:, :, c] = np.clip(roi_f[:, :, c] + delta, 0, 255)
        return roi_f.astype(np.uint8)

    if transform == "sharpen":
        kernel = np.array([[0, -0.2, 0], [-0.2, 1.8, -0.2], [0, -0.2, 0]])
        return cv2.filter2D(roi, -1, kernel)

    return roi


def perturb_pixel_region(image, bbox, transform=None, rng=None):
    """
    Apply one subtle transform inside bbox region.
    If transform is None, pick random from TRANSFORMS.
    """
    rng = rng or np.random.default_rng()
    x, y, w, h = [int(v) for v in bbox]
    H, W = image.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)
    if x2 <= x1 or y2 <= y1:
        return image, None

    roi = image[y1:y2, x1:x2].copy()
    if transform is None:
        transform = rng.choice(TRANSFORMS)
    roi_new = apply_transform(roi, transform, rng=rng)
    image[y1:y2, x1:x2] = roi_new
    return image, transform


def translate_pixels(image, old_bbox, new_bbox):
    """
    Shift pixel content from old_bbox to new_bbox.
    Used in S2, S3, S5, S6.
    """
    x_old, y_old, w_old, h_old = [int(v) for v in old_bbox]
    x_new, y_new, w_new, h_new = [int(v) for v in new_bbox]
    H, W = image.shape[:2]

    # Crop from old
    x1o, y1o = max(0, x_old), max(0, y_old)
    x2o, y2o = min(W, x_old + w_old), min(H, y_old + h_old)
    if x2o <= x1o or y2o <= y1o:
        return image

    patch = image[y1o:y2o, x1o:x2o].copy()

    # Paste to new (resize to fit)
    patch = cv2.resize(patch, (max(1, w_new), max(1, h_new)))

    x1n, y1n = max(0, x_new), max(0, y_new)
    x2n, y2n = min(W, x_new + w_new), min(H, y_new + h_new)

    ph, pw = y2n - y1n, x2n - x1n
    if ph <= 0 or pw <= 0:
        return image

    image[y1n:y2n, x1n:x2n] = patch[:ph, :pw]
    return image