"""
Six attack scenarios:
S1: BBox only
S2: BBox + Pixel
S3: BBox + Pixel + Augment
S4: Text only
S5: BBox + Text
S6: BBox + Pixel + Text
"""
import numpy as np

from .bbox_perturb import perturb_bboxes
from .text_perturb import perturb_texts, edit_rate
from .pixel_perturb import perturb_pixel_region, translate_pixels, TRANSFORMS


SCENARIOS = ["S1", "S2", "S3", "S4", "S5", "S6"]

# Skenario yang memakai augmentasi RoDLA
AUGMENT_SCENARIOS = {"S3"}
# Skenario yang memakai translasi piksel
PIXEL_SCENARIOS = {"S2", "S3", "S6"}
# Skenario yang memakai perturbasi teks
TEXT_SCENARIOS = {"S4", "S5", "S6"}
# Skenario yang memakai perturbasi BBox
BBOX_SCENARIOS = {"S1", "S2", "S3", "S5", "S6"}


def apply_scenario(image, bboxes, texts, scenario, config, rng=None):
    rng = rng or np.random.default_rng()

    new_image = image.copy()
    new_bboxes = [list(b) for b in bboxes]
    new_texts = list(texts)
    metadata = {
        "scenario": scenario,
        "bbox_method": None,
        "tau": None,
        "text_method": None,
        "rho": None,
        "pixel_transform": None,
        "ious": [],
        "edit_rates": [],
    }

    # ---- Text only (S4) ----
    if scenario == "S4":
        text_method = config.get("text_method", "random")
        rho = config.get("rho", 0.1)
        new_texts = perturb_texts(texts, method=text_method, rho=rho, rng=rng)
        metadata["text_method"] = text_method
        metadata["rho"] = rho
        metadata["edit_rates"] = [edit_rate(o, n) for o, n in zip(texts, new_texts)]
        return new_image, new_bboxes, new_texts, metadata

    # ---- BBox perturbation ----
    if scenario in BBOX_SCENARIOS:
        bbox_method = config.get("bbox_method", "random")
        tau = config.get("tau", 0.6)
        bbox_results = perturb_bboxes(bboxes, method=bbox_method, tau=tau, rng=rng)
        new_bboxes = [r[0] for r in bbox_results]
        metadata["ious"] = [r[1] for r in bbox_results]
        metadata["bbox_method"] = bbox_method
        metadata["tau"] = tau

    # S1 berhenti di sini
    if scenario == "S1":
        return new_image, new_bboxes, new_texts, metadata

    # ---- Pixel step (S2, S3, S6) ----
    if scenario in PIXEL_SCENARIOS:
        pixel_transform = config.get("pixel_transform", None)
        used_transforms = []

        for old_b, new_b in zip(bboxes, new_bboxes):
            # Translasi piksel dari bbox lama ke bbox baru
            new_image = translate_pixels(new_image, old_b, new_b)

            # Augment hanya untuk skenario yang diizinkan
            if scenario in AUGMENT_SCENARIOS:
                t = pixel_transform if pixel_transform else rng.choice(TRANSFORMS)
                new_image, used = perturb_pixel_region(
                    new_image, new_b, transform=t, rng=rng
                )
                used_transforms.append(used)

        metadata["pixel_transform"] = used_transforms or None

    # ---- Text step (S4, S5, S6) ----
    if scenario in TEXT_SCENARIOS and scenario != "S4":
        text_method = config.get("text_method", "random")
        rho = config.get("rho", 0.1)
        new_texts = perturb_texts(texts, method=text_method, rho=rho, rng=rng)
        metadata["text_method"] = text_method
        metadata["rho"] = rho
        metadata["edit_rates"] = [edit_rate(o, n) for o, n in zip(texts, new_texts)]

    return new_image, new_bboxes, new_texts, metadata