import json
import numpy as np
from pathlib import Path

root = Path("./Datasets/CORD_adv")

for scenario in ["S1", "S2", "S3", "S4", "S5", "S6"]:
    stats = json.load(open(root / scenario / "stats.json"))
    ssim = np.mean([s["ssim"] for s in stats])
    psnr = np.mean([s["psnr"] for s in stats])
    ious = [s["mean_iou"] for s in stats if s["mean_iou"] is not None]
    mean_iou = np.mean(ious) if ious else None
    print(f"{scenario}: n={len(stats)}, SSIM={ssim:.4f}, PSNR={psnr:.2f} dB, "
          f"mean_IoU={mean_iou if mean_iou is None else round(mean_iou, 4)}")