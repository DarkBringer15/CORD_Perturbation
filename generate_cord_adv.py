"""
Generate adversarial CORD dataset for S1-S6.

Usage:
    python generate_cord_adv.py \
        --input_root ./Datasets/CORD/CORD/CORD/train \
        --output_root ./Datasets/CORD_adv \
        --scenarios S1,S2,S3,S4,S5,S6 \
        --tau 0.6 \
        --rho 0.1 \
        --bbox_method random \
        --text_method random \
        --seed 42
"""
import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from perturbations.scenarios import apply_scenario
from loaders.cord_loader import load_cord_document, save_cord_document
from utils.metrics import compute_ssim, compute_psnr


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_root", required=True,
                   help="Folder berisi image/ dan json/")
    p.add_argument("--output_root", required=True)
    p.add_argument("--scenarios", default="S1,S2,S3,S4,S5,S6")
    p.add_argument("--tau", type=float, default=0.6)
    p.add_argument("--rho", type=float, default=0.1)
    p.add_argument("--bbox_method", default="random",
                   choices=["random", "pgd"])
    p.add_argument("--text_method", default="random",
                   choices=["random", "unicode"])
    p.add_argument("--pixel_transform", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    image_dir = input_root / "image"
    json_dir = input_root / "json"

    # Kumpulkan semua pasangan (image, json)
    pairs = []
    for img_path in sorted(image_dir.glob("*")):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        json_path = json_dir / f"{img_path.stem}.json"
        if json_path.exists():
            pairs.append((img_path, json_path))

    if args.limit:
        pairs = pairs[:args.limit]

    print(f"Found {len(pairs)} documents")

    scenarios = args.scenarios.split(",")

    for scenario in scenarios:
        print(f"\n=== Scenario {scenario} ===")
        out_img_dir = output_root / scenario / "image"
        out_json_dir = output_root / scenario / "json"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_json_dir.mkdir(parents=True, exist_ok=True)

        stats = []

        for img_path, json_path in tqdm(pairs):
            image, words, raw_json = load_cord_document(img_path, json_path)

            bboxes = [w["bbox"] for w in words]
            texts = [w["text"] for w in words]

            config = {
                "bbox_method": args.bbox_method,
                "tau": args.tau,
                "text_method": args.text_method,
                "rho": args.rho,
                "pixel_transform": args.pixel_transform,
            }

            new_image, new_bboxes, new_texts, meta = apply_scenario(
                image, bboxes, texts, scenario, config, rng=rng
            )

            # Update words
            new_words = []
            for i, w in enumerate(words):
                nw = dict(w)
                nw["original_bbox"] = bboxes[i]
                nw["original_text"] = texts[i]
                nw["bbox"] = new_bboxes[i]
                nw["text"] = new_texts[i]
                nw["iou"] = meta["ious"][i] if meta["ious"] else None
                nw["edit_rate"] = meta["edit_rates"][i] if meta["edit_rates"] else None
                new_words.append(nw)

            # Save
            img_out = out_img_dir / f"{img_path.stem}.jpg"
            json_out = out_json_dir / f"{img_path.stem}.json"
            save_cord_document(new_image, new_words, raw_json, img_out, json_out)

            # Metrics
            ssim_val = compute_ssim(image, new_image)
            psnr_val = compute_psnr(image, new_image)
            stats.append({
                "file": img_path.stem,
                "scenario": scenario,
                "ssim": float(ssim_val),
                "psnr": float(psnr_val),
                "n_words": len(words),
                "mean_iou": float(np.mean(meta["ious"])) if meta["ious"] else None,
            })

        # Save stats
        with open(output_root / scenario / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        print(f"Saved {len(stats)} docs to {output_root / scenario}")


if __name__ == "__main__":
    main()