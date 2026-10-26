#!/usr/bin/env python3
"""
PlantVillage -> SAM leaf segmentation pipeline.

Purpose
-------
Create transparent PNGs containing the leaf/plant foreground while preserving
the original PlantVillage class-directory structure.

Strategy
--------
1. Load SAM ViT-B.
2. Generate automatic SAM masks.
3. Generate SAM masks from several center/near-center point prompts.
4. Score candidate masks using:
   - reasonable foreground area
   - distance from image center
   - plant-like color/saturation
   - SAM predicted IoU
   - stability score
   - penalty for touching the image border
   - penalty for masks that look like the whole background
5. Select the best candidate.
6. Save an RGBA PNG with the selected mask as alpha.
7. Write per-image validation statistics to CSV.
8. Write a run summary to JSON and failures to a log.

The scoring intentionally does NOT require the leaf to be green. PlantVillage
contains diseased leaves that can be yellow/brown, so green is only one part of
the plant-likeness score.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from PIL import Image

import cv2
import numpy as np
import torch
from tqdm import tqdm
from segment_anything import (
    SamAutomaticMaskGenerator,
    SamPredictor,
    sam_model_registry,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Candidate mask should normally occupy a meaningful fraction of PlantVillage.
MIN_AREA_RATIO = 0.03
MAX_AREA_RATIO = 0.95

# Multiple prompt locations make the center-prompt branch less brittle.
PROMPT_LOCATIONS = (
    (0.50, 0.50),
    (0.50, 0.40),
    (0.50, 0.60),
    (0.40, 0.50),
    (0.60, 0.50),
)

CSV_FIELDS = [
    "input_path",
    "output_path",
    "status",
    "method",
    "area_ratio",
    "center_score",
    "plant_color_score",
    "border_penalty",
    "sam_confidence",
    "final_score",
    "inference_seconds",
]


def configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment PlantVillage leaves with SAM and save transparent PNGs."
    )

    parser.add_argument(
        "--input",
        default=r"C:\Users\agroeye\Desktop\agro-eye\datasets\raw\PlantVillage",
        help="PlantVillage root directory.",
    )
    parser.add_argument(
        "--output",
        default=r"C:\Users\agroeye\Desktop\agro-eye\datasets\segmented\images",
        help="Output directory for transparent PNGs.",
    )
    parser.add_argument(
        "--checkpoint",
        default=r"C:\Users\agroeye\Desktop\agro-eye\datasets\models\sam_vit_b_01ec64.pth",
        help="Path to SAM ViT-B checkpoint.",
    )
    parser.add_argument(
        "--stats-dir",
        default=r"C:\Users\agroeye\Desktop\agro-eye\datasets\segmented\stats",
        help="Directory for CSV/JSON/log files.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Inference device.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess outputs that already exist.",
    )
    parser.add_argument(
        "--no-center-prompts",
        action="store_true",
        help="Disable center/near-center SAM prompting.",
    )
    parser.add_argument(
        "--save-debug",
        action="store_true",
        help="Save a visual overlay for each successful segmentation.",
    )
    parser.add_argument(
        "--debug-limit",
        type=int,
        default=50,
        help="Maximum number of debug overlays to save. 0 = unlimited.",
    )
    parser.add_argument(
    "--limit",
    type=int,
    default=0,
    help="Maximum number of images to process. 0 = process all images.",
)

    return parser.parse_args()


def choose_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
        return "cuda"

    return "cuda" if torch.cuda.is_available() else "cpu"


def normalize_area_score(area_ratio: float) -> float:
    """
    Prefer masks that are neither tiny nor almost the entire image.
    This is intentionally a soft score rather than a hard 'largest mask' rule.
    """
    if area_ratio < MIN_AREA_RATIO or area_ratio > MAX_AREA_RATIO:
        return 0.0

    # Peak preference around a moderate foreground size.
    # PlantVillage leaves vary considerably, so this is deliberately broad.
    if area_ratio <= 0.50:
        return min(area_ratio / 0.30, 1.0)

    return max(0.0, 1.0 - (area_ratio - 0.50) / 0.45)


def center_score(mask: np.ndarray) -> float:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0.0

    h, w = mask.shape
    cx = float(xs.mean())
    cy = float(ys.mean())

    dx = (cx - w / 2.0) / (w / 2.0)
    dy = (cy - h / 2.0) / (h / 2.0)
    distance = float(np.sqrt(dx * dx + dy * dy))

    return max(0.0, 1.0 - min(distance, 1.0))


def border_penalty(mask: np.ndarray) -> float:
    """
    1.0 = strongly touches image border
    0.0 = no border contact

    PlantVillage's leaf is generally isolated from the image edge. This helps
    reject masks that are actually large background regions.
    """
    h, w = mask.shape
    border_pixels = np.concatenate(
        [
            mask[0, :],
            mask[-1, :],
            mask[:, 0],
            mask[:, -1],
        ]
    )
    return float(np.mean(border_pixels))


def plant_color_score(image_rgb: np.ndarray, mask: np.ndarray) -> float:
    """
    Estimate plant-likeness from HSV.

    Green is useful, but diseased leaves can be yellow/brown. Therefore this
    score gives credit to:
      - green/yellow hues
      - moderate/high saturation
      - non-extreme brightness

    It is NOT a hard green-leaf detector.
    """
    pixels = image_rgb[mask]

    if len(pixels) == 0:
        return 0.0

    hsv = cv2.cvtColor(
        pixels.reshape(-1, 1, 3),
        cv2.COLOR_RGB2HSV,
    ).reshape(-1, 3)

    hue = hsv[:, 0].astype(np.float32)
    sat = hsv[:, 1].astype(np.float32) / 255.0
    val = hsv[:, 2].astype(np.float32) / 255.0

    # OpenCV hue range is 0..179.
    # Green is roughly 35..95; yellow-green extends toward ~25.
    green = (hue >= 25) & (hue <= 95)

    # Brown/orange diseased tissue can still be part of the leaf.
    brown_yellow = (hue >= 5) & (hue < 25) & (sat >= 0.20)

    hue_score = np.maximum(green.astype(np.float32), brown_yellow.astype(np.float32))

    # Avoid giving white/gray background pixels a high plant score.
    saturation_score = np.clip((sat - 0.10) / 0.45, 0.0, 1.0)

    # Extremely dark pixels are less useful, but don't reject them completely.
    brightness_score = np.clip((val - 0.05) / 0.25, 0.0, 1.0)

    pixel_score = (
        0.60 * hue_score
        + 0.25 * saturation_score
        + 0.15 * brightness_score
    )

    return float(np.mean(pixel_score))


def mask_from_prediction(pred_mask: np.ndarray) -> np.ndarray:
    """
    Convert SAM predictor output to a single 2-D boolean mask.
    """
    arr = np.asarray(pred_mask)

    if arr.ndim == 3:
        # Predictor returns N x H x W.
        arr = arr[0]

    return arr.astype(bool)


def score_mask(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    sam_confidence: float,
    stability: float = 0.0,
) -> dict[str, float]:
    h, w = mask.shape
    area_ratio = float(mask.sum() / (h * w))

    a_score = normalize_area_score(area_ratio)
    c_score = center_score(mask)
    p_score = plant_color_score(image_rgb, mask)
    b_penalty = border_penalty(mask)

    # Strongly reject obvious background-sized masks.
    if area_ratio > 0.95:
        final = -1.0
    elif area_ratio < MIN_AREA_RATIO:
        final = -1.0
    else:
        # Main signal: center + plant appearance.
        # SAM confidence is useful but should not dominate the domain heuristics.
        final = (
            0.20 * a_score
            + 0.28 * c_score
            + 0.30 * p_score
            + 0.12 * float(np.clip(sam_confidence, 0.0, 1.0))
            + 0.10 * float(np.clip(stability, 0.0, 1.0))
            - 0.25 * b_penalty
        )

    return {
        "area_ratio": area_ratio,
        "area_score": a_score,
        "center_score": c_score,
        "plant_color_score": p_score,
        "border_penalty": b_penalty,
        "sam_confidence": float(sam_confidence),
        "stability_score": float(stability),
        "final_score": float(final),
    }


def automatic_candidates(
    mask_generator: SamAutomaticMaskGenerator,
    image_rgb: np.ndarray,
) -> list[dict[str, Any]]:
    masks = mask_generator.generate(image_rgb)

    candidates = []

    for m in masks:
        segmentation = np.asarray(m["segmentation"]).astype(bool)

        scores = score_mask(
            image_rgb,
            segmentation,
            float(m.get("predicted_iou", 0.0)),
            float(m.get("stability_score", 0.0)),
        )

        if scores["final_score"] < 0:
            continue

        candidates.append(
            {
                "mask": segmentation,
                "method": "automatic",
                **scores,
            }
        )

    return candidates


def center_prompt_candidates(
    predictor: SamPredictor,
    image_rgb: np.ndarray,
) -> list[dict[str, Any]]:
    predictor.set_image(image_rgb)

    h, w, _ = image_rgb.shape
    candidates = []

    for fx, fy in PROMPT_LOCATIONS:
        point = np.array([[int(w * fx), int(h * fy)]], dtype=np.float32)
        label = np.array([1], dtype=np.int32)

        masks, iou_predictions, _ = predictor.predict(
            point_coords=point,
            point_labels=label,
            multimask_output=True,
        )

        for i in range(len(masks)):
            segmentation = mask_from_prediction(masks[i])

            scores = score_mask(
                image_rgb,
                segmentation,
                float(iou_predictions[i]),
                0.0,
            )

            if scores["final_score"] < 0:
                continue

            candidates.append(
                {
                    "mask": segmentation,
                    "method": "center_prompt",
                    **scores,
                }
            )

    predictor.reset_image()
    return candidates


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """
    Mild cleanup only. SAM already provides the actual boundary; this removes
    tiny isolated specks without aggressively altering the leaf boundary.
    """
    mask_u8 = (mask.astype(np.uint8) * 255)

    kernel = np.ones((3, 3), np.uint8)

    mask_u8 = cv2.morphologyEx(
        mask_u8,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1,
    )

    mask_u8 = cv2.morphologyEx(
        mask_u8,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1,
    )

    # Keep the largest connected component.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8,
        connectivity=8,
    )

    if count <= 1:
        return mask_u8 > 0

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest_label


def save_rgba(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rgba = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask.astype(np.uint8) * 255

    if not cv2.imwrite(str(output_path), rgba):
        raise IOError(f"Failed to write: {output_path}")


def save_debug_overlay(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    output_path: Path,
) -> None:
    overlay = image_rgb.copy()

    # Make background darker while retaining enough context for validation.
    overlay[~mask] = (
        overlay[~mask].astype(np.float32) * 0.20
    ).astype(np.uint8)

    # Draw the boundary.
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.drawContours(overlay_bgr, contours, -1, (0, 255, 0), 2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay_bgr)


def load_models(
    checkpoint: str,
    device: str,
) -> tuple[Any, SamAutomaticMaskGenerator, SamPredictor]:
    checkpoint_path = Path(checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"SAM checkpoint not found: {checkpoint_path}"
        )

    logging.info("Loading SAM ViT-B...")
    logging.info("Checkpoint: %s", checkpoint_path)
    logging.info("Device: %s", device)

    sam = sam_model_registry["vit_b"](
        checkpoint=str(checkpoint_path)
    )
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=32,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.90,
        crop_n_layers=1,
        min_mask_region_area=100,
    )

    predictor = SamPredictor(sam)

    return sam, mask_generator, predictor


def main() -> None:
    args = parse_args()

    input_root = Path(args.input)
    output_root = Path(args.output)
    mask_root = output_root.parent / "masks"
    stats_root = Path(args.stats_dir)

    stats_root.mkdir(parents=True, exist_ok=True)

    log_path = stats_root / "segmentation.log"
    csv_path = stats_root / "segmentation_results.csv"
    summary_path = stats_root / "summary.json"

    configure_logging(log_path)

    if not input_root.exists():
        raise FileNotFoundError(f"Input directory not found: {input_root}")

    device = choose_device(args.device)

    if device == "cuda":
        logging.info("GPU: %s", torch.cuda.get_device_name(0))
        logging.info(
            "CUDA capability: %s",
            torch.cuda.get_device_capability(0),
        )

    _, mask_generator, predictor = load_models(
        args.checkpoint,
        device,
    )

    image_paths = sorted(
        p for p in input_root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    total_found = len(image_paths)

    if args.limit > 0:
        image_paths = image_paths[:args.limit]

    logging.info("Images found: %d", total_found)
    logging.info("Images selected for processing: %d", len(image_paths))

    logging.info("Images found: %d", len(image_paths))

    processed = 0
    skipped_existing = 0
    failed = 0
    automatic_count = 0
    center_count = 0
    start_all = time.perf_counter()

    fieldnames = CSV_FIELDS

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        progress = tqdm(
            enumerate(image_paths, start=1),
            total=len(image_paths),
            desc="Segmenting PlantVillage",
            unit="img",
            dynamic_ncols=True,
        )

        for index, image_path in progress:
            relative = image_path.relative_to(input_root)
            mask_relative = relative.with_suffix(".png")
            mask_path = mask_root / mask_relative


            progress.set_postfix(
                processed=processed,
                failed=failed,
                skipped=skipped_existing,
                refresh=False,
            )
            output_path = (
                output_root
                / relative.parent
                / f"{image_path.stem}.png"
            )

            if (
                output_path.exists()
                and mask_path.exists()
                and not args.overwrite
            ):
                logging.info(
                    "[%d/%d] exists, skipping: %s",
                    index,
                    len(image_paths),
                    relative,
                )
                skipped_existing += 1
                continue

            t0 = time.perf_counter()

            try:
                image_bgr = cv2.imread(
                    str(image_path),
                    cv2.IMREAD_COLOR,
                )

                if image_bgr is None:
                    raise ValueError("OpenCV could not read the image.")

                image_rgb = cv2.cvtColor(
                    image_bgr,
                    cv2.COLOR_BGR2RGB,
                )

                candidates = automatic_candidates(
                    mask_generator,
                    image_rgb,
                )

                if not args.no_center_prompts:
                    candidates.extend(
                        center_prompt_candidates(
                            predictor,
                            image_rgb,
                        )
                    )

                if not candidates:
                    raise RuntimeError(
                        "SAM produced no usable leaf candidate."
                    )

                best = max(
                    candidates,
                    key=lambda x: x["final_score"],
                )

                cleaned = clean_mask(best["mask"])

                # Recalculate statistics after cleanup so the CSV reflects the
                # actual mask written to disk.
                final_scores = score_mask(
                    image_rgb,
                    cleaned,
                    best["sam_confidence"],
                    best["stability_score"],
                )

                save_rgba(
                    image_bgr,
                    cleaned,
                    output_path,
                )


                mask_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                mask_image = Image.fromarray(
                    cleaned.astype(np.uint8) * 255,
                    mode="L",
                )

                mask_image.save(mask_path)




                if args.save_debug:
                    debug_path = (
                        stats_root
                        / "debug"
                        / relative.parent
                        / f"{image_path.stem}.jpg"
                    )

                    current_debug_count = processed + 1

                    if args.debug_limit == 0 or current_debug_count <= args.debug_limit:
                        save_debug_overlay(
                            image_rgb,
                            cleaned,
                            debug_path,
                        )

                elapsed = time.perf_counter() - t0

                method = best["method"]

                if method == "automatic":
                    automatic_count += 1
                else:
                    center_count += 1

                writer.writerow(
                    {
                        "input_path": str(relative),
                        "output_path": str(
                            output_path.relative_to(output_root)
                        ),
                        "status": "processed",
                        "method": method,
                        "area_ratio": round(
                            final_scores["area_ratio"], 6
                        ),
                        "center_score": round(
                            final_scores["center_score"], 6
                        ),
                        "plant_color_score": round(
                            final_scores["plant_color_score"], 6
                        ),
                        "border_penalty": round(
                            final_scores["border_penalty"], 6
                        ),
                        "sam_confidence": round(
                            final_scores["sam_confidence"], 6
                        ),
                        "final_score": round(
                            final_scores["final_score"], 6
                        ),
                        "inference_seconds": round(elapsed, 3),
                    }
                )

                processed += 1

                progress.set_postfix(
                    processed=processed,
                    failed=failed,
                    skipped=skipped_existing,
                    method=method,
                    refresh=False,
                )

            except Exception as exc:
                failed += 1

                elapsed = time.perf_counter() - t0

                logging.exception(
                    "FAILED: %s | %s",
                    relative,
                    exc,
                )

                writer.writerow(
                    {
                        "input_path": str(relative),
                        "output_path": "",
                        "status": f"failed: {exc}",
                        "method": "",
                        "area_ratio": "",
                        "center_score": "",
                        "plant_color_score": "",
                        "border_penalty": "",
                        "sam_confidence": "",
                        "final_score": "",
                        "inference_seconds": round(elapsed, 3),
                    }
                )

    total_seconds = time.perf_counter() - start_all

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "checkpoint": str(Path(args.checkpoint)),
        "device": device,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "images_found": len(image_paths),
        "processed": processed,
        "skipped": skipped_existing,
        "failed": failed,
        "automatic_selected": automatic_count,
        "center_prompt_selected": center_count,
        "total_seconds": round(total_seconds, 3),
        "images_per_second": round(
            processed / total_seconds,
            4,
        ) if total_seconds > 0 else 0,
        "automatic_mask_settings": {
            "points_per_side": 32,
            "pred_iou_thresh": 0.88,
            "stability_score_thresh": 0.90,
            "crop_n_layers": 1,
            "min_mask_region_area": 100,
        },
        "scoring_weights": {
            "area": 0.20,
            "center": 0.28,
            "plant_color": 0.30,
            "sam_confidence": 0.12,
            "stability": 0.10,
            "border_penalty": -0.25,
        },
    }

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    logging.info("========================================")
    logging.info("SEGMENTATION COMPLETE")
    logging.info("Processed: %d", processed)
    logging.info("Skipped: %d", skipped_existing)
    logging.info("Failed: %d", failed)
    logging.info("Automatic selected: %d", automatic_count)
    logging.info("Center prompt selected: %d", center_count)
    logging.info("Results CSV: %s", csv_path)
    logging.info("Summary JSON: %s", summary_path)
    logging.info("Log: %s", log_path)
    logging.info("========================================")


if __name__ == "__main__":
    main()
