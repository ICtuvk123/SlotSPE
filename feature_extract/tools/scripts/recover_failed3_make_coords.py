#!/usr/bin/env python3
import argparse
import os

import cv2
import h5py
import numpy as np
import openslide
import pandas as pd


def slide_display_id(slide_path, raw_dir):
    parent = os.path.basename(os.path.dirname(slide_path))
    name = os.path.basename(slide_path)
    return f"{parent}--{name}"


def slide_id_for_outputs(slide_path):
    return os.path.splitext(os.path.basename(slide_path))[0]


def is_tissue(tile_rgb, min_tissue_frac):
    hsv = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    tissue = (saturation > 20) & (value < 245)
    return float(np.count_nonzero(tissue)) / float(tissue.size) >= min_tissue_frac


def generate_coords_by_sampling(slide, patch_size, step_size, sample_size, coarse_step, min_tissue_frac):
    width, height = slide.dimensions
    coords = set()

    for y in range(0, max(1, height - patch_size + 1), coarse_step):
        print(f"[info] scanned coarse y={y}/{height}, kept={len(coords)}", flush=True)
        for x in range(0, max(1, width - patch_size + 1), coarse_step):
            sx = min(width - sample_size, max(0, x + coarse_step // 2 - sample_size // 2))
            sy = min(height - sample_size, max(0, y + coarse_step // 2 - sample_size // 2))
            tile = slide.read_region((sx, sy), 0, (sample_size, sample_size)).convert("RGB")
            if is_tissue(np.asarray(tile), min_tissue_frac):
                x_end = min(width - patch_size + 1, x + coarse_step)
                y_end = min(height - patch_size + 1, y + coarse_step)
                for yy in range(y, max(y + 1, y_end), step_size):
                    for xx in range(x, max(x + 1, x_end), step_size):
                        coords.add((xx, yy))

    if not coords:
        return np.empty((0, 2), dtype=np.int32)
    return np.asarray(sorted(coords, key=lambda xy: (xy[1], xy[0])), dtype=np.int32)


def write_preview_mask(slide, coords, out_path, max_dim=4096):
    width, height = slide.dimensions
    scale = max(width / max_dim, height / max_dim, 1.0)
    preview = np.full((max(1, int(height / scale)), max(1, int(width / scale)), 3), 255, dtype=np.uint8)
    for x, y in coords:
        px = int(x / scale)
        py = int(y / scale)
        cv2.circle(preview, (px, py), 1, (0, 0, 255), -1)
    cv2.imwrite(out_path, preview)


def write_h5(path, coords, slide_id, patch_size, slide_dims):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        dset = f.create_dataset(
            "coords",
            data=coords,
            maxshape=(None, 2),
            chunks=(max(1, min(len(coords), 1024)), 2),
            dtype=np.int32,
        )
        dset.attrs["patch_size"] = patch_size
        dset.attrs["patch_level"] = 0
        dset.attrs["wsi_name"] = slide_id
        dset.attrs["name"] = slide_id
        dset.attrs["downsample"] = np.array([1.0, 1.0])
        dset.attrs["level_dim"] = np.array(slide_dims, dtype=np.int64)
        dset.attrs["downsampled_level_dim"] = np.array(slide_dims, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--step-size", type=int, default=512)
    parser.add_argument("--sample-size", type=int, default=128)
    parser.add_argument("--coarse-step", type=int, default=4096)
    parser.add_argument("--min-tissue-frac", type=float, default=0.15)
    args = parser.parse_args()

    patch_dir = os.path.join(args.save_dir, "patches")
    mask_dir = os.path.join(args.save_dir, "masks")
    os.makedirs(patch_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    rows = []
    for parent in sorted(os.listdir(args.raw_dir)):
        pdir = os.path.join(args.raw_dir, parent)
        if not os.path.isdir(pdir):
            continue
        for filename in sorted(os.listdir(pdir)):
            if not filename.endswith(".svs"):
                continue
            slide_path = os.path.join(pdir, filename)
            display_id = slide_display_id(slide_path, args.raw_dir)
            output_id = slide_id_for_outputs(slide_path)
            print(f"[info] processing {display_id}", flush=True)

            slide = openslide.OpenSlide(slide_path)
            coords = generate_coords_by_sampling(
                slide,
                args.patch_size,
                args.step_size,
                args.sample_size,
                args.coarse_step,
                args.min_tissue_frac,
            )
            print(f"[info] {output_id}: generated {len(coords)} coords", flush=True)
            if len(coords) == 0:
                status = "failed_seg"
            else:
                h5_path = os.path.join(patch_dir, output_id + ".h5")
                write_h5(h5_path, coords, output_id, args.patch_size, slide.dimensions)
                status = "processed"

            write_preview_mask(slide, coords, os.path.join(mask_dir, output_id + ".jpg"))

            rows.append(
                {
                    "slide_id": display_id,
                    "process": 0,
                    "status": status,
                    "seg_level": 0,
                    "sthresh": 20,
                    "mthresh": 0,
                    "close": 5,
                    "use_otsu": False,
                    "keep_ids": "none",
                    "exclude_ids": "none",
                    "a_t": 0,
                    "a_h": 0,
                    "max_n_holes": 0,
                    "vis_level": 0,
                    "line_thickness": 100,
                    "use_padding": True,
                    "contour_fn": "thumbnail_mask",
                    "magnification": 40,
                    "seg_ref_size": 512,
                    "patch_ref_size": args.patch_size,
                    "patch_ref_step": args.step_size,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.save_dir, "process_list_autogen.csv"), index=False)
    if not rows:
        raise SystemExit("no .svs files found")
    if any(r["status"] != "processed" for r in rows):
        raise SystemExit("some slides did not produce coordinates")


if __name__ == "__main__":
    main()
