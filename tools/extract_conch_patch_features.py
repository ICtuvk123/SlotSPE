#!/usr/bin/env python3
"""Extract ordered CONCH patch embeddings with explicit alignment metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

try:
    from .conch_utils import CONCH_MODEL_NAME, load_frozen_conch
except ImportError:  # direct script execution
    from conch_utils import CONCH_MODEL_NAME, load_frozen_conch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "third_party/CONCH/checkpoints/conch/pytorch_model.bin"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}


class ImagePatchDataset(Dataset):
    def __init__(self, paths: Sequence[Path], preprocess: Any) -> None:
        self.paths = list(paths)
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as image:
            return self.preprocess(image.convert("RGB"))


class WSICoordinateDataset(Dataset):
    def __init__(
        self,
        wsi_path: Path,
        coordinates: np.ndarray,
        patch_level: int,
        patch_size: int,
        preprocess: Any,
    ) -> None:
        self.wsi_path = str(wsi_path)
        self.coordinates = coordinates
        self.patch_level = int(patch_level)
        self.patch_size = int(patch_size)
        self.preprocess = preprocess
        self._wsi = None

    def __len__(self) -> int:
        return len(self.coordinates)

    def _slide(self) -> Any:
        if self._wsi is None:
            try:
                import openslide
            except ImportError as exc:
                raise RuntimeError("openslide-python is required for WSI extraction") from exc
            self._wsi = openslide.OpenSlide(self.wsi_path)
        return self._wsi

    def __getitem__(self, index: int) -> torch.Tensor:
        x, y = (int(value) for value in self.coordinates[index])
        image = self._slide().read_region(
            (x, y), self.patch_level, (self.patch_size, self.patch_size)
        ).convert("RGB")
        return self.preprocess(image)


def _load_h5_coordinates(
    path: Path, *, require_patch_metadata: bool = True
) -> tuple[np.ndarray, int | None, int | None]:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required to read CLAM coordinate files") from exc
    with h5py.File(path, "r") as handle:
        if "coords" not in handle:
            raise ValueError(f"HDF5 file lacks 'coords': {path}")
        coords = np.asarray(handle["coords"])
        attrs = handle["coords"].attrs
        if require_patch_metadata:
            if "patch_level" not in attrs or "patch_size" not in attrs:
                raise ValueError(f"Coordinate HDF5 lacks patch_level/patch_size attrs: {path}")
            patch_level = int(attrs["patch_level"])
            patch_size = int(attrs["patch_size"])
        else:
            patch_level = int(attrs["patch_level"]) if "patch_level" in attrs else None
            patch_size = int(attrs["patch_size"]) if "patch_size" in attrs else None
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must have shape [N,2]: {path}")
    return coords, patch_level, patch_size


def _slot_feature_count(path: Path) -> int:
    if path.suffix.casefold() in {".h5", ".hdf5"}:
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError("h5py is required to inspect SlotSPE HDF5 features") from exc
        with h5py.File(path, "r") as handle:
            key = "features" if "features" in handle else next(iter(handle.keys()))
            return int(handle[key].shape[0])
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if isinstance(value, dict):
        value = value.get("features", value.get("embeddings"))
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise ValueError(f"Slot feature file must contain a [N,D] tensor: {path}")
    return int(value.shape[0])


def _hash_order(patch_ids: Sequence[str], coordinates: np.ndarray | None) -> str:
    digest = hashlib.sha256()
    for patch_id in patch_ids:
        digest.update(patch_id.encode("utf-8"))
        digest.update(b"\0")
    if coordinates is not None:
        digest.update(np.ascontiguousarray(coordinates.astype(np.int64)).tobytes())
    return digest.hexdigest()


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Manifest is empty: {path}")
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            payload = payload.get("slides", [payload])
        if not isinstance(payload, list):
            raise ValueError("Manifest JSON must be a slide object or a list of slides")
        records = payload
    except json.JSONDecodeError:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("Every manifest record must be an object")
    base = path.parent
    for record in records:
        record["_base"] = str(base)
    return records


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _record_from_patch_dir(path: Path, slide_id: str | None) -> dict[str, Any]:
    files = sorted(item for item in path.rglob("*") if item.suffix.casefold() in IMAGE_SUFFIXES)
    if not files:
        raise ValueError(f"No supported patch images found in {path}")
    return {
        "slide_id": slide_id or path.name,
        "patch_paths": [str(item.resolve()) for item in files],
        "patch_ids": [str(item.relative_to(path)) for item in files],
        "_base": str(path.resolve()),
    }


def _prepare_record(record: dict[str, Any], preprocess: Any) -> tuple[Dataset, dict[str, Any]]:
    if not record.get("slide_id"):
        raise ValueError("Every record needs slide_id")
    base = Path(record.get("_base", "."))
    coordinates: np.ndarray | None = None
    alignment_verified = False
    alignment_method = "unverified"

    if "patch_paths" in record:
        paths = [_resolve(base, value) for value in record["patch_paths"]]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing patch images, first entries: {missing[:3]}")
        patch_ids = record.get("patch_ids") or [path.name for path in paths]
        if len(patch_ids) != len(paths):
            raise ValueError("patch_ids and patch_paths lengths differ")
        dataset: Dataset = ImagePatchDataset(paths, preprocess)
        slot_patch_ids = record.get("slot_patch_ids")
        if slot_patch_ids is not None:
            alignment_verified = list(slot_patch_ids) == list(patch_ids)
            alignment_method = "exact_patch_id_match" if alignment_verified else "patch_id_mismatch"
        source_patch_size = record.get("patch_size")
        patch_level = record.get("patch_level")
    elif "wsi_path" in record and "coords_h5" in record:
        wsi_path = _resolve(base, record["wsi_path"])
        coords_path = _resolve(base, record["coords_h5"])
        if not wsi_path.is_file() or not coords_path.is_file():
            raise FileNotFoundError(f"Missing WSI or coords file for {record['slide_id']}")
        coordinates, patch_level, source_patch_size = _load_h5_coordinates(coords_path)
        patch_ids = [f"{int(x)}_{int(y)}" for x, y in coordinates]
        assert patch_level is not None and source_patch_size is not None
        dataset = WSICoordinateDataset(
            wsi_path, coordinates, patch_level, source_patch_size, preprocess
        )
        if record.get("slot_feature_h5"):
            slot_coords_path = _resolve(base, record["slot_feature_h5"])
            slot_coords, _, _ = _load_h5_coordinates(
                slot_coords_path, require_patch_metadata=False
            )
            alignment_verified = np.array_equal(coordinates, slot_coords)
            alignment_method = "exact_coordinate_match" if alignment_verified else "coordinate_mismatch"
    else:
        raise ValueError(
            "Record must contain ordered patch_paths, or wsi_path plus coords_h5"
        )

    if record.get("slot_feature_path"):
        slot_path = _resolve(base, record["slot_feature_path"])
        if not slot_path.is_file():
            raise FileNotFoundError(f"Slot feature file does not exist: {slot_path}")
        if _slot_feature_count(slot_path) != len(dataset):
            raise ValueError(f"Slot feature count differs for {record['slide_id']}")

    metadata = {
        "slide_id": str(record["slide_id"]),
        "patch_ids": list(patch_ids),
        "coordinates": None if coordinates is None else torch.from_numpy(coordinates).long(),
        "patch_level": patch_level,
        "source_patch_size": source_patch_size,
        "target_patch_size": 448,
        "magnification": record.get("magnification", 20),
        "alignment_verified": alignment_verified,
        "alignment_method": alignment_method,
        "patch_order_sha256": _hash_order(patch_ids, coordinates),
    }
    return dataset, metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--patch-dir", type=Path)
    parser.add_argument("--slide-id")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--from-hf", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        records = (
            _read_manifest(args.manifest)
            if args.manifest
            else [_record_from_patch_dir(args.patch_dir.resolve(), args.slide_id)]
        )
        model, preprocess, _tokenizer, _tokenize, device = load_frozen_conch(
            checkpoint=args.checkpoint, from_hf=args.from_hf, device=args.device
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for record in records:
            slide_id = str(record.get("slide_id", "unknown"))
            output = args.output_dir / f"{slide_id}.pt"
            if output.exists() and not args.overwrite:
                if args.resume:
                    print(f"Skipping existing: {output}")
                    continue
                raise FileExistsError(
                    f"Output exists: {output}; use --overwrite or --resume"
                )
            dataset, metadata = _prepare_record(record, preprocess)
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
            )
            encoded: list[torch.Tensor] = []
            for images in loader:
                images = images.to(device, non_blocking=True)
                with torch.inference_mode():
                    features = model.encode_image(
                        images, proj_contrast=True, normalize=True
                    )
                encoded.append(F.normalize(features.float(), dim=-1).cpu())
            embeddings = torch.cat(encoded, dim=0)
            if len(embeddings) != len(dataset):
                raise RuntimeError(f"Encoded patch count mismatch for {slide_id}")
            if args.save_dtype == "float16":
                embeddings = embeddings.half()
            artifact = {
                **metadata,
                "conch_patch_embeddings": embeddings,
                "model_name": CONCH_MODEL_NAME,
                "feature_space": "conch_contrastive",
                "normalized": True,
            }
            torch.save(artifact, output)
            status = "verified" if metadata["alignment_verified"] else "UNVERIFIED"
            print(f"Saved {tuple(embeddings.shape)} to {output} ({status} alignment)")
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"CONCH patch extraction failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
