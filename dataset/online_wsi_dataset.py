"""Online raw-WSI patch loading for end-to-end CONCH adaptation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import h5py
import numpy as np
import openslide
import torch
from PIL import Image
from torchvision import transforms

from dataset.dataset_survival import SurvivalDataset


class OnlineWSISurvivalDataset(SurvivalDataset):
    """Return sampled raw patch tensors instead of precomputed embeddings."""

    def __init__(
        self,
        dataset_factory,
        raw_wsi_dir,
        patch_coords_dir,
        split_key="train",
        fold=None,
        target_patch_size=448,
        sample_seed=3,
    ):
        super().__init__(
            dataset_factory,
            wsi_path="",
            split_key=split_key,
            fold=fold,
            encoding_dim=768,
        )
        if dataset_factory.num_patches is None or dataset_factory.num_patches <= 0:
            raise ValueError("online WSI training requires a positive num_patches")
        self.raw_wsi_dir = Path(raw_wsi_dir).expanduser().resolve()
        self.patch_coords_dir = Path(patch_coords_dir).expanduser().resolve()
        self.target_patch_size = int(target_patch_size)
        self.sample_seed = int(sample_seed)
        if self.target_patch_size <= 0:
            raise ValueError("target_patch_size must be positive")

        self.slide_paths = self._index_unique_files(self.raw_wsi_dir, (".svs",))
        self.coords_paths = self._index_unique_files(self.patch_coords_dir, (".h5",))
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    self.target_patch_size,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                transforms.CenterCrop((self.target_patch_size, self.target_patch_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        self._validate_slide_assets()

    @staticmethod
    def _index_unique_files(root: Path, suffixes: tuple[str, ...]) -> dict[str, Path]:
        if not root.is_dir():
            raise FileNotFoundError(f"directory not found: {root}")
        index = {}
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.casefold() not in suffixes:
                continue
            stem = path.stem
            if stem in index:
                raise ValueError(f"duplicate slide stem {stem!r}: {index[stem]} and {path}")
            index[stem] = path
        return index

    def _validate_slide_assets(self) -> None:
        missing_raw = []
        missing_coords = []
        for slides in self.label_df["wsi"]:
            for slide_id in self._slide_ids(slides):
                stem = self._slide_stem(slide_id)
                if stem not in self.slide_paths:
                    missing_raw.append(stem)
                if stem not in self.coords_paths:
                    missing_coords.append(stem)
        if missing_raw or missing_coords:
            raise FileNotFoundError(
                "online WSI assets are incomplete; "
                f"missing_raw={missing_raw[:5]}, missing_coords={missing_coords[:5]}"
            )

    def _rng_for_case(self, case_id: str) -> np.random.Generator:
        if self.split_key == "train":
            return np.random.default_rng(np.random.randint(0, 2**32 - 1))
        digest = hashlib.sha256(
            f"{self.sample_seed}:{self.fold}:{case_id}".encode("utf-8")
        ).digest()
        return np.random.default_rng(int.from_bytes(digest[:8], "little"))

    def _sample_locations(self, slide_ids: list[str], case_id: str):
        coordinate_parts = []
        patch_metadata = []
        for slide_index, slide_id in enumerate(slide_ids):
            stem = self._slide_stem(slide_id)
            with h5py.File(self.coords_paths[stem], "r") as handle:
                dataset = handle["coords"]
                coords = np.asarray(dataset, dtype=np.int64)
                patch_level = int(dataset.attrs["patch_level"])
                patch_size = int(dataset.attrs["patch_size"])
            if len(coords) == 0:
                continue
            coordinate_parts.append(coords)
            patch_metadata.append((slide_index, patch_level, patch_size, len(coords)))

        total = sum(item[3] for item in patch_metadata)
        if total == 0:
            raise ValueError(f"no patch coordinates found for case {case_id}")
        count = int(self.dataset_factory.num_patches)
        rng = self._rng_for_case(case_id)
        selected = np.sort(rng.choice(total, size=count, replace=total < count))

        locations = []
        offset = 0
        for coords, (slide_index, patch_level, patch_size, length) in zip(
            coordinate_parts, patch_metadata
        ):
            local = selected[(selected >= offset) & (selected < offset + length)] - offset
            locations.extend(
                (slide_index, coords[index], patch_level, patch_size) for index in local
            )
            offset += length
        return locations

    def _load_patches(self, slide_ids: list[str], case_id: str) -> torch.Tensor:
        locations = self._sample_locations(slide_ids, case_id)
        grouped = {}
        for output_index, location in enumerate(locations):
            grouped.setdefault(location[0], []).append((output_index, *location[1:]))

        patches = [None] * len(locations)
        for slide_index, requests in grouped.items():
            stem = self._slide_stem(slide_ids[slide_index])
            slide = openslide.OpenSlide(str(self.slide_paths[stem]))
            try:
                for output_index, coord, patch_level, patch_size in requests:
                    image = slide.read_region(
                        tuple(map(int, coord)),
                        int(patch_level),
                        (int(patch_size), int(patch_size)),
                    ).convert("RGB")
                    patches[output_index] = self.transform(image)
            finally:
                slide.close()
        return torch.stack(patches)

    def __getitem__(self, idx):
        case_id = self.label_df.loc[idx, "case id"]
        slide_ids = self._slide_ids(self.label_df.loc[idx, "wsi"])
        label, event_time, censorship = self.get_label(case_id)
        patches = self._load_patches(slide_ids, case_id)
        genes = self.load_genes(case_id)
        return patches, genes, label, event_time, censorship

