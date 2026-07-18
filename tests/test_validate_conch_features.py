from __future__ import annotations

import json
import sys

import torch

from tools import validate_conch_features


def _write_manifest(path, slide_ids):
    rows = ["id\tfilename\tmd5\tsize\tstate"]
    rows.extend(
        f"id-{index}\t{slide_id}.svs\tunused\t1\treleased"
        for index, slide_id in enumerate(slide_ids)
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_normalized_float16_conch_directory_validates(tmp_path, monkeypatch):
    feature_dir = tmp_path / "pt_files"
    feature_dir.mkdir()
    manifest = tmp_path / "manifest.tsv"
    report = tmp_path / "report.json"
    slide_ids = ["slide_a", "slide_b"]
    _write_manifest(manifest, slide_ids)

    for slide_id in slide_ids:
        tensor = torch.randn(7, 512)
        tensor = torch.nn.functional.normalize(tensor, dim=-1).half()
        torch.save(tensor, feature_dir / f"{slide_id}.pt")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_conch_features.py",
            "--feature-dir",
            str(feature_dir),
            "--manifest",
            str(manifest),
            "--report",
            str(report),
        ],
    )
    assert validate_conch_features.main() == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["valid_slides"] == 2
    assert payload["total_patches"] == 14
    assert payload["invalid_slides"] == {}


def test_unnormalized_conch_tensor_is_rejected(tmp_path, monkeypatch):
    feature_dir = tmp_path / "pt_files"
    feature_dir.mkdir()
    manifest = tmp_path / "manifest.tsv"
    _write_manifest(manifest, ["slide_a"])
    torch.save(torch.ones(3, 512, dtype=torch.float16), feature_dir / "slide_a.pt")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_conch_features.py",
            "--feature-dir",
            str(feature_dir),
            "--manifest",
            str(manifest),
        ],
    )
    assert validate_conch_features.main() == 1
