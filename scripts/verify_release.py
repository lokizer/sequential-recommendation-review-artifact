"""Run inexpensive checks before training or archiving the artifact."""

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

from scripts.evaluate_duospectra import model_args  # noqa: E402
from model.isolated_ablation import IsolatedAblationModel  # noqa: E402
from model.sasfreqrec_v4_1 import SASFreqRecV41Model  # noqa: E402


EXPECTED_HASHES = {
    "Beauty.txt": "f85a84ab47d70e82bf0e7301804270af10fb51e127ab8f0885697d593f6f6d43",
    "LastFM.txt": "46767eacbf73bce98cc6c402671933a22ede8d73d3433c0e164d6b1868775a62",
    "Toys_and_Games.txt": "fb28d2f6bc01d38e0da8ad88265d62043364f1a908e8dbf3591d5bba68c9d789",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    config = json.loads((ROOT / "configs" / "paper.json").read_text(encoding="utf-8"))
    for filename, expected in EXPECTED_HASHES.items():
        path = SRC / "data" / filename
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"Dataset checksum mismatch: {filename}: {actual}")

    args = model_args(item_size=100, dataset="Beauty")
    original = SASFreqRecV41Model(args)
    validity = IsolatedAblationModel(args)
    if not hasattr(original.item_encoder.blocks[0].layer, "branch_gate"):
        raise RuntimeError("Boundary-preserving view lost its adaptive gate.")
    if hasattr(validity.item_encoder.blocks[0].layer, "branch_gate"):
        raise RuntimeError("Validity-constrained view unexpectedly has a gate.")
    print(f"PASS: {config['model_id']}")
    print("PASS: three reference datasets match the recorded SHA-256 checksums")


if __name__ == "__main__":
    main()
