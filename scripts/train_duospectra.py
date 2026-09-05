"""Train the two asymmetric DuoSpectra views under the paper protocol."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CONFIG = json.loads((ROOT / "configs" / "paper.json").read_text(encoding="utf-8"))


def command(dataset, view, seed, gpu_id):
    common = CONFIG["common"]
    dataset_config = CONFIG["datasets"][dataset]
    model_type = "duospectra_original" if view == "original" else "duospectra_validity"
    train_name = f"DuoSpectra_{dataset}_{view}_seed{seed}"
    cmd = [
        sys.executable,
        "-u",
        "main.py",
        "--model_type",
        model_type,
        "--data_name",
        dataset,
        "--train_name",
        train_name,
        "--lr",
        str(common["learning_rate"]),
        "--batch_size",
        str(common["batch_size"]),
        "--epochs",
        str(common["epochs"]),
        "--patience",
        str(common["patience"]),
        "--num_workers",
        "0",
        "--seed",
        str(seed),
        "--gpu_id",
        str(gpu_id),
        "--max_seq_length",
        str(common["max_sequence_length"]),
        "--hidden_size",
        str(common["hidden_size"]),
        "--num_hidden_layers",
        str(common["num_hidden_layers"]),
        "--num_attention_heads",
        str(common["num_attention_heads"]),
        "--hidden_dropout_prob",
        str(dataset_config["dropout"]),
        "--attention_probs_dropout_prob",
        str(dataset_config["dropout"]),
        "--freq_cutoff",
        str(common["frequency_cutoff"]),
        "--spectral_alpha",
        str(common["spectral_alpha"]),
        "--high_pass_init",
        str(common["high_pass_init"]),
        "--wavelet_init",
        str(common["wavelet_init"]),
    ]
    if view == "validity":
        cmd.extend(["--isolated_ablation_mode", "without_time_frequency_gate"])
    return train_name, cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=CONFIG["datasets"], required=True)
    parser.add_argument("--view", choices=("original", "validity", "both"), default="both")
    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    views = ("original", "validity") if args.view == "both" else (args.view,)
    for view in views:
        train_name, cmd = command(args.dataset, view, args.seed, args.gpu_id)
        print(f"[{view}] {' '.join(cmd)}", flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=SRC, check=True)
            checkpoint = SRC / "output" / f"{train_name}.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Training finished without checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
