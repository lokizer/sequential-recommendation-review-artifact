"""Validation-calibrated full-ranking evaluation for DuoSpectra."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

from dataset import (  # noqa: E402
    RecDataset,
    generate_rating_matrix_test,
    generate_rating_matrix_valid,
    get_user_seqs,
)
from model.isolated_ablation import IsolatedAblationModel  # noqa: E402
from model.sasfreqrec_v4_1 import SASFreqRecV41Model  # noqa: E402


CONFIG = json.loads((ROOT / "configs" / "paper.json").read_text(encoding="utf-8"))


def model_args(item_size, dataset):
    common = CONFIG["common"]
    dropout = CONFIG["datasets"][dataset]["dropout"]
    return SimpleNamespace(
        item_size=item_size,
        batch_size=common["batch_size"],
        hidden_size=common["hidden_size"],
        max_seq_length=common["max_sequence_length"],
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout,
        num_attention_heads=common["num_attention_heads"],
        num_hidden_layers=common["num_hidden_layers"],
        hidden_act="gelu",
        initializer_range=0.02,
        freq_cutoff=common["frequency_cutoff"],
        spectral_alpha=common["spectral_alpha"],
        high_pass_init=common["high_pass_init"],
        wavelet_init=common["wavelet_init"],
        isolated_ablation_mode="without_time_frequency_gate",
        model_type=CONFIG["model_id"],
        padding_strategy="zero",
    )


def normalize_candidates(scores, excluded):
    scores = np.asarray(scores, dtype=np.float64)
    excluded = np.asarray(excluded, dtype=bool)
    included = ~excluded
    counts = included.sum(axis=1, keepdims=True)
    if np.any(counts == 0):
        raise ValueError("Each user must have at least one reportable candidate.")
    means = np.where(included, scores, 0.0).sum(axis=1, keepdims=True) / counts
    centered = scores - means
    variances = np.where(included, centered * centered, 0.0).sum(
        axis=1, keepdims=True
    ) / counts
    normalized = centered / np.sqrt(variances).clip(1e-8)
    normalized[excluded] = 0.0
    return normalized


def batch_ranks(scores, answers, k=20):
    candidates = np.argpartition(scores, -k, axis=1)[:, -k:]
    candidate_scores = np.take_along_axis(scores, candidates, axis=1)
    order = np.argsort(candidate_scores, axis=1)[:, ::-1]
    topk = np.take_along_axis(candidates, order, axis=1)
    matches = topk == answers[:, None]
    ranks = np.full(len(answers), k + 1, dtype=np.int64)
    found = matches.any(axis=1)
    ranks[found] = matches[found].argmax(axis=1) + 1
    return ranks


def metrics(ranks):
    result = {}
    for k in (5, 10, 20):
        hits = ranks <= k
        result[f"HR@{k}"] = float(hits.mean())
        result[f"NDCG@{k}"] = float(
            np.where(hits, 1.0 / np.log2(ranks + 1), 0.0).mean()
        )
    return result


@torch.no_grad()
def evaluate_grid(original, validity, loader, rating_matrix, device, alphas):
    rank_batches = {float(alpha): [] for alpha in alphas}
    original.eval()
    validity.eval()
    for user_ids, input_ids, answers, _, _ in loader:
        user_ids = user_ids.to(device)
        input_ids = input_ids.to(device)
        answers = answers.numpy().reshape(-1)
        original_hidden = original.predict(input_ids, user_ids)[:, -1, :]
        validity_hidden = validity.predict(input_ids, user_ids)[:, -1, :]
        original_scores = (original_hidden @ original.item_embeddings.weight.T).cpu().numpy()
        validity_scores = (validity_hidden @ validity.item_embeddings.weight.T).cpu().numpy()

        excluded = rating_matrix[user_ids.cpu().numpy()].toarray() > 0
        excluded[:, 0] = True
        if np.any(excluded[np.arange(len(answers)), answers]):
            raise ValueError("An evaluation target was excluded from ranking.")
        original_scores = normalize_candidates(original_scores, excluded)
        validity_scores = normalize_candidates(validity_scores, excluded)
        for alpha in alphas:
            scores = float(alpha) * original_scores + (1.0 - float(alpha)) * validity_scores
            scores[excluded] = -np.inf
            rank_batches[float(alpha)].append(batch_ranks(scores, answers))
    return {
        alpha: metrics(np.concatenate(batches))
        for alpha, batches in rank_batches.items()
    }


def load_state(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=CONFIG["datasets"], required=True)
    parser.add_argument("--original-checkpoint", type=Path)
    parser.add_argument("--validity-checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    original_path = args.original_checkpoint or (
        SRC / "output" / f"DuoSpectra_{args.dataset}_original_seed{CONFIG['seed']}.pt"
    )
    validity_path = args.validity_checkpoint or (
        SRC / "output" / f"DuoSpectra_{args.dataset}_validity_seed{CONFIG['seed']}.pt"
    )
    sequences, max_item, num_users = get_user_seqs(SRC / "data" / f"{args.dataset}.txt")
    model_config = model_args(max_item + 1, args.dataset)
    model_config.batch_size = args.batch_size
    original = SASFreqRecV41Model(model_config)
    validity = IsolatedAblationModel(model_config)
    original.load_state_dict(load_state(original_path))
    validity.load_state_dict(load_state(validity_path))

    if not hasattr(original.item_encoder.blocks[0].layer, "branch_gate"):
        raise RuntimeError("The boundary-preserving view must retain its adaptive gate.")
    if hasattr(validity.item_encoder.blocks[0].layer, "branch_gate"):
        raise RuntimeError("The validity-constrained view must use fixed equal fusion.")

    device = torch.device(args.device)
    original.to(device)
    validity.to(device)
    valid_dataset = RecDataset(model_config, sequences, data_type="valid")
    valid_loader = DataLoader(
        valid_dataset,
        sampler=SequentialSampler(valid_dataset),
        batch_size=args.batch_size,
    )
    test_dataset = RecDataset(model_config, sequences, data_type="test")
    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
    )
    valid_matrix = generate_rating_matrix_valid(sequences, num_users, max_item + 1)
    test_matrix = generate_rating_matrix_test(sequences, num_users, max_item + 1)
    alpha_grid = np.linspace(0.0, 1.0, 21)
    validation = evaluate_grid(
        original, validity, valid_loader, valid_matrix, device, alpha_grid
    )
    selected_alpha = max(
        alpha_grid, key=lambda alpha: validation[float(alpha)]["NDCG@20"]
    )
    test = evaluate_grid(
        original,
        validity,
        test_loader,
        test_matrix,
        device,
        (0.0, float(selected_alpha), 1.0),
    )
    result = {
        "model_id": CONFIG["model_id"],
        "dataset": args.dataset,
        "seed": CONFIG["seed"],
        "selection_metric": "validation NDCG@20",
        "alpha_original": float(selected_alpha),
        "alpha_validity": 1.0 - float(selected_alpha),
        "validation": validation[float(selected_alpha)],
        "test": test[float(selected_alpha)],
        "original_only_test": test[1.0],
        "validity_only_test": test[0.0],
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
