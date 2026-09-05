# DuoSpectra

Anonymous reproducibility artifact for **DuoSpectra: Padding-Aware Dual-View
Spectral Learning for Sequential Recommendation**.

DuoSpectra uses two asymmetric encoders:

- a boundary-preserving view that retains the context-adaptive temporal-frequency gate;
- a validity-constrained view that masks invalid positions before rFFT and Haar processing and uses fixed equal temporal-frequency fusion.

The two full-ranking score vectors are standardized over reportable candidate
items. Their fusion coefficient is selected on validation NDCG@20 and is then
fixed for the test set.

## Repository scope

This repository is a research fork of the official BSARec implementation. It
contains only the final paper model and the ablations needed to reproduce the
reported analysis. Historical development variants, manuscript files,
checkpoints, local paths, and training logs are intentionally excluded.

The paper evaluates Beauty, LastFM, and Toys and Games with leave-one-out data
splits and full-item ranking. Seed 42 is used for the main comparison and the
strict single-component ablations.

## Environment

The reported environment uses Python 3.10, PyTorch 2.7.0, NumPy 1.24.3,
SciPy 1.11.1, and tqdm 4.65.0. Create it with:

```bash
conda env create -f environment.yml
conda activate duospectra
```

CUDA is used automatically when available. The code also runs on CPU.

## Verify the artifact

Run the model-identity and dataset-integrity checks before training:

```bash
python scripts/verify_release.py
```

The command verifies the asymmetric encoder definition and the SHA-256 hashes
of the three processed reference datasets.

## Train

Train both views for one dataset:

```bash
python scripts/train_duospectra.py --dataset Beauty --view both --gpu-id 0
python scripts/train_duospectra.py --dataset LastFM --view both --gpu-id 0
python scripts/train_duospectra.py --dataset Toys_and_Games --view both --gpu-id 0
```

Checkpoints and logs are written to `src/output/`. Each view can also be trained
independently with `--view original` or `--view validity`.

## Evaluate

After both checkpoints are available, run validation-calibrated full-ranking
evaluation:

```bash
python scripts/evaluate_duospectra.py --dataset Beauty --output results/beauty.json
python scripts/evaluate_duospectra.py --dataset LastFM --output results/lastfm.json
python scripts/evaluate_duospectra.py --dataset Toys_and_Games --output results/toys.json
```

The evaluator excludes item 0 and each user's observed history before
candidate-wise standardization and Top-K retrieval. It reports HR and NDCG at
5, 10, and 20 for the fused model and both individual views.

## Expected results

`configs/paper.json` records the paper protocol, validation-selected fusion
coefficients, and expected seed-42 test metrics. Small differences may occur
across CUDA, cuDNN, and hardware versions.

## Data

The processed interaction files inherited from BSARec are stored in
`src/data/`. They are reference data derived from public Amazon Review and
HetRec 2011 LastFM datasets, not data collected by the DuoSpectra authors.
Sources, citations, transformations, and checksums are documented in
[`DATA.md`](DATA.md).

## Attribution

This artifact builds on the official BSARec implementation:

> Y. Shin, J. Choi, H. Wi, and N. Park. An Attentive Inductive Bias for
> Sequential Recommendation beyond the Self-Attention. AAAI, 2024.

The BSARec repository in turn acknowledges FMLP-Rec. Rights in upstream code
and third-party datasets remain with their respective owners; see
[`NOTICE.md`](NOTICE.md).

Author identities and a citation entry are omitted from this review artifact
to preserve double-anonymous peer review. They should be added after the review
stage.
