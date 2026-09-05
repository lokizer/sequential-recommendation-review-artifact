# Reproducibility checklist

- Final model identifier: `duospectra_asymmetric_gate_v1`
- Paper datasets: Beauty, LastFM, Toys and Games
- Main seed: 42
- Split: leave-one-out validation and test targets
- Ranking: full item set after excluding padding item 0 and observed history
- Model selection: early stopping with patience 10
- Fusion selection: validation NDCG@20 over coefficients from 0.00 to 1.00 in 0.05 steps
- Test use: selected coefficient is fixed before test evaluation
- Metrics: HR@5/10/20 and NDCG@5/10/20
- Maximum sequence length: 50
- Hidden size: 64
- Encoder layers: 2 per view
- Attention heads: 2
- Batch size: 256
- Optimizer: Adam
- Learning rate: 0.0005
- Epoch cap: 200

The exact dataset-dependent dropout rates and expected outputs are stored in
`configs/paper.json`. Run `python scripts/verify_release.py` before training.
