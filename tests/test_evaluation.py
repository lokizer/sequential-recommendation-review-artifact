import unittest

import numpy as np

from scripts.evaluate_duospectra import batch_ranks, metrics, normalize_candidates


class EvaluationTest(unittest.TestCase):
    def test_normalization_ignores_excluded_scores(self):
        scores = np.array([[1000.0, 1.0, 2.0, 3.0]])
        excluded = np.array([[True, False, False, False]])
        changed = scores.copy()
        changed[0, 0] = -1000.0
        np.testing.assert_allclose(
            normalize_candidates(scores, excluded),
            normalize_candidates(changed, excluded),
        )

    def test_excluded_items_cannot_enter_topk(self):
        scores = np.array([[99.0, 5.0, 4.0, 3.0]])
        excluded = np.array([[True, False, False, False]])
        scores[excluded] = -np.inf
        rank = batch_ranks(scores, np.array([1]), k=3)
        self.assertEqual(rank.tolist(), [1])

    def test_metrics_report_all_cutoffs(self):
        result = metrics(np.array([1, 6, 21]))
        self.assertEqual(
            set(result),
            {"HR@5", "NDCG@5", "HR@10", "NDCG@10", "HR@20", "NDCG@20"},
        )


if __name__ == "__main__":
    unittest.main()
