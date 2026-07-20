import unittest

from block_size_sweep import annotate_recommendations, percentile


def result_row(block_size, median_ms, p95_ms):
    return {
        "model_class": "SLM",
        "model_name": "synthetic/slm",
        "framework": "pace",
        "phase": "decode_generation",
        "requested_block_size": block_size,
        "effective_block_size": block_size,
        "input_tokens": 512,
        "output_tokens": 128,
        "batch_size": 1,
        "kv_cache_type": "SLAB_POOL",
        "median_generation_latency_ms": median_ms,
        "p95_generation_latency_ms": p95_ms,
    }


class TestBlockSizeSweep(unittest.TestCase):
    def test_p95_uses_linear_interpolation(self):
        self.assertAlmostEqual(percentile(list(range(1, 11)), 0.95), 9.55)

    def test_complete_workload_gets_stable_recommendation(self):
        rows = [
            result_row(16, 12.0, 12.5),
            result_row(32, 11.0, 11.5),
            result_row(64, 10.0, 10.4),
            result_row(128, 10.7, 11.0),
            result_row(256, 11.5, 12.0),
        ]
        annotate_recommendations(rows, {16, 32, 64, 128, 256}, 5.0)
        winner = next(row for row in rows if row["is_recommended"])
        self.assertEqual(winner["recommended_block_size"], 64)
        self.assertEqual(winner["rank_by_median"], 1)
        self.assertTrue(winner["recommended_p95_not_worse"])
        self.assertTrue(winner["recommendation_stable"])

    def test_incomplete_workload_has_no_recommendation(self):
        rows = [result_row(16, 12.0, 12.5), result_row(32, 11.0, 11.5)]
        annotate_recommendations(rows, {16, 32, 64, 128, 256}, 5.0)
        self.assertTrue(all(row["recommended_block_size"] == "" for row in rows))


if __name__ == "__main__":
    unittest.main()
