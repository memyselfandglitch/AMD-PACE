import unittest

from analyze_block_size_latency import (
    autotune_block_size,
    extract_readings,
    percentile,
)


class TestBlockSizeLatencyAnalysis(unittest.TestCase):
    def test_p95_uses_linear_interpolation(self):
        self.assertAlmostEqual(percentile([1, 2, 3, 4, 5], 0.95), 4.8)

    def test_raw_generation_times_support_distribution_statistics(self):
        readings, source = extract_readings(
            {"generation_times": [1.2, 1.0, 1.4], "average_gen_time": 1.2}
        )
        self.assertEqual(readings, [1.2, 1.0, 1.4])
        self.assertEqual(source, "raw_generation_times")

    def test_old_results_are_identified_as_aggregate_only(self):
        readings, source = extract_readings({"average_gen_time": 1.25})
        self.assertEqual(readings, [1.25])
        self.assertEqual(source, "aggregate_mean_only")

    def test_job_7055_auto_block_sizes_for_one_mib_l2(self):
        one_mib = 1024 * 1024
        self.assertEqual(autotune_block_size("Qwen/Qwen2.5-0.5B", one_mib), 256)
        self.assertEqual(
            autotune_block_size("Qwen/Qwen2.5-7B-Instruct", one_mib), 128
        )


if __name__ == "__main__":
    unittest.main()
