import math
import unittest

from pipeline import evaluate, predict, sample_input, train_model


class PipelineTests(unittest.TestCase):
    def test_evaluation_reports_measured_metrics(self):
        metrics = evaluate()
        self.assertGreater(metrics["test_rows"], 0)
        for key, value in metrics.items():
            if key != "test_rows":
                self.assertTrue(math.isfinite(value), key)

    def test_train_and_predict(self):
        result = predict(train_model(), sample_input())
        self.assertIsInstance(result["duplicate"], bool)
        self.assertTrue(math.isfinite(result["estimated_resolution_hours"]))


if __name__ == "__main__":
    unittest.main()
