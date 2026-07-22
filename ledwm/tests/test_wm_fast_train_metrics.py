import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "messenger-emma"))

from ledwm.WM import fast_train_metrics  # noqa: E402


class FastTrainMetricsTest(unittest.TestCase):
    def test_keeps_replay_priority_fields_only(self):
        data = {
            "sample_id": np.array([[1, 2], [3, 4]], dtype=np.int64),
            "reward": np.ones((2, 3), dtype=np.float32),
        }
        priority_loss = np.array([1.0, 2.0], dtype=np.float32)
        sup_loss = np.array([3.0, 4.0], dtype=np.float32)
        rollout_loss = np.array([5.0, 6.0], dtype=np.float32)

        metrics = fast_train_metrics(data, priority_loss, sup_loss, rollout_loss)

        self.assertEqual(
            set(metrics),
            {
                "priority_loss_per_batch",
                "sup_loss_per_batch",
                "rollout_loss_per_batch",
                "sample_id",
            },
        )
        self.assertIs(metrics["priority_loss_per_batch"], priority_loss)
        self.assertIs(metrics["sup_loss_per_batch"], sup_loss)
        self.assertIs(metrics["rollout_loss_per_batch"], rollout_loss)
        self.assertIs(metrics["sample_id"], data["sample_id"])
