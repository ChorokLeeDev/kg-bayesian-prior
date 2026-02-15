import unittest

import torch
import torch.nn as nn

from src.utils.training import EarlyStopping


class EarlyStoppingSnapshotTest(unittest.TestCase):
    def test_best_snapshot_is_deep_cloned(self) -> None:
        model = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)

        stopper = EarlyStopping(patience=2, mode="max")
        stopper(0.50, model)

        saved_weight = stopper.best_model_state["weight"].clone()

        with torch.no_grad():
            model.weight.fill_(7.0)

        # Snapshot must remain unchanged after model mutation.
        self.assertTrue(torch.allclose(stopper.best_model_state["weight"], saved_weight))

        stopper.load_best_model(model)
        self.assertTrue(torch.allclose(model.weight, saved_weight))


if __name__ == "__main__":
    unittest.main()
