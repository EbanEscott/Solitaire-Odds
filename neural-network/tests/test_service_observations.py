"""Check engine observation ingestion without a checkpoint or running HTTP service."""

import unittest
from types import SimpleNamespace

import torch

from src.action_encoding import ActionSpace
from src.service import _pick_action_and_scores


class ObservationIngestionTest(unittest.TestCase):
    def test_unseen_inventory_reaches_existing_features_in_both_request_formats(self):
        for inventory_key in ("unknownCards", "unknown_cards"):
            with self.subTest(inventory_key=inventory_key):
                captured = []

                def model(state):
                    captured.append(state)
                    return {
                        "policy": torch.zeros((1, 2)),
                        "value": torch.zeros((1, 1)),
                    }

                bundle = SimpleNamespace(
                    model=model,
                    model_family="mlp",
                    device="cpu",
                    action_space=ActionSpace(["turn", "quit"], {"turn": 0, "quit": 1}),
                )
                payload = {
                    "talon": ["??", "??", "A♣"],
                    "stockSize": 21,
                    "legalMoves": ["turn", "quit"],
                    inventory_key: ["2♣", "3♥"],
                    "rules": {"drawCount": 3, "stockVisibility": "topOnly", "maxStockPasses": None},
                }
                command, probability, _ = _pick_action_and_scores(bundle, payload)
                self.assertEqual("turn", command)
                self.assertEqual(0.5, probability)
                # The trailing 236 features are 52 inventory flags plus 184 guess features.
                inventory = captured[0][0, -236:-184]
                self.assertEqual(2, inventory.sum().item())
                self.assertEqual(1, inventory[1].item())  # 2♣
                self.assertEqual(1, inventory[41].item())  # 3♥
                self.assertTrue(torch.isfinite(captured[0]).all())


if __name__ == "__main__":
    unittest.main()
