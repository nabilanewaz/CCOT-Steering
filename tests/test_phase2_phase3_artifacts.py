import os
import tempfile
import unittest
from unittest.mock import patch

import torch

from phase2.run import _hidden_state_cache_usable, run_phase2_source
from phase3.evaluate import _require_phase2_inputs


class HiddenStateCacheTests(unittest.TestCase):
    def test_rejects_cache_without_contrastive_layers(self):
        self.assertFalse(_hidden_state_cache_usable({"H_pos": {}, "H_neg": {}}))
        self.assertFalse(
            _hidden_state_cache_usable(
                {"H_pos": {0: torch.ones(2, 3)}, "H_neg": {}}
            )
        )

    def test_accepts_cache_with_both_classes(self):
        self.assertTrue(
            _hidden_state_cache_usable(
                {
                    "H_pos": {0: torch.ones(2, 3)},
                    "H_neg": {0: torch.ones(2, 3)},
                }
            )
        )

    @patch("phase2.run.collect_hidden_states")
    def test_insufficient_collection_fails_phase2_without_caching(self, collect):
        collection_diag = {
            "per_layer": {
                "0": {"h_pos": 18, "h_neg": 201},
                "1": {"h_pos": 18, "h_neg": 201},
            }
        }
        collect.return_value = ({}, {}, collection_diag)
        model = torch.nn.Linear(2, 2)

        with tempfile.TemporaryDirectory() as vectors_dir:
            with self.assertRaisesRegex(
                RuntimeError, r"largest observed H\+=18, H-=201"
            ):
                run_phase2_source(
                    model=model,
                    tokenizer=object(),
                    D_steer=[{"question": "q", "answer": "#### 1"}],
                    model_tag="test_model",
                    source_tag="ccot",
                    boundary_idx_fn=lambda *_: 0,
                    device="cpu",
                    vectors_dir=vectors_dir,
                    prompt_mode="latent_prompt",
                    N=1,
                    min_samples=200,
                )

            self.assertFalse(
                os.path.exists(
                    os.path.join(vectors_dir, "ccot_hstates_cache.pt")
                )
            )
            self.assertTrue(
                os.path.exists(os.path.join(vectors_dir, "ccot_diagnostics.json"))
            )


class Phase3PreflightTests(unittest.TestCase):
    def test_reports_missing_phase2_vector_before_evaluation(self):
        with tempfile.TemporaryDirectory() as vectors_dir:
            for name in ("phase2_meta.json", "base_dom.pt"):
                with open(os.path.join(vectors_dir, name), "wb") as fp:
                    fp.write(b"present")

            with self.assertRaisesRegex(RuntimeError, "ccot_dom\\.pt"):
                _require_phase2_inputs(vectors_dir)

    def test_accepts_nonempty_required_artifacts(self):
        with tempfile.TemporaryDirectory() as vectors_dir:
            for name in ("phase2_meta.json", "ccot_dom.pt", "base_dom.pt"):
                with open(os.path.join(vectors_dir, name), "wb") as fp:
                    fp.write(b"present")

            _require_phase2_inputs(vectors_dir)


if __name__ == "__main__":
    unittest.main()
