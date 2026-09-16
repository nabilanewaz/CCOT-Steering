import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils import dataset_paths
from utils.experiment_config import expected_count
from scripts.build_splits import build_all_splits
from evaluate_final import merge_final_summaries


class DatasetWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.previous = dataset_paths._ACTIVE

    def tearDown(self):
        dataset_paths._ACTIVE = self.previous

    def test_explicit_svamp_overrides_inherited_gsm8k_environment(self):
        with patch.dict(os.environ, {"CCOT_DATASET": "gsm8k"}):
            dataset_paths.init_project_dataset("svamp", persist=False)
            self.assertEqual(dataset_paths.get_train_pool_path(), "svamp/train.jsonl")
            self.assertEqual(dataset_paths.phase4_subprocess_env()["CCOT_DATASET"], "svamp")

    def test_independent_paths_preserve_gsm8k_layout(self):
        for kind in ("checkpoints", "vectors", "results", "cache"):
            self.assertEqual(dataset_paths.artifact_root(kind, "gsm8k"), kind)
            self.assertEqual(dataset_paths.artifact_root(kind, "svamp"), f"{kind}/svamp")
        self.assertEqual(dataset_paths.selected_config_path("gsm8k"), "configs/selected.yaml")
        self.assertEqual(dataset_paths.selected_config_path("svamp"), "configs/svamp/selected.yaml")
        self.assertEqual(dataset_paths.split_output_dir("svamp"), "configs/svamp/splits")

    def test_full_local_svamp_partition(self):
        dataset_paths.set_active_dataset("svamp", persist=False)
        root = Path(__file__).resolve().parents[1]
        parts = build_all_splits(root / "svamp/train.jsonl")["S3"]
        self.assertEqual({key: len(rows) for key, rows in parts.items()},
                         {"D_train": 420, "D_steer": 70, "D_val": 210})
        for role, rows in parts.items():
            self.assertEqual(len(rows), expected_count(role))
            self.assertTrue(all(row["question"] and "####" in row["answer"] for row in rows))
        with (root / "svamp/test.jsonl").open() as stream:
            test = [json.loads(line) for line in stream if line.strip()]
        self.assertEqual(len(test), expected_count("D_test"))
        key = lambda row: " ".join(row["question"].split()).casefold()
        self.assertFalse({key(row) for rows in parts.values() for row in rows}
                         & {key(row) for row in test})

    def test_selection_write_is_scoped_to_svamp(self):
        import pipeline
        dataset_paths.set_active_dataset("svamp", persist=False)
        previous_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                pipeline._update_selected_phase3_best("tiny", {"best_condition": "dom_L6_ccot"})
                self.assertTrue(Path("configs/svamp/selected.yaml").exists())
                self.assertFalse(Path("configs/selected.yaml").exists())
                self.assertIn("training_dataset: svamp", Path("configs/svamp/selected.yaml").read_text())
            finally:
                os.chdir(previous_cwd)

    def test_transfer_and_independent_results_cannot_be_merged(self):
        previous = {"provenance": {"eval_dataset": "svamp", "training_dataset": "gsm8k"}}
        current = {"provenance": {"eval_dataset": "svamp", "training_dataset": "svamp"}}
        with self.assertRaisesRegex(RuntimeError, "training_dataset"):
            merge_final_summaries(previous, current)

    def check_final_routing(self, training_dataset, expected_root, output_dir):
        import evaluate_final
        argv = ["evaluate_final.py", "--dataset", "svamp", "--model", "qwen25_0.5b"]
        if training_dataset == "gsm8k":
            argv += ["--training-dataset", "gsm8k"]
        with patch("sys.argv", argv), \
             patch.object(evaluate_final, "init_project_dataset", side_effect=lambda *a, **k: dataset_paths.set_active_dataset("svamp", persist=False)), \
             patch.object(evaluate_final.os.path, "exists", return_value=False), \
             patch.object(evaluate_final, "_load_selected_yaml", return_value={"winning_config": "S3", "training_dataset": training_dataset}) as selected, \
             patch.object(evaluate_final, "load_test_set", return_value=[]), \
             patch.object(evaluate_final, "run_final_evaluation", return_value={}) as run, \
             patch.object(evaluate_final, "save_final_results") as save:
            evaluate_final.main()
        selected.assert_called_once_with(dataset_paths.selected_config_path(training_dataset))
        self.assertEqual(run.call_args.kwargs["checkpoints_base"], expected_root)
        self.assertEqual(save.call_args.args[1], output_dir)
        self.assertEqual(save.call_args.kwargs["provenance"]["training_dataset"], training_dataset)

    def test_independent_svamp_final_uses_svamp_checkpoints(self):
        self.check_final_routing("svamp", "checkpoints/svamp", "results/svamp/final")

    def test_explicit_transfer_uses_gsm8k_checkpoints(self):
        self.check_final_routing("gsm8k", "checkpoints", "results/final_svamp_transfer")


if __name__ == "__main__":
    unittest.main()
