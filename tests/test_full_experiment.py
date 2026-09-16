import json
import os
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from phase1.modeling import Coconut
from phase2.balance import stratified_balance
from phase2.collect import _register_all_hooks, extraction_position
from phase2.config import get_model_config
from phase2.dom import save_multilayer_dom_vectors
from phase2.probe import _gate_check
from phase2.probe_heads import probe_all_heads
from phase3.alpha import tune_alpha
from phase3.hooks import (
    generation_scope, installed_hooks, intervention_positions, make_dom_hook,
    make_cpca_hook, make_iti_hook_for_layer,
)
from phase3.select import select_best_steered_config
from scripts.build_splits import build_all_splits
from utils.artifacts import atomic_json, dataset_fingerprint
from utils.data import select_test_examples
from utils.experiment_config import require_exact_count, split_counts
from evaluate_final import merge_final_summaries


class FullSplitTests(unittest.TestCase):
    def test_full_gsm8k_counts(self):
        self.assertEqual(split_counts(7473), {"D_train": 4484, "D_steer": 747, "D_val": 2242})

    def test_all_pool_rows_are_used_without_overlap(self):
        rows = [{"id": index, "question": f"question {index}", "answer": "#### 1"} for index in range(100)]
        with tempfile.TemporaryDirectory() as root:
            pool = os.path.join(root, "pool.jsonl")
            with open(pool, "w") as stream:
                stream.writelines(json.dumps(row) + "\n" for row in rows)
            parts = build_all_splits(pool, out_dir=root)["S3"]
            repeated = build_all_splits(pool)["S3"]
            self.assertEqual(parts, repeated)
            self.assertEqual([len(parts[role]) for role in ("D_train", "D_steer", "D_val")], [60, 10, 30])
            self.assertEqual({row["id"] for subset in parts.values() for row in subset}, set(range(100)))
            with open(os.path.join(root, "split_manifest.json")) as stream:
                manifest = json.load(stream)
            self.assertEqual(manifest["fingerprints"]["D_val"], dataset_fingerprint(parts["D_val"]))

    @patch("utils.experiment_config.expected_count", return_value=1319)
    def test_test_loader_keeps_entire_test_in_order(self, expected):
        rows = [{"question": str(index)} for index in range(1319)]
        self.assertEqual(select_test_examples(rows), rows)
        with self.assertRaises(ValueError):
            select_test_examples(rows[:300])

    @patch("utils.experiment_config.expected_count", return_value=4484)
    def test_rejects_old_training_sample_count(self, expected):
        with self.assertRaises(ValueError):
            require_exact_count([None] * 300, "D_train")


class IdentityBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Identity()])

    def forward(self, inputs_embeds, **kwargs):
        return self.model.layers[0](inputs_embeds)


class ExtractionTests(unittest.TestCase):
    def test_generated_window_excludes_prompt(self):
        model = IdentityBackbone()
        handles, captured = _register_all_hooks(model)
        try:
            hidden = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)
            model(inputs_embeds=hidden)
            self.assertNotIn(0, captured)
            captured["boundary_range"] = (3, 5)
            model(inputs_embeds=hidden)
            self.assertTrue(torch.equal(captured[0], hidden[:, 3:5].mean(1)))
        finally:
            for handle in handles:
                handle.remove()

    def test_extraction_modes_and_short_rollouts(self):
        ids = torch.ones(1, 8, dtype=torch.long)
        self.assertEqual(extraction_position(ids, 4, None, None, "mean_gen", 20),
                         {"boundary_range": (4, 8)})
        self.assertIsNone(extraction_position(ids, 6, None, None, "mean_gen", 20))
        self.assertEqual(extraction_position(ids, 4, None, None, "first_gen", 20), {"boundary_idx": 4})
        self.assertEqual(extraction_position(ids, 4, None, lambda *_: 6, "boundary", 20), {"boundary_idx": 6})

    def test_balance_downsamples_either_majority(self):
        positive = {0: [torch.tensor(index) for index in range(5)]}
        negative = {0: [torch.tensor(index) for index in range(3)]}
        pos, neg = stratified_balance(positive, negative, {0: [0, 0, 0, 1, 1]}, {0: [0, 1, 1]})
        self.assertEqual(len(pos[0]), 3)
        self.assertEqual(len(neg[0]), 3)

    def test_probe_gate_is_strict_when_requested(self):
        with self.assertRaises(RuntimeError):
            _gate_check({0: 0.55}, 0.55, enforce_gate=True)
        with self.assertRaises(RuntimeError):
            _gate_check({}, 0.55, enforce_gate=True)

    def test_phase2_backbone_settings_match_spec(self):
        math = get_model_config("qwen25_math1.5b")
        self.assertEqual((math["N"], math["r_final"], math["threshold_multiplier"], math["min_samples"]),
                         (10, 8, 0.4, 200))
        small = get_model_config("qwen25_0.5b")
        self.assertEqual((small["r_per_layer"], small["r_final"]), (3, 10))

    def test_multilayer_vectors_are_layer_local(self):
        vectors = {index: torch.eye(4)[index] for index in range(4)}
        with tempfile.TemporaryDirectory() as root:
            path = save_multilayer_dom_vectors(vectors, {0: .6, 1: .8, 2: .7, 3: .9}, "tiny", "ccot", root)
            payload = torch.load(path, weights_only=False)
            self.assertEqual(payload["top_layers"], [3, 1, 2])
            self.assertTrue(torch.equal(payload["layer_vectors"][1], vectors[1]))

    def test_head_probe_saves_projected_scales(self):
        generator = torch.Generator().manual_seed(1)
        pos = {(0, 0): torch.randn(40, 4, generator=generator) + 2}
        neg = {(0, 0): torch.randn(40, 4, generator=generator) - 2}
        payload = probe_all_heads(pos, neg, top_k=1)
        self.assertEqual(payload["top_heads"], [(0, 0)])
        self.assertGreater(payload["head_sigmas"][(0, 0)], 0)
        self.assertAlmostEqual(float(payload["head_directions"][(0, 0)].norm()), 1, places=5)


class SteeringTests(unittest.TestCase):
    def test_uncached_generation_changes_only_generated_positions(self):
        model = IdentityBackbone()
        hook = make_dom_hook(0, torch.tensor([1., 0., 0., 0.]), 2, "cpu")
        with generation_scope(model, 3), installed_hooks(model, [(0, hook)]):
            prompt = model(inputs_embeds=torch.ones(1, 3, 4))
            decoded = model(inputs_embeds=torch.ones(1, 5, 4))
        self.assertTrue(torch.equal(prompt, torch.ones_like(prompt)))
        self.assertTrue(torch.equal(decoded[:, :3], torch.ones(1, 3, 4)))
        self.assertTrue(torch.equal(decoded[:, 3:, 0], torch.full((1, 2), 3.)))

    def test_cached_positions_distinguish_latents_and_decode(self):
        model = IdentityBackbone()
        hook = make_dom_hook(0, torch.ones(4), 1, "cpu")
        hidden = torch.ones(1, 1, 4)
        with generation_scope(model, 5), installed_hooks(model, [(0, hook)]):
            latent = model(inputs_embeds=hidden, position_ids=torch.tensor([[3]]))
            decoded = model(inputs_embeds=hidden, position_ids=torch.tensor([[5]]))
        self.assertTrue(torch.equal(latent, hidden))
        self.assertFalse(torch.equal(decoded, hidden))

    def test_bfloat16_cpca_hook(self):
        model = IdentityBackbone()
        hidden = torch.ones(1, 4, 4, dtype=torch.bfloat16)
        hook = make_cpca_hook(0, torch.eye(4)[:, :2], 1, "cpu")
        with generation_scope(model, 3), installed_hooks(model, [(0, hook)]):
            output = model(inputs_embeds=hidden)
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(output[:, :3], hidden[:, :3]))
        self.assertFalse(torch.equal(output[:, 3:], hidden[:, 3:]))

    def test_iti_only_changes_selected_head(self):
        payload = {"num_heads": 2, "top_heads": [(0, 1)],
                   "head_directions": {(0, 1): torch.tensor([1., 0.])},
                   "head_sigmas": {(0, 1): 2.0}}
        hook = make_iti_hook_for_layer(0, payload, 3, "cpu")
        hidden = torch.ones(1, 1, 4)
        modified = hook(None, (hidden,))[0]
        self.assertTrue(torch.equal(modified[..., :2], hidden[..., :2]))
        self.assertEqual(float(modified[0, 0, 2]), 7.)
        self.assertEqual(float(modified[0, 0, 3]), 1.)
        self.assertIsNone(hook(None, (torch.ones(1, 3, 4),)))

    def test_hooks_removed_after_exception(self):
        model = IdentityBackbone()
        with self.assertRaises(ValueError):
            with generation_scope(model, 3), installed_hooks(model, [(0, make_dom_hook(0, torch.ones(4), 1, "cpu"))]):
                raise ValueError("interrupted")
        self.assertFalse(model._forward_pre_hooks)
        self.assertFalse(model.model.layers[0]._forward_hooks)


def tiny_coconut():
    torch.manual_seed(7)
    config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         eos_token_id=None, pad_token_id=0)
    base = LlamaForCausalLM(config).eval()
    return Coconut(base, latent_token_id=4, start_latent_id=3, end_latent_id=5, eos_token_id=None)


class CoconutIntegrationTests(unittest.TestCase):
    def test_real_coconut_loop_steers_decode_not_latent_prefill(self):
        model = tiny_coconut()
        ids = torch.tensor([[1, 3, 4, 4, 5]])
        positions_seen = []
        def capture(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            positions_seen.append((hidden.shape[-2], intervention_positions(hidden)))
        direction = torch.ones(16)
        handles = [(0, make_dom_hook(0, direction, 1.0, "cpu")), (1, capture)]
        with generation_scope(model, ids.shape[1]), installed_hooks(model, handles):
            output = model.generate(input_ids=ids, max_new_tokens=3)
        self.assertEqual(output.shape[1], 8)
        self.assertTrue(all(not positions for length, positions in positions_seen if length <= 5))
        self.assertIn((6, [5]), positions_seen)
        self.assertIn((7, [5, 6]), positions_seen)

    def test_alpha_has_real_gradient_through_coconut(self):
        model = tiny_coconut()
        vocabulary = {"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "<|start-latent|>": 3,
                      "<|latent|>": 4, "<|end-latent|>": 5, "####": 6,
                      "1": 7, "question": 8, "reason": 9}
        backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
                                            pad_token="[PAD]", eos_token="[EOS]",
                                            additional_special_tokens=["<|start-latent|>", "<|latent|>", "<|end-latent|>"])
        data = [{"question": "question", "answer": "reason\n#### 1"} for _ in range(2)]
        def generate(input_ids, **kwargs):
            return torch.cat([input_ids, torch.tensor([[9, 6, 7]])], dim=1)
        with patch.object(model, "generate", side_effect=generate):
            alpha, history = tune_alpha(model, tokenizer, data, torch.ones(16), 0, "cpu",
                                         latent_tokens=2, max_epochs=1)
        self.assertTrue(torch.isfinite(alpha))
        self.assertGreater(history[0]["grad_norm_theta"], 0)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))


class FinalSummaryTests(unittest.TestCase):
    def summary(self, model, dataset="gsm8k"):
        return {"models": [model], "n_test": 1319,
                "accuracy_table": {model: {"accuracy": 0.5}},
                "provenance": {"eval_dataset": dataset, "dataset_fingerprint": "abc",
                               "experiment_version": "full_60_10_30_v1", "winning_config": "S3"}}

    def test_sequential_models_preserve_previous_results(self):
        merged = merge_final_summaries(self.summary("first"), self.summary("second"))
        self.assertEqual(merged["models"], ["first", "second"])
        self.assertEqual(set(merged["accuracy_table"]), {"first", "second"})
        self.assertEqual(merged["provenance"]["models"], merged["models"])

    def test_different_evaluation_datasets_cannot_be_combined(self):
        with self.assertRaises(RuntimeError):
            merge_final_summaries(self.summary("first"), self.summary("second", "svamp"))


class SelectionTests(unittest.TestCase):
    def test_multilayer_and_iti_are_eligible_but_trimmed_and_controls_are_not(self):
        with tempfile.TemporaryDirectory() as root:
            atomic_json(os.path.join(root, "phase2_meta.json"), {"ccot_probe_gate": {"gate_passed": True}})
            records = []
            for condition, method, accuracy in (
                ("ccot_L6", None, .5), ("dom_L6_ccot", "dom", .6),
                ("multilayer_dom_L6_ccot", "multilayer_dom", .7),
                ("iti_L6_ccot", "iti", .8), ("noise_L6_ccot", "noise", .99),
                ("trimmed_dom_L6", "dom", 1.0),
            ):
                records.append({"condition": condition, "vector_method": method,
                                "accuracy": accuracy, "vector_source": "ccot",
                                "n_examples": 2242, "flip_rate": .1, "alpha": 1.})
            atomic_json(os.path.join(root, "phase3_val.json"), records)
            selected = select_best_steered_config(root, "tiny")
            self.assertEqual(selected["best_condition"], "iti_L6_ccot")


if __name__ == "__main__":
    unittest.main()
