import json
import os
import tempfile
import types
import unittest

import torch

from phase1.data import SimpleDataset
from phase1.modeling import Coconut
from phase1.train import (
    BEST_DIRNAME,
    COT_BEST_DIRNAME,
    CURRICULUM_VERSION,
    C_THOUGHT,
    MAX_LATENT_TOKENS,
    MODEL_HPARAMS,
    VALIDATION_EPOCHS,
    _DEFAULT_HP,
    _build_stage_dataset,
    _get_stage_info,
    _materialize_alias_dir,
    _should_run_validation,
    export_compat_checkpoints,
)


class CurriculumDatasetTests(unittest.TestCase):
    def setUp(self):
        self.sample = SimpleDataset([
            {
                "question_tokenized": [10, 11],
                "steps_tokenized": [[20], [21], [22], [23]],
                "answer_tokenized": [30],
            }
        ])
        self.start_id = 97
        self.latent_id = 98
        self.end_id = 99

    def build(self, stage, drop_remaining=False):
        return _build_stage_dataset(
            self.sample,
            stage,
            drop_remaining,
            self.start_id,
            self.latent_id,
            self.end_id,
        )[0]

    def test_stage_zero_is_full_visible_cot(self):
        item = self.build(0)
        self.assertEqual(item["input_ids"], [10, 11, 20, 21, 22, 23, 30])
        self.assertEqual(item["labels"], [-100, -100, 20, 21, 22, 23, 30])
        self.assertNotIn(self.latent_id, item["input_ids"])

    def test_latent_stages_remove_steps_from_the_front(self):
        stage_one = self.build(1)
        self.assertEqual(
            stage_one["input_ids"],
            [10, 11, 97, 98, 98, 99, 21, 22, 23, 30],
        )
        self.assertEqual(stage_one["labels"][: 2 + C_THOUGHT + 2], [-100] * 6)

        stage_three = self.build(3)
        self.assertEqual(stage_three["input_ids"].count(self.latent_id), 6)
        self.assertEqual(stage_three["input_ids"][-2:], [23, 30])

    def test_final_stage_is_fully_latent(self):
        final = self.build(4, drop_remaining=True)
        self.assertEqual(final["input_ids"].count(self.latent_id), MAX_LATENT_TOKENS)
        self.assertEqual(
            final["input_ids"],
            [10, 11, 97] + [98] * MAX_LATENT_TOKENS + [99, 30],
        )
        self.assertEqual(final["labels"][-1], 30)
        self.assertTrue(all(label == -100 for label in final["labels"][:-1]))

    def test_phase1_runs_for_30_epochs(self):
        self.assertEqual(_DEFAULT_HP["epochs"], 30)
        self.assertEqual({hp["epochs"] for hp in MODEL_HPARAMS.values()}, {30})

    def test_paper_epoch_schedule(self):
        expected = {
            0: (0, False, True),
            5: (0, False, False),
            6: (1, False, True),
            9: (2, False, True),
            12: (3, False, True),
            15: (4, True, True),
            29: (4, True, False),
        }
        for epoch, stage_info in expected.items():
            self.assertEqual(_get_stage_info(epoch), stage_info)

    def test_validation_runs_at_stage_zero_end_and_every_ten_epochs(self):
        scheduled = [
            epoch for epoch in range(1, 31) if _should_run_validation(epoch)
        ]
        self.assertEqual(VALIDATION_EPOCHS, (6, 10, 20, 30))
        self.assertEqual(scheduled, [6, 10, 20, 30])


class CompatibilityExportTests(unittest.TestCase):
    def make_checkpoint(self, root, dirname, uses_wrapper):
        path = os.path.join(root, dirname)
        os.makedirs(path)
        with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
            json.dump({}, f)
        with open(os.path.join(path, "coconut_meta.json"), "w", encoding="utf-8") as f:
            json.dump({
                "curriculum_version": CURRICULUM_VERSION,
                "uses_coconut_wrapper": uses_wrapper,
            }, f)

    def test_export_keeps_cot_and_coconut_sources_distinct(self):
        with tempfile.TemporaryDirectory() as root:
            self.make_checkpoint(root, BEST_DIRNAME, True)
            self.make_checkpoint(root, COT_BEST_DIRNAME, False)
            export_compat_checkpoints(root, latent_token_counts=[3])
            with open(os.path.join(root, "cot", "coconut_meta.json"), encoding="utf-8") as f:
                cot_meta = json.load(f)
            with open(os.path.join(root, "ccot_L3", "coconut_meta.json"), encoding="utf-8") as f:
                ccot_meta = json.load(f)
            self.assertFalse(cot_meta["uses_coconut_wrapper"])
            self.assertTrue(ccot_meta["uses_coconut_wrapper"])
            self.assertTrue(os.path.samefile(
                os.path.join(root, BEST_DIRNAME, "config.json"),
                os.path.join(root, "ccot_L3", "config.json"),
            ))

    def test_export_rejects_mislabeled_coconut_checkpoint(self):
        with tempfile.TemporaryDirectory() as root:
            self.make_checkpoint(root, BEST_DIRNAME, False)
            self.make_checkpoint(root, COT_BEST_DIRNAME, False)
            with self.assertRaises(RuntimeError):
                export_compat_checkpoints(root, latent_token_counts=[3])

    def test_alias_replaces_a_stale_physical_checkpoint_copy(self):
        with tempfile.TemporaryDirectory() as root:
            source = os.path.join(root, "source")
            alias = os.path.join(root, "alias")
            os.makedirs(source)
            os.makedirs(alias)
            with open(os.path.join(source, "model.safetensors"), "wb") as f:
                f.write(b"selected-best")
            with open(os.path.join(alias, "stale.safetensors"), "wb") as f:
                f.write(b"stale-copy")

            _materialize_alias_dir(source, alias)

            self.assertFalse(os.path.exists(os.path.join(alias, "stale.safetensors")))
            self.assertTrue(os.path.samefile(
                os.path.join(source, "model.safetensors"),
                os.path.join(alias, "model.safetensors"),
            ))


class FakeCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(12, 4)
        with torch.no_grad():
            values = torch.arange(48, dtype=torch.float32).reshape(12, 4)
            self.embedding.weight.copy_(values)
        self.config = types.SimpleNamespace(pad_token_id=0, eos_token_id=9)
        self.generation_config = types.SimpleNamespace(pad_token_id=0, eos_token_id=9)
        self.embed_calls = []

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        del kwargs
        if input_ids is not None:
            raise AssertionError("Generation reverted to literal input_ids")
        self.embed_calls.append(inputs_embeds.detach().clone())
        hidden = inputs_embeds + 0.5
        logits = torch.zeros(
            inputs_embeds.shape[0], inputs_embeds.shape[1], 12,
            device=inputs_embeds.device,
        )
        next_id = 9 if inputs_embeds.shape[1] >= 4 else 7
        logits[:, -1, next_id] = 10.0
        return types.SimpleNamespace(
            logits=logits,
            hidden_states=(hidden,),
            past_key_values=None,
        )


class RecurrentGenerationTests(unittest.TestCase):
    def test_generation_keeps_substituted_latent_embeddings(self):
        base = FakeCausalLM()
        model = Coconut(
            base,
            latent_token_id=2,
            start_latent_id=1,
            end_latent_id=3,
            eos_token_id=9,
        )
        input_ids = torch.tensor([[1, 2, 3]])
        output = model.generate(input_ids=input_ids, max_new_tokens=2)

        self.assertEqual(output.tolist(), [[1, 2, 3, 7, 9]])
        self.assertEqual(output.device, input_ids.device)
        self.assertGreaterEqual(len(base.embed_calls), 3)
        expected_latent = base.embed_calls[0][0, -1] + 0.5
        self.assertTrue(torch.equal(base.embed_calls[1][0, 1], expected_latent))


if __name__ == "__main__":
    unittest.main()
