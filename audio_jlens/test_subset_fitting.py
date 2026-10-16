import json, os, tempfile
from collections.abc import Sequence
import random

import torch
from torch import nn

from audio_jlens.subset_fitting import subset_fit, subset_fit_experiment
from jacobian_lens.jlens.lens import JacobianLens


class _Tokenizer:
    def __call__(self, text, *, add_special_tokens, return_tensors):
        assert not add_special_tokens
        assert return_tensors == "pt"
        return {"input_ids": torch.zeros((1, len(text.split())), dtype=torch.long)}


class _Processor:
    def apply_chat_template(
        self, conversation, *, add_generation_prompt, tokenize
    ):
        assert add_generation_prompt
        assert not tokenize
        text = conversation[0]["content"][1]["text"]
        return f"<|audio_eos|>{text}<|im_end|><|im_start|>assistant\n"


class _ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)

    def forward(self, hidden):
        return hidden + 0.1 * self.linear(hidden)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_layers = 2
        self.d_model = 4
        self.embedding = nn.Embedding(32, self.d_model)
        self.layers: Sequence[nn.Module] = [_ResidualBlock(), _ResidualBlock()]
        self.tokenizer = _Tokenizer()
        self.encode_calls = 0

    def encode(self, text, *, max_length=300):
        self.encode_calls += 1
        assert max_length == 24
        return torch.arange(24, dtype=torch.long).unsqueeze(0)

    def forward(self, input_ids):
        hidden = self.embedding(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden

    def unembed(self, residual):
        return residual


def _raw_prompt():
    return json.dumps(
        {
            "id": "1",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "path": "sample.wav"},
                        {"type": "text", "text": "Describe this sound."},
                    ],
                }
            ],
        }
    )


def test_subset_fit_encodes_each_raw_record_once():
    model = _Model()

    lens = subset_fit(
        model=model,
        prompts=[_raw_prompt()],
        processor=_Processor(),
        mode="S2T",
        source_layers=[0],
        target_layer=1,
        dim_batch=2,
        max_seq_len=24,
        skip_first=16,
    )

    assert model.encode_calls == 1
    assert lens.n_prompts == 1
    assert lens.jacobians[0].shape == (4, 4)


def test_full_fit_path_needs_no_processor():
    model = _Model()

    lens = subset_fit(
        model=model,
        prompts=[_raw_prompt()],
        mode=None,
        source_layers=[0],
        target_layer=1,
        dim_batch=2,
        max_seq_len=24,
        skip_first=16,
    )

    assert model.encode_calls == 1
    assert lens.n_prompts == 1


def _sample_prompt(sample_id: str, dataset_name: str):
    return json.dumps(
        {
            "id": sample_id,
            "dataset_name": dataset_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "path": f"sample_{sample_id}.wav"},
                        {"type": "text", "text": f"Prompt {sample_id}"},
                    ],
                }
            ],
        }
    )


def test_subset_fit_experiment():
    model = _Model()
    prompts = [
        _sample_prompt("1", "ESC50"),
        _sample_prompt("2", "ESC50"),
        _sample_prompt("3", "AnimalQA"),
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        combined_lens, dataset_lenses, milestone_lenses = subset_fit_experiment(
            model=model,
            prompts=prompts,
            seed=42,
            mode="S2T",
            save_prompt_counts=[1, 2],
            source_layers=[0],
            target_layer=1,
            dim_batch=2,
            max_seq_len=24,
            skip_first=16,
            output_dir=tmp_dir,
            checkpoint_path=os.path.join(tmp_dir, "ckpt.pt"),
            checkpoint_every=1,
            resume=False,
            processor=_Processor(),
        )

        assert combined_lens.n_prompts == 3
        assert set(dataset_lenses.keys()) == {"esc50", "animalqa"}
        assert dataset_lenses["esc50"].n_prompts == 2
        assert dataset_lenses["animalqa"].n_prompts == 1
        assert set(milestone_lenses.keys()) == {1, 2}
        assert milestone_lenses[1].n_prompts == 1
        assert milestone_lenses[2].n_prompts == 2

        # Check saved files
        assert os.path.exists(os.path.join(tmp_dir, "lens_full.pt"))
        assert os.path.exists(os.path.join(tmp_dir, "milestones", "lens_count_1.pt"))
        assert os.path.exists(os.path.join(tmp_dir, "milestones", "lens_count_2.pt"))
        assert os.path.exists(os.path.join(tmp_dir, "datasets", "lens_esc50.pt"))
        assert os.path.exists(os.path.join(tmp_dir, "datasets", "lens_animalqa.pt"))
        assert os.path.exists(os.path.join(tmp_dir, "metadata.json"))

        # Verify dataset lenses merge into combined lens
        merged = JacobianLens.merge([dataset_lenses["esc50"], dataset_lenses["animalqa"]])
        assert torch.allclose(merged.jacobians[0], combined_lens.jacobians[0], atol=1e-6)


def test_subset_fit_experiment_caps_shuffled_prompts_and_validates_resume():
    model = _Model()
    prompts = [
        _sample_prompt(str(sample_id), "ESC50" if sample_id % 2 else "Clotho")
        for sample_id in range(1, 7)
    ]
    seed = 42
    cap = 3
    expected_prompts = prompts.copy()
    random.Random(seed).shuffle(expected_prompts)
    expected_dataset_counts = {}
    for prompt in expected_prompts[:cap]:
        sample = json.loads(prompt)
        dataset = sample["dataset_name"].lower()
        expected_dataset_counts[dataset] = expected_dataset_counts.get(dataset, 0) + 1

    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint_path = os.path.join(tmp_dir, "ckpt.pt")
        combined_lens, dataset_lenses, _ = subset_fit_experiment(
            model=model,
            prompts=prompts,
            seed=seed,
            mode=None,
            save_prompt_counts=[1, cap],
            source_layers=[0],
            target_layer=1,
            dim_batch=2,
            max_seq_len=24,
            skip_first=16,
            output_dir=tmp_dir,
            checkpoint_path=checkpoint_path,
            checkpoint_every=1,
            resume=False,
            max_prompts=cap,
        )

        assert combined_lens.n_prompts == cap
        assert {
            dataset: lens.n_prompts for dataset, lens in dataset_lenses.items()
        } == expected_dataset_counts

        with open(os.path.join(tmp_dir, "metadata.json"), encoding="utf-8") as f:
            metadata = json.load(f)
        assert metadata["total_prompts_input"] == len(prompts)
        assert metadata["total_prompts_selected"] == cap
        assert metadata["max_prompts"] == cap
        assert sum(metadata["dataset_counts"].values()) == cap

        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        assert state["max_prompts"] == cap

        try:
            subset_fit_experiment(
                model=model,
                prompts=prompts,
                seed=seed,
                mode=None,
                source_layers=[0],
                target_layer=1,
                dim_batch=2,
                max_seq_len=24,
                skip_first=16,
                checkpoint_path=checkpoint_path,
                resume=True,
                max_prompts=cap - 1,
            )
        except ValueError as exc:
            assert "max_prompts" in str(exc)
        else:
            raise AssertionError("resuming with a different cap should fail")
