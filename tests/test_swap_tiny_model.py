"""Run swaps through the real adapters with tiny, randomly initialized Qwen models.

The architecture, processor, chat template, and tokenizer are real, but the
weights are random, so these tests check the wiring and not any model behavior.
Only processor and tokenizer files are downloaded. Tests skip without network
access or the real transformers and librosa packages.
"""

import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

if any(isinstance(sys.modules.get(name), MagicMock) for name in ("transformers", "librosa")):
    pytest.skip("transformers and librosa are required", allow_module_level=True)

pytest.importorskip("jlens")
pytest.importorskip("jacobian_lens.jlens.lens")
soundfile = pytest.importorskip("soundfile")

from transformers import (
    Qwen2AudioConfig,
    Qwen2AudioForConditionalGeneration,
    Qwen2AudioProcessor,
)

from audio_jlens.jlens import Jlens
from model.adapters import qwen2_audio, qwen2_audio_instruct
from model.model import Model
from tools.experiments import Experiments

N_LAYERS = 4
D_MODEL = 32
AUDIO_MODELS = {
    "Qwen/Qwen2-Audio-7B": qwen2_audio,
    "Qwen/Qwen2-Audio-7B-Instruct": qwen2_audio_instruct,
}


def _tiny_hf_model(dtype: torch.dtype):
    torch.manual_seed(0)
    config = Qwen2AudioConfig(
        audio_config={
            "num_mel_bins": 128,
            "encoder_layers": 1,
            "encoder_attention_heads": 2,
            "encoder_ffn_dim": 16,
            "d_model": 8,
        },
        text_config={
            "hidden_size": D_MODEL,
            "intermediate_size": 64,
            "num_hidden_layers": N_LAYERS,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "vocab_size": 156032,
        },
        audio_token_index=151646,
    )
    return Qwen2AudioForConditionalGeneration(config).to(dtype).eval()


@pytest.fixture(scope="module", params=list(AUDIO_MODELS))
def tiny(request, tmp_path_factory):
    model_id = request.param
    try:
        processor = Qwen2AudioProcessor.from_pretrained(model_id)
    except OSError as exc:
        pytest.skip(f"Could not download the processor: {exc}")

    root = tmp_path_factory.mktemp("tiny_audio")
    soundfile.write(root / "audio.wav", np.sin(np.linspace(0, 800, 16000)), 16000)
    line = json.dumps({
        "id": "0",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "audio", "path": "audio.wav"},
                {"type": "text", "text": "What animal is making this sound?"},
            ],
        }],
        "transcript": "dog",
        "dataset_name": "tiny",
    })

    module = AUDIO_MODELS[model_id]
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        module,
        "load_model",
        lambda _id, _do_4bit=False: (_tiny_hf_model(torch.bfloat16), processor),
    )
    model = Model(logging.getLogger("test"), model_id, model_id)
    instance = model.load_model()
    generator = torch.Generator().manual_seed(1)
    lens = Jlens(logging.getLogger("test"), instance)
    lens.lens = SimpleNamespace(
        jacobians={
            layer: torch.randn(D_MODEL, D_MODEL, generator=generator)
            for layer in range(N_LAYERS - 1)
        },
        source_layers=list(range(N_LAYERS - 1)),
        d_model=D_MODEL,
    )
    model.lens = lens
    yield SimpleNamespace(model=model, instance=instance, root=root, line=line)
    monkeypatch.undo()


def _token(tiny, word: str) -> int:
    token_id = Experiments.swap_single_token(tiny.instance.tokenizer, word)
    assert token_id is not None
    return token_id


def _swap(tiny, **overrides):
    arguments = {
        "jsonl_line": tiny.line,
        "data_root": str(tiny.root),
        "source_token": _token(tiny, "cat"),
        "target_token": _token(tiny, "dog"),
        "layers": [1, 2],
    }
    return tiny.model.run_swap_inference(**{**arguments, **overrides})


@pytest.mark.parametrize("direction_source", ["jlens", "logit", "random"])
def test_swap_changes_the_logits_on_the_real_architecture(tiny, direction_source):
    result = _swap(tiny, direction_source=direction_source)
    vocab = tiny.instance.hf_model.lm_head.weight.shape[0]
    assert result.baseline_logits.shape == (1, vocab)
    assert result.swapped_logits.shape == (1, vocab)
    assert torch.isfinite(result.swapped_logits).all()
    assert not torch.equal(result.baseline_logits, result.swapped_logits)
    assert set(result.direction_cosines) == {1, 2}
    assert all(not layer._forward_hooks for layer in tiny.instance.layers)


def test_repeated_baselines_are_identical(tiny):
    first = _swap(tiny, direction_source="logit")
    second = _swap(tiny, direction_source="logit")
    assert torch.equal(first.baseline_logits, second.baseline_logits)
    assert torch.equal(first.swapped_logits, second.swapped_logits)


def test_swapping_a_later_position_cannot_change_an_earlier_one(tiny):
    """The decoder is causal, so a later patch must leave earlier logits untouched."""
    result = _swap(
        tiny,
        direction_source="logit",
        token_pos=[-1],
        read_positions=[-2, -1],
    )
    assert torch.equal(result.baseline_logits[0], result.swapped_logits[0])
    assert not torch.equal(result.baseline_logits[1], result.swapped_logits[1])


def test_audio_positions_match_the_real_prompt(tiny):
    tiny.instance.load_data(str(tiny.root))
    input_ids = tiny.instance.encode(tiny.line)
    tokenizer = tiny.instance.tokenizer
    ids = input_ids[0].tolist()

    audio = Experiments.swap_token_positions(input_ids, tokenizer, "audio")
    after = Experiments.swap_token_positions(input_ids, tokenizer, "after_audio")
    assert len(audio) > 2
    assert audio[-1] + 1 == after[0]
    assert after[-1] == len(ids) - 1
    audio_token = tokenizer.convert_tokens_to_ids("<|AUDIO|>")
    assert all(ids[position] == audio_token for position in audio[1:-1])
    assert "assistant" in tokenizer.decode([ids[position] for position in after])
    assert not any(ids[position] == audio_token for position in after)


def test_default_candidate_words_are_single_tokens(tiny):
    settings_path = Path(__file__).resolve().parents[1] / "settings.json"
    with open(settings_path, encoding="utf-8") as file:
        words = json.load(file)["experiment"]["swap"]["candidate_words"]
    tokenizer = tiny.instance.tokenizer
    multi_token = [
        form
        for word in words
        for form in (word, word.capitalize())
        if Experiments.swap_single_token(tokenizer, form) is None
    ]
    assert multi_token == []


def test_full_experiment_runs_on_the_real_architecture(tiny, tmp_path, monkeypatch):
    tiny.instance.load_data(str(tiny.root))
    input_ids = tiny.instance.encode(
        Experiments.swap_prompt_line(tiny.line, "Answer in one word.")
    )
    with torch.no_grad():
        logits = tiny.instance.unembed(tiny.instance.forward(input_ids)[0][[-1]])[0]
    answer = tiny.instance.tokenizer.decode([int(logits.argmax())])

    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(tiny.line, encoding="utf-8")
    lens = tiny.model.lens
    monkeypatch.setattr(
        tiny.model, "load_Jlens", lambda lens_path=None: setattr(tiny.model, "lens", lens)
    )
    report = Experiments.swap_verbal_report(
        logger=logging.getLogger("test"),
        model=tiny.model,
        run_name="tiny",
        lens_path="lens.pt",
        prompts_path=str(prompts),
        data_root=str(tiny.root),
        instruction="Answer in one word.",
        # Random weights answer with an arbitrary token, so accept that token.
        candidate_words=[answer.strip(), "dog", "cat"],
        layer_bands_percent=[[0, 50], [38, 92]],
        candidates_per_prompt=1,
        position_policies=["all", "audio", "after_audio", "last"],
        max_direction_cosine=1.1,
    )
    # 1 candidate x 2 bands x 3 sources x 4 policies.
    assert len(report["rows"]) == 24
    assert len(report["summary"]) == 24
    assert all(row["source_text"] == answer for row in report["rows"])
    assert report["skipped"] == []
