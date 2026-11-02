"""Tests for the swap experiment, Master.eval_swap, and the eval-swap CLI on a fake model."""

import csv
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("jlens")
pytest.importorskip("jacobian_lens.jlens.lens")

import main
from audio_jlens.jlens import Jlens
from Master import Master
from model.adapters.Instance import Instance
from model.model import Model
from tools.experiments import Experiments

VOCABULARY = [
    "cat", "dog", "pig", "cow", "Cat", "Dog", " dog", " cat",
    "the", "<|audio_bos|>", "<|AUDIO|>", "<|audio_eos|>",
]
TOKEN_IDS = {token: index for index, token in enumerate(VOCABULARY)}
D_MODEL = len(VOCABULARY)
N_LAYERS = 5
# <|audio_bos|> <|AUDIO|> <|AUDIO|> <|audio_eos|> the the
INPUT_IDS = [9, 10, 10, 11, 8, 8]


class FakeTokenizer:
    """Treats known strings as one token, and anything else as two."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [TOKEN_IDS[text]] if text in TOKEN_IDS else [0, 0]}

    def decode(self, token_ids):
        return "".join(VOCABULARY[token_id] for token_id in token_ids)

    def convert_tokens_to_ids(self, token):
        return TOKEN_IDS[token]


class IdentityLayer(nn.Module):
    def forward(self, hidden):
        return hidden


class FakeInstance(Instance):
    def __init__(self, answer: str = "cat"):
        self.answer = answer
        super().__init__("fake")

    def _initialize_model(self, _model_id: str, _do_4bit: bool) -> None:
        self.layer_modules = nn.ModuleList(IdentityLayer() for _ in range(N_LAYERS))
        self.layers = list(self.layer_modules)
        self.n_layers = N_LAYERS
        self.d_model = D_MODEL
        self.tokenizer = FakeTokenizer()
        self.hf_model = nn.Identity()
        self.processor = SimpleNamespace(tokenizer=self.tokenizer)
        self.forward_count = 0

    def encode(self, text: str, *, max_length: int = 3000) -> torch.Tensor:
        return torch.tensor([INPUT_IDS])

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.forward_count += 1
        captured = {}
        handle = self.layers[-1].register_forward_hook(
            lambda _module, _args, output: captured.update(residual=output)
        )
        try:
            # With orthonormal unembedding rows the answer's logit is its coordinate.
            hidden = torch.zeros(1, len(INPUT_IDS), D_MODEL)
            hidden[:, :, TOKEN_IDS[self.answer]] = 5.0
            hidden[:, :, TOKEN_IDS["the"]] = 1.0
            for layer in self.layers:
                hidden = layer(hidden)
        finally:
            handle.remove()
        return captured["residual"]

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        return residual.float()

    def get_unembedding_direction(self, token_id: int) -> torch.Tensor:
        return torch.eye(D_MODEL)[token_id]


def _record(sample_id: int, transcript: str = "cat") -> dict:
    return {
        "id": str(sample_id),
        "messages": [{
            "role": "user",
            "content": [
                {"type": "audio", "path": "audio.wav"},
                {"type": "text", "text": "original question"},
            ],
        }],
        "transcript": transcript,
        "dataset_name": "SAKURA_AnimalQA",
    }


def _prompts(tmp_path, records=None):
    path = tmp_path / "prompts.jsonl"
    records = records if records is not None else [_record(0), _record(1, "dog")]
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    return path


def _model(answer: str = "cat") -> Model:
    logger = logging.getLogger("test")
    model = Model(logger, "Qwen/Qwen2-Audio-7B", "Qwen/Qwen2-Audio-7B")
    model._model = FakeInstance(answer)

    def load_lens(lens_path=None, lens_name=None):
        lens = Jlens(logger, model._model)
        # J = I makes the J lens directions equal the unembedding directions.
        lens.lens = SimpleNamespace(
            jacobians={layer: torch.eye(D_MODEL) for layer in range(N_LAYERS - 1)},
            source_layers=list(range(N_LAYERS - 1)),
            d_model=D_MODEL,
        )
        model.lens = lens

    model.load_Jlens = load_lens
    return model


def _run(model, tmp_path, **overrides):
    arguments = {
        "logger": logging.getLogger("test"),
        "model": model,
        "run_name": "test",
        "lens_path": "lens.pt",
        "prompts_path": overrides.get("prompts_path") or str(_prompts(tmp_path)),
        "data_root": str(tmp_path),
        "instruction": "What animal? Answer in one word.",
        "candidate_words": ["cat", "dog", "pig", "horse"],
        "layer_bands_percent": [[25, 50]],
        "candidates_per_prompt": 2,
    }
    return Experiments.swap_verbal_report(**{**arguments, **overrides})


@pytest.mark.parametrize(
    "policy, expected",
    [
        ("all", [0, 1, 2, 3, 4, 5]),
        ("audio", [0, 1, 2, 3]),
        ("after_audio", [4, 5]),
        ("last", [5]),
    ],
)
def test_position_policies(policy, expected):
    positions = Experiments.swap_token_positions(
        torch.tensor([INPUT_IDS]), FakeTokenizer(), policy
    )
    assert positions == expected


@pytest.mark.parametrize(
    "input_ids, message",
    [
        ([8, 8], "exactly one audio segment"),
        ([9, 11, 9, 11], "exactly one audio segment"),
        ([11, 9, 8], "before"),
        ([9, 10, 11], "No tokens follow"),
    ],
)
def test_invalid_audio_segments_raise(input_ids, message):
    with pytest.raises(ValueError, match=message):
        Experiments.swap_token_positions(
            torch.tensor([input_ids]), FakeTokenizer(), "after_audio"
        )


def test_unknown_position_policy_raises():
    with pytest.raises(ValueError, match="Unknown position policy"):
        Experiments.swap_token_positions(torch.tensor([[1]]), FakeTokenizer(), "other")


def test_layer_bands_use_depth_percentages_and_lens_layers():
    bands = Experiments.swap_layer_bands(
        [[0, 25], [38, 92], [75, 100]], n_layers=5, source_layers=[0, 1, 2, 3]
    )
    assert bands == {"L0-25": [0, 1], "L38-92": [2, 3], "L75-100": [3]}


@pytest.mark.parametrize("band", [[50, 25], [-1, 10], [0, 101], [1]])
def test_invalid_layer_bands_raise(band):
    with pytest.raises(ValueError, match="Each band"):
        Experiments.swap_layer_bands([band], 5, [0, 1, 2, 3])


def test_empty_layer_band_raises():
    with pytest.raises(ValueError, match="contains no lens layers"):
        Experiments.swap_layer_bands([[80, 85]], 5, [0, 1, 2, 3])


@pytest.mark.parametrize(
    "source_text, expected",
    [
        ("cat", {"dog": 1, "pig": 2}),
        ("Cat", {"dog": 5}),
        (" cat", {"dog": 6}),
    ],
)
def test_candidates_match_the_source_form(source_text, expected):
    assert (
        Experiments.swap_candidate_tokens(
            FakeTokenizer(), source_text, ["cat", "dog", "pig", "horse"], 2
        )
        == expected
    )


def test_candidates_skip_multi_token_words_and_respect_count():
    assert Experiments.swap_candidate_tokens(
        FakeTokenizer(), "dog", ["horse", "cat", "pig", "cow"], 2
    ) == {"cat": 0, "pig": 2}


def test_prompt_line_only_replaces_the_user_text():
    record = _record(7)
    line = Experiments.swap_prompt_line(json.dumps(record), "新問題")
    updated = json.loads(line)
    assert updated["messages"][0]["content"][1]["text"] == "新問題"
    assert "新問題" in line
    record["messages"][0]["content"][1]["text"] = "新問題"
    assert updated == record


def test_swap_replaces_the_model_answer(tmp_path):
    report = _run(_model(), tmp_path, direction_sources=["jlens"])
    rows = report["rows"]

    assert report["prompt_count"] == 2
    assert report["skipped"] == []
    assert report["bands"] == {"L25-50": [1, 2]}
    assert len(rows) == 4
    assert {row["target_word"] for row in rows} == {"dog", "pig"}
    assert all(row["swapped_top_text"] == row["target_text"] for row in rows)
    assert all(row["target_rank"] == 1 and row["source_rank"] > 1 for row in rows)
    # The gap goes from -5 to +5, because the coordinates are exchanged.
    assert all(row["logit_gap_shift"] == pytest.approx(10.0) for row in rows)
    assert {row["label_match"] for row in rows} == {True, False}


def test_summary_groups_every_condition(tmp_path):
    report = _run(
        _model(),
        tmp_path,
        direction_sources=["jlens", "random"],
        position_policies=["all", "last"],
        layer_bands_percent=[[0, 25], [25, 50]],
    )
    # 2 prompts x 2 candidates x 2 bands x 2 sources x 2 policies.
    assert len(report["rows"]) == 32
    assert len(report["summary"]) == 8
    assert all(group["swap_count"] == 4 for group in report["summary"])

    jlens = next(
        group
        for group in report["summary"]
        if (group["direction_source"], group["band"], group["position_policy"])
        == ("jlens", "L25-50", "all")
    )
    assert jlens["target_top1_rate"] == 1.0
    assert jlens["source_displaced_rate"] == 1.0
    assert jlens["median_target_rank"] == 1
    assert jlens["mean_logit_gap_shift"] == pytest.approx(10.0)

    random_group = next(
        group
        for group in report["summary"]
        if (group["direction_source"], group["band"], group["position_policy"])
        == ("random", "L25-50", "all")
    )
    assert random_group["target_top1_rate"] == 0.0


def test_high_cosine_pairs_are_excluded_from_the_summary_only(tmp_path):
    report = _run(
        _model(), tmp_path, direction_sources=["jlens"], max_direction_cosine=-1.0
    )
    assert len(report["rows"]) == 4
    assert all(row["excluded_high_cosine"] for row in report["rows"])
    group = report["summary"][0]
    assert group["swap_count"] == 0
    assert group["excluded_high_cosine"] == 4
    assert group["target_top1_rate"] is None


def test_answers_outside_the_vocabulary_are_skipped(tmp_path):
    report = _run(_model(answer="the"), tmp_path)
    assert report["rows"] == []
    assert report["summary"] == []
    assert [entry["reason"] for entry in report["skipped"]] == [
        "answer_not_in_candidates"
    ] * 2


def test_prompts_without_usable_candidates_are_skipped(tmp_path):
    report = _run(_model(), tmp_path, candidate_words=["cat", "horse"])
    assert report["rows"] == []
    assert [entry["reason"] for entry in report["skipped"]] == [
        "no_single_token_candidates"
    ] * 2


def test_max_prompts_is_deterministic(tmp_path):
    prompts = _prompts(tmp_path, [_record(index) for index in range(10)])
    first = _run(_model(), tmp_path, prompts_path=str(prompts), max_prompts=3)
    again = _run(_model(), tmp_path, prompts_path=str(prompts), max_prompts=3)
    assert first["prompt_count"] == 3
    assert [row["sample_id"] for row in first["rows"]] == [
        row["sample_id"] for row in again["rows"]
    ]
    assert len({row["sample_id"] for row in first["rows"]}) == 3


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"direction_sources": ["other"]}, "Unknown direction source"),
        ({"position_policies": ["other"]}, "Unknown position policy"),
        ({"candidates_per_prompt": 0}, "candidates_per_prompt"),
        ({"max_prompts": 0}, "max_prompts"),
    ],
)
def test_invalid_experiment_arguments_raise_before_any_forward(tmp_path, overrides, message):
    model = _model()
    with pytest.raises(ValueError, match=message):
        _run(model, tmp_path, **overrides)
    assert model._model.forward_count == 0


def _master(tmp_path, model: Model) -> Master:
    with open(Path(main.__file__).with_name("settings.json"), encoding="utf-8") as file:
        settings = json.load(file)
    settings["fitting"]["root"] = str(tmp_path)
    settings["experiment"]["swap"]["output_root"] = str(tmp_path / "swap_output")
    settings["experiment"]["swap"]["candidate_words"] = ["cat", "dog", "pig"]
    settings["experiment"]["swap"]["layer_bands_percent"] = [[25, 50]]
    master = Master.__new__(Master)
    master.logger = logging.getLogger("test")
    master.settings = settings
    master.model = model
    master.run_name = "smoke"
    return master


def test_master_writes_the_report_and_csv(tmp_path):
    output_root = _master(tmp_path, _model()).eval_swap(
        lens_path="lens.pt", prompts_path=str(_prompts(tmp_path))
    )
    report = json.loads(
        open(f"{output_root}/smoke_swap_report.json", encoding="utf-8").read()
    )
    assert report["lens_path"] == "lens.pt"
    assert "rows" not in report
    assert len(report["summary"]) == 3
    with open(f"{output_root}/smoke_swap_rows.csv", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    # 2 prompts x 2 candidates x 1 band x 3 sources.
    assert len(rows) == 12
    assert {row["direction_source"] for row in rows} == {"jlens", "logit", "random"}


def test_master_arguments_override_settings(tmp_path):
    prompts = _prompts(tmp_path, [_record(index) for index in range(5)])
    output_root = _master(tmp_path, _model()).eval_swap(
        lens_path="lens.pt",
        prompts_path=str(prompts),
        max_prompts=1,
        layer_bands_percent=[[0, 25]],
    )
    report = json.loads(
        open(f"{output_root}/smoke_swap_report.json", encoding="utf-8").read()
    )
    assert report["prompt_count"] == 1
    assert list(report["bands"]) == ["L0-25"]


def test_master_handles_runs_with_no_swaps(tmp_path):
    output_root = _master(tmp_path, _model(answer="the")).eval_swap(
        lens_path="lens.pt", prompts_path=str(_prompts(tmp_path))
    )
    report = json.loads(
        open(f"{output_root}/smoke_swap_report.json", encoding="utf-8").read()
    )
    assert len(report["skipped"]) == 2


def test_cli_parses_eval_swap():
    args = main.build_parser().parse_args([
        "eval-swap",
        "--lens-path", "lens.pt",
        "--max-prompts", "5",
        "--layer-band", "38:92",
        "--layer-band", "0:25",
        "--run-name", "smoke",
    ])
    assert args.command == "eval-swap"
    assert args.lens_path == "lens.pt"
    assert args.max_prompts == 5
    assert args.layer_band == [[38.0, 92.0], [0.0, 25.0]]
    assert args.prompts_path == main.DEFAULT_PROMPTS_PATH


def test_cli_requires_a_lens_and_rejects_bad_bands():
    with pytest.raises(SystemExit):
        main.build_parser().parse_args(["eval-swap"])
    with pytest.raises(SystemExit):
        main.build_parser().parse_args(
            ["eval-swap", "--lens-path", "lens.pt", "--layer-band", "38"]
        )


def test_existing_cli_commands_still_parse():
    args = main.build_parser().parse_args(["eval-jlens", "--layer", "3"])
    assert args.command == "eval-jlens" and args.layer == 3
    args = main.build_parser().parse_args(["generate"])
    assert args.command == "generate"
    args = main.build_parser().parse_args(["info-jlens", "--lens-path", "x.pt"])
    assert args.command == "info-jlens"
