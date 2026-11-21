"""Tests for stroop.experiments.readout. Run from the repo root:

    python -m unittest discover -s stroop/tests -t .

Torch-dependent tests are skipped when torch is not installed (e.g. locally).
"""

import importlib.util
import unittest

from stroop.experiments.readout import (
    build_token_sets,
    region_stats,
    single_token_ids,
    summarize,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None


class FakeTokenizer:
    """Single token for known strings, two tokens otherwise."""

    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text, add_special_tokens=False):
        if text in self.vocab:
            return {"input_ids": [self.vocab[text]]}
        return {"input_ids": [0, 0]}


class TokenSetTest(unittest.TestCase):
    def test_keeps_only_single_token_forms(self):
        tok = FakeTokenizer({"male": 1, " male": 2, "Male": 3})
        self.assertEqual(single_token_ids(tok, ["male"]), [1, 2, 3])

    def test_empty_set_raises(self):
        tok = FakeTokenizer({"female": 1})  # no male forms at all
        with self.assertRaises(ValueError):
            build_token_sets(tok)


class RegionStatsTest(unittest.TestCase):
    def test_values(self):
        stats = region_stats([1, 5, 3], [2, 4, 10])
        self.assertAlmostEqual(stats["win_rate"], 2 / 3)
        self.assertEqual(stats["correct_best_rank"], 1)
        self.assertEqual(stats["wrong_best_rank"], 2)
        self.assertEqual(stats["correct_median_rank"], 3)

    def test_ties_are_not_wins(self):
        self.assertEqual(region_stats([4, 4], [4, 4])["win_rate"], 0.0)

    def test_mismatch_raises(self):
        with self.assertRaises(ValueError):
            region_stats([1], [1, 2])


def row(gender, win_rate, best=5, layer=3, lens="jlens", region="audio", token_set="attr"):
    return {"gender": gender, "win_rate": win_rate, "correct_best_rank": best, "wrong_best_rank": 50,
            "layer": layer, "lens": lens, "region": region, "token_set": token_set}


class SummarizeTest(unittest.TestCase):
    def test_balanced_win_rate_cancels_gender_bias(self):
        # A lens that always prefers "male": wins on every male clip, loses on every female clip.
        rows = [row("male", 1.0)] * 3 + [row("female", 0.0)] * 1
        [entry] = summarize(rows)
        self.assertAlmostEqual(entry["mean_win_rate"], 0.75)
        self.assertAlmostEqual(entry["balanced_win_rate"], 0.5)

    def test_groups_and_hits(self):
        rows = [row("male", 0.8, best=1), row("female", 0.6, best=50),
                row("male", 0.5, region="text"), row("male", 0.5, layer=4)]
        summary = summarize(rows)
        self.assertEqual(len(summary), 3)
        audio3 = next(e for e in summary if e["region"] == "audio" and e["layer"] == 3)
        self.assertEqual(audio3["n_prompts"], 2)
        self.assertEqual(audio3["hit@1"], 0.5)
        self.assertEqual(audio3["hit@100"], 1.0)


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class BestSetRankTest(unittest.TestCase):
    def test_ranks(self):
        import torch

        from stroop.experiments.readout import best_set_rank

        logits = torch.tensor([[0.1, 0.9, 0.5, 0.3],
                               [0.8, 0.1, 0.2, 0.9]])
        self.assertEqual(best_set_rank(logits, [2]).tolist(), [2, 2])
        self.assertEqual(best_set_rank(logits, [2, 1]).tolist(), [1, 2])


if __name__ == "__main__":
    unittest.main()
