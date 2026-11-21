"""Tests for stroop.data_prep.make_sakura_sets. Run from the repo root:

    python -m unittest discover -s stroop/tests -t .
"""

import json
import tempfile
import unittest
from pathlib import Path

from stroop.data_prep.make_sakura_sets import (
    build,
    check_prompts,
    load_prompts,
    parse_args,
    split_halves,
)


def trunk_record(index: int, gender: str) -> dict:
    """A record shaped like data/get_data.py output."""
    audio = f"audio/sample-{index:06d}.wav"
    return {
        "id": f"sample-{index:06d}",
        "source_index": index,
        "audio_path": audio,
        "messages": [{"role": "user", "content": [
            {"type": "audio", "path": audio},
            {"type": "text", "text": "Is the speaker (a) Male or (b) Female?"},
        ]}],
        "transcript": gender,
    }


class CheckPromptsTest(unittest.TestCase):
    def test_rejects_gender_words(self):
        for prompt in ("Is the speaker male or female?", "Is this a man?", "Did she speak?"):
            with self.assertRaises(ValueError):
                check_prompts([prompt], "gender")

    def test_accepts_clean_prompts(self):
        check_prompts(["What is the speaker's gender? Answer in one word."], "gender")

    def test_repo_prompt_files_are_clean(self):
        prompt_dir = Path(__file__).resolve().parent.parent / "prompts"
        for name in ("ask_gender.txt", "transcribe.txt"):
            check_prompts(load_prompts(prompt_dir / name), name)


class SplitTest(unittest.TestCase):
    def records(self, n_female, n_male):
        genders = ["female"] * n_female + ["male"] * n_male
        return [{"id": f"r{i:03d}", "label": {"speaker_gender": g}} for i, g in enumerate(genders)]

    def test_disjoint_balanced_and_complete(self):
        records = self.records(251, 249)
        a, b = split_halves(records, seed=0)
        self.assertFalse(set(a) & set(b))
        self.assertEqual(len(a) + len(b), 500)
        self.assertEqual((len(a), len(b)), (250, 250))
        gender = {r["id"]: r["label"]["speaker_gender"] for r in records}
        for half in (a, b):
            females = sum(gender[i] == "female" for i in half)
            self.assertLessEqual(abs(females - (len(half) - females)), 2)

    def test_deterministic(self):
        records = self.records(10, 10)
        self.assertEqual(split_halves(records, 1), split_halves(records, 1))
        self.assertNotEqual(split_halves(records, 1), split_halves(records, 2))


class BuildTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / "data"
        self.dataset_dir = self.root / "datasets" / "stroop" / "sakura_gender"
        self.dataset_dir.mkdir(parents=True)
        records = [trunk_record(i, "female" if i % 2 else "male") for i in range(20)]
        (self.dataset_dir / "prompts.jsonl").write_text(
            "\n".join(json.dumps(r) for r in records), encoding="utf-8")
        self.output = self.root / "assembled" / "stroop" / "sakura_gender"

    def run_build(self, *extra):
        return build(parse_args([
            "--dataset-dir", str(self.dataset_dir),
            "--data-root", str(self.root),
            "--output", str(self.output),
            *extra,
        ]))

    def read(self, name):
        lines = (self.output / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines]

    def test_outputs(self):
        counts = self.run_build()
        self.assertEqual(counts["fit_gender_all"], 20)
        self.assertEqual(counts["fit_gender_A"] + counts["fit_gender_B"], 20)

        pilot = self.read("pilot_single")
        self.assertEqual(pilot[0]["audio_path"], "datasets/stroop/sakura_gender/audio/sample-000000.wav")
        self.assertEqual(pilot[0]["messages"][0]["content"][0]["path"], pilot[0]["audio_path"])
        self.assertIn("(a) Male", pilot[0]["messages"][0]["content"][1]["text"])
        self.assertEqual(pilot[0]["label"], {"speaker_gender": "male"})
        self.assertEqual([r["id"] for r in pilot], [str(i) for i in range(20)])
        self.assertEqual(pilot[0]["source_id"], "sample-000000")

    def test_tasks_differ_only_in_prompt(self):
        self.run_build()
        for half in ("all", "A", "B"):
            gender, trans = self.read(f"fit_gender_{half}"), self.read(f"fit_trans_{half}")
            self.assertEqual([r["id"] for r in gender], [r["id"] for r in trans])
            for g, t in zip(gender, trans):
                self.assertEqual(g["audio_path"], t["audio_path"])
                g_text = g["messages"][0]["content"][1]["text"]
                t_text = t["messages"][0]["content"][1]["text"]
                self.assertNotEqual(g_text, t_text)
                check_prompts([g_text, t_text], "output")

    def test_halves_match_across_tasks(self):
        self.run_build()
        split = json.loads((self.output / "split.json").read_text(encoding="utf-8"))
        self.assertEqual({r["id"] for r in self.read("fit_gender_A")}, set(split["A"]))
        self.assertEqual({r["id"] for r in self.read("fit_trans_B")}, set(split["B"]))

    def test_refuses_overwrite(self):
        self.run_build()
        with self.assertRaises(FileExistsError):
            self.run_build()
        self.run_build("--overwrite")

    def test_rejects_unexpected_label(self):
        bad = trunk_record(99, "unknown")
        with (self.dataset_dir / "prompts.jsonl").open("a", encoding="utf-8") as handle:
            handle.write("\n" + json.dumps(bad))
        with self.assertRaises(ValueError):
            self.run_build()


if __name__ == "__main__":
    unittest.main()
