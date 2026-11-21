"""Tests for stroop gender labelling. Run from the repo root:

    python -m unittest discover -s stroop/tests -t .
"""

import tempfile
import unittest
from pathlib import Path

from stroop.data_prep.gender_labels import (
    FEMALE,
    MALE,
    MIXED,
    NONE,
    cell_name,
    content_gender,
    is_conflict,
    parse_speakers_file,
)
from stroop.data_prep.get_stroop_libri import select

SPEAKERS_TXT = """\
; Some comment
;ID  |SEX| SUBSET           |MINUTES| NAME
14   | F | train-clean-360  | 25.03 | Kristin LeMoine
16   | F | train-clean-360  | 25.11 | Alys AtteWater
17   | M | train-clean-360  | 25.04 | Gord Mackenzie
60   | M | train-clean-100  | 20.18 | |CBW|Simon
"""


class ContentGenderTest(unittest.TestCase):
    def test_third_person_male(self):
        found = content_gender("AND HE SAID THAT HIS FATHER WAS AWAY")
        self.assertEqual(found.gender, MALE)
        self.assertEqual(found.person, "third")
        self.assertEqual(found.male_words, ("he", "his", "father"))

    def test_third_person_female(self):
        self.assertEqual(content_gender("SHE TOOK HER MOTHER'S HAND").gender, FEMALE)

    def test_mixed_and_none(self):
        self.assertEqual(content_gender("HE TOLD HER").gender, MIXED)
        self.assertEqual(content_gender("THE RIVER RAN COLD").gender, NONE)
        self.assertEqual(content_gender("THE RIVER RAN COLD").person, "none")

    def test_word_boundaries(self):
        # "manner", "there", "shed", "hiss" contain gendered substrings but are not gendered words.
        self.assertEqual(content_gender("IN THAT MANNER THERE WAS A SHED AND A HISS").gender, NONE)

    def test_first_person_claim_overrides_other_words(self):
        found = content_gender("SHE KNOWS I AM A MAN OF MY WORD")
        self.assertEqual(found.gender, MALE)
        self.assertEqual(found.person, "first")
        self.assertEqual(found.self_gender, MALE)

    def test_first_person_variants(self):
        self.assertEqual(content_gender("I'M A POOR GIRL").self_gender, FEMALE)
        self.assertEqual(content_gender("I AM HIS WIFE").self_gender, FEMALE)
        self.assertEqual(content_gender("I AM BUT A YOUNG MAN").self_gender, MALE)

    def test_negation_is_not_a_claim(self):
        found = content_gender("I AM NOT A MAN TO BE TRIFLED WITH")
        self.assertEqual(found.person, "third")
        self.assertIsNone(found.self_gender)


class SpeakersFileTest(unittest.TestCase):
    def write(self, text):
        handle = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
        handle.write(text)
        handle.close()
        self.addCleanup(Path(handle.name).unlink)
        return handle.name

    def test_parse(self):
        speakers = parse_speakers_file(self.write(SPEAKERS_TXT))
        self.assertEqual(speakers, {14: FEMALE, 16: FEMALE, 17: MALE, 60: MALE})

    def test_bad_sex(self):
        with self.assertRaises(ValueError):
            parse_speakers_file(self.write("14 | X | train | 1.0 | Name\n"))

    def test_empty(self):
        with self.assertRaises(ValueError):
            parse_speakers_file(self.write("; only comments\n"))


class SelectTest(unittest.TestCase):
    speakers = {14: FEMALE, 17: MALE}

    def test_conflict_cell(self):
        label = select("HE SAID SO", 14, self.speakers, first_person_only=False, min_gender_words=1)
        self.assertEqual(label["cell"], "F_voice-M_text")
        self.assertTrue(label["conflict"])
        self.assertEqual(label["gender_words"], ["he"])

    def test_congruent_cell(self):
        label = select("HE SAID SO", 17, self.speakers, first_person_only=False, min_gender_words=1)
        self.assertFalse(label["conflict"])

    def test_rejections(self):
        kwargs = dict(first_person_only=False, min_gender_words=1)
        self.assertIsNone(select("HE SAID SO", 999, self.speakers, **kwargs))      # unknown speaker
        self.assertIsNone(select("HE TOLD HER", 14, self.speakers, **kwargs))      # mixed
        self.assertIsNone(select("IT RAINED", 14, self.speakers, **kwargs))        # no gender
        self.assertIsNone(select("HE SAID SO", 14, self.speakers,
                                 first_person_only=True, min_gender_words=1))    # not first person
        self.assertIsNone(select("HE SAID SO", 14, self.speakers,
                                 first_person_only=False, min_gender_words=2))   # too few words

    def test_first_person_kept(self):
        label = select("I AM A MAN", 14, self.speakers, first_person_only=True, min_gender_words=5)
        self.assertEqual(label["person"], "first")
        self.assertTrue(label["conflict"])


class HelpersTest(unittest.TestCase):
    def test_cell_and_conflict(self):
        self.assertEqual(cell_name(MALE, FEMALE), "M_voice-F_text")
        self.assertTrue(is_conflict(MALE, FEMALE))
        self.assertFalse(is_conflict(FEMALE, FEMALE))


if __name__ == "__main__":
    unittest.main()
