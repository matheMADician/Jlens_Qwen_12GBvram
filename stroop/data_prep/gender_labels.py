"""Gender labelling for stroop stimuli: speaker gender (acoustic) and content gender (text).

Pure functions with no heavy dependencies, so they can be unit-tested anywhere.

* Speaker gender comes from LibriSpeech ``SPEAKERS.TXT`` (``F`` / ``M``).
* Content gender comes from gendered words in the transcript.
* ``person`` separates a true Stroop conflict about the speaker ("I am a man")
  from third-person mentions ("he said"). See stroop/AGENTS.md, section 5.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

MALE = "male"
FEMALE = "female"
MIXED = "mixed"
NONE = "none"

# Words whose gender is unambiguous in LibriSpeech's normalised text
# (upper case, no punctuation except apostrophes, numbers spelled out).
# "miss" is left out because it is also a verb.
MALE_WORDS = frozenset({
    "he", "him", "his", "himself",
    "man", "men", "boy", "boys", "gentleman", "gentlemen",
    "father", "son", "sons", "brother", "brothers", "husband", "uncle",
    "king", "prince", "sir", "mister", "lord",
})
FEMALE_WORDS = frozenset({
    "she", "her", "hers", "herself",
    "woman", "women", "girl", "girls", "lady", "ladies",
    "mother", "daughter", "daughters", "sister", "sisters", "wife", "aunt",
    "queen", "princess", "madam", "missus",
})

# Nouns that make "I am a <noun>" a statement about the speaker's own gender.
_SELF_NOUNS = {
    MALE: ("man", "boy", "gentleman", "father", "son", "brother", "husband", "king", "prince"),
    FEMALE: ("woman", "girl", "lady", "mother", "daughter", "sister", "wife", "queen", "princess"),
}
_SELF_PATTERNS = {
    gender: re.compile(
        # Negations ("I am not a man") are deliberately not matched.
        r"\bi(?: am|'m)(?: only| but| just)?(?: a| an| the| your| his| her| their)?"
        r"(?: \w+)? (?:" + "|".join(nouns) + r")\b"
    )
    for gender, nouns in _SELF_NOUNS.items()
}
_WORD = re.compile(r"[a-z']+")

SPEAKER_SEX = {"F": FEMALE, "M": MALE}


@dataclass(frozen=True)
class ContentGender:
    """Gender evidence found in one transcript."""

    gender: str                  # male | female | mixed | none
    male_words: tuple[str, ...]
    female_words: tuple[str, ...]
    person: str                  # first | third | none
    self_gender: str | None      # gender claimed in "I am a ...", if any


def words(text: str) -> list[str]:
    """Lower-case word tokens; keeps apostrophes inside words."""
    return _WORD.findall(text.lower())


def content_gender(text: str) -> ContentGender:
    """Classify the gender referred to by a transcript.

    A first-person self-reference ("I am a man") decides ``gender`` on its own,
    because that is the claim that conflicts with the speaker's voice. Otherwise
    the gender is male or female only when every gendered word agrees.
    """
    tokens = words(text)
    male = tuple(t for t in tokens if t in MALE_WORDS)
    female = tuple(t for t in tokens if t in FEMALE_WORDS)

    lowered = " ".join(tokens)
    claims = [g for g, pattern in _SELF_PATTERNS.items() if pattern.search(lowered)]
    if len(claims) == 1:
        return ContentGender(claims[0], male, female, "first", claims[0])
    if len(claims) > 1:
        return ContentGender(MIXED, male, female, "first", None)

    if male and female:
        gender = MIXED
    elif male:
        gender = MALE
    elif female:
        gender = FEMALE
    else:
        gender = NONE
    person = "third" if gender != NONE else "none"
    return ContentGender(gender, male, female, person, None)


def parse_speakers_file(path: str | Path) -> dict[int, str]:
    """Map LibriSpeech speaker id to ``male`` / ``female``.

    ``SPEAKERS.TXT`` lines look like ``ID | SEX | SUBSET | MINUTES | NAME``;
    lines starting with ``;`` are comments. Names may contain ``|``, so only the
    first two fields are used.
    """
    speakers: dict[int, str] = {}
    for line_number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith(";"):
            continue
        fields = [field.strip() for field in line.split("|")]
        if len(fields) < 2:
            raise ValueError(f"{path}:{line_number}: expected 'ID | SEX | ...', got {raw!r}")
        try:
            speaker_id = int(fields[0])
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: speaker id {fields[0]!r} is not an integer") from exc
        sex = fields[1].upper()
        if sex not in SPEAKER_SEX:
            raise ValueError(f"{path}:{line_number}: unknown sex {fields[1]!r}")
        speakers[speaker_id] = SPEAKER_SEX[sex]
    if not speakers:
        raise ValueError(f"No speakers found in {path}")
    return speakers


def cell_name(speaker_gender: str, text_gender: str) -> str:
    """Name of a 2x2 cell, e.g. ``F_voice-M_text``; ``congruent`` cells have matching genders."""
    return f"{speaker_gender[0].upper()}_voice-{text_gender[0].upper()}_text"


def is_conflict(speaker_gender: str, text_gender: str) -> bool:
    return speaker_gender != text_gender
