import json
import unittest

from json_parser import Parser


def _record(**fields: object) -> str:
    record: dict[str, object] = {
        "id": "sample-1",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "path": "audio/sample-1.wav"},
                    {"type": "text", "text": "Describe this audio."},
                ],
            }
        ],
    }
    record.update(fields)
    return json.dumps(record)


class JsonParserLabelTests(unittest.TestCase):
    def test_parses_named_labels_separately_from_transcript(self) -> None:
        sample = Parser.parse_json_line(
            _record(
                transcript="recognized words",
                label={"emotion": "anger", "language": "zh"},
            )
        )

        self.assertEqual(sample.transcript, "recognized words")
        self.assertEqual(sample.label, {"emotion": "anger", "language": "zh"})

    def test_parses_multi_value_label_fields(self) -> None:
        sample = Parser.parse_json_line(
            _record(label={"tags": ["bird", "animal"]})
        )

        self.assertEqual(sample.label, {"tags": ["bird", "animal"]})

    def test_legacy_record_without_labels_defaults_to_empty_mapping(self) -> None:
        sample = Parser.parse_json_line(_record(transcript="recognized words"))

        self.assertEqual(sample.label, {})

    def test_rejects_non_mapping_labels(self) -> None:
        with self.assertRaisesRegex(ValueError, "label"):
            Parser.parse_json_line(_record(label=["anger"]))


if __name__ == "__main__":
    unittest.main()
