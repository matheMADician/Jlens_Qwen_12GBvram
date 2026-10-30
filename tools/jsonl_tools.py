from __future__ import annotations

import json, os
from pathlib import Path
from typing import Any

def load_jsonl(path: Path) -> list[dict[str, Any]]:
	"""Load a JSONL file, reporting malformed rows with their line numbers."""
	if not path.is_file():
		raise FileNotFoundError(f"JSONL file not found: {path}")

	records: list[dict[str, Any]] = []
	with path.open("r", encoding="utf-8") as handle:
		for line_number, line in enumerate(handle, start=1):
			line = line.strip()
			if not line:
				continue
			try:
				record = json.loads(line)
			except json.JSONDecodeError as exc:
				raise ValueError(
					f"Invalid JSON in {path}, line {line_number}: {exc.msg}"
				) from exc
			if not isinstance(record, dict):
				raise ValueError(
					f"Expected a JSON object in {path}, line {line_number}"
				)
			records.append(record)
	return records

def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
	"""Write a list of dictionaries to a JSONL file."""
	os.makedirs(path.parent, exist_ok=True)
	with path.open("w", encoding="utf-8") as handle:
		for record in records:
			json_line = json.dumps(record, ensure_ascii=False)
			handle.write(json_line + "\n")