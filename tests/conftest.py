"""Shared test setup. Run from the repo root: python -m pytest tests"""

import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

# main.py defaults HF_HOME to a server path, so choose the local cache first.
os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))

# Tests on fake models can run without these; test_swap_tiny_model.py needs the real ones.
for _name in ("transformers", "librosa"):
    if importlib.util.find_spec(_name) is None:
        sys.modules[_name] = MagicMock()
