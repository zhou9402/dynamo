#!/usr/bin/env python3

from pathlib import Path
import sys

import pytest
from pydantic import ValidationError


ROOT = Path(__file__).resolve().parents[1]
WORKERS_DIR = ROOT / "workers"
ORCH_DIR = ROOT / "orchestrator"
sys.path.insert(0, str(WORKERS_DIR))
sys.path.insert(0, str(ORCH_DIR))

from request_validation import validate_generate_http_body  # noqa: E402


def test_validate_generate_http_body_accepts_minimal_payload():
    out = validate_generate_http_body({"prompt": "a cat"})
    assert out["prompt"] == "a cat"
    assert out["response_format"] == "url"
    assert out["seed"] is None


def test_validate_generate_http_body_rejects_invalid_response_format():
    with pytest.raises(ValueError, match="response_format"):
        validate_generate_http_body(
            {
                "prompt": "a cat",
                "response_format": "invalid",
            }
        )


def test_validate_generate_http_body_rejects_invalid_schema():
    with pytest.raises(ValidationError):
        validate_generate_http_body({"height": 544})
