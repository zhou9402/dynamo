#!/usr/bin/env python3

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
WORKERS_DIR = ROOT / "workers"
sys.path.insert(0, str(WORKERS_DIR))

from protocol import DenoiserResponse, EncoderResponse, GenerateRequest, VAEDecodeRequest  # noqa: E402


def test_protocol_mutable_defaults_are_isolated():
    a = EncoderResponse()
    b = EncoderResponse()
    a.transfer_meta["x"] = 1
    a.shapes["latents"] = [1, 2, 3]

    assert b.transfer_meta == {}
    assert b.shapes == {}


def test_protocol_other_dict_defaults_are_isolated():
    a = DenoiserResponse()
    b = DenoiserResponse()
    a.transfer_meta["k"] = "v"
    a.shape.append(42)

    assert b.transfer_meta == {}
    assert b.shape == []

    x = VAEDecodeRequest(transfer_meta={})
    y = VAEDecodeRequest(transfer_meta={})
    x.tensor_data["t"] = "blob"
    assert y.tensor_data == {}


def test_generate_request_seed_is_optional():
    req = GenerateRequest(prompt="hello")
    assert req.seed is None
