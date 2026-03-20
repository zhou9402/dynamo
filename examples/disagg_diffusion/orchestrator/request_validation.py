#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request validation helpers for disaggregated diffusion orchestrator."""

from protocol import GenerateRequest


def validate_generate_http_body(body: dict) -> dict:
    """Validate and normalize POST /v1/videos/generations request body."""
    req = GenerateRequest.model_validate(body)
    req_dict = req.model_dump()
    resp_format = body.get("response_format", "url")
    if resp_format not in {"url", "b64_json"}:
        raise ValueError("response_format must be 'url' or 'b64_json'")
    req_dict["response_format"] = resp_format
    return req_dict
