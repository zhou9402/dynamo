# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Handler base class with graceful fallback.

Tries to import the canonical ``BaseGenerativeHandler`` from
``dynamo.sglang``.  When the Dynamo Python components are not installed
(common in bare-metal / container setups that only ship the Rust runtime),
provides a local implementation with the same interface.

This ensures the disagg diffusion handlers work in both environments:
  - Full Dynamo install  → real BaseGenerativeHandler with full tracing/metrics
  - Minimal install      → local fallback with the same API surface
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, Optional

logger = logging.getLogger(__name__)

try:
    from dynamo.sglang.request_handlers.handler_base import (
        BaseGenerativeHandler,
    )
    from dynamo.sglang.args import Config, DynamoConfig

    HAS_DYNAMO_SGLANG = True
    logger.debug("Using dynamo.sglang BaseGenerativeHandler")

except ImportError:
    HAS_DYNAMO_SGLANG = False
    logger.info(
        "dynamo.sglang not available — using local BaseGenerativeHandler fallback"
    )

    class _StubConfig:
        """Minimal Config stub when dynamo.sglang is not installed."""

        def __init__(self, server_args=None):
            self.server_args = server_args
            self.dynamo_args = None
            self.serving_mode = None

    Config = _StubConfig  # type: ignore[misc]
    DynamoConfig = None  # type: ignore[misc,assignment]

    class BaseGenerativeHandler(ABC):  # type: ignore[no-redef]
        """Local fallback matching dynamo.sglang.BaseGenerativeHandler API."""

        def __init__(self, config, publisher=None):
            self.config = config
            self.metrics_publisher = None
            self.kv_publisher = None

        @abstractmethod
        async def generate(
            self, request: Dict[str, Any], context: Any
        ) -> AsyncGenerator[Dict[str, Any], None]:
            yield {}  # pragma: no cover

        def cleanup(self) -> None:
            pass

        def _get_trace_header(self, context) -> Optional[Dict[str, str]]:
            trace_id = getattr(context, "trace_id", None)
            span_id = getattr(context, "span_id", None)
            if not trace_id or not span_id:
                return None
            return {"traceparent": f"00-{trace_id}-{span_id}-01"}
