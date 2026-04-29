# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import importlib
import logging
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[2]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

service_main = importlib.import_module("service.main")


class _DummyRootLogger:
    def __init__(self, handlers: list[object]) -> None:
        self.handlers = handlers
        self.level = None

    def setLevel(self, level: int) -> None:
        self.level = level


def test_configure_logging_uses_basic_config_without_existing_handlers(monkeypatch):
    root_logger = _DummyRootLogger([])
    captured: dict[str, object] = {}

    def fake_basic_config(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(service_main.logging, "basicConfig", fake_basic_config)

    service_main._configure_logging(root_logger=root_logger)

    assert captured["level"] == logging.INFO
    assert "handlers" in captured


def test_configure_logging_reuses_existing_root_handlers(monkeypatch):
    root_logger = _DummyRootLogger([object()])
    basic_config_calls = 0

    def fake_basic_config(**kwargs):
        nonlocal basic_config_calls
        basic_config_calls += 1

    monkeypatch.setattr(service_main.logging, "basicConfig", fake_basic_config)

    service_main._configure_logging(root_logger=root_logger)

    assert root_logger.level == logging.INFO
    assert basic_config_calls == 0
