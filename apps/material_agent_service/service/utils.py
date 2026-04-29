# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import sys
import time
from collections.abc import Generator
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

logger = logging.getLogger(__name__)


def get_version() -> str:
    # In a PyInstaller bundle, read the baked-in version from _version.txt
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass is not None:
        version_file = Path(meipass) / "_version.txt"
        if version_file.exists():
            return version_file.read_text().strip()

    try:
        return version("material-agent-service")
    except PackageNotFoundError:
        return "0.0.1-dev"


class AccessLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Exclude healthchecks from access logs
        return (
            record.getMessage().find("/health") == -1
            and record.getMessage().find("/metrics") == -1
        )


@contextmanager
def timer(label: str) -> Generator[None, None, None]:
    start = time.perf_counter()
    yield
    logger.info("%s: %fs", label, time.perf_counter() - start)
