# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from importlib.metadata import PackageNotFoundError, version


def get_version() -> str:
    try:
        return version("world-understanding")
    except PackageNotFoundError:
        return "0.0.1-dev"
