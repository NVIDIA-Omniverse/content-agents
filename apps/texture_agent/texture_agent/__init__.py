# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dotenv import load_dotenv

# Load .env before submodules cache environment-derived settings at import time.
load_dotenv()

from .utils import get_version  # noqa: E402

__version__ = get_version()
