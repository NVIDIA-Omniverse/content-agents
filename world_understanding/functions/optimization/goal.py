# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Goal ABC for blackbox optimization."""

from abc import ABC, abstractmethod
from typing import Any, Literal


class Goal(ABC):
    @property
    @abstractmethod
    def metric_name(self) -> str: ...

    @property
    @abstractmethod
    def metric_direction(self) -> Literal["minimize", "maximize"]: ...

    @property
    @abstractmethod
    def time_budget(self) -> float: ...  # wall-clock seconds

    @property
    @abstractmethod
    def n_dims(self) -> int: ...

    @property
    @abstractmethod
    def bounds(self) -> tuple[float, float]: ...

    @abstractmethod
    def evaluate(self, **context: Any) -> float: ...

    def is_improvement(self, new_val: float, old_val: float) -> bool:
        direction = self.metric_direction
        if direction == "minimize":
            return new_val < old_val
        elif direction == "maximize":
            return new_val > old_val
        else:
            raise ValueError(
                f"Invalid metric_direction: {direction!r}. "
                "Must be 'minimize' or 'maximize'."
            )
