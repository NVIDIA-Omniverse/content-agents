# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for NewtonSimulator wiring that do not require Newton."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from math import inf, nan
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from physics_agent.tuning.errors import NewtonUnavailableError
from physics_agent.tuning.newton_simulator import NewtonSimulator


class _FakeArray:
    def __init__(self, values: Any) -> None:
        self.values = values

    def numpy(self) -> Any:
        if isinstance(self.values, list):
            return [
                list(item) if isinstance(item, list | tuple) else item
                for item in self.values
            ]
        return self.values

    def assign(self, values: Any) -> None:
        if isinstance(values, list):
            self.values = [
                list(item) if isinstance(item, list | tuple) else item
                for item in values
            ]
        else:
            self.values = values


class _FakeState:
    def __init__(self) -> None:
        self.body_q = _FakeArray([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        self.body_qd = _FakeArray([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        self.joint_q = _FakeArray([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        self.joint_qd = _FakeArray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.clear_force_calls = 0

    def clear_forces(self) -> None:
        self.clear_force_calls += 1


class _FakeModel:
    body_count = 1

    def __init__(self) -> None:
        self.states: list[_FakeState] = []
        self.collide_calls = 0
        self.joint_child = _FakeArray([0])
        self.joint_qd_start = _FakeArray([0, 6])
        self.joint_articulation = _FakeArray([0])
        self.device = "cpu"

    def state(self) -> _FakeState:
        state = _FakeState()
        self.states.append(state)
        return state

    def control(self) -> object:
        return object()

    def contacts(self) -> object:
        return object()

    def collide(self, state: _FakeState, contacts: object) -> None:
        self.collide_calls += 1


class _FakeSolver:
    def __init__(
        self,
        model: _FakeModel,
        *,
        njmax: int | None = None,
        use_mujoco_contacts: bool = True,
        use_mujoco_cpu: bool = False,
    ) -> None:
        self.model = model
        self.njmax = njmax
        self.use_mujoco_contacts = use_mujoco_contacts
        self.use_mujoco_cpu = use_mujoco_cpu

    def step(
        self,
        state_in: _FakeState,
        state_out: _FakeState,
        control: object,
        contacts: object,
        dt: float,
    ) -> None:
        state_out.body_q.assign(state_in.body_q.numpy())
        state_out.body_qd.assign(state_in.body_qd.numpy())
        state_out.joint_q.assign(state_in.joint_q.numpy())
        state_out.joint_qd.assign(state_in.joint_qd.numpy())


@dataclass
class _FakeShapeConfig:
    ke: float = 2500.0
    kd: float = 100.0
    kf: float = 1000.0
    ka: float = 0.0
    mu: float = 1.0
    restitution: float = 0.0
    mu_torsional: float = 0.005
    mu_rolling: float = 0.0001


class _FakeBuilder:
    ShapeConfig = _FakeShapeConfig

    def __init__(self) -> None:
        self.add_usd_kwargs: dict[str, Any] | None = None
        self.model = _FakeModel()
        self.ground_planes = 0
        self.ground_plane_cfgs: list[_FakeShapeConfig | None] = []
        self.reject_up_axis_kwarg = False
        self.reject_load_visual_shapes_kwarg = False
        self.shape_material_ke = [10.0, 20.0]
        self.shape_material_kd = [11.0, 21.0]
        self.shape_material_kf = [12.0, 22.0]
        self.shape_material_ka = [13.0, 23.0]
        self.shape_material_mu = [0.42, 0.84]
        self.shape_material_restitution = [0.18, 0.36]
        self.shape_material_mu_torsional = [0.015, 0.025]
        self.shape_material_mu_rolling = [0.001, 0.002]

    def add_usd(self, scene_usd: str, **kwargs: Any) -> dict[str, Any]:
        if self.reject_up_axis_kwarg and "apply_up_axis_from_stage" in kwargs:
            raise TypeError("unexpected keyword argument 'apply_up_axis_from_stage'")
        if self.reject_load_visual_shapes_kwarg and "load_visual_shapes" in kwargs:
            raise TypeError("unexpected keyword argument 'load_visual_shapes'")
        self.add_usd_kwargs = kwargs
        return {
            "path_body_map": {"/World/Body": 0},
            "path_shape_map": {"/World/GroundPlane": 0, "/World/Body": 1},
        }

    def add_ground_plane(self, *, cfg: _FakeShapeConfig | None = None) -> None:
        self.ground_planes += 1
        self.ground_plane_cfgs.append(cfg)

    def finalize(self, *, device: str) -> _FakeModel:
        return self.model


def _array_like_to_list(value: Any) -> list[Any]:
    if hasattr(value, "numpy"):
        value = value.numpy()
    return list(value)


def _install_fake_warp(monkeypatch: pytest.MonkeyPatch) -> None:
    warp = ModuleType("warp")
    warp.int32 = int  # type: ignore[attr-defined]
    warp.array = lambda values, **kwargs: _FakeArray(values)  # type: ignore[attr-defined]
    warp.transform_identity = lambda: object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "warp", warp)


def test_inject_initial_velocity_syncs_body_and_joint_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_warp(monkeypatch)
    state = _FakeState()
    calls: list[tuple[Any, Any, Any, Any, dict[str, Any]]] = []

    def eval_ik(
        model: object,
        state_arg: object,
        joint_q: object,
        joint_qd: object,
        **kwargs: Any,
    ) -> None:
        calls.append((model, state_arg, joint_q, joint_qd, kwargs))
        joint_qd.assign([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])

    fake_newton = SimpleNamespace(eval_ik=eval_ik)
    model = _FakeModel()

    NewtonSimulator._inject_initial_velocity(
        fake_newton,
        model,
        state,
        0,
        initial_linear_velocity=(1.0, 2.0, 3.0),
        initial_angular_velocity=(0.1, 0.2, 0.3),
    )

    assert state.body_qd.values[0] == [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]
    assert state.joint_qd.values == [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]
    assert calls[0][:4] == (model, state, state.joint_q, state.joint_qd)
    assert _array_like_to_list(calls[0][4]["indices"]) == [0]


def test_inject_initial_velocity_requires_movable_joint_dofs() -> None:
    state = _FakeState()
    model = _FakeModel()
    model.joint_child = _FakeArray([1])
    fake_newton = SimpleNamespace(eval_ik=lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="no movable Newton joint DOFs"):
        NewtonSimulator._inject_initial_velocity(
            fake_newton,
            model,
            state,
            0,
            initial_linear_velocity=(1.0, 0.0, 0.0),
            initial_angular_velocity=None,
        )


def test_inject_initial_velocity_rejects_constrained_joint_projection() -> None:
    state = _FakeState()
    model = _FakeModel()
    model.joint_qd_start = _FakeArray([0, 3])
    fake_newton = SimpleNamespace(eval_ik=lambda *args, **kwargs: None)
    original_qd = [list(row) for row in state.body_qd.values]

    with pytest.raises(RuntimeError, match="only 3 movable DOFs"):
        NewtonSimulator._inject_initial_velocity(
            fake_newton,
            model,
            state,
            0,
            initial_linear_velocity=(1.0, 0.0, 0.0),
            initial_angular_velocity=None,
        )
    assert state.body_qd.values == original_qd


def test_inject_initial_velocity_requires_joint_state_attrs() -> None:
    state = SimpleNamespace(
        body_qd=_FakeArray([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
    )
    model = _FakeModel()
    fake_newton = SimpleNamespace(eval_ik=lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="joint_q and joint_qd"):
        NewtonSimulator._inject_initial_velocity(
            fake_newton,
            model,
            state,
            0,
            initial_linear_velocity=(1.0, 0.0, 0.0),
            initial_angular_velocity=None,
        )


def test_inject_initial_velocity_validates_triplet_lengths() -> None:
    state = _FakeState()
    model = _FakeModel()
    fake_newton = SimpleNamespace(eval_ik=lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="exactly 3 values"):
        NewtonSimulator._inject_initial_velocity(
            fake_newton,
            model,
            state,
            0,
            initial_linear_velocity=(1.0, 2.0),
            initial_angular_velocity=None,
        )


def test_inject_initial_velocity_preserves_other_body_velocities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_warp(monkeypatch)
    state = _FakeState()
    state.body_qd = _FakeArray(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [9.0, 8.0, 7.0, 0.9, 0.8, 0.7],
        ]
    )
    model = _FakeModel()
    calls: list[list[list[float]]] = []

    def eval_ik(
        model: object,
        state_arg: _FakeState,
        joint_q: object,
        joint_qd: object,
        **kwargs: Any,
    ) -> None:
        assert _array_like_to_list(kwargs["indices"]) == [0]
        calls.append(state_arg.body_qd.numpy())

    fake_newton = SimpleNamespace(eval_ik=eval_ik)

    NewtonSimulator._inject_initial_velocity(
        fake_newton,
        model,
        state,
        0,
        initial_linear_velocity=(1.0, 2.0, 3.0),
        initial_angular_velocity=(0.1, 0.2, 0.3),
    )

    assert state.body_qd.values[0] == [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]
    assert state.body_qd.values[1] == [9.0, 8.0, 7.0, 0.9, 0.8, 0.7]
    assert calls == [state.body_qd.values]


def test_array_to_list_handles_numpy_like_values_and_rejects_scalars() -> None:
    assert NewtonSimulator._array_to_list(_FakeArray((1, 2, 3)), name="values") == [
        1,
        2,
        3,
    ]
    with pytest.raises(RuntimeError, match="model.joint_child"):
        NewtonSimulator._array_to_list(_FakeArray(5), name="model.joint_child")


def test_eval_ik_indices_falls_back_when_warp_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = __import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "warp":
            raise ImportError("test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    assert NewtonSimulator._eval_ik_indices_for_joint(_FakeModel(), 0) is None


def test_joint_info_uses_terminal_joint_qd_start_for_final_joint() -> None:
    model = _FakeModel()
    model.joint_child = _FakeArray([0, 1])
    model.joint_qd_start = _FakeArray([0, 6, 12])

    assert NewtonSimulator._joint_info_for_body(model, 1) == (1, 6)


def test_evaluate_preserves_stage_up_axis_when_importing_usd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    builders: list[_FakeBuilder] = []

    class _ModelBuilder(_FakeBuilder):
        def __init__(self) -> None:
            super().__init__()
            builders.append(self)

    fake_newton = ModuleType("newton")
    fake_newton.ModelBuilder = _ModelBuilder  # type: ignore[attr-defined]
    fake_newton.solvers = SimpleNamespace(SolverMuJoCo=_FakeSolver)  # type: ignore[attr-defined]
    fake_newton.eval_ik = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "newton", fake_newton)
    monkeypatch.setattr(
        NewtonSimulator,
        "_add_ground_plane",
        staticmethod(lambda builder, **kwargs: None),
    )

    result = NewtonSimulator(device="cpu").evaluate(
        scene_usd=tmp_path / "y_up_scene.usda",
        body_pattern="/World/Body",
        duration_s=0.01,
        dt=0.01,
        sample_fps=100,
    )

    assert result["n_steps"] == 1
    assert builders[0].add_usd_kwargs is not None
    assert builders[0].add_usd_kwargs["collapse_fixed_joints"] is True
    assert builders[0].add_usd_kwargs["apply_up_axis_from_stage"] is True
    assert builders[0].add_usd_kwargs["load_visual_shapes"] is False


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"duration_s": -0.1}, "duration_s must be finite and > 0"),
        ({"duration_s": 0.0}, "duration_s must be finite and > 0"),
        ({"duration_s": nan}, "duration_s must be finite and > 0"),
        ({"dt": 0.0}, "dt must be finite and > 0"),
        ({"dt": inf}, "dt must be finite and > 0"),
        ({"sample_fps": 0}, "sample_fps must be finite and > 0"),
    ],
)
def test_evaluate_validates_timing_before_importing_newton(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    kwargs: dict[str, float],
    message: str,
) -> None:
    monkeypatch.delitem(sys.modules, "newton", raising=False)
    params: dict[str, Any] = {
        "scene_usd": tmp_path / "scene.usda",
        "body_pattern": "/World/Body",
        "duration_s": 0.01,
        "dt": 0.01,
        "sample_fps": 30,
    }
    params.update(kwargs)

    with pytest.raises(ValueError, match=message):
        NewtonSimulator(device="cpu").evaluate(**params)


@pytest.mark.parametrize(
    "reject_attr",
    ("reject_up_axis_kwarg", "reject_load_visual_shapes_kwarg"),
)
def test_evaluate_reports_stale_newton_without_required_usd_import_kwarg(
    reject_attr: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _ModelBuilder(_FakeBuilder):
        def __init__(self) -> None:
            super().__init__()
            setattr(self, reject_attr, True)

    fake_newton = ModuleType("newton")
    fake_newton.ModelBuilder = _ModelBuilder  # type: ignore[attr-defined]
    fake_newton.solvers = SimpleNamespace(SolverMuJoCo=_FakeSolver)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "newton", fake_newton)

    with pytest.raises(NewtonUnavailableError, match=r"newton\[sim,importers\]"):
        NewtonSimulator(device="cpu").evaluate(
            scene_usd=tmp_path / "scene.usda",
            body_pattern="/World/Body",
            duration_s=0.01,
        )


def test_evaluate_reports_newton_missing_simulator_apis(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_newton = ModuleType("newton")
    monkeypatch.setitem(sys.modules, "newton", fake_newton)

    with pytest.raises(RuntimeError, match="missing required simulator APIs"):
        NewtonSimulator(device="cpu").evaluate(
            scene_usd=tmp_path / "scene.usda",
            body_pattern="/World/Body",
            duration_s=0.01,
        )


def test_evaluate_reports_newton_missing_usd_importer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _ModelBuilder:
        pass

    fake_newton = ModuleType("newton")
    fake_newton.ModelBuilder = _ModelBuilder  # type: ignore[attr-defined]
    fake_newton.solvers = SimpleNamespace(SolverMuJoCo=_FakeSolver)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "newton", fake_newton)

    with pytest.raises(RuntimeError, match="missing ModelBuilder.add_usd"):
        NewtonSimulator(device="cpu").evaluate(
            scene_usd=tmp_path / "scene.usda",
            body_pattern="/World/Body",
            duration_s=0.01,
        )


def test_warmup_reports_newton_missing_usd_importer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_warp(monkeypatch)

    class _ModelBuilder:
        pass

    fake_newton = ModuleType("newton")
    fake_newton.ModelBuilder = _ModelBuilder  # type: ignore[attr-defined]
    fake_newton.solvers = SimpleNamespace(SolverMuJoCo=_FakeSolver)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "newton", fake_newton)

    with pytest.raises(RuntimeError, match="missing ModelBuilder.add_usd"):
        NewtonSimulator(device="cpu").warmup()


def test_evaluate_reports_missing_newton_add_usd_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _ModelBuilder(_FakeBuilder):
        def add_usd(self, scene_usd: str, **kwargs: Any) -> None:
            self.add_usd_kwargs = kwargs
            return None

    fake_newton = ModuleType("newton")
    fake_newton.ModelBuilder = _ModelBuilder  # type: ignore[attr-defined]
    fake_newton.solvers = SimpleNamespace(SolverMuJoCo=_FakeSolver)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "newton", fake_newton)

    with pytest.raises(RuntimeError, match="add_usd returned None"):
        NewtonSimulator(device="cpu").evaluate(
            scene_usd=tmp_path / "scene.usda",
            body_pattern="/World/Body",
            duration_s=0.01,
        )


def test_warmup_uses_mujoco_cpu_when_device_is_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_warp(monkeypatch)
    captured: dict[str, Any] = {}

    class _ModelBuilder:
        def add_usd(self, scene_usd: str, **kwargs: Any) -> dict[str, Any]:
            return {}

        def add_body(self, **kwargs: Any) -> None:
            pass

        def add_shape_box(self, **kwargs: Any) -> None:
            pass

        def add_shape_plane(self, **kwargs: Any) -> None:
            pass

        def finalize(self, *, device: str) -> _FakeModel:
            captured["finalize_device"] = device
            return _FakeModel()

    def solver_factory(model: _FakeModel, **kwargs: Any) -> _FakeSolver:
        captured["solver_kwargs"] = kwargs
        return _FakeSolver(model, **kwargs)

    fake_newton = ModuleType("newton")
    fake_newton.ModelBuilder = _ModelBuilder  # type: ignore[attr-defined]
    fake_newton.solvers = SimpleNamespace(SolverMuJoCo=solver_factory)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "newton", fake_newton)

    NewtonSimulator(device="cpu").warmup()

    assert captured["finalize_device"] == "cpu"
    assert captured["solver_kwargs"]["use_mujoco_cpu"] is True
    assert captured["solver_kwargs"]["njmax"] == 128


def test_mujoco_cpu_device_detection_rejects_cuda_and_prefix_collisions() -> None:
    assert NewtonSimulator(device="cpu")._use_mujoco_cpu() is True
    assert NewtonSimulator(device="cpu:0")._use_mujoco_cpu() is True
    assert NewtonSimulator(device="cuda")._use_mujoco_cpu() is False
    assert NewtonSimulator(device="cuda:0")._use_mujoco_cpu() is False
    assert NewtonSimulator(device="cpu_foo")._use_mujoco_cpu() is False


def test_evaluate_uses_mujoco_cpu_when_device_is_cpu(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    class _ModelBuilder(_FakeBuilder):
        pass

    def solver_factory(model: _FakeModel, **kwargs: Any) -> _FakeSolver:
        captured["solver_kwargs"] = kwargs
        return _FakeSolver(model, **kwargs)

    fake_newton = ModuleType("newton")
    fake_newton.ModelBuilder = _ModelBuilder  # type: ignore[attr-defined]
    fake_newton.solvers = SimpleNamespace(SolverMuJoCo=solver_factory)  # type: ignore[attr-defined]
    fake_newton.eval_ik = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "newton", fake_newton)

    NewtonSimulator(device="cpu").evaluate(
        scene_usd=tmp_path / "scene.usda",
        body_pattern="/World/Body",
        duration_s=0.01,
        dt=0.01,
        sample_fps=100,
    )

    assert captured["solver_kwargs"]["use_mujoco_contacts"] is False
    assert captured["solver_kwargs"]["use_mujoco_cpu"] is True
    assert captured["solver_kwargs"]["njmax"] == 128


def test_evaluate_preserves_authored_ground_contact_material(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    builders: list[_FakeBuilder] = []

    class _ModelBuilder(_FakeBuilder):
        def __init__(self) -> None:
            super().__init__()
            builders.append(self)

    fake_newton = ModuleType("newton")
    fake_newton.ModelBuilder = _ModelBuilder  # type: ignore[attr-defined]
    fake_newton.solvers = SimpleNamespace(SolverMuJoCo=_FakeSolver)  # type: ignore[attr-defined]
    fake_newton.eval_ik = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "newton", fake_newton)

    NewtonSimulator(device="cpu").evaluate(
        scene_usd=tmp_path / "scene.usda",
        body_pattern="/World/Body",
        duration_s=0.01,
        dt=0.01,
        sample_fps=100,
    )

    cfg = builders[0].ground_plane_cfgs[0]
    assert cfg is not None
    assert cfg.ke == pytest.approx(10.0)
    assert cfg.kd == pytest.approx(11.0)
    assert cfg.kf == pytest.approx(12.0)
    assert cfg.ka == pytest.approx(13.0)
    assert cfg.mu == pytest.approx(0.42)
    assert cfg.restitution == pytest.approx(0.18)
    assert cfg.mu_torsional == pytest.approx(0.015)
    assert cfg.mu_rolling == pytest.approx(0.001)


def test_ground_plane_shape_config_falls_back_without_shape_config() -> None:
    class _Builder:
        shape_material_mu = [0.42]

    assert (
        NewtonSimulator._ground_plane_shape_config(
            _Builder(),
            {"path_shape_map": {"/World/GroundPlane": 0}},
        )
        is None
    )


def test_ground_plane_shape_config_falls_back_without_material_arrays() -> None:
    class _Builder:
        ShapeConfig = _FakeShapeConfig

    assert (
        NewtonSimulator._ground_plane_shape_config(
            _Builder(),
            {"path_shape_map": {"/World/GroundPlane": 0}},
        )
        is None
    )


def test_ground_plane_shape_config_falls_back_on_shape_config_type_error() -> None:
    class _RejectingShapeConfig:
        def __init__(self, **kwargs: Any) -> None:
            raise TypeError("bad config")

    class _Builder:
        ShapeConfig = _RejectingShapeConfig
        shape_material_mu = [0.42]

    assert (
        NewtonSimulator._ground_plane_shape_config(
            _Builder(),
            {"path_shape_map": {"/World/GroundPlane": 0}},
        )
        is None
    )


def test_ground_plane_shape_config_ignores_malformed_material_arrays() -> None:
    class _Builder:
        ShapeConfig = _FakeShapeConfig
        shape_material_mu = object()
        shape_material_restitution = ["not-a-number"]

    assert (
        NewtonSimulator._ground_plane_shape_config(
            _Builder(),
            {"path_shape_map": {"/World/GroundPlane": 0}},
        )
        is None
    )


def test_ground_shape_index_prefers_ground_plane_case_insensitively() -> None:
    assert (
        NewtonSimulator._ground_shape_index(
            {
                "path_shape_map": {
                    "/World/Ground": 5,
                    "/World/groundplane": 7,
                }
            }
        )
        == 7
    )


def test_ground_shape_index_prefers_ground_plane_with_underscore() -> None:
    assert (
        NewtonSimulator._ground_shape_index(
            {
                "path_shape_map": {
                    "/World/Ground": 5,
                    "/World/Ground_Plane": 7,
                }
            }
        )
        == 7
    )


def test_ground_shape_index_returns_none_without_ground_candidate() -> None:
    assert NewtonSimulator._ground_shape_index({"path_shape_map": {}}) is None
    assert NewtonSimulator._ground_shape_index({"path_shape_map": None}) is None
    assert (
        NewtonSimulator._ground_shape_index(
            {"path_shape_map": {"/World/Body": 0, "/World/Floor": 1}}
        )
        is None
    )


def test_ground_plane_helper_uses_builder_up_axis_ground_plane() -> None:
    builder = _FakeBuilder()
    NewtonSimulator._add_ground_plane(builder)

    assert builder.ground_planes == 1


def test_ground_plane_helper_falls_back_to_shape_plane() -> None:
    class _Builder:
        up_vector = (0.0, 1.0, 0.0)

        def __init__(self) -> None:
            self.planes: list[dict[str, Any]] = []

        def add_shape_plane(self, **kwargs: Any) -> None:
            self.planes.append(kwargs)

    builder = _Builder()
    NewtonSimulator._add_ground_plane(builder)

    assert builder.planes == [
        {
            "plane": (0.0, 1.0, 0.0, 0.0),
            "width": 0.0,
            "length": 0.0,
        }
    ]


def test_ground_plane_helper_falls_back_when_cfg_kwarg_is_unsupported() -> None:
    class _Builder:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def add_ground_plane(self, **kwargs: Any) -> None:
            if "cfg" in kwargs:
                raise TypeError("got an unexpected keyword argument 'cfg'")
            self.calls.append("default")

    builder = _Builder()
    NewtonSimulator._add_ground_plane(builder, cfg=_FakeShapeConfig(mu=0.2))

    assert builder.calls == ["default"]


def test_shape_plane_helper_falls_back_when_cfg_kwarg_is_unsupported() -> None:
    class _Builder:
        up_vector = (0.0, 1.0, 0.0)

        def __init__(self) -> None:
            self.planes: list[dict[str, Any]] = []

        def add_shape_plane(self, **kwargs: Any) -> None:
            if "cfg" in kwargs:
                raise TypeError("got an unexpected keyword argument 'cfg'")
            self.planes.append(kwargs)

    builder = _Builder()
    NewtonSimulator._add_ground_plane(builder, cfg=_FakeShapeConfig(mu=0.2))

    assert builder.planes == [
        {"plane": (0.0, 1.0, 0.0, 0.0), "width": 0.0, "length": 0.0}
    ]


def test_unknown_cfg_kwarg_detection_is_narrow() -> None:
    assert NewtonSimulator._looks_like_unknown_cfg_kwarg(
        TypeError("got an unexpected keyword argument 'cfg'")
    )
    assert NewtonSimulator._looks_like_unknown_cfg_kwarg(
        TypeError("GOT AN UNEXPECTED KEYWORD ARGUMENT 'CFG'")
    )
    assert not NewtonSimulator._looks_like_unknown_cfg_kwarg(
        TypeError("expected cfg to be ShapeConfig")
    )
    assert not NewtonSimulator._looks_like_unknown_cfg_kwarg(
        TypeError("unexpected failure while processing cfg")
    )


def test_resolve_body_index_supports_single_wildcard_match() -> None:
    assert (
        NewtonSimulator._resolve_body_index(
            "/World/*/Body",
            {"/World/A/Body": 3, "/World/B/Wheel": 4},
        )
        == 3
    )


def test_resolve_body_index_refuses_ambiguous_wildcard_match() -> None:
    with pytest.raises(RuntimeError, match="matched multiple"):
        NewtonSimulator._resolve_body_index(
            "/World/*/Body",
            {"/World/A/Body": 3, "/World/B/Body": 4},
        )
