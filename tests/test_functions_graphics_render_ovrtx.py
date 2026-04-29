# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for OvRTX rendering backend.

Unit tests run without GPU/ovrtx. Integration tests require ovrtx + RTX GPU.
"""

import os
import subprocess
import unittest.mock

import pytest

from world_understanding.functions.graphics.render_ovrtx import (
    DEFAULT_NUM_SENSOR_UPDATES,
    _build_render_products_usda,
    _copy_exported_relative_assets,
    _ensure_lights,
    _map_sensor_to_render_var,
    _OvRTXDaemon,
    _parse_frames,
)

# ---------------------------------------------------------------------------
# Unit tests (no GPU required)
# ---------------------------------------------------------------------------


class TestParseFrames:
    """Test _parse_frames() for all three formats."""

    def test_single_frame(self):
        assert _parse_frames("0") == [0]
        assert _parse_frames("42") == [42]

    def test_frame_range(self):
        assert _parse_frames("0:3") == [0, 1, 2, 3]
        assert _parse_frames("5:7") == [5, 6, 7]

    def test_comma_separated(self):
        assert _parse_frames("0,5,10") == [0, 5, 10]
        assert _parse_frames("10,0,5") == [0, 5, 10]  # sorted

    def test_single_frame_range(self):
        assert _parse_frames("3:3") == [3]

    def test_whitespace_handling(self):
        assert _parse_frames(" 0 ") == [0]
        assert _parse_frames("0, 5, 10") == [0, 5, 10]

    def test_invalid_frames_raises(self):
        with pytest.raises(ValueError):
            _parse_frames("not_a_number")


class TestBuildRenderProductsUsda:
    """Test _build_render_products_usda() generates valid USDA."""

    def test_single_camera(self):
        usda, paths = _build_render_products_usda(
            cameras=["/Cameras/Camera1"],
            image_width=512,
            image_height=512,
        )

        assert len(paths) == 1
        assert "/Render/" in paths[0]
        assert "#usda 1.0" in usda
        assert "RenderProduct" in usda
        assert "resolution = (512, 512)" in usda
        assert "LdrColor" in usda
        assert "rel camera = </Cameras/Camera1>" in usda

    def test_multiple_cameras(self):
        cameras = ["/Cameras/Cam1", "/Cameras/Cam2", "/Cameras/Cam3"]
        usda, paths = _build_render_products_usda(
            cameras=cameras,
            image_width=1024,
            image_height=768,
        )

        assert len(paths) == 3
        assert "resolution = (1024, 768)" in usda
        for camera in cameras:
            assert f"rel camera = <{camera}>" in usda

    def test_with_depth_sensor(self):
        usda, paths = _build_render_products_usda(
            cameras=["/Camera"],
            image_width=512,
            image_height=512,
            sensors=["depth"],
        )

        assert "Depth" in usda
        assert "LdrColor" in usda
        assert len(paths) == 1

    def test_render_scope(self):
        usda, _ = _build_render_products_usda(
            cameras=["/Camera"],
            image_width=256,
            image_height=256,
        )

        assert 'def Scope "Render"' in usda


class TestMapSensorToRenderVar:
    """Test _map_sensor_to_render_var() mapping correctness."""

    def test_depth_mapping(self):
        assert _map_sensor_to_render_var("depth") == "Depth"

    def test_normal_mapping(self):
        assert _map_sensor_to_render_var("normal") == "Normal"

    def test_albedo_mapping(self):
        assert _map_sensor_to_render_var("albedo") == "Albedo"

    def test_unknown_returns_none(self):
        assert _map_sensor_to_render_var("unknown_sensor") is None
        assert _map_sensor_to_render_var("") is None


class TestOvRTXBackendImportError:
    """Test graceful error when ovrtx venv cannot be provisioned."""

    def test_backend_error_when_venv_fails(self):
        """OvRTXRenderingBackend should raise RuntimeError when venv provisioning fails."""
        from world_understanding.functions.graphics.rendering import (
            OvRTXRenderingBackend,
        )

        with unittest.mock.patch(
            "world_understanding.functions.graphics.render_ovrtx._get_ovrtx_python",
            side_effect=RuntimeError("ovrtx venv creation failed"),
        ):
            with pytest.raises(RuntimeError, match="ovrtx venv creation failed"):
                OvRTXRenderingBackend()


class TestOvRTXBackendSensorSupport:
    """Test sensor capability methods without requiring ovrtx."""

    def test_supported_sensor_modes_class_var(self):
        from world_understanding.functions.graphics.rendering import (
            OvRTXRenderingBackend,
        )

        assert "depth" in OvRTXRenderingBackend.SUPPORTED_SENSOR_MODES


class TestEnsureLights:
    """Test _ensure_lights() adds default lights when needed."""

    def test_adds_hdri_dome_with_bundled_default(self, monkeypatch):
        """No env override: DomeLight at intensity=1 with the bundled EXR.

        ``_DEFAULT_HDRI_PATH`` resolves to
        ``world_understanding/data/env_maps/SmartMaterials_Environment_with_Lights.exr``
        — the same brightly-exposed HDRI Kit's
        ``stage_manager._create_default_light`` binds. Intensity is 1
        (Kit's value) because the EXR is already correctly exposed.
        """
        monkeypatch.delenv("WU_OVRTX_DEFAULT_HDRI", raising=False)
        from pxr import Usd, UsdGeom, UsdLux

        from world_understanding.functions.graphics import render_ovrtx

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Cube.Define(stage, "/World/Cube")

        # No lights initially
        light_prims = [p for p in stage.Traverse() if "Light" in p.GetTypeName()]
        assert len(light_prims) == 0

        _ensure_lights(stage)

        dome_prims = [p for p in stage.Traverse() if p.IsA(UsdLux.DomeLight)]
        assert len(dome_prims) == 1
        dome = UsdLux.DomeLight(dome_prims[0])
        # Kit's default: intensity=1 paired with the brightly-exposed
        # SmartMaterials EXR gives correct scene exposure out of the box.
        assert dome.GetIntensityAttr().Get() == render_ovrtx._DEFAULT_HDRI_INTENSITY
        tex = dome_prims[0].GetAttribute("inputs:texture:file")
        assert tex is not None
        tex_value = tex.Get()
        resolved = str(tex_value.resolvedPath or tex_value.path)
        assert resolved == render_ovrtx._DEFAULT_HDRI_PATH
        # No distant light — env map provides direction.
        assert [p for p in stage.Traverse() if p.IsA(UsdLux.DistantLight)] == []

    def test_env_var_overrides_default_hdri(self, monkeypatch, tmp_path):
        """``WU_OVRTX_DEFAULT_HDRI`` overrides the hardcoded bundled EXR.

        Operators can point at a different local ``.exr``/``.hdr`` or an
        S3 URL when they want to swap lighting (e.g. a public HDRI for
        restricted-network hosts). The override replaces the texture
        path but keeps every other DomeLight attribute identical.
        """
        fake_hdri = tmp_path / "env.exr"
        fake_hdri.write_bytes(b"fake-exr-contents")
        monkeypatch.setenv("WU_OVRTX_DEFAULT_HDRI", str(fake_hdri))

        from pxr import Usd, UsdGeom, UsdLux

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Cube.Define(stage, "/World/Cube")

        _ensure_lights(stage)

        dome_prims = [p for p in stage.Traverse() if p.IsA(UsdLux.DomeLight)]
        assert len(dome_prims) == 1
        tex = dome_prims[0].GetAttribute("inputs:texture:file")
        tex_value = tex.Get()
        resolved = str(tex_value.resolvedPath or tex_value.path)
        assert resolved == str(fake_hdri)

    def test_does_not_add_lights_when_present(self):
        from pxr import Usd, UsdLux

        stage = Usd.Stage.CreateInMemory()
        UsdLux.DomeLight.Define(stage, "/Existing/DomeLight")

        _ensure_lights(stage)

        # Should still only have the original light
        light_prims = [p for p in stage.Traverse() if "Light" in p.GetTypeName()]
        assert len(light_prims) == 1


class TestCopyExportedRelativeAssets:
    """Test local texture mirroring for exported OVRTX stages."""

    def test_copies_relative_texture_assets_to_export_dir(self, tmp_path):
        from pxr import Sdf, Usd, UsdShade

        source_dir = tmp_path / "source"
        texture_dir = source_dir / "textures"
        texture_dir.mkdir(parents=True)
        texture = texture_dir / "checker.png"
        texture.write_bytes(b"fake-png")

        stage_path = source_dir / "stage.usda"
        stage = Usd.Stage.CreateNew(str(stage_path))
        shader = UsdShade.Shader.Define(stage, "/World/Looks/Tex")
        shader.GetPrim().CreateAttribute(
            "inputs:file",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("textures/checker.png"))
        stage.GetRootLayer().Save()

        reopened = Usd.Stage.Open(str(stage_path))
        export_dir = tmp_path / "render"
        export_dir.mkdir()

        copied = _copy_exported_relative_assets(reopened, export_dir)

        assert copied == 1
        assert (export_dir / "textures" / "checker.png").read_bytes() == b"fake-png"


class TestDefaultNumSensorUpdates:
    """Test DEFAULT_NUM_SENSOR_UPDATES constant."""

    def test_default_num_sensor_updates(self):
        # 500 is the convergence plateau for PT-mode renders on the
        # kit-gen-ai-service golden scene — see the cap sweep in
        # /tmp/ovrtx_cap.py. num_sensor_updates here is step-loop count, not
        # samples-per-pixel (ovrtx 0.2.0 ignores the SPP schema attr).
        assert DEFAULT_NUM_SENSOR_UPDATES == 500


class TestOvRTXDaemonLifecycle:
    """Test _OvRTXDaemon start / render / shutdown / crash-recovery."""

    @staticmethod
    def _make_fake_daemon_script(tmp_path):
        """Write a tiny Python script that speaks the daemon JSON protocol."""
        script = tmp_path / "fake_daemon.py"
        script.write_text(
            "import json, sys\n"
            'sys.stdout.write(json.dumps({"status": "ready"}) + "\\n")\n'
            "sys.stdout.flush()\n"
            "for line in sys.stdin:\n"
            "    req = json.loads(line.strip())\n"
            '    if req.get("command") == "shutdown":\n'
            "        break\n"
            '    if req.get("command") == "render":\n'
            '        manifest = [{"camera": c, "image_files": [], '
            '"sensor_files": {}, "frame_count": 0} '
            'for c in req["cameras"]]\n'
            '        sys.stdout.write(json.dumps({"status": "ok", '
            '"manifest": manifest}) + "\\n")\n'
            "        sys.stdout.flush()\n"
        )
        return str(script)

    def test_start_and_shutdown(self, tmp_path):
        """Daemon starts, is running, then shuts down cleanly."""
        import sys

        script = self._make_fake_daemon_script(tmp_path)
        daemon = _OvRTXDaemon(ovrtx_python=sys.executable, daemon_script_path=script)
        daemon.ensure_running()
        assert daemon._is_running()
        daemon.shutdown()
        assert not daemon._is_running()

    def test_render_returns_manifest(self, tmp_path):
        """daemon.render() returns the manifest list."""
        import sys

        script = self._make_fake_daemon_script(tmp_path)
        daemon = _OvRTXDaemon(ovrtx_python=sys.executable, daemon_script_path=script)
        daemon.ensure_running()

        manifest = daemon.render(
            {
                "cameras": ["/Camera"],
                "usd_path": "/fake/stage.usdc",
                "fps": 24.0,
                "frames": [],
                "sensors": [],
                "output_dir": str(tmp_path),
                "product_paths": [],
            }
        )

        assert len(manifest) == 1
        assert manifest[0]["camera"] == "/Camera"
        daemon.shutdown()

    def test_auto_restart_on_crash(self, tmp_path):
        """Daemon auto-restarts if the subprocess dies between calls."""
        import sys

        script = self._make_fake_daemon_script(tmp_path)
        daemon = _OvRTXDaemon(ovrtx_python=sys.executable, daemon_script_path=script)
        daemon.ensure_running()

        # Kill the subprocess to simulate a crash
        daemon._process.kill()
        daemon._process.wait()
        assert not daemon._is_running()

        # Next render should auto-restart
        manifest = daemon.render(
            {
                "cameras": ["/Cam"],
                "usd_path": "/fake/stage.usdc",
                "fps": 24.0,
                "frames": [],
                "sensors": [],
                "output_dir": str(tmp_path),
                "product_paths": [],
            }
        )
        assert len(manifest) == 1
        assert daemon._is_running()
        daemon.shutdown()

    def test_shutdown_when_not_running_is_noop(self, tmp_path):
        """Calling shutdown() when the daemon is not started does not raise."""
        import sys

        script = self._make_fake_daemon_script(tmp_path)
        daemon = _OvRTXDaemon(ovrtx_python=sys.executable, daemon_script_path=script)
        daemon.shutdown()  # should not raise

    def test_render_error_propagates(self, tmp_path):
        """Daemon error responses become RuntimeError in the caller."""
        import sys

        # Script that always returns an error for render commands
        script = tmp_path / "err_daemon.py"
        script.write_text(
            "import json, sys\n"
            'sys.stdout.write(json.dumps({"status": "ready"}) + "\\n")\n'
            "sys.stdout.flush()\n"
            "for line in sys.stdin:\n"
            "    req = json.loads(line.strip())\n"
            '    if req.get("command") == "shutdown":\n'
            "        break\n"
            '    sys.stdout.write(json.dumps({"status": "error", '
            '"error": "boom"}) + "\\n")\n'
            "    sys.stdout.flush()\n"
        )

        daemon = _OvRTXDaemon(
            ovrtx_python=sys.executable, daemon_script_path=str(script)
        )
        daemon.ensure_running()

        with pytest.raises(RuntimeError, match="boom"):
            daemon.render(
                {
                    "cameras": [],
                    "usd_path": "/fake/stage.usdc",
                    "fps": 24.0,
                    "frames": [],
                    "sensors": [],
                    "output_dir": str(tmp_path),
                    "product_paths": [],
                }
            )
        daemon.shutdown()


class TestRenderAllCamerasBackwardCompat:
    """Verify render_all_cameras(daemon=None) still uses subprocess.run."""

    def test_daemon_none_uses_subprocess(self):
        """When daemon=None, the old subprocess.run path should be taken."""
        with (
            unittest.mock.patch(
                "world_understanding.functions.graphics.render_ovrtx._get_ovrtx_python",
                return_value="/fake/python",
            ),
            unittest.mock.patch(
                "world_understanding.functions.graphics.render_ovrtx.subprocess.run",
            ) as mock_run,
        ):
            # Make subprocess.run return a failure so we can catch it quickly
            mock_run.return_value = unittest.mock.Mock(
                returncode=1, stdout="", stderr="test"
            )
            from pxr import Usd

            stage = Usd.Stage.CreateInMemory()
            from world_understanding.functions.graphics.render_ovrtx import (
                render_all_cameras,
            )

            with pytest.raises(RuntimeError, match="OvRTX subprocess failed"):
                render_all_cameras(stage=stage, daemon=None)

            mock_run.assert_called_once()


class TestOvRTXBackendCreatesDaemon:
    """Verify OvRTXRenderingBackend creates and shuts down a daemon."""

    def test_backend_creates_daemon(self):
        """OvRTXRenderingBackend.__init__ should create a _OvRTXDaemon."""
        from world_understanding.functions.graphics.rendering import (
            OvRTXRenderingBackend,
        )

        with unittest.mock.patch(
            "world_understanding.functions.graphics.render_ovrtx._get_ovrtx_python",
            return_value="/fake/python",
        ):
            backend = OvRTXRenderingBackend()
            assert hasattr(backend, "_daemon")
            assert isinstance(backend._daemon, _OvRTXDaemon)


class TestOvRTXBackendNumSensorUpdatesPrecedence:
    """Verify OvRTXRenderingBackend.render's per-call num_sensor_updates override."""

    def _make_backend(self, num_sensor_updates: int):
        from world_understanding.functions.graphics.rendering import (
            OvRTXRenderingBackend,
        )

        with unittest.mock.patch(
            "world_understanding.functions.graphics.render_ovrtx._get_ovrtx_python",
            return_value="/fake/python",
        ):
            return OvRTXRenderingBackend(num_sensor_updates=num_sensor_updates)

    def test_none_falls_back_to_instance_default(self):
        """num_sensor_updates=None must pass the instance-level value through."""
        from pxr import Usd

        backend = self._make_backend(num_sensor_updates=7)
        stage = Usd.Stage.CreateInMemory()

        with unittest.mock.patch(
            "world_understanding.functions.graphics.render_ovrtx.render_all_cameras",
            return_value={"results": []},
        ) as mock_render:
            backend.render(stage=stage, num_sensor_updates=None)

        kwargs = mock_render.call_args.kwargs
        assert kwargs["num_sensor_updates"] == 7

    def test_explicit_value_overrides_instance_default(self):
        """An explicit num_sensor_updates must beat the instance-level value."""
        from pxr import Usd

        backend = self._make_backend(num_sensor_updates=7)
        stage = Usd.Stage.CreateInMemory()

        with unittest.mock.patch(
            "world_understanding.functions.graphics.render_ovrtx.render_all_cameras",
            return_value={"results": []},
        ) as mock_render:
            backend.render(stage=stage, num_sensor_updates=42)

        kwargs = mock_render.call_args.kwargs
        assert kwargs["num_sensor_updates"] == 42


# ---------------------------------------------------------------------------
# Integration tests (require RTX GPU + ovrtx installed)
# ---------------------------------------------------------------------------

_has_ovrtx = False
_ovrtx_skip_reason = "ovrtx not available"
try:
    from world_understanding.functions.graphics.render_ovrtx import _get_ovrtx_python

    _python = _get_ovrtx_python()
    # Verify ovrtx can actually initialize (needs GPU + Vulkan + libGL).
    # Strip PYTHONPATH so usd-core doesn't leak into the isolated venv.
    import subprocess

    _env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    _probe = subprocess.run(
        [_python, "-c", "import ovrtx; ovrtx.Renderer()"],
        capture_output=True,
        timeout=60,
        env=_env,
    )
    _has_ovrtx = _probe.returncode == 0
    if not _has_ovrtx:
        _ovrtx_skip_reason = f"ovrtx Renderer() failed: {_probe.stderr.decode()[-200:]}"
except (ImportError, RuntimeError, subprocess.TimeoutExpired) as exc:
    _ovrtx_skip_reason = f"ovrtx probe error: {exc}"

requires_ovrtx = pytest.mark.skipif(not _has_ovrtx, reason=_ovrtx_skip_reason)


@pytest.fixture
def simple_usd_stage():
    """Create a simple USD stage with a cube and camera for testing."""
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)

    # Add a simple cube
    cube = UsdGeom.Cube.Define(stage, "/World/Cube")
    cube.GetSizeAttr().Set(1.0)

    # Add a camera
    camera = UsdGeom.Camera.Define(stage, "/Camera")
    camera.GetFocalLengthAttr().Set(50.0)
    camera.GetHorizontalApertureAttr().Set(36.0)

    # Position camera to see the cube
    xform = UsdGeom.Xformable(camera.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(3.0, 3.0, 3.0))

    return stage


@requires_ovrtx
class TestOvRTXIntegrationSingleCamera:
    """Integration test: single camera rendering with OvRTX."""

    def test_render_single_camera(self, simple_usd_stage):
        from world_understanding.functions.graphics.render_ovrtx import (
            render_all_cameras,
        )

        result = render_all_cameras(
            stage=simple_usd_stage,
            image_width=256,
            image_height=256,
            cameras=["/Camera"],
            frames="0",
        )

        assert result["total_cameras"] == 1
        assert result["successful_cameras"] == 1
        assert result["failed_cameras"] == 0
        assert len(result["results"]) == 1
        assert result["results"][0]["frame_count"] >= 1
        assert len(result["results"][0]["images"]) >= 1

        # Check that the image is a valid PIL Image
        img = result["results"][0]["images"][0]
        assert img.size == (256, 256)


@requires_ovrtx
class TestOvRTXIntegrationMultipleCameras:
    """Integration test: multi-camera rendering with OvRTX."""

    def test_render_multiple_cameras(self, simple_usd_stage):
        from pxr import Gf, UsdGeom

        # Add a second camera
        camera2 = UsdGeom.Camera.Define(simple_usd_stage, "/Camera2")
        camera2.GetFocalLengthAttr().Set(50.0)
        camera2.GetHorizontalApertureAttr().Set(36.0)
        xform2 = UsdGeom.Xformable(camera2.GetPrim())
        xform2.AddTranslateOp().Set(Gf.Vec3d(-3.0, 3.0, 3.0))

        from world_understanding.functions.graphics.render_ovrtx import (
            render_all_cameras,
        )

        result = render_all_cameras(
            stage=simple_usd_stage,
            image_width=256,
            image_height=256,
            cameras=["/Camera", "/Camera2"],
            frames="0",
        )

        assert result["total_cameras"] == 2
        assert result["successful_cameras"] == 2
        assert len(result["results"]) == 2
