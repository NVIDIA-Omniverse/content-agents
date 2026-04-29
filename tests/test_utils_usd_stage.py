# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for USD Stage utility functions."""

import os

import pytest

# Skip all tests in this module if pxr is not installed
pxr = pytest.importorskip("pxr")

from pxr import Sdf, Usd, UsdGeom  # noqa: E402

from world_understanding.utils.usd.stage import (  # noqa: E402
    MAX_PATH_COMPONENT_LEN,
    create_stage,
    create_stage_with_file,
    create_temp_stage,
    duplicate_stage,
    export_stage_to_string,
    flatten_stage,
    get_stage_info,
    get_stage_info_from_path,
    load_stage,
    load_stage_from_string,
    merge_stages,
    remove_animation,
    save_stage,
    shorten_for_filesystem,
)


def test_sanitize_name_for_filesystem():
    """Test the sanitize_name_for_filesystem function."""
    from world_understanding.utils.usd.stage import (
        sanitize_name_for_filesystem,
    )

    # Test basic path sanitization
    assert sanitize_name_for_filesystem("/World/Camera_001") == "World_Camera_001"

    # Test with spaces
    assert sanitize_name_for_filesystem("/World/Camera 001") == "World_Camera_001"

    # Test with special characters
    assert sanitize_name_for_filesystem("Camera:Main (HD)") == "Camera_Main__HD"

    # Test with multiple slashes
    assert (
        sanitize_name_for_filesystem("/World/Lights/Key Light")
        == "World_Lights_Key_Light"
    )

    # Test empty string
    assert sanitize_name_for_filesystem("") == "unnamed"

    # Test string that becomes empty after sanitization
    assert sanitize_name_for_filesystem("///") == "unnamed"

    # Test with dots and hyphens (should be preserved)
    assert sanitize_name_for_filesystem("camera-main.v2") == "camera-main.v2"


def test_shorten_for_filesystem_leaves_short_names_unchanged():
    """Short names should only be sanitized, not truncated."""
    assert shorten_for_filesystem("short_name", max_len=32) == "short_name"
    assert shorten_for_filesystem("Camera:Main (HD)", max_len=64) == "Camera_Main__HD"


def test_shorten_for_filesystem_is_deterministic_and_bounded():
    """Long names should be stably truncated with hash suffix and length cap."""
    long_name = "part_" + ("abc123XYZ" * 50)
    shortened_once = shorten_for_filesystem(long_name, max_len=MAX_PATH_COMPONENT_LEN)
    shortened_twice = shorten_for_filesystem(long_name, max_len=MAX_PATH_COMPONENT_LEN)

    assert shortened_once == shortened_twice
    assert len(shortened_once) <= MAX_PATH_COMPONENT_LEN
    assert "_" in shortened_once


def test_shorten_for_filesystem_distinguishes_colliding_prefixes():
    """Different long names with same prefix should produce different outputs."""
    shared_prefix = "same_prefix_" + ("x" * 200)
    long_name_a = shared_prefix + "_A"
    long_name_b = shared_prefix + "_B"

    shortened_a = shorten_for_filesystem(long_name_a, max_len=64)
    shortened_b = shorten_for_filesystem(long_name_b, max_len=64)

    assert shortened_a != shortened_b
    assert len(shortened_a) <= 64
    assert len(shortened_b) <= 64


def test_create_stage():
    """Test creating a stage in memory."""
    # Test with identifier
    stage = create_stage("test.usda")
    assert stage is not None
    assert isinstance(stage, Usd.Stage)

    # Test without identifier
    stage = create_stage()
    assert stage is not None
    assert isinstance(stage, Usd.Stage)


def test_create_stage_with_file(tmp_path):
    """Test creating a stage with an associated file."""
    test_file = tmp_path / "test.usda"
    stage = create_stage_with_file(test_file)

    assert stage is not None
    assert isinstance(stage, Usd.Stage)
    assert os.path.exists(test_file)


def test_load_stage(tmp_path):
    """Test loading a stage from file."""
    # Create a test file with a real USD stage
    test_file = tmp_path / "test.usda"
    test_stage = Usd.Stage.CreateNew(str(test_file))
    test_stage.Save()

    stage = load_stage(test_file)
    assert stage is not None
    assert isinstance(stage, Usd.Stage)

    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        load_stage(tmp_path / "nonexistent.usda")


def test_save_stage(tmp_path):
    """Test saving a stage to file."""
    # Create a stage with a file
    existing_path = tmp_path / "existing.usda"
    stage = Usd.Stage.CreateNew(str(existing_path))
    stage.Save()

    # Test saving to new path
    new_path = tmp_path / "new.usda"
    result = save_stage(stage, new_path)
    assert result == str(new_path)
    assert os.path.exists(new_path)

    # Test saving to existing path
    result = save_stage(stage)
    assert result == str(existing_path)

    # Test in-memory stage without path
    # Note: In-memory stages have anonymous identifiers like "anon:0x..."
    # The save_stage function should raise an error for anonymous layers
    in_memory_stage = Usd.Stage.CreateInMemory()
    # Check if the identifier is anonymous
    assert in_memory_stage.GetRootLayer().anonymous
    with pytest.raises(ValueError):
        save_stage(in_memory_stage)


def test_export_stage_to_string():
    """Test exporting stage to string."""
    stage = Usd.Stage.CreateInMemory()
    # Add a simple prim to make the stage non-empty
    stage.DefinePrim("/TestPrim", "Xform")

    # Test with comment
    result = export_stage_to_string(stage, add_comment=True)
    assert result is not None
    assert "#usda 1.0" in result
    assert "TestPrim" in result

    # Test without comment
    result = export_stage_to_string(stage, add_comment=False)
    assert result is not None
    assert "#usda 1.0" in result
    assert "TestPrim" in result


def test_load_stage_from_string():
    """Test creating stage from string."""
    usda_content = '#usda 1.0\ndef Xform "Test" {}'

    stage = load_stage_from_string(usda_content, "test.usda")

    assert stage is not None
    assert isinstance(stage, Usd.Stage)
    # Verify the content was loaded
    test_prim = stage.GetPrimAtPath("/Test")
    assert test_prim.IsValid()
    assert test_prim.GetTypeName() == "Xform"


def test_duplicate_stage():
    """Test duplicating a stage."""
    source_stage = Usd.Stage.CreateInMemory()
    source_stage.DefinePrim("/SourcePrim", "Xform")

    # Duplicate with identifier
    dup_stage = duplicate_stage(source_stage, "duplicate.usda")
    assert dup_stage is not None
    assert isinstance(dup_stage, Usd.Stage)
    # Verify the content was duplicated
    dup_prim = dup_stage.GetPrimAtPath("/SourcePrim")
    assert dup_prim.IsValid()

    # Duplicate without identifier
    dup_stage = duplicate_stage(source_stage)
    assert dup_stage is not None
    assert isinstance(dup_stage, Usd.Stage)
    dup_prim = dup_stage.GetPrimAtPath("/SourcePrim")
    assert dup_prim.IsValid()


def test_create_temp_stage():
    """Test creating temporary stage."""
    stage, temp_path = create_temp_stage()

    assert stage is not None
    assert isinstance(stage, Usd.Stage)
    assert os.path.exists(temp_path)
    assert temp_path.endswith(".usda")

    # Clean up
    os.unlink(temp_path)


def test_get_stage_info(tmp_path):
    """Test getting stage information."""
    # Create a stage with some properties
    test_file = tmp_path / "test.usda"
    stage = Usd.Stage.CreateNew(str(test_file))

    # Set up stage properties
    stage.SetStartTimeCode(1.0)
    stage.SetEndTimeCode(100.0)
    stage.SetTimeCodesPerSecond(24.0)

    # Add some prims
    world_prim = stage.DefinePrim("/World", "Xform")
    stage.DefinePrim("/World/Child1", "Xform")
    stage.DefinePrim("/World/Child2", "Xform")
    stage.SetDefaultPrim(world_prim)

    # Set up axis and units
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)

    info = get_stage_info(stage)

    assert info["root_layer_path"] == str(test_file)
    assert info["up_axis"] == "Y"
    assert info["meters_per_unit"] == 0.01
    assert info["start_time_code"] == 1.0
    assert info["end_time_code"] == 100.0
    assert info["time_codes_per_second"] == 24.0
    assert info["prim_count"] == 3  # World, Child1, Child2
    assert info["default_prim"] == "/World"
    assert info["layer_count"] >= 1


def test_get_stage_info_from_path(tmp_path):
    """Test getting stage info from a USD file path."""
    test_file = tmp_path / "count_test.usda"
    stage = Usd.Stage.CreateNew(str(test_file))

    stage.DefinePrim("/World", "Xform")
    stage.DefinePrim("/World/Child", "Xform")
    stage.Save()

    info = get_stage_info_from_path(test_file)
    assert info is not None
    assert info["prim_count"] == 2
    assert get_stage_info_from_path(tmp_path / "missing.usda") is None


def test_flatten_stage(tmp_path):
    """Test flattening a stage."""
    # Create a source stage with some hierarchy
    source_stage = Usd.Stage.CreateInMemory()
    source_stage.DefinePrim("/World", "Xform")
    source_stage.DefinePrim("/World/Child", "Xform")

    # Add a sublayer to test flattening
    sublayer = Sdf.Layer.CreateAnonymous()
    source_stage.GetRootLayer().subLayerPaths.append(sublayer.identifier)

    # Test with output path
    output_path = tmp_path / "flattened.usda"
    result = flatten_stage(source_stage, output_path, add_comment=True)
    assert result is not None
    assert isinstance(result, Usd.Stage)
    assert os.path.exists(output_path)

    # Test without output path (in-memory)
    result = flatten_stage(source_stage)
    assert result is not None
    assert isinstance(result, Usd.Stage)
    # Verify the prims exist in the flattened stage
    assert result.GetPrimAtPath("/World").IsValid()
    assert result.GetPrimAtPath("/World/Child").IsValid()


def test_merge_stages():
    """Test merging stages."""
    # Create base stage
    base_stage = Usd.Stage.CreateInMemory()
    base_stage.DefinePrim("/World", "Xform")
    base_stage.DefinePrim("/World/Base", "Xform")

    # Create overlay stage
    overlay_stage = Usd.Stage.CreateInMemory()
    overlay_stage.DefinePrim("/Overlay", "Xform")
    overlay_stage.DefinePrim("/Overlay/Child", "Xform")

    # Test merge
    result = merge_stages(base_stage, overlay_stage, "/World/Merged")
    assert result is not None
    assert isinstance(result, Usd.Stage)

    # Verify base content is preserved
    assert result.GetPrimAtPath("/World").IsValid()
    assert result.GetPrimAtPath("/World/Base").IsValid()

    # Verify overlay content is merged under the specified path
    assert result.GetPrimAtPath("/World/Merged").IsValid()
    # The merge should have created the parent prim for the overlay content
    merged_prim = result.GetPrimAtPath("/World/Merged")
    assert merged_prim.IsValid()


def test_remove_animation_basic():
    """Test basic functionality of remove_animation."""
    stage = Usd.Stage.CreateInMemory()

    # Create an Xform prim with animated translate attribute
    xform = UsdGeom.Xform.Define(stage, "/AnimatedPrim")
    translate_op = xform.AddTranslateOp()

    # Set time-sampled values
    translate_op.Set((0.0, 0.0, 0.0), Usd.TimeCode(0))
    translate_op.Set((10.0, 10.0, 10.0), Usd.TimeCode(24))
    translate_op.Set((20.0, 20.0, 20.0), Usd.TimeCode(48))

    # Verify animation exists
    attr = xform.GetPrim().GetAttribute("xformOp:translate")
    assert attr.GetNumTimeSamples() == 3

    # Remove animation (defaults to time 0 since no start time code set)
    num_removed = remove_animation(stage)

    # Verify animation was removed
    assert num_removed == 1
    assert attr.GetNumTimeSamples() == 0

    # Verify static value is set (sampled from time 0)
    value = attr.Get()
    assert value is not None
    assert value == (0.0, 0.0, 0.0)


def test_remove_animation_custom_reference_time():
    """Test remove_animation with custom reference time."""
    stage = Usd.Stage.CreateInMemory()

    # Create an Xform prim with animated translate attribute
    xform = UsdGeom.Xform.Define(stage, "/AnimatedPrim")
    translate_op = xform.AddTranslateOp()

    # Set time-sampled values
    translate_op.Set((0.0, 0.0, 0.0), Usd.TimeCode(0))
    translate_op.Set((10.0, 10.0, 10.0), Usd.TimeCode(24))
    translate_op.Set((20.0, 20.0, 20.0), Usd.TimeCode(48))

    # Remove animation, sampling at time 24
    num_removed = remove_animation(stage, reference_time=Usd.TimeCode(24))

    # Verify animation was removed and value is from time 24
    attr = xform.GetPrim().GetAttribute("xformOp:translate")
    assert num_removed == 1
    assert attr.GetNumTimeSamples() == 0
    value = attr.Get()
    assert value == (10.0, 10.0, 10.0)


def test_remove_animation_default_time_code():
    """Test that remove_animation uses stage.GetStartTimeCode() or time 0 by default."""
    stage = Usd.Stage.CreateInMemory()

    # Set stage start time code to test that it's used as the default
    stage.SetStartTimeCode(24)

    # Create an Xform prim with animated translate attribute
    xform = UsdGeom.Xform.Define(stage, "/AnimatedPrim")
    translate_op = xform.AddTranslateOp()

    # Set time-sampled values at different times
    translate_op.Set((5.0, 5.0, 5.0), Usd.TimeCode(0))
    translate_op.Set((15.0, 15.0, 15.0), Usd.TimeCode(24))

    # Remove animation without specifying reference_time (uses stage start time)
    num_removed = remove_animation(stage)

    # Verify animation was removed
    attr = xform.GetPrim().GetAttribute("xformOp:translate")
    assert num_removed == 1
    assert attr.GetNumTimeSamples() == 0
    # Should use value from stage start time (24), which is (15, 15, 15)
    value = attr.Get()
    assert value is not None
    assert value == (15.0, 15.0, 15.0)


def test_remove_animation_multiple_attributes():
    """Test remove_animation with multiple animated attributes."""
    stage = Usd.Stage.CreateInMemory()

    # Create multiple prims with animation
    xform1 = UsdGeom.Xform.Define(stage, "/Prim1")
    translate_op1 = xform1.AddTranslateOp()
    translate_op1.Set((0.0, 0.0, 0.0), Usd.TimeCode(0))
    translate_op1.Set((10.0, 10.0, 10.0), Usd.TimeCode(24))

    xform2 = UsdGeom.Xform.Define(stage, "/Prim2")
    translate_op2 = xform2.AddTranslateOp()
    translate_op2.Set((5.0, 5.0, 5.0), Usd.TimeCode(0))
    translate_op2.Set((15.0, 15.0, 15.0), Usd.TimeCode(24))

    rotate_op2 = xform2.AddRotateXYZOp()
    rotate_op2.Set((0.0, 0.0, 0.0), Usd.TimeCode(0))
    rotate_op2.Set((90.0, 0.0, 0.0), Usd.TimeCode(24))

    # Remove all animation
    num_removed = remove_animation(stage)

    # Verify all animations were removed (3 animated attributes)
    assert num_removed == 3

    # Verify all attributes are now static
    attr1 = xform1.GetPrim().GetAttribute("xformOp:translate")
    attr2 = xform2.GetPrim().GetAttribute("xformOp:translate")
    attr3 = xform2.GetPrim().GetAttribute("xformOp:rotateXYZ")
    assert attr1.GetNumTimeSamples() == 0
    assert attr2.GetNumTimeSamples() == 0
    assert attr3.GetNumTimeSamples() == 0


def test_remove_animation_no_animation():
    """Test remove_animation when stage has no animation."""
    stage = Usd.Stage.CreateInMemory()

    # Create a static prim with no animation
    xform = UsdGeom.Xform.Define(stage, "/StaticPrim")
    translate_op = xform.AddTranslateOp()
    translate_op.Set((5.0, 5.0, 5.0))  # Static value, no time code

    # Remove animation (should find nothing)
    num_removed = remove_animation(stage)

    # Verify no attributes were modified
    assert num_removed == 0

    # Verify static value is preserved
    attr = xform.GetPrim().GetAttribute("xformOp:translate")
    assert attr.Get() == (5.0, 5.0, 5.0)


def test_remove_animation_returns_correct_count():
    """Test that remove_animation returns accurate count of modified attributes."""
    stage = Usd.Stage.CreateInMemory()

    # Create prims: 2 animated, 1 static
    xform1 = UsdGeom.Xform.Define(stage, "/Animated1")
    op1 = xform1.AddTranslateOp()
    op1.Set((0.0, 0.0, 0.0), Usd.TimeCode(0))
    op1.Set((10.0, 10.0, 10.0), Usd.TimeCode(24))

    xform2 = UsdGeom.Xform.Define(stage, "/Animated2")
    op2 = xform2.AddScaleOp()
    op2.Set((1.0, 1.0, 1.0), Usd.TimeCode(0))
    op2.Set((2.0, 2.0, 2.0), Usd.TimeCode(24))

    xform3 = UsdGeom.Xform.Define(stage, "/Static")
    op3 = xform3.AddTranslateOp()
    op3.Set((5.0, 5.0, 5.0))  # Static, no time samples

    # Remove animation
    num_removed = remove_animation(stage)

    # Should only count the 2 animated attributes
    assert num_removed == 2
