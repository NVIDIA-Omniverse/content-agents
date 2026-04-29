# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for _collect_mdl_deps helper in material_agent.manifest."""

from pathlib import Path

from material_agent.manifest import _collect_mdl_deps


class TestCollectMdlDepsNoImports:
    """Tests for MDL files with no imports."""

    def test_single_mdl_no_imports(self, tmp_path: Path):
        """Single MDL with no import statements → returns just that file."""
        mdl = tmp_path / "Simple.mdl"
        mdl.write_text(
            "mdl 1.0;\nexport material Simple() = material() {};",
            encoding="utf-8",
        )

        result = _collect_mdl_deps(mdl)

        assert result == {mdl}

    def test_non_existent_path_returns_empty(self, tmp_path: Path):
        """Non-existent MDL path → returns empty set."""
        result = _collect_mdl_deps(tmp_path / "does_not_exist.mdl")

        assert result == set()


class TestCollectMdlDepsSiblingImport:
    """Tests for sibling (same-directory) imports."""

    def test_sibling_import_includes_dependency(self, tmp_path: Path):
        """import Foo::bar → includes Foo.mdl if it exists in same dir."""
        foo = tmp_path / "Foo.mdl"
        foo.write_text(
            "mdl 1.0;\nexport float bar() { return 1.0; }",
            encoding="utf-8",
        )

        main = tmp_path / "Main.mdl"
        main.write_text(
            "mdl 1.0;\nimport Foo::bar;\nexport material M() = material() {};",
            encoding="utf-8",
        )

        result = _collect_mdl_deps(main)

        assert main in result
        assert foo in result

    def test_sibling_import_missing_file_ignored(self, tmp_path: Path):
        """import Foo::bar but Foo.mdl doesn't exist → only returns the main file."""
        main = tmp_path / "Main.mdl"
        main.write_text(
            "mdl 1.0;\nimport MissingModule::func;\n"
            "export material M() = material() {};",
            encoding="utf-8",
        )

        result = _collect_mdl_deps(main)

        assert result == {main}


class TestCollectMdlDepsRelativeImport:
    """Tests for relative parent imports (..::)."""

    def test_relative_parent_import(self, tmp_path: Path):
        """..::Templates::Glass → resolves to parent/Templates/Glass.mdl."""
        templates_dir = tmp_path / "Templates"
        templates_dir.mkdir()
        glass = templates_dir / "Glass.mdl"
        glass.write_text(
            "mdl 1.0;\nexport material Glass() = material() {};",
            encoding="utf-8",
        )

        sub_dir = tmp_path / "sub"
        sub_dir.mkdir()
        main = sub_dir / "Main.mdl"
        main.write_text(
            "mdl 1.0;\nimport ..::Templates::Glass;\n"
            "export material M() = material() {};",
            encoding="utf-8",
        )

        result = _collect_mdl_deps(main)

        assert main in result
        assert glass in result


class TestCollectMdlDepsCommentStripping:
    """Tests that imports inside comments are ignored."""

    def test_single_line_comment_ignored(self, tmp_path: Path):
        """// import Foo::bar → does NOT include Foo.mdl."""
        foo = tmp_path / "Foo.mdl"
        foo.write_text(
            "mdl 1.0;\nexport float bar() { return 1.0; }",
            encoding="utf-8",
        )

        main = tmp_path / "Main.mdl"
        main.write_text(
            "mdl 1.0;\n// import Foo::bar;\nexport material M() = material() {};",
            encoding="utf-8",
        )

        result = _collect_mdl_deps(main)

        assert result == {main}
        assert foo not in result

    def test_block_comment_ignored(self, tmp_path: Path):
        """/* import Foo::bar */ → does NOT include Foo.mdl."""
        foo = tmp_path / "Foo.mdl"
        foo.write_text(
            "mdl 1.0;\nexport float bar() { return 1.0; }",
            encoding="utf-8",
        )

        main = tmp_path / "Main.mdl"
        main.write_text(
            "mdl 1.0;\n/* import Foo::bar; */\nexport material M() = material() {};",
            encoding="utf-8",
        )

        result = _collect_mdl_deps(main)

        assert result == {main}
        assert foo not in result

    def test_multiline_block_comment_ignored(self, tmp_path: Path):
        """Multi-line block comment with import → does NOT include dependency."""
        foo = tmp_path / "Foo.mdl"
        foo.write_text(
            "mdl 1.0;\nexport float bar() { return 1.0; }",
            encoding="utf-8",
        )

        main = tmp_path / "Main.mdl"
        main.write_text(
            "mdl 1.0;\n"
            "/*\n"
            " * import Foo::bar;\n"
            " * This import is disabled.\n"
            " */\n"
            "export material M() = material() {};",
            encoding="utf-8",
        )

        result = _collect_mdl_deps(main)

        assert result == {main}
        assert foo not in result


class TestCollectMdlDepsCircular:
    """Tests for circular import handling."""

    def test_circular_imports_do_not_loop(self, tmp_path: Path):
        """A imports B, B imports A → terminates without infinite loop."""
        a = tmp_path / "A.mdl"
        b = tmp_path / "B.mdl"

        a.write_text(
            "mdl 1.0;\nimport B::func;\nexport float x() { return 0; }",
            encoding="utf-8",
        )
        b.write_text(
            "mdl 1.0;\nimport A::func;\nexport float func() { return 1; }",
            encoding="utf-8",
        )

        result = _collect_mdl_deps(a)

        assert a in result
        assert b in result
        assert len(result) == 2


class TestCollectMdlDepsTransitive:
    """Tests for transitive dependency collection."""

    def test_transitive_deps_collected(self, tmp_path: Path):
        """A → B → C: starting from A collects all three."""
        c = tmp_path / "C.mdl"
        c.write_text(
            "mdl 1.0;\nexport float helper() { return 0; }",
            encoding="utf-8",
        )

        b = tmp_path / "B.mdl"
        b.write_text(
            "mdl 1.0;\nimport C::helper;\nexport float func() { return 1; }",
            encoding="utf-8",
        )

        a = tmp_path / "A.mdl"
        a.write_text(
            "mdl 1.0;\nimport B::func;\nexport material M() = material() {};",
            encoding="utf-8",
        )

        result = _collect_mdl_deps(a)

        assert result == {a, b, c}
