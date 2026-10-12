from pathlib import Path

import astro_retrieval


def test_package_imports_from_source_tree():
    project_root = Path(__file__).resolve().parents[1]
    expected = project_root / "src" / "astro_retrieval" / "__init__.py"
    assert Path(astro_retrieval.__file__).resolve() == expected
