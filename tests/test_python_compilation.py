"""Test that all Python files in the project compile without syntax errors.

This catches:
- Syntax errors (missing colons, parentheses, etc.)
- Import errors at module level (missing dependencies)
- Undefined names that are referenced at import time
"""

import ast
from pathlib import Path

import pytest

# Project root
ROOT = Path(__file__).parent.parent

# Directories to scan
SCAN_DIRS = ["src", "tests", "lab", "scripts", "scratch", "playground"]

# Files to skip (known issues or external)
SKIP_FILES = {
    "benchmark_opt8bit.py",  # requires GPU
}


def discover_python_files():
    """Discover all Python files in the project."""
    files = []
    for dir_name in SCAN_DIRS:
        dir_path = ROOT / dir_name
        if dir_path.exists():
            for py_file in dir_path.rglob("*.py"):
                if py_file.name not in SKIP_FILES:
                    files.append(py_file)
    return sorted(files)


def compile_file(filepath: Path) -> tuple[bool, str]:
    """Try to compile a Python file. Returns (success, error_message)."""
    try:
        source = filepath.read_text(encoding="utf-8")
        # Compile to AST (catches syntax errors)
        ast.parse(source, filename=str(filepath))
        # Also compile to bytecode (catches more issues)
        compile(source, str(filepath), "exec")
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


@pytest.mark.parametrize("filepath", discover_python_files(), ids=lambda p: str(p.relative_to(ROOT)))
def test_python_compiles(filepath: Path):
    """Each Python file must compile without errors."""
    success, error = compile_file(filepath)
    assert success, f"Failed to compile {filepath.relative_to(ROOT)}: {error}"


def test_import_main_modules():
    """Test that main modules can be imported without errors."""
    modules_to_test = [
        "src.bitnet.model.modeling_bitnet",
        "src.bitnet.training.train_sovereign_school",
        "src.bitnet.training.modules.strategy",
        "src.bitnet.training.modules.state_manager",
        "src.bitnet.training.modules.corpus",
        "src.bitnet.training.modules.exam_compiler",
        "src.bitnet.training.modules.partitioner",
        "src.bitnet.training.modules.stage_config",
        "src.bitnet.training.modules.tokenization",
    ]
    errors = []
    for module_name in modules_to_test:
        try:
            __import__(module_name)
        except ImportError as e:
            errors.append(f"{module_name}: {e}")
        except Exception as e:
            errors.append(f"{module_name}: {type(e).__name__}: {e}")

    assert not errors, "Import errors found:\n" + "\n".join(errors)
