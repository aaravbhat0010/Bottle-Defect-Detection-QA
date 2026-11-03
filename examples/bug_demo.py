"""Run from repository root: python examples/bug_demo.py"""
from pathlib import Path
import subprocess
import sys
import tempfile

source = Path("bottle_qa/data.py").read_text()
assert "/ 255.0" in source
with tempfile.TemporaryDirectory() as folder:
    root = Path(folder)
    (root / "loader.py").write_text(source.replace("/ 255.0", "", 1))
    test = Path("tests/test_data.py").read_text().split("def test_corrupt_image_rejected")[0]
    test = test.replace("from bottle_qa.data import load_image", "from loader import load_image")
    (root / "test_scaling.py").write_text(test)
    print("Deliberate missing-normalization bug: expect failure", flush=True)
    failed = subprocess.run([sys.executable, "-B", "-m", "pytest", "-q", str(root / "test_scaling.py")], cwd=root)
    assert failed.returncode == 1, "Expected an assertion failure"
    (root / "loader.py").write_text(source)
    print("Normalization restored: expect pass", flush=True)
    fixed = subprocess.run([sys.executable, "-B", "-m", "pytest", "-q", str(root / "test_scaling.py")], cwd=root)
    assert fixed.returncode == 0
