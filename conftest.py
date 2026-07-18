# conftest.py — make `shared` and each stage's modules importable under pytest.
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
for path in (ROOT, os.path.join(ROOT, "stage1"), os.path.join(ROOT, "stage2"),
             os.path.join(ROOT, "stage1", "remote"), os.path.join(ROOT, "stage2", "remote")):
    if path not in sys.path:
        sys.path.insert(0, path)
