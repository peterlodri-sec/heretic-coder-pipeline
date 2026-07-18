# stage2/conftest.py — put stage2's own dirs on sys.path for its bare imports.
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for path in (HERE, os.path.join(HERE, "remote")):
    if path not in sys.path:
        sys.path.insert(0, path)
