# stage5/conftest.py — put stage5's own dirs on sys.path for its bare imports.
import importlib
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
_STAGE_PATHS = (HERE, os.path.join(HERE, "remote"))
# Bare top-level module names stage5 SHARES with sibling stages (stage4). Running
# several stages in ONE pytest session (`pytest stage4 stage5 ...`) means they'd
# fight over a single sys.modules entry, so string-target patches, test-local
# imports, and reloads could bind the wrong stage's copy. We load THIS stage's
# copies once, then RESTORE them into sys.modules before this stage's modules are
# collected and before each of its tests runs. (run_stage5/rlvr_train/reward names
# are unique across stages, so they need no isolation.)
_SHARED_MODULES = ("enums", "status_io", "verdict", "controller")


def _front_paths():
    for path in _STAGE_PATHS:
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)


def _load_own():
    _front_paths()
    cache = {}
    for name in _SHARED_MODULES:
        sys.modules.pop(name, None)  # evict a sibling's copy, import ours fresh
        cache[name] = importlib.import_module(name)
    return cache


_OWN = _load_own()


def _restore_own():
    _front_paths()
    sys.modules.update(_OWN)


def pytest_collectstart(collector):
    node = getattr(collector, "path", None) or getattr(collector, "fspath", None)
    if node is not None and HERE in str(node):
        _restore_own()


@pytest.fixture(autouse=True)
def _stage5_import_isolation():
    _restore_own()
