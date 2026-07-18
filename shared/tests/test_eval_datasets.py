import sys
import types
from unittest.mock import MagicMock, patch


def _fake_datasets(rows):
    """A stand-in for the (uninstalled) `datasets` package.

    The loaders `import datasets` and call `datasets.load_dataset(...)`, so we
    inject a fake module into sys.modules. This keeps the loaders patchable
    without the heavy `datasets` dependency being installed.
    """
    module = types.ModuleType("datasets")
    module.load_dataset = MagicMock(return_value=rows)
    return module


def test_load_refusal_prompts_returns_instruction_strings():
    from shared.eval import datasets as eval_datasets

    rows = [
        {"instruction": "harm one ", "response": "sure"},
        {"instruction": " harm two", "response": "sure"},
        {"instruction": "harm three", "response": "sure"},
    ]
    fake = _fake_datasets(rows)
    with patch.dict(sys.modules, {"datasets": fake}):
        prompts = eval_datasets.load_refusal_prompts(limit=2)

    assert prompts == ["harm one", "harm two"]
    fake.load_dataset.assert_called_once()


def test_load_refusal_prompts_default_limit_caps_length():
    from shared.eval import datasets as eval_datasets

    rows = [{"instruction": f"harm {i}", "response": "x"} for i in range(400)]
    fake = _fake_datasets(rows)
    with patch.dict(sys.modules, {"datasets": fake}):
        prompts = eval_datasets.load_refusal_prompts()

    assert len(prompts) == 150
    assert prompts[0] == "harm 0"


def test_load_bfcl_cases_maps_first_answer_call():
    from shared.eval import datasets as eval_datasets

    rows = [
        {
            "query": "q1",
            "answers": '[{"name": "f", "arguments": {"x": 1}}, {"name": "g", "arguments": {}}]',
            "tools": "[]",
        },
        {
            "query": "q2",
            "answers": [{"name": "h", "arguments": {"y": 2}}],
            "tools": "[]",
        },
    ]
    fake = _fake_datasets(rows)
    with patch.dict(sys.modules, {"datasets": fake}):
        cases = eval_datasets.load_bfcl_cases(limit=10)

    assert cases == [
        {"prompt": "q1", "expected": {"name": "f", "arguments": {"x": 1}}},
        {"prompt": "q2", "expected": {"name": "h", "arguments": {"y": 2}}},
    ]
    fake.load_dataset.assert_called_once()


def test_load_bfcl_cases_honors_limit():
    from shared.eval import datasets as eval_datasets

    rows = [
        {"query": f"q{i}", "answers": f'[{{"name": "f{i}", "arguments": {{}}}}]'}
        for i in range(50)
    ]
    fake = _fake_datasets(rows)
    with patch.dict(sys.modules, {"datasets": fake}):
        cases = eval_datasets.load_bfcl_cases(limit=5)

    assert len(cases) == 5
    assert cases[0] == {"prompt": "q0", "expected": {"name": "f0", "arguments": {}}}
