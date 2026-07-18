import json

from shared.dataprep.schema import (TrainingExample, tool_call_block,
                             tool_response_block, validate_example)


def test_tool_call_block_is_hermes_json():
    block = tool_call_block("bash", {"cmd": "ls"})
    assert block.startswith("<tool_call>") and block.endswith("</tool_call>")
    inner = block[len("<tool_call>"):-len("</tool_call>")].strip()
    assert json.loads(inner) == {"name": "bash", "arguments": {"cmd": "ls"}}


def test_tool_response_block_roundtrips():
    block = tool_response_block("ok")
    assert json.loads(block.split(">", 1)[1].rsplit("<", 1)[0].strip()) == {"output": "ok"}


def test_valid_example_passes_validation():
    ex = TrainingExample(
        source="magicoder",
        messages=[{"role": "user", "content": "hi"},
                  {"role": "assistant", "content": "hello"}],
    )
    validate_example(ex)  # no raise


def test_empty_messages_rejected():
    import pytest
    with pytest.raises(ValueError):
        validate_example(TrainingExample(source="x", messages=[]))


def test_bad_role_rejected():
    import pytest
    ex = TrainingExample(source="x", messages=[{"role": "wizard", "content": "?"}])
    with pytest.raises(ValueError):
        validate_example(ex)
