from unittest.mock import patch

from dataprep.sources.magicoder import MagicoderSource


def test_magicoder_maps_rows_to_examples():
    rows = [{"problem": "Write add()", "solution": "def add(a,b): return a+b"}]
    with patch("dataprep.sources.magicoder.load_rows", return_value=rows):
        exs = list(MagicoderSource().examples())
    assert len(exs) == 1
    ex = exs[0]
    assert ex.source == "magicoder"
    assert ex.messages[0]["role"] == "user" and "add()" in ex.messages[0]["content"]
    assert ex.messages[1]["role"] == "assistant" and "def add" in ex.messages[1]["content"]
    assert ex.is_negative is False
