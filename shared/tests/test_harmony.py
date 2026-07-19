from shared.harmony import extract_final


def test_extract_final_from_channelled_output():
    t = "<|channel|>analysis<|message|>reasoning...<|channel|>final<|message|>print(1)<|return|>"
    assert extract_final(t) == "print(1)"


def test_extract_final_plain_text_passthrough():
    assert extract_final("just some code") == "just some code"


def test_extract_final_strips_code_fence():
    t = "<|channel|>final<|message|>```python\nx = 1\n```<|end|>"
    assert extract_final(t) == "x = 1"


def test_extract_final_uses_last_final_marker():
    t = ("<|channel|>final<|message|>first<|end|>"
         "<|channel|>final<|message|>second<|return|>")
    assert extract_final(t) == "second"


def test_extract_final_no_stop_token():
    assert extract_final("<|channel|>final<|message|>tail with no stop") == "tail with no stop"
