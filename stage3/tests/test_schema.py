import pytest
from dataprep.schema import PreferencePair, validate_pair


def _pair(chosen="A", rejected="B"):
    return PreferencePair(prompt=[{"role": "user", "content": "q"}],
                          chosen=chosen, rejected=rejected, source="bfcl")


def test_to_record_has_orpo_fields():
    rec = _pair().to_record()
    assert set(rec) == {"prompt", "chosen", "rejected", "source", "weight"}
    assert rec["chosen"] == "A" and rec["rejected"] == "B"


def test_valid_pair_passes():
    validate_pair(_pair())


def test_empty_prompt_rejected():
    p = PreferencePair(prompt=[], chosen="A", rejected="B", source="x")
    with pytest.raises(ValueError):
        validate_pair(p)


def test_identical_chosen_rejected_rejected():
    with pytest.raises(ValueError):
        validate_pair(_pair(chosen="same", rejected="same"))
