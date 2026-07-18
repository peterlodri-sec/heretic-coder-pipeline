import pytest
from dataprep.schema import PreferencePair, validate_pair


def _pair(chosen="A", rejected="B"):
    return PreferencePair(
        prompt=[{"role": "user", "content": "q"}],
        chosen=[{"role": "assistant", "content": chosen}],
        rejected=[{"role": "assistant", "content": rejected}],
        source="xlam",
    )


def test_to_record_has_orpo_fields():
    rec = _pair().to_record()
    assert set(rec) == {"prompt", "chosen", "rejected"}
    assert rec["chosen"] == [{"role": "assistant", "content": "A"}]
    assert rec["rejected"] == [{"role": "assistant", "content": "B"}]


def test_conversational_shapes():
    p = _pair()
    assert isinstance(p.prompt, list) and p.prompt[0]["role"] == "user"
    assert p.chosen[0]["role"] == "assistant"
    assert p.rejected[0]["role"] == "assistant"


def test_valid_pair_passes():
    validate_pair(_pair())


def test_empty_prompt_rejected():
    p = PreferencePair(prompt=[], chosen=[{"role": "assistant", "content": "A"}],
                       rejected=[{"role": "assistant", "content": "B"}], source="x")
    with pytest.raises(ValueError):
        validate_pair(p)


def test_empty_chosen_rejected():
    p = PreferencePair(prompt=[{"role": "user", "content": "q"}], chosen=[],
                       rejected=[{"role": "assistant", "content": "B"}], source="x")
    with pytest.raises(ValueError):
        validate_pair(p)


def test_identical_chosen_rejected_rejected():
    with pytest.raises(ValueError):
        validate_pair(_pair(chosen="same", rejected="same"))
