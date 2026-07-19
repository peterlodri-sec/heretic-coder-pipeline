import pytest
import reward


def test_reward_is_stubbed_interface():
    # Verifiable exec-test reward finalized from research; calling it raises.
    with pytest.raises(NotImplementedError):
        reward.code_execution_reward(["p"], ["c"], ["t"])
