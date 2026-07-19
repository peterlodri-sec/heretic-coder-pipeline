from enums import Stage


def test_loop_stages_present():
    # RFT loop cycles generate -> verify -> train per round.
    assert {Stage.GENERATING, Stage.VERIFYING, Stage.TRAINING} <= set(Stage)
    assert Stage.SETUP == "setup" and Stage.DONE == "done"
