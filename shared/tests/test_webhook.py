import json
from unittest.mock import MagicMock, patch

from shared import webhook


def test_notify_noop_when_url_unset(monkeypatch):
    monkeypatch.delenv("PIPELINE_WEBHOOK_URL", raising=False)
    with patch("urllib.request.urlopen") as u:
        assert webhook.notify({"status": "training"}) is False  # disabled
        u.assert_not_called()  # never touches the network


def test_notify_posts_stage_status_and_token(monkeypatch):
    monkeypatch.setenv("PIPELINE_WEBHOOK_URL", "http://monitor.local/webhook")
    monkeypatch.setenv("PIPELINE_WEBHOOK_TOKEN", "s3cr3t")
    monkeypatch.setenv("PIPELINE_STAGE_KEY", "sft")
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["token"] = req.get_header("X-webhook-token")
        seen["body"] = json.loads(req.data)
        return MagicMock(close=lambda: None)

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        assert webhook.notify({"status": "training", "train_loss": 0.42}) is True
    assert seen["url"] == "http://monitor.local/webhook"
    assert seen["token"] == "s3cr3t"
    assert seen["body"]["stage_key"] == "sft"           # dashboard card key, distinct from status stage
    assert seen["body"]["status"] == "training" and seen["body"]["train_loss"] == 0.42


def test_notify_swallows_errors(monkeypatch):
    # A down / blocked / slow monitor must never break a run.
    monkeypatch.setenv("PIPELINE_WEBHOOK_URL", "http://monitor.local/webhook")
    with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
        assert webhook.notify({"status": "x"}) is False  # returns, does not raise
