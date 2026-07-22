"""Best-effort live-status webhook push.

Every ``Status.write()`` (the single status chokepoint) optionally POSTs the new
status to a monitor webhook, so the dashboard reflects the box in real time
instead of SSH-polling it. FULLY best-effort and env-gated:

  * unset ``PIPELINE_WEBHOOK_URL`` -> no-op (the default; tests never touch the net);
  * any network / HTTP / serialization error is swallowed — a down, blocked, or
    slow monitor must NEVER affect a training run.

stdlib only (``urllib``), GPU-free. The payload is the status dict plus a ``stage_key``
(``PIPELINE_STAGE_KEY``, e.g. "sft") so the dashboard knows which card to update
— a separate key from the status dict's own ``stage`` field (the run phase, e.g.
"training"/"done"), so neither clobbers the other. The token
(``PIPELINE_WEBHOOK_TOKEN``) authenticates against the public tunnel.
"""
import json
import os
import urllib.request

URL_ENV = "PIPELINE_WEBHOOK_URL"
TOKEN_ENV = "PIPELINE_WEBHOOK_TOKEN"
STAGE_ENV = "PIPELINE_STAGE_KEY"


def notify(payload: dict, timeout: float = 3.0) -> bool:
    """POST ``{stage, **payload}`` to the monitor webhook. Returns True if sent,
    False if disabled or on any failure (never raises)."""
    url = os.environ.get(URL_ENV)
    if not url:
        return False
    try:
        body = json.dumps({"stage_key": os.environ.get(STAGE_ENV), **payload}).encode()
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={"Content-Type": "application/json",
                     "X-Webhook-Token": os.environ.get(TOKEN_ENV, ""),
                     # bypass the ngrok-free browser interstitial for the POST
                     "ngrok-skip-browser-warning": "true"},
        )
        urllib.request.urlopen(req, timeout=timeout).close()
        return True
    except Exception:  # noqa: BLE001 — a monitor outage must not break a run
        return False
