# Model cards

Documentation for the models produced by the
[heretic-coder-pipeline](https://github.com/peterlodri-sec/heretic-coder-pipeline)
(`heretic → SFT → RFT → RLVR`). **Weights are private/gated** — these cards
document method + evaluation only.

| Card | Model | Status |
|---|---|---|
| [qwen2.5-coder-32b-instruct-heretic](qwen2.5-coder-32b-instruct-heretic.md) | abliterated 32B base | run |
| [qwen2.5-coder-32b-instruct-heretic-sft](qwen2.5-coder-32b-instruct-heretic-sft.md) | + SFT (validation baseline) | run |
| [gpt-oss-120b-heretic](gpt-oss-120b-heretic.md) | frontier chain | configured, pending |

## Related data

- [`PeetPedro/ultrawhale-dogfood`](https://huggingface.co/datasets/PeetPedro/ultrawhale-dogfood)
  — self-generated, self-hosted silver-label Q&A corpus from a continuous dogfeed
  loop (gated).

Pipeline source, design, and reproduction:
<https://github.com/peterlodri-sec/heretic-coder-pipeline>
