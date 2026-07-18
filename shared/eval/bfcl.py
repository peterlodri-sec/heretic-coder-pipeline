import json


def generate_tool_call(model, prompt) -> str:
    from transformers import pipeline
    gen = pipeline("text-generation", model=model)
    text = gen(prompt, max_new_tokens=256)[0]["generated_text"]
    start, end = text.find("<tool_call>"), text.find("</tool_call>")
    return text[start + len("<tool_call>"):end].strip() if start >= 0 else text


def _matches(pred_json: str, expected: dict) -> bool:
    try:
        return json.loads(pred_json) == expected
    except (json.JSONDecodeError, TypeError):
        return False


def accuracy(model, cases) -> float:
    if not cases:
        return 0.0
    hits = sum(1 for c in cases if _matches(generate_tool_call(model, c["prompt"]), c["expected"]))
    return hits / len(cases)
