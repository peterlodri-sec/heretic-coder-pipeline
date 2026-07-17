MMLU_TASK = "mmlu"
MMLU_LIMIT_PER_SUBJECT = 5
GSM8K_TASK = "gsm8k"
GSM8K_LIMIT = 300


def compute_deltas(base_results: dict, candidate_results: dict) -> dict:
    base_mmlu = base_results["mmlu"]["acc,none"]
    candidate_mmlu = candidate_results["mmlu"]["acc,none"]
    base_gsm8k = base_results["gsm8k"]["exact_match,strict-match"]
    candidate_gsm8k = candidate_results["gsm8k"]["exact_match,strict-match"]
    return {
        "mmlu_delta": base_mmlu - candidate_mmlu,
        "gsm8k_delta": base_gsm8k - candidate_gsm8k,
    }


def run_benchmarks(model_path_or_id: str) -> dict:
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    hflm = HFLM(pretrained=model_path_or_id, batch_size="auto")

    mmlu_out = lm_eval.simple_evaluate(model=hflm, tasks=[MMLU_TASK], limit=MMLU_LIMIT_PER_SUBJECT)
    gsm8k_out = lm_eval.simple_evaluate(model=hflm, tasks=[GSM8K_TASK], limit=GSM8K_LIMIT)

    return {
        "mmlu": mmlu_out["results"][MMLU_TASK],
        "gsm8k": gsm8k_out["results"][GSM8K_TASK],
    }
