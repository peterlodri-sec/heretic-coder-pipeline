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
    import gc

    import lm_eval
    from lm_eval.models.huggingface import HFLM

    # parallelize=True shards the model across ALL visible GPUs via accelerate
    # (device_map="auto"). Without it HFLM loads the whole model onto ONE GPU,
    # so a 120B (~240GB bf16) OOMs a single 140GB H200. parallelize is the exact
    # sharding flag on lm_eval 0.4.12's HFLM.__init__ ("only used if
    # parallelize=True" per its own docstring).
    hflm = HFLM(pretrained=model_path_or_id, batch_size="auto", parallelize=True)
    try:
        mmlu_out = lm_eval.simple_evaluate(model=hflm, tasks=[MMLU_TASK], limit=MMLU_LIMIT_PER_SUBJECT)
        gsm8k_out = lm_eval.simple_evaluate(model=hflm, tasks=[GSM8K_TASK], limit=GSM8K_LIMIT)

        return {
            "mmlu": mmlu_out["results"][MMLU_TASK],
            "gsm8k": gsm8k_out["results"][GSM8K_TASK],
        }
    finally:
        # Free this 120B before the NEXT run_benchmarks (base -> candidate)
        # loads: two ~240GB models cannot both be resident on 2xH200 (282GB).
        del hflm
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
