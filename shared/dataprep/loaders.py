# Raw dataset row loaders, isolated so tests patch them and neither stage loads
# real data. Each stage's adapters map these rows to their own schema. The heavy
# `from datasets import load_dataset` stays INSIDE each function so importing this
# module never requires `datasets` to be installed.


def load_magicoder_rows():
    from datasets import load_dataset
    # cols: problem (str), solution (str)
    return load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")


def load_xlam_rows():
    from datasets import load_dataset
    # cols: query (str), tools (JSON str), answers (JSON str), id
    return load_dataset("NobodyExistsOnTheInternet/xlam-function-calling-60k", split="train")


def load_toolace_rows():
    from datasets import load_dataset
    # cols: system (str), conversations (list of {from, value})
    return load_dataset("Team-ACE/ToolACE", split="train")


def load_swebench_rows():
    from datasets import load_dataset
    # eval only; split="test", no `resolved` column
    return load_dataset("princeton-nlp/SWE-bench_Verified", split="test")


def load_nebius_rows():
    from datasets import load_dataset
    # Multi-turn OpenHands agent trajectories (Qwen3-Coder-480B, OpenHands v0.54.0)
    # resolving real GitHub issues — the dominant SWE-bench SFT lever (+13-26pp,
    # per the deep-research report). cols: trajectory_id, instance_id, repo,
    # trajectory (list of chat msgs w/ tool_calls), tools, model_patch,
    # exit_status, resolved (int64: 1 == the trajectory's patch passed the tests).
    return load_dataset("nebius/SWE-rebench-openhands-trajectories", split="train")


def load_swegym_rows():
    from datasets import load_dataset
    # SWE-Gym: real GitHub task instances built as a TRAINING substrate — each has
    # a gold `patch` + executable tests (FAIL_TO_PASS / PASS_TO_PASS). Disjoint
    # from SWE-bench Verified by construction, but the source still runs rows
    # through the decontamination guard as a hard backstop. cols: instance_id,
    # repo, base_commit, problem_statement, patch, test_patch, hints_text.
    return load_dataset("SWE-Gym/SWE-Gym", split="train")


def load_verifiable_coding_rows():
    # Coding problems that ship WITH executable tests — the substrate for stage4
    # RFT filtering + stage5 RLVR rewards (reward = fraction of tests passing).
    # cols (target schema): prompt (str), tests (str), entry_point (str, optional).
    # NOTE: exact source(s) finalize from research — MBPP+/HumanEval+/
    # LiveCodeBench-style verifiable subsets, or SWE-Gym repo tasks with
    # FAIL_TO_PASS. Heavy `load_dataset` stays function-local.
    from datasets import load_dataset
    return load_dataset("evalplus/mbppplus", split="test")


def load_traces(trace_dir):
    import glob
    import json
    traces = []
    for path in glob.glob(f"{trace_dir}/*.json"):
        with open(path) as f:
            traces.append(json.load(f))
    return traces
