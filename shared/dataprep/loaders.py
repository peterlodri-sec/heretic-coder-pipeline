# Raw dataset row loaders, isolated so tests patch them and neither stage loads
# real data. Each stage's adapters map these rows to their own schema.


def load_magicoder_rows():
    from datasets import load_dataset
    return load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")


def load_bfcl_rows():
    from datasets import load_dataset
    return load_dataset("gorilla-llm/Berkeley-Function-Calling-Leaderboard", split="train")


def load_toolace_rows():
    from datasets import load_dataset
    return load_dataset("Team-ACE/ToolACE", split="train")


def load_swebench_rows():
    from datasets import load_dataset
    return load_dataset("princeton-nlp/SWE-bench_Verified", split="test")


def load_traces(trace_dir):
    import glob
    import json
    traces = []
    for path in glob.glob(f"{trace_dir}/*.json"):
        with open(path) as f:
            traces.append(json.load(f))
    return traces
