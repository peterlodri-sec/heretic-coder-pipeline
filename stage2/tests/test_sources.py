import json
from unittest.mock import patch

from dataprep.sources.magicoder import MagicoderSource
from dataprep.sources.xlam import XLAMSource
from dataprep.sources.toolace import ToolACESource
from dataprep.sources.crabcc import CrabccSource
from shared.dataprep.schema import render_for_family


# --- Magicoder: cols problem/solution -> user/assistant -------------------

def test_magicoder_maps_rows_to_examples():
    rows = [{"problem": "Write add()", "solution": "def add(a,b): return a+b"}]
    with patch("shared.dataprep.loaders.load_magicoder_rows", return_value=rows):
        exs = list(MagicoderSource().examples())
    assert len(exs) == 1
    ex = exs[0]
    assert ex.source == "magicoder"
    assert ex.messages[0]["role"] == "user" and "add()" in ex.messages[0]["content"]
    assert ex.messages[1]["role"] == "assistant" and "def add" in ex.messages[1]["content"]


# --- xLAM: JSON-string tools/answers -> system + Hermes tool_call ----------

def test_xlam_puts_tools_in_system_and_answers_as_tool_calls():
    tools = [{"name": "get_weather", "description": "weather",
              "parameters": {"city": {"type": "string"}}}]
    answers = [{"name": "get_weather", "arguments": {"city": "NYC"}}]
    rows = [{"query": "weather in NYC?",
             "tools": json.dumps(tools), "answers": json.dumps(answers), "id": "1"}]
    with patch("shared.dataprep.loaders.load_xlam_rows", return_value=rows):
        exs = list(XLAMSource().examples())
    assert len(exs) == 1
    ex = exs[0]
    assert ex.source == "xlam"
    roles = [m["role"] for m in ex.messages]
    assert roles == ["system", "user", "assistant"]
    # tools land in the system message
    assert "get_weather" in ex.messages[0]["content"]
    assert ex.messages[1]["content"] == "weather in NYC?"
    # gold answer recorded as neutral structured tool call
    assert ex.messages[2]["tool_calls"] == [
        {"name": "get_weather", "arguments": {"city": "NYC"}}]
    # qwen render collapses it to one Hermes <tool_call> block
    assistant = render_for_family(ex.messages, "qwen")[2]["content"]
    assert assistant.startswith("<tool_call>") and assistant.endswith("</tool_call>")
    inner = assistant[len("<tool_call>"):-len("</tool_call>")].strip()
    assert json.loads(inner) == {"name": "get_weather", "arguments": {"city": "NYC"}}


def test_xlam_renders_multiple_calls_as_separate_blocks():
    tools = [{"name": "a", "parameters": {}}, {"name": "b", "parameters": {}}]
    answers = [{"name": "a", "arguments": {"x": 1}}, {"name": "b", "arguments": {"y": 2}}]
    rows = [{"query": "do both", "tools": json.dumps(tools),
             "answers": json.dumps(answers), "id": "2"}]
    with patch("shared.dataprep.loaders.load_xlam_rows", return_value=rows):
        ex = next(iter(XLAMSource().examples()))
    assert len(ex.messages[2]["tool_calls"]) == 2
    assert render_for_family(ex.messages, "qwen")[2]["content"].count("<tool_call>") == 2


# --- ToolACE: from/value mapping + bracket -> Hermes -----------------------

def test_toolace_maps_from_value_and_prepends_system():
    rows = [{
        "system": "You can call tools.",
        "conversations": [
            {"from": "user", "value": "weather in NYC?"},
            {"from": "assistant", "value": '[get_weather(city="NYC", days=3)]'},
            {"from": "tool", "value": '{"temp": 20}'},
            {"from": "assistant", "value": "It is 20 degrees."},
        ],
    }]
    with patch("shared.dataprep.loaders.load_toolace_rows", return_value=rows):
        exs = list(ToolACESource().examples())
    assert len(exs) == 1
    ex = exs[0]
    assert ex.source == "toolace"
    roles = [m["role"] for m in ex.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    assert ex.messages[0]["content"] == "You can call tools."
    # bracket assistant call recorded as neutral structured tool call
    assert ex.messages[2]["tool_calls"] == [
        {"name": "get_weather", "arguments": {"city": "NYC", "days": 3}}]
    # qwen render normalizes it to Hermes
    call = render_for_family(ex.messages, "qwen")[2]["content"]
    assert call.startswith("<tool_call>")
    assert json.loads(call[len("<tool_call>"):-len("</tool_call>")].strip()) == {
        "name": "get_weather", "arguments": {"city": "NYC", "days": 3}}
    # tool result + plain assistant turn preserved verbatim (plain text, no name)
    assert ex.messages[3]["role"] == "tool" and ex.messages[3]["content"] == '{"temp": 20}'
    assert "name" not in ex.messages[3]
    assert ex.messages[4]["content"] == "It is 20 degrees."


def test_toolace_keeps_unparseable_bracket_text_as_is():
    rows = [{
        "system": "sys",
        "conversations": [
            {"from": "user", "value": "hi"},
            # not a parseable keyword-call list -> kept verbatim
            {"from": "assistant", "value": "[this is not a call]"},
        ],
    }]
    with patch("shared.dataprep.loaders.load_toolace_rows", return_value=rows):
        ex = next(iter(ToolACESource().examples()))
    assert ex.messages[2]["content"] == "[this is not a call]"


# --- crabcc: local traces -> Hermes tool_call/tool_response ----------------

def test_crabcc_reads_local_traces():
    trace = {"turns": [
        {"role": "user", "content": "run tests"},
        {"role": "assistant", "tool": "bash", "arguments": {"cmd": "pytest"}},
        {"role": "tool", "output": "ok"},
    ]}
    with patch("shared.dataprep.loaders.load_traces", return_value=[trace]):
        exs = list(CrabccSource(trace_dir="/x").examples())
    msgs = exs[0].messages
    # neutral structured tool call + named tool result (name threaded from call)
    assert msgs[1]["tool_calls"] == [{"name": "bash", "arguments": {"cmd": "pytest"}}]
    assert msgs[2]["role"] == "tool" and msgs[2]["name"] == "bash" and msgs[2]["content"] == "ok"
    # qwen render restores Hermes <tool_call>/<tool_response>
    contents = [m["content"] for m in render_for_family(msgs, "qwen")]
    assert any("<tool_call>" in c for c in contents)
    assert any("<tool_response>" in c for c in contents)


# --- SWEBench source: HARD-decontaminated against SWE-bench Verified -----------

def test_swebench_source_drops_verified_and_keeps_clean_resolved():
    from dataprep.sources.swebench import SWEBenchSource
    rows = [
        {"instance_id": "django__django-99999", "problem_statement": "leak",
         "patch": "p", "resolved": True},                       # Verified -> DROP
        {"instance_id": "swegym__x-1", "problem_statement": "fix the bug",
         "patch": "diff --git", "resolved": True},              # clean, resolved -> keep
        {"instance_id": "swegym__x-2", "problem_statement": "q",
         "patch": "p2", "resolved": False},                     # clean but unresolved -> skip
    ]
    verified = (frozenset({"django__django-99999"}), frozenset())
    with patch("shared.dataprep.loaders.load_swebench_rows", return_value=rows), \
         patch("shared.dataprep.decontaminate.load_verified_keys", return_value=verified):
        exs = list(SWEBenchSource().examples())
    assert len(exs) == 1
    assert exs[0].source == "swebench"
    assert exs[0].messages[0]["content"] == "fix the bug"
    assert exs[0].messages[1]["content"] == "diff --git"


# --- SWE-Gym source: (issue -> gold diff), decontaminated against Verified ------

def test_swegym_maps_and_decontaminates():
    from dataprep.sources.swegym import SWEGymSource
    rows = [
        {"instance_id": "django__django-99999", "repo": "django/django",
         "base_commit": "beef", "problem_statement": "leak", "patch": "p"},  # Verified -> DROP
        {"instance_id": "swegym__ok-1", "repo": "acme/widgets",
         "base_commit": "0001", "problem_statement": "Timeout not honored",
         "patch": "diff --git a/req.py b/req.py"},                            # keep
        {"instance_id": "swegym__nopatch", "repo": "a/b", "base_commit": "1",
         "problem_statement": "x", "patch": ""},                             # no patch -> skip
    ]
    verified = (frozenset({"django__django-99999"}), frozenset())
    with patch("shared.dataprep.loaders.load_swegym_rows", return_value=rows), \
         patch("shared.dataprep.decontaminate.load_verified_keys", return_value=verified):
        exs = list(SWEGymSource().examples())
    assert len(exs) == 1
    ex = exs[0]
    assert ex.source == "swegym"
    assert ex.messages[0]["role"] == "system"
    assert ex.messages[1]["role"] == "user" and "acme/widgets" in ex.messages[1]["content"]
    assert "Timeout not honored" in ex.messages[1]["content"]
    assert ex.messages[2]["role"] == "assistant" and ex.messages[2]["content"].startswith("diff --git")


# --- Nebius OpenHands trajectories: multi-turn, verified-passing, decontaminated -

def test_nebius_maps_trajectory_and_filters_resolved():
    from dataprep.sources.nebius import NebiusSource
    rows = [
        {  # resolved trajectory in a CLEAN repo -> kept + mapped
            "instance_id": "swegym__ok-1", "repo": "some/lib", "base_commit": "c1",
            "resolved": 1,
            "trajectory": [
                {"role": "system", "content": "You are a coding agent."},
                {"role": "user", "content": "Fix the timeout bug."},
                {"role": "assistant", "content": "I'll edit the file.",
                 "tool_calls": [{"id": "a", "type": "function",
                                 "function": {"name": "str_replace",
                                              "arguments": '{"path": "x.py", "old_str": "1", "new_str": "2"}'}}]},
                {"role": "tool", "name": "str_replace", "tool_call_id": "a",
                 "content": "edit applied"},
                {"role": "assistant", "content": "Done."},
            ],
        },
        {"instance_id": "u", "repo": "some/lib", "base_commit": "c2", "resolved": 0,
         "trajectory": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]},  # unresolved -> skip
        {"instance_id": "django__django-1", "repo": "django/django", "base_commit": "c",
         "resolved": 1, "trajectory": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]},  # blocklisted repo -> drop
    ]
    with patch("shared.dataprep.loaders.load_nebius_rows", return_value=rows), \
         patch("shared.dataprep.decontaminate.load_verified_keys",
               return_value=(frozenset(), frozenset())):
        exs = list(NebiusSource().examples())
    assert len(exs) == 1
    ex = exs[0]
    assert ex.source == "nebius-openhands"
    # roles preserved, tool_call converted to neutral {name, arguments-dict}
    asst = [m for m in ex.messages if m["role"] == "assistant" and m.get("tool_calls")][0]
    tc = asst["tool_calls"][0]
    assert tc["name"] == "str_replace"
    assert tc["arguments"] == {"path": "x.py", "old_str": "1", "new_str": "2"}  # JSON-parsed
    tool_msg = [m for m in ex.messages if m["role"] == "tool"][0]
    assert tool_msg["name"] == "str_replace" and tool_msg["content"] == "edit applied"


def test_nebius_keeps_unparseable_tool_args_verbatim():
    from dataprep.sources.nebius import NebiusSource
    rows = [{"instance_id": "ok", "repo": "a/b", "base_commit": "c", "resolved": 1,
             "trajectory": [
                 {"role": "user", "content": "go"},
                 {"role": "assistant", "content": "",
                  "tool_calls": [{"function": {"name": "run", "arguments": "not json"}}]},
             ]}]
    with patch("shared.dataprep.loaders.load_nebius_rows", return_value=rows), \
         patch("shared.dataprep.decontaminate.load_verified_keys",
               return_value=(frozenset(), frozenset())):
        ex = list(NebiusSource().examples())[0]
    tc = [m for m in ex.messages if m.get("tool_calls")][0]["tool_calls"][0]
    assert tc["arguments"] == {"_raw_arguments": "not json"}
