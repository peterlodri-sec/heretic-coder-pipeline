# Deep Research Report: How to Dramatically Increase SWE-bench & Autonomous SWE Agent Scores

## Executive Summary

Achieving SOTA performance on benchmarks like **SWE-bench Verified**, **SWE-Gym**, and real-world repository resolution requires moving beyond standard multi-turn chat models. While baseline base models (e.g. `Qwen2.5-Coder-32B`, `gpt-oss-120b`, `Claude 3.5 Sonnet`) possess strong zero-shot code generation, their standalone resolve rates on complex software engineering tasks cap out around **30%–45%**. 

To reach **60%–85%+ SWE-bench resolve rates**, an autonomous agent system must integrate five synergistic pillars:

1. **Agentic Test-Time Compute & Search (MCTS / Best-of-N / Trajectory Scaling)**
2. **Repository-Level Knowledge Graph Indexing (AST / Caller-Callee Blast Radius)**
3. **Execution-Feedback RLVR & Post-Training (GSPO, RFT, & Dense Rewards)**
4. **Environment-Grounded Multi-Turn Self-Correction Loops**
5. **Hardened Tooling & Deterministic Evaluation Sandboxes**

---

## 1. Agentic Test-Time Compute & Search

### A. Best-of-N & Execution-Gated Selection
Generating $N$ candidate patch trajectories ($N \in [8, 64]$) and running each against the repository test suite drastically boosts the probability of discovering a valid solution.
- **Pass@N Scaling:** Empirical results show that Pass@8 improves resolve rates by **+14%–18%** over Pass@1, provided execution feedback can evaluate the candidates.
- **Verifier Models:** Train a outcome/process verifier (ORM/PRM) to score intermediate trajectory steps (file localization accuracy, test edit precision) before full execution.

### B. Agentic Monte Carlo Tree Search (MCTS)
Rather than linear multi-turn turns:
- **Node definition:** A state consists of the current git diff, environment state, and conversation history.
- **Actions:** Localize files -> Search symbols -> Propose patch -> Run test suite -> Inspect error log.
- **Pruning:** If a proposed edit introduces syntax errors or breaks pre-existing unit tests, prune the branch immediately.

---

## 2. Repository-Level Knowledge Graph Indexing

A major cause of SWE failure is **context dilution**—stuffing entire 2,000-line files into the context window, causing key details to be lost in attention mechanisms.

### A. Graph-Aware Context Slicing
- Use Tree-sitter to parse the entire codebase into a structural dependency graph (e.g., `code-review-graph` or `codebase-memory-mcp`).
- Trace **inbound/outbound caller paths** to compute the exact **blast radius** of an issue before making any code edits.
- Feed the agent only the target function signature, docstring, body, and immediate callers rather than full files.

### B. Two-Phase Localization
1. **Coarse Slicing (File/Module Level):** Use BM25 + dense embedding retrieval over commit messages, file paths, and symbol definitions to select top-5 relevant files.
2. **Fine Slicing (Symbol/AST Level):** Use graph traversal to pull exact class definitions, interface contracts, and related test fixtures.

---

## 3. Post-Training: RLVR, GSPO, & High-Quality Synthetic Data

### A. Sequence-Level RLVR (GSPO for MoE Bases)
For Mixture-of-Experts (MoE) model architectures such as `gpt-oss-120b`:
- **Token-Level GRPO Collapse:** Standard token-level importance sampling causes MoE expert router weights to collapse during RL.
- **Group Sequence Policy Optimization (GSPO):** Using sequence-level importance sampling ($\beta=0$, sequence-level clipping) stabilizes expert routing while optimizing directly for test execution pass rates.

### B. Rejection-Sampling Fine-Tuning (RFT)
1. Sample $N=16$ candidate trajectories per training issue on SWE-Gym datasets.
2. Execute candidates in isolated sandboxes.
3. Retain only trajectories that pass 100% of existing tests AND holdout tests.
4. Perform SFT on the passing trajectories with completion-only loss masking and BFD packing.

### C. Synthetic Holdout Unit Test Generation
To combat **reward hacking** (where a model modifies existing unit tests to pass or writes trivial assertions):
- Prompt a separate test-gen model to synthesize hidden holdout test cases from the issue description *before* the patch candidate is evaluated.
- Penalize any candidate patch that modifies existing test files without explicit requirement.

---

## 4. Multi-Stage Environment Self-Repair

### A. The 4-Step Fix Cycle
1. **Reproduce:** Automatically convert the issue description into a minimal reproducing test case (`reproduce_issue.py`).
2. **Localize & Patch:** Generate candidate diffs using AST context.
3. **Verify & Diff Audit:** Run regression tests. If a failure occurs, append the raw stdout/stderr trace and feed it back to the agent for self-repair.
4. **Clean & Format:** Run `git diff` inspection to ensure no scratch files, debug `print()` statements, or formatting breakages are left behind.

---

## 5. Quantitative Targets & Capability Verdict Gate

To track progress toward SOTA performance, enforce a strict automated capability gate (`shared/verdict.py`):

| Capability Metric | Target Floor | Impact on SWE Score |
|---|---|---|
| **Refusal Rate** | `< 0.05` | Prevents false refusals on security/vulnerability tasks |
| **BFCL Tool Accuracy** | `> 0.90` | Ensures multi-step tool calls don't fail due to JSON syntax |
| **HumanEval Delta** | `< 0.01` | Preserves baseline Python syntax & logic competence |
| **SWE-bench Verified** | `> 0.60` | Target benchmark resolve rate |

---

## Action Plan & Immediate Roadmap

1. **Integrate Graph Context Indexer in Stage 4 RFT Rollouts:** Connect `code-review-graph` parsing into `shared/exec_sandbox.py` to index repos dynamically during rollouts.
2. **Implement Best-of-N Candidate Reranking:** Add an ORM verifier pass before execution to select top-4 candidates for full sandbox execution.
3. **Scale GSPO RLVR Training:** Train `PeetPedro/gpt-oss-120b-heretic-sft` with GSPO on 4×H200 with 1,000+ SWE-Gym task environments.
