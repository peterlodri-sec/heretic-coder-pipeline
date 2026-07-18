from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass

from pipeline.config import BASE_MODEL, STAGES, StageSpec


@dataclass(frozen=True, slots=True)
class StageResult:
    name: str
    input_model: str
    returncode: int

    @property
    def passed(self) -> bool:
        return self.returncode == 0


def run_pipeline(
    stages: tuple[StageSpec, ...] = STAGES,
    base_model: str = BASE_MODEL,
    python_exe: str | None = None,
    repo_root: str | None = None,
) -> list[StageResult]:
    if python_exe is None:
        python_exe = sys.executable
    if repo_root is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    results: list[StageResult] = []
    input_model = base_model
    for stage in stages:
        cmd = [
            python_exe,
            os.path.join(repo_root, stage.controller),
            "--model",
            input_model,
        ]
        proc = subprocess.run(cmd)
        results.append(StageResult(stage.name, input_model, proc.returncode))
        if proc.returncode != 0:
            break
        input_model = stage.output_repo
    return results


def main(argv: list[str] | None = None) -> int:
    results = run_pipeline()
    for r in results:
        if r.passed:
            print(f"[{r.name}] input={r.input_model} -> PASS")
        else:
            print(f"[{r.name}] input={r.input_model} -> FAIL (rc={r.returncode})")
    ok = len(results) == len(STAGES) and all(r.passed for r in results)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
