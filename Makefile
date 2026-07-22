# entropy-om/heretic-coder-pipeline — dev tasks.
#
# Each stage is a self-contained package deployed independently to a GPU box, and
# several share module NAMES (dataprep / enums / status_io) with different
# contents. Running one pytest process over the whole tree collides on those
# names, so `make test` runs each group in its OWN isolated process — mirroring CI.

GROUPS := shared stage1 stage2 stage3 stage4 stage5 frontier pipeline
# Only the deps the modules-under-test import at load time; the heavy ML stack is
# mocked in the tests and intentionally not installed.
PYTEST := uv run --with pytest --with vastai --with pexpect --with pyyaml --python 3.11 python -m pytest
RUFF := uv run --with ruff --python 3.11 ruff

.PHONY: help test lint fix check
.DEFAULT_GOAL := help

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-8s\033[0m %s\n", $$1, $$2}'

test:  ## run the full test suite (each group isolated)
	@set -e; for g in $(GROUPS); do echo "=== $$g ==="; $(PYTEST) $$g -q; done

lint:  ## ruff lint (E/F/W + bugbear)
	$(RUFF) check .

fix:  ## ruff auto-fix
	$(RUFF) check . --fix

check: lint test  ## the CI gate: lint + full test suite
