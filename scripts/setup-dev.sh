#!/usr/bin/env bash
# Development setup for QGRE Engine.
#
# Idempotent — safe to re-run any time .pre-commit-config.yaml or pyproject.toml
# changes. Run this once after a fresh clone (or any time the toolchain moves):
#
#     bash scripts/setup-dev.sh
#
# What it does:
#   1. uv sync --extra dev           — populate .venv with main + dev deps
#                                      (dev deps include pre-commit, pytest, ruff)
#   2. uv tool install pyright       — pyright on PATH for the pre-commit hook
#   3. clear local core.hooksPath    — pre-commit refuses to install otherwise
#   4. pre-commit install            — wire .git/hooks/pre-commit
#   5. pre-commit install --hook-type pre-push — wire .git/hooks/pre-push
#
# After this, every `git commit` and `git push` automatically runs ruff,
# ruff-format, pyright, bandit, and the standard hygiene hooks.

set -euo pipefail

# Run from repo root regardless of where the script was invoked from.
cd "$(dirname "$0")/.."

if ! command -v uv > /dev/null 2>&1; then
    echo "error: uv is not installed. Install it from https://docs.astral.sh/uv/" >&2
    exit 1
fi

echo "==> 1/5  uv sync --extra dev"
# --extra dev pulls in pre-commit, pytest, ruff, pyright, bandit (all the
# tooling the hooks expect). Without it, the bare `uv sync` only installs
# the main dependencies and the hook install in step 4 fails.
uv sync --extra dev

echo "==> 2/5  uv tool install pyright"
# `uv tool install` is idempotent: re-running just no-ops if already current.
uv tool install pyright

# pre-commit refuses to install when core.hooksPath is set on the local repo,
# even if it points at the default .git/hooks. Some clones inherit this from
# templates or earlier tooling. Clear it so step 4 can proceed.
if git config --local --get core.hooksPath > /dev/null 2>&1; then
    echo "==> 3/5  clearing local core.hooksPath"
    git config --unset-all core.hooksPath
else
    echo "==> 3/5  core.hooksPath already clean"
fi

echo "==> 4/5  pre-commit install (commit stage)"
uv run pre-commit install

echo "==> 5/5  pre-commit install --hook-type pre-push"
uv run pre-commit install --hook-type pre-push

cat <<'EOF'

Dev setup complete.

Verify the toolchain is wired correctly:
    uv run pre-commit run --all-files

If that passes, every future `git commit` and `git push` will run ruff,
ruff-format, pyright, bandit, and the standard hygiene hooks automatically.
EOF
