# Working agreements for this repo

See [ROADMAP.md](ROADMAP.md) for the staged plan. This project is a hobby learning project (see repo owner's memory profile) — prioritize clarity and incremental, verifiable steps over speed.

## Workflow rules

- When starting a new roadmap stage, create a context file for it under `docs/stages/` (e.g. `docs/stages/stage0.md`) describing scope, decisions made, and what "done" looks like for that stage.
- Whenever a project/architecture decision needs a human call (tooling, config format, algorithm choice, API shape, etc.), ask before proceeding — don't assume.
- Commit and push to `origin/main` after each major change (a stage's scaffolding, a working feature, a passing test suite) — not after every tiny edit.
- Keep [README.md](README.md) up to date whenever it's needed — new setup/run steps, status, or stack changes — as part of the same change, not as an afterthought.
- Document code thoroughly: docstrings/comments should explain the underlying concept and *why* a choice was made (e.g. which physics/ML technique, what trade-off), not just restate what the code does — the repo owner is using this project to learn these concepts, not just to ship a working simulation.

## Confirmed stack decisions

- **Env/dependency manager:** `uv` (pyproject.toml + uv.lock)
- **Config system:** plain Python dataclasses with defaults, optionally overridden by YAML
- **Lint/format:** Ruff + Black, wired via pre-commit
- **Package layout:** `soccersim/` with subpackages `physics/`, `viz/`, `agents/`, `tactics/`, `env/`, `training/`, `eval/` (see ROADMAP.md for the full repo structure and dependency direction rules)
