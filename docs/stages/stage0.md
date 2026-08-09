# Stage 0 — Scaffolding

## Scope

Repo skeleton, dependency management, config system, and lint/test tooling — no simulation logic yet. Goal: a clean, minimal foundation that later stages build on without needing to be restructured.

## Decisions

- **Env/dependency manager:** `uv` — fast, single `pyproject.toml` + `uv.lock`, `uv run`/`uv sync` workflow.
- **Config system:** Python dataclasses with defaults, optionally overridden by a YAML file. Chosen over Hydra for simplicity while the physics/env APIs are still in flux.
- **Lint/format:** Ruff (lint) + Black (format), wired via pre-commit. No mypy for now — can be added later once interfaces stabilize.
- **Package layout:** `soccersim/` package with empty (or near-empty) subpackages `physics/`, `viz/`, `agents/`, `tactics/`, `env/`, `training/`, `eval/`, matching the structure in [ROADMAP.md](../../ROADMAP.md).

## Done criteria

- `uv sync` installs a working environment from `pyproject.toml`/`uv.lock`.
- `soccersim/` package skeleton exists with all Stage 1+ subpackages as importable (empty) modules.
- `configs/` holds a `default.py` (or similar) dataclass config plus an example YAML override.
- `pytest` runs (even with just a placeholder/sanity test) via `uv run pytest`.
- Pre-commit hooks (Ruff + Black) are installed and pass on the initial commit.
- Everything committed and pushed to `origin/main`.
