# Stage 2 — Visualization & replay

For a concept-level and implementation-level walkthrough (with diagrams), see [stage2-concepts.md](stage2-concepts.md).

## Scope

A `pygame` renderer that draws a `MatchState` (pitch, goals, players, ball, score/time HUD), plus a replay recorder/loader that persists `(state, action, event)` sequences to disk. Two ways to watch a match: live, while `step()` is actually running, or played back later from a recorded file — both go through the same drawing code. No new physics, no `Agent` interface yet (Stage 3); a keyboard-controlled player in the live viewer exists purely as a debug tool for eyeballing the Stage 1 kernel, not as a preview of real gameplay.

## Decisions

- **Replay serialization format: JSON Lines.** Chosen over pickle (breaks across dataclass shape changes, not human-readable) and a fixed-schema numpy array (would require locking in an array layout before it's clear what training pipelines will actually need). JSONL is diffable, resilient to schema evolution, and line-oriented (streamable later if replays get large) — worth the verbosity/parse-speed cost at this project's current scale. See `viz/replay.py`'s module docstring for the full reasoning.
- **Renderer is a pure function of `(surface, state, config)`, shared by both live and replay playback.** `viz/render.draw_match_state()` has no knowledge of *where* the `MatchState` came from — a running `step()` loop or a loaded replay file. This avoids building two renderers and keeps `physics/` → `viz/` a one-way dependency (per ROADMAP.md's dependency-direction rule); headless training in later stages never needs to import pygame at all unless it explicitly wants to render.
- **Replay files are self-describing.** The first line of every `.jsonl` replay is a `"meta"` record containing the full `SimConfig` the match was run with, so `scripts/watch_replay.py` doesn't need the original config passed in separately — pitch dimensions, `dt`, etc. all travel with the file.
- **Keyboard-controlled debug player, not a real agent.** `scripts/run_match.py` drives one player from arrow keys + spacebar purely so a human can visually sanity-check the physics kernel — acceleration, speed clamping, friction, kicking, dribbling, bounds/goals — in real time. This is *not* the Stage 3 `Agent` interface and shouldn't be mistaken for one; every other player just sits still.
- **Addendum (added before Stage 3): kick charging, facing-based aim, and player switching.** Space now *charges* a kick (a HUD bar shows the current power fraction) rather than firing an always-full-power shot the instant it's pressed, and releasing it kicks in whatever direction the controlled player is currently `facing` (see `stage1-concepts.md` §9) rather than always toward the opponent's goal — so the debug tool can exercise passing in any direction, not just shooting. Tab cycles which of team 0's players is controlled. All of this (`_KickCharger`, the charge bar, the switch logic) lives entirely in `run_match.py` — it's an input/UI concern layered on top of `PlayerAction`, not a physics-kernel change; see `physics/state.py`'s docstring on `PlayerAction.kick`.
- **`load_replay()` reads the whole file into memory.** Simpler than streaming, and matches are short/low-roster enough right now that this isn't a real cost. Revisit only if later stages produce large enough replay files for it to matter.

## Done criteria

- `viz/render.draw_match_state()` runs correctly against a headless (no real display) `pygame.Surface`, so it's testable without a windowing system.
- `viz/replay.ReplayWriter`/`load_replay()` round-trip a recorded match exactly: same states, actions (including `kick=None`), and events after writing and reading back.
- `scripts/run_match.py` opens a live window, steps the Stage 1 kernel each frame, and (optionally, via `--record`) writes a replay to disk.
- `scripts/watch_replay.py` loads and plays back a recorded replay using the same renderer.
- `uv run pytest`, `uv run ruff check .`, and `uv run black --check .` all pass.
