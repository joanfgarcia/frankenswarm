# lab/experiments — Fossilized Experiment Scripts

Standalone entrypoints from past research epochs (proto-syntax, math, conversational, resonance, grids 033-034). Moved out of `src/bitnet/` on 2026-07-03: zero inbound imports, zero references from REPRODUCE.md, DEMO_GUIDE.md, configs or the experiment queue.

They remain runnable from the repo root (imports are absolute):

```bash
PYTHONPATH=. .venv/bin/python lab/experiments/<script>.py
```

Rules:
- `src/bitnet/` holds importable modules and entrypoints pinned by REPRODUCE.md / DEMO_GUIDE.md (the paper's reproducibility surface).
- New one-off experiments start here (or in `scratch/`), not in `src/`.
- Nothing in `src/`, `tests/` or `scripts/` may import from this directory.
