# Frankenswarm Conventions

## 🚨 RULE 1: Experimental-First Philosophy

Frankenswarm is a **research laboratory**, not a production system.
Every component must be:
- **Replaceable**: No tight coupling. Swapping BitNet for another substrate, or the Python router for Prolog, must not require rewriting other modules.
- **Measurable**: Every experiment must define a fitness metric before implementation.
- **Documented at the decision point**: Use inline `# TODO: Upgrade to X when Y` comments. Never leave architectural decisions implicit.

---

## 🚨 RULE 2: No Premature Neural Complexity

Do not use a neural component (BitNet node, transformer layer) for a task that can be solved deterministically.

| Task type | Correct tool |
|---|---|
| Routing / classification | `src/router/prolog_router.py` (Python rule-based) |
| Aggregation / combination | Plain Python — no neural overhead |
| Verification / validation | Python assertions or a real interpreter/compiler |
| **Generation** (code, text) | BitNet Transformer node |

> **TODO (future):** Replace `src/router/prolog_router.py` with SWI-Prolog via `pyswip`
> if routing rules exceed ~50 clauses or require recursive inference.
> Interface contract: `route(task: str) -> NodeTarget` stays identical.

---

## 🚨 RULE 3: Ternary Weights Are Sacred

All neural weight matrices in this project use the BitNet 1.58b ternary constraint: `{-1, 0, +1}`.

- **NEVER** initialize weights as float32/float16 unless in a clearly marked `# BASELINE` comparison branch.
- **NEVER** use standard AdamW/SGD on ternary weights. Use NEAT mutation or straight-through estimators explicitly.
- All weight matrices must pass `assert_ternary(matrix)` before being registered as an Expert.

---

## 🚨 RULE 4: Shared Vocabulary

All nodes in the swarm share **one tokenizer** (`src/tokenizer/sovtok.py`).
- Vocabulary size: **8,192 tokens** (2¹³)
- Algorithm: **Unigram LM**
- Domain: Python code + technical English + Red-Pill ecosystem terminology
- **NEVER** create per-node vocabularies — cross-node vector communication requires a shared embedding space.

---

## 🚨 RULE 5: Every Expert Has a Fitness Score

No Expert BitNet node may be promoted to the MoE router pool without a recorded fitness score on a defined benchmark task.

```python
# Required metadata on every registered Expert
expert = ExpertNode(
    id="code_expert_v1",
    fitness=0.87,           # 0.0 - 1.0
    benchmark="python_next_token_prediction",
    params=7_000_000,
    ternary=True,
)
```

---

## 🚨 RULE 6: Documentation Standards

Every document in the repository must follow strict formatting and naming rules:

- **Repo Root & Main Docs**: Use `UPPER_SNAKE_CASE.md` (e.g., [ARCHITECTURE.md](file:///home/joan/Documents/IA/frankenswarm/docs/ARCHITECTURE.md), [CONVENTIONS.md](file:///home/joan/Documents/IA/frankenswarm/CONVENTIONS.md), [EXPERTS_ROSTER.md](file:///home/joan/Documents/IA/frankenswarm/docs/EXPERTS_ROSTER.md)).
- **External Audits (`docs/extern/`)**: Named `STAGE_AUDITOR.md` where `AUDITOR` is one of: `CLAUDE`, `DEEPSEEK`, `GROK`, `LUMO` (e.g. [POST_EXP_006_DEEPSEEK.md](file:///home/joan/Documents/IA/frankenswarm/docs/extern/POST_EXP_006_DEEPSEEK.md)).
- **Standard Metadata Header**: Every audit report must start with a YAML-style blockquote specifying Auditor, Role, Target Stage, and Date, separated by a horizontal line.
- **Raw Outputs**: Relocated to the `docs/extern/raw/` subdirectory (e.g. `EXP_005_PROTO_SYNTAX.md`) to keep the main audits list clean.

---

## 📁 Project Structure

```
frankenswarm/
├── ARCHITECTURE.md       # Core pillars: BitNet + NEAT + Net2Net + MoE + TurboQuant
├── CHANGELOG.md          # Version history
├── CONVENTIONS.md        # This file
├── README.md             # Overview
├── pyproject.toml        # Dependencies
├── src/
│   ├── tokenizer/        # SovTok — Unigram LM sovereign tokenizer
│   ├── router/           # Prolog-style deterministic router
│   ├── nodes/            # BitNet Expert node definitions
│   └── swarm/            # NEAT Mutator, Net2Net Expander, MoE Orchestrator
└── tests/                # Fitness benchmarks + unit tests
```
