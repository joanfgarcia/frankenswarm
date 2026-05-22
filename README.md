# Frankenswarm — Heterogeneous Hardware Mixture of Experts

> *"No esperes a tener el hardware perfecto. Conquista el hardware que tienes."*

## What Is This?

Frankenswarm is a **physically-distributed Mixture of Experts** that turns every accelerator in your machine into a specialized inference node. Instead of one big model on one big GPU, it runs multiple models across **all available silicon** — discrete GPU, integrated GPU, CPU, and Neural Processing Unit — each one an expert with different strengths, speeds, and energy costs.

A Prolog router classifies intent. A Lisp orchestrator evolves the topology in real-time. BitNet micro-experts breed via genetic algorithms on the NPU at 2 watts while the GPU serves your actual queries.

**The system grows new neurons while answering your questions.**

## The Equation (Evolved)

```
Frankenswarm v2 = Hardware MoE (GPU + iGPU + CPU + NPU)
                + Prolog Gate (intent → silicon routing)
                + Lisp REPL (live topology mutation)
                + BitNet 1.58b (ternary micro-experts)
                + NEAT (genetic evolution)
                + Net2Net (capacity expansion)
                + TurboQuant (3-bit KV cache compression)
```

## The Hardware (Measured — Strix Point, 2026-05-22)

| Silicon | Model | tok/s | Power | Role |
|---------|-------|:-----:|:-----:|------|
| RTX 5070 (CUDA) | Falcon3-10B | 23 | ~80W | Heavyweight Reasoning |
| Ryzen AI 9 (CPU) | Falcon3-10B | 12.8 | ~45W | Long Context Scholar |
| Radeon 880M (Vulkan) | Falcon3-10B | 4.8 | ~15W | Background Sentinel |
| XDNA2 NPU | Qwen3-0.6B | 96 | ~2W | Fast Scout / NEAT Breeder |
| XDNA2 NPU | Qwen3-8B | 10.6 | ~2W | Orchestrator / Triage |

## The Three Evolutionary Loops

| Loop | Timescale | What Mutates |
|------|-----------|--------------|
| **FAST** (Prolog) | Per-query (~200ms) | Routing policy — which silicon handles what |
| **MEDIUM** (NEAT) | Per-session (~minutes) | Expert weights — ternary mutations |
| **SLOW** (Lisp) | Per-sleep (~hours) | Network topology + hardware affinity |

## Why Lisp?

Because **code is data**. A neural network topology is an S-expression. A Lisp function can read it, mutate it, and deploy it — without stopping the system. The network rewrites itself at 3 AM and you wake up to a smarter machine.

→ See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full thesis.

## Energy Sovereignty

| Strategy | Energy per 1000 queries |
|----------|:-----------------------:|
| All-Cloud (GPT-4) | ~5-10 kWh |
| All-CUDA (local) | ~2.2 kWh |
| **Frankenswarm Cascade** | **~0.05 kWh** |

80% of queries are answered by the NPU at 2 watts. The GPU only fires for the hard questions. **200x more efficient than cloud inference.**

## Project Status

| Phase | Status |
|-------|--------|
| Translator (text ↔ vectors) | ✅ Done |
| Ghost Swarm (mocks + routing) | ✅ Done |
| Hardware Discovery (benchmarks) | ✅ Done |
| Prolog Gate (intent routing) | 🔲 Next |
| Aggregator (multi-expert consensus) | 🔲 Planned |
| BitNet Real (ternary micro-experts) | 🔲 Planned |
| NEAT + Lisp Evolution | 🔲 Planned |

→ See [ROADMAP.md](docs/ROADMAP.md) for the full plan.

## Docs

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) — Full system design, Lisp thesis, lifecycle diagrams
- [EXPERTS_ROSTER.md](docs/EXPERTS_ROSTER.md) — Expert ↔ silicon bindings with benchmarks
- [ROADMAP.md](docs/ROADMAP.md) — Phase-by-phase implementation plan
- [REFERENCES.md](docs/REFERENCES.md) — Prior art, bibliography, and novelty analysis
- [PROLOG_RESEARCH.md](docs/PROLOG_RESEARCH.md) — Prolog router research and design
- [FRANKENSWARM_SCHEMA.md](docs/FRANKENSWARM_SCHEMA.md) — Visual topology diagrams

## Part of the Red-Pill Ecosystem

Frankenswarm integrates with [Red-Pill](../sharing) via the `ProviderRegistry`. The `FastFlowLMInferenceProvider` (NPU), `SipInferenceProvider` (CUDA/CPU), and `BitNetInferenceProvider` (ternary) are already wired. Frankenswarm's Prolog Gate is the evolutionary successor to the static `InferenceRouter`.
