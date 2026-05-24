# Frankenswarm — Heterogeneous Hardware Mixture of Experts

> *"No esperes a tener el hardware perfecto. Conquista el hardware que tienes."*

## What Is This?

Frankenswarm is a **physically-distributed Mixture of Experts** that turns every accelerator in your machine into a specialized inference node. Instead of one big model on one big GPU, it runs multiple models across **all available silicon** — discrete GPU, integrated GPU, CPU, and Neural Processing Unit — each one an expert with different strengths, speeds, and energy costs.

A Prolog router classifies intent. A Python orchestrator manages the active topology dynamically. Specialized LoRA micro-experts run on the NPU at 2 watts while the GPU serves your heavyweight queries.

**The system adapts its active routing and fine-tunes specialized adapters dynamically.**

## The Equation (Recalibrated)

```
Frankenswarm v2 = Hardware MoE (GPU + iGPU + CPU + NPU)
                + Prolog Gate (intent → silicon routing)
                + Python Orchestrator (dynamic adapter management)
                + LoRA / PEFT (task-specific micro-experts)
                + NAS (Network Architecture Search for thresholds and fallbacks)
                + Knowledge Distillation (heavy model → micro-expert)
```

## The Hardware (Measured — Strix Point, 2026-05-22)

| Silicon | Model | tok/s | Power | Role |
|---------|-------|:-----:|:-----:|------|
| RTX 5070 (CUDA) | Falcon3-10B | 23 | ~80W | Heavyweight Reasoning |
| Ryzen AI 9 (CPU) | Falcon3-10B | 12.8 | ~45W | Long Context Scholar |
| Radeon 880M (Vulkan) | Falcon3-10B | 4.8 | ~15W | Background Sentinel |
| XDNA2 NPU | Qwen3-0.6B | 96 | ~2W | Fast Scout / NEAT Breeder |
| XDNA2 NPU | Qwen3-8B | 10.6 | ~2W | Orchestrator / Triage |

## The Three Adaptation Loops

| Loop | Timescale | What Mutates |
|------|-----------|--------------|
| **FAST** (Prolog) | Per-query (~200ms) | Routing policy — which silicon handles what |
| **MEDIUM** (Orchestrator) | Per-session (~minutes) | Activation thresholds and cascade fallback weights (NAS) |
| **SLOW** (Metabolic) | Per-sleep (~hours) | Local fine-tuning (QLoRA) of specialized micro-experts |

## Why Python + ONNX/GGUF?

By sticking to standard python-based orchestrations and standardized formats like ONNX and GGUF, we gain native access to hardware accelerator runtimes (CUDA, ROCm, Vulkan, NPU) without adding compilation overhead. Topology is managed dynamically as a standard JSON schema, and hot-swapping adapters is a simple lightweight weight-loading call rather than hot-swapping runtimes.

→ See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full thesis.

## Energy Sovereignty

| Strategy | Energy per 1000 queries |
|----------|:-----------------------:|
| All-Cloud (GPT-4) | ~5-10 kWh |
| All-CUDA (local) | ~2.2 kWh |
| **Frankenswarm Cascade** | **~0.05 kWh** |

For queries where a specialized local micro-expert (0.6B-1.5B) provides acceptable quality, we run at 2 watts on the NPU. The GPU (80W+) is kept in sleep mode, firing only when the cascade's confidence threshold is violated. This makes the local swarm **up to 200x more efficient** for routine triage and classification tasks.

## Project Status

| Phase | Status |
|-------|--------|
| Translator (text ↔ vectors) | ✅ Done |
| Ghost Swarm (mocks + routing) | ✅ Done |
| Hardware Discovery (benchmarks) | ✅ Done |
| Prolog Gate (intent routing) | 🔲 Next |
| Aggregator (multi-expert consensus) | 🔲 Planned |
| LoRA Specialization & Distillation | 🔲 Planned |
| NAS & Python Orchestration | 🔲 Planned |

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
