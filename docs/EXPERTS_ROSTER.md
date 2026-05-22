# Frankenswarm Experts Roster v2 — Hardware-Bound Specialists

> *"Cada chip tiene una personalidad. El router las conoce todas."*

## Physical Topology

```
     ┌─────────────────────────────────────────────┐
     │           EXPERT → SILICON BINDING           │
     ├─────────────┬────────────┬───────────────────┤
     │   Expert    │  Silicon   │  Why This Pairing │
     ├─────────────┼────────────┼───────────────────┤
     │ Orchestrator│  NPU       │  Fast triage,     │
     │             │  (2W)      │  always-on, cheap │
     ├─────────────┼────────────┼───────────────────┤
     │ Engineer    │  CUDA      │  Complex code gen │
     │             │  (80W)     │  needs raw power  │
     ├─────────────┼────────────┼───────────────────┤
     │ Scholar     │  CPU       │  128K context,    │
     │             │  (45W)     │  DDR5 bandwidth   │
     ├─────────────┼────────────┼───────────────────┤
     │ Sentinel    │  Vulkan    │  Background watch │
     │             │  (15W)     │  low priority     │
     └─────────────┴────────────┴───────────────────┘
```

---

## 0. The Orchestrator — NPU (XDNA2, ~2W)
*Model: Qwen3-8B via FastFlowLM @ 10.6 tok/s*

- **Mission**: First contact. Every query hits the Orchestrator first. It classifies intent, estimates complexity, and either answers directly or escalates.
- **Why NPU**: Ultra-low latency for classification (96 tok/s with 0.6B). The NPU is always on, costs nothing, and never competes for VRAM. It's the receptionist that never sleeps.
- **Input**: Raw human language.
- **Output**: Either a direct answer (simple queries) or a routing decision `{expert: "engineer", confidence: 0.3, reason: "complex_code"}`.
- **Cascade Rule**: If self-confidence < 0.6, escalate to CUDA. If context > 16K tokens, escalate to CPU.

## 1. The Engineer — CUDA (RTX 5070, ~80W)
*Model: Falcon3-10B-1.58b @ 23 tok/s / Qwen3-8B GGUF @ 30+ tok/s*

- **Mission**: Heavy lifting. Complex code generation, multi-step reasoning, architectural planning, debugging. The brain.
- **Why CUDA**: 8GB GDDR7 enables 10B+ models with full GPU offload. Fastest decoding for complex tasks.
- **Input**: Structured prompts from the Orchestrator with context.
- **Output**: Code, analysis, multi-paragraph reasoning.
- **Energy Note**: Only activates when the Orchestrator escalates. For a typical session, CUDA fires ~20% of the time.

## 2. The Scholar — CPU (Ryzen AI 9 365, ~45W)
*Model: Falcon3-10B via build_vulkan/llama-cli @ 12.8 tok/s*

- **Mission**: Long-context specialist. When the query requires analyzing entire files, long documents, or maintaining massive conversation history, the Scholar uses the full 32GB DDR5 as a context buffer.
- **Why CPU**: No VRAM limit. Can hold 128K+ token contexts that would OOM on the GPU. Slower but unbounded.
- **Input**: Long documents, multi-file analysis, codebase-wide searches.
- **Output**: Summarization, cross-reference analysis, long-form synthesis.

## 3. The Sentinel — Vulkan iGPU (Radeon 880M, ~15W)
*Model: Falcon3-10B via Vulkan @ 4.8 tok/s*

- **Mission**: Background watchdog. Runs continuous health checks, code audits, and slow-burn tasks that don't need speed. The Sentinel never interrupts the other experts.
- **Why Vulkan iGPU**: Shared DDR5 memory means it doesn't steal VRAM from CUDA. Slow but independent — can run while the GPU handles the Engineer's workload.
- **Input**: Automated audit requests, background analysis queues.
- **Output**: Health reports, code smell alerts, low-priority summaries.

## 4. The Evolved Micro-Experts — NPU / CPU (Future)
*Model: BitNet 1.58b micro-networks, 1-7M params, bred via NEAT*

- **Mission**: Ultra-specialized domain experts evolved through genetic algorithms. One might be a JSON validator. Another a regex engine. Another a commit message generator. Each does ONE thing perfectly.
- **Why BitNet**: Ternary weights mean the forward pass is pure addition. No FP16. Runs at near-native speed on NPU and CPU.
- **Evolution**: NEAT mutates populations. Net2Net expands survivors. The Lisp Orchestrator deploys the fittest to the most idle silicon.
- **Status**: Phase 3+ (not yet implemented — requires NEAT loop activation).

---

## Routing Decision Matrix

| Query Type | Primary | Fallback | Strategy | Est. Latency |
|------------|---------|----------|----------|:------------:|
| "What is X?" | NPU | — | Route | 200ms |
| "Write a function that..." | CUDA | CPU | Cascade | 2-5s |
| "Analyze this 50KB file" | CPU | CUDA | Route | 10-30s |
| "Run nightly audit" | Vulkan | CPU | Route | 60s+ |
| "Review this PR" | ALL | — | Consensus | 5-15s |
| "Fix this crash" | NPU→CUDA | — | Cascade | 1-5s |
| "Summarize today" | NPU | CUDA | Cascade | 1-3s |

## Energy Budget (Per 1000 Queries, Measured)

| Strategy | Distribution | Total Energy | Cost vs Cloud |
|----------|:------------:|:------------:|:-------------:|
| All-CUDA (naive) | 100% GPU | 2.2 kWh | 1x |
| All-Cloud (GPT-4) | 100% API | ~5-10 kWh | 2-5x |
| **Frankenswarm Cascade** | 80% NPU, 15% CUDA, 5% all | **0.05 kWh** | **0.02x** |

---

*Documento vivo. Hardware bindings se actualizan cuando se añade nuevo silicio al host.*
