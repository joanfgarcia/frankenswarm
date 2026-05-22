# Frankenswarm: Roadmap v2 — Heterogeneous Hardware MoE

> *Evolved from single-GPU logical MoE to physically-distributed inference across all available silicon.*

## Phase 0: The Translator ✅ (Completed 2026-05-18)
Bidirectional bridge between human language and the embedding space.
- [x] Integrate `all-MiniLM-L6-v2` as the Universal Dictionary
- [x] Text → Vector → Similarity Search → Text roundtrip validated

## Phase 1: The Ghost Swarm ✅ (Completed 2026-05-18)
Mock pipeline with fake experts and static Prolog routing.
- [x] Mock BitNet nodes (perturbation-based forward pass)
- [x] Mock Prolog router (static rules)
- [x] Pipeline: `Text → Translator → Vector → Router → Mock Expert → Text`
- [x] SWI-Prolog router prototype

## Phase 1.5: Hardware Discovery ✅ (Completed 2026-05-22)
Benchmark all available silicon and confirm inference capabilities.
- [x] CUDA (RTX 5070): 23 tok/s with Falcon3-10B — confirmed
- [x] CPU (Ryzen AI 9): 12.8 tok/s with Falcon3-10B — confirmed
- [x] Vulkan iGPU (Radeon 880M): 4.8 tok/s with Falcon3-10B — confirmed
- [x] NPU (XDNA2): 96 tok/s (0.6B) / 10.6 tok/s (8B) — confirmed
- [x] FastFlowLMInferenceProvider integrated in Red-Pill ProviderRegistry
- [x] InferenceRouter with `npu` tier and cascade fallback

## Phase 2: The Prolog Gate (Hardware-Aware Routing)
Replace mock routing with real intent classification → silicon binding.
- [ ] Define Prolog ontology: query taxonomy (code, reasoning, recall, triage, background)
- [ ] Implement hardware affinity predicates: `route(Query, Silicon)`
- [ ] Add confidence scoring: Orchestrator (NPU) self-evaluates and escalates
- [ ] Energy-aware routing: prefer 2W (NPU) over 80W (CUDA) when quality is equal
- [ ] Integrate with Red-Pill `InferenceRouter` — replace static tier list with Prolog decisions

## Phase 2.5: The Cage (Sandboxing)
Mandatory before granting autonomous mutation capabilities.
- [ ] `systemd-run --user --scope` isolation for NEAT processes
- [ ] `MemoryMax`, `CPUQuota`, `PrivateNetwork=yes` constraints
- [ ] Demonstrate that a mutant trying to escape the sandbox is killed by the kernel
- [ ] Audit trail: every mutation logged to SQLite with rollback capability

## Phase 3: The Aggregator
Multiple experts respond — the system decides who wins.
- [ ] Implement aggregation strategies: Route, Cascade, Race, Consensus
- [ ] Semantic similarity voting for Consensus mode
- [ ] Confidence threshold (θ) for Cascade escalation: NPU → CUDA
- [ ] Latency budgets: if NPU doesn't respond in 500ms, fire CUDA in parallel
- [ ] Telemetry: log per-query energy cost, latency, and expert selection

## Phase 4: The First Spark (BitNet Real)
Replace mock experts with real ternary micro-networks.
- [ ] Implement `BitNet158Linear_Dynamic` with shape mutation support
- [ ] Train initial micro-experts (1-7M params) on deterministic tasks
- [ ] Deploy on NPU via FastFlowLM custom model loading
- [ ] Benchmark: micro-expert tok/s on NPU (target: 500+ tok/s for 1M params)

## Phase 5: Natural Selection (NEAT + Net2Net + Lisp)
The system evolves itself.
- [ ] Implement NEAT genetic loop: selection, crossover, ternary weight mutation
- [ ] Implement Net2Net zero-padding for capacity expansion
- [ ] Lisp REPL orchestrator: topology as S-expressions, live mutation
- [ ] The Three Loops:
  - **FAST** (per-query): Prolog adjusts routing weights
  - **MEDIUM** (per-session): NEAT mutates BitNet weights
  - **SLOW** (per-sleep): Lisp restructures topology + hardware affinity
- [ ] NPU as evolution accelerator: NEAT fitness eval at 1000+ tok/s while GPU serves
- [ ] Self-modifying routing: Lisp rewrites Prolog rules based on fitness observations

## Phase 6: Convergence with Red-Pill Sleep Cycle
Frankenswarm becomes the metabolic engine of the Bünker.
- [ ] Hook into `metabolism/sleep.py` — evolution runs during 3 AM window
- [ ] Fitness function derived from daily interaction quality (Qdrant engram scores)
- [ ] Expert deployment pipeline: NEAT winner → FastFlowLM model → ProviderRegistry
- [ ] Dream mode: Lisp generates hypothetical queries, evaluates micro-experts, prunes the weak

---

## Status Summary

| Phase | Status | Hardware |
|-------|--------|----------|
| 0 — Translator | ✅ Done | — |
| 1 — Ghost Swarm | ✅ Done | — |
| 1.5 — Hardware Discovery | ✅ Done | All 4 confirmed |
| 2 — Prolog Gate | 🔲 Next | NPU + CUDA |
| 2.5 — Cage | 🔲 Planned | systemd |
| 3 — Aggregator | 🔲 Planned | All 4 |
| 4 — BitNet Real | 🔲 Planned | NPU + CPU |
| 5 — Evolution | 🔲 Planned | NPU (evolution) + GPU (serving) |
| 6 — Red-Pill Convergence | 🔲 Future | Full stack |
