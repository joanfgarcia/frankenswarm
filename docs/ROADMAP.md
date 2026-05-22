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
## ── END OF PHASE A: PoC (Existing Models) ──
## ── BEGIN PHASE B: THE ARENA (Custom-Bred Experts) ──

## Phase 4: The Babel Fish (AI-Native Tokenization)
Replace human-language tokenization with a sovereign latent language.
- [ ] Train or adopt a multilingual Translator: `Human Text (any lang) → D-dim vector`
- [ ] Implement `BabelFishTranslator` with bidirectional encode/decode
- [ ] Validate: same query in Spanish and English produces near-identical routing decisions
- [ ] Define the swarm's native embedding dimensionality (384D for PoC, target 768D for Arena)
- [ ] Prolog Gate operates on vectors only — zero human text past the Translator boundary

## Phase 5: The Three-Layer Architecture
Build the expert anatomy: Common Substrate → Adapter → Specialist Core.
- [ ] Implement `CommonSubstrate`: shared frozen weights across all experts
- [ ] Implement `AdapterLayer`: lightweight LoRA-scale bridge (~1% of total params)
- [ ] Implement `SpecialistCore`: domain-specific ternary weights, NEAT-evolvable
- [ ] Validate decoupled evolution: mutating a Specialist doesn't break other experts
- [ ] Adapter auto-retrain: detect distribution shift when Specialist mutates, retrain bridge in ~2 min
- [ ] Common Substrate upgrade path: change embedding dim → ALL Adapters retrain → Specialists untouched

## Phase 6: The First Spark (BitNet Real in Three Layers)
Replace mock experts with real ternary micro-networks using the three-layer anatomy.
- [ ] Implement `BitNet158Linear_Dynamic` with shape mutation support
- [ ] Train initial micro-experts (1-7M params) on deterministic tasks
- [ ] Each expert = Common (shared) + Adapter (learned) + Specialist (evolved)
- [ ] Deploy on NPU via FastFlowLM custom model loading
- [ ] Benchmark: micro-expert tok/s on NPU (target: 500+ tok/s for 1M params)

## Phase 7: Natural Selection (NEAT + Net2Net + Lisp)
The system evolves itself — Specialist Cores mutate independently.
- [ ] Implement NEAT genetic loop: selection, crossover, ternary weight mutation on Layer 3 only
- [ ] Implement Net2Net zero-padding for Specialist capacity expansion
- [ ] Lisp REPL orchestrator: topology as S-expressions, live mutation
- [ ] The Three Loops:
  - **FAST** (per-query): Prolog adjusts routing weights
  - **MEDIUM** (per-session): NEAT mutates Specialist Core weights
  - **SLOW** (per-sleep): Lisp restructures topology, grows layers, migrates hardware affinity
- [ ] NPU as evolution accelerator: NEAT fitness eval at 1000+ tok/s while GPU serves
- [ ] Self-modifying routing: Lisp rewrites Prolog rules based on fitness observations
- [ ] Adapter auto-retrain triggered after every NEAT generation that produces a new champion

## Phase 8: Convergence with Red-Pill Sleep Cycle
Frankenswarm becomes the metabolic engine of the Bünker.
- [ ] Hook into `metabolism/sleep.py` — evolution runs during 3 AM window
- [ ] Fitness function derived from daily interaction quality (Qdrant engram scores)
- [ ] Expert deployment pipeline: NEAT winner → Adapter retrain → FastFlowLM → ProviderRegistry
- [ ] Dream mode: Lisp generates hypothetical queries, evaluates micro-experts, prunes the weak
- [ ] Common Substrate versioning: major upgrades trigger full Adapter re-train across swarm

## Phase 9: The Sovereign Translator (Custom Babel Fish)
Graduate from pre-trained embeddings to a custom AI-native language.
- [ ] Train a domain-specific encoder: optimized for the swarm's actual query distribution
- [ ] Train a multilingual decoder: latent vector → any human language (es, en, zh, ja, ar)
- [ ] The embedding space becomes *ours* — not constrained by English Wikipedia
- [ ] Inter-expert communication bandwidth drops to O(D) instead of O(tokens)
- [ ] Full language sovereignty: the swarm thinks in its own tongue, translates at the edges

---

## Status Summary

| Phase | Status | Stage | Hardware |
|-------|--------|-------|----------|
| 0 — Translator | ✅ Done | PoC | — |
| 1 — Ghost Swarm | ✅ Done | PoC | — |
| 1.5 — Hardware Discovery | ✅ Done | PoC | All 4 confirmed |
| 2 — Prolog Gate | 🔲 Next | PoC | NPU + CUDA |
| 2.5 — Cage | 🔲 Planned | PoC | systemd |
| 3 — Aggregator | 🔲 Planned | PoC | All 4 |
| **── PoC validated ──** | | | |
| 4 — Babel Fish | 🔲 Planned | Arena | Translator |
| 5 — Three-Layer Architecture | 🔲 Planned | Arena | All 4 |
| 6 — BitNet Real | 🔲 Planned | Arena | NPU + CPU |
| 7 — Evolution | 🔲 Planned | Arena | NPU (breed) + GPU (serve) |
| 8 — Red-Pill Convergence | 🔲 Future | Arena | Full stack |
| 9 — Sovereign Translator | 🔲 Future | Arena | Custom training |

