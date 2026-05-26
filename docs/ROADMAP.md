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
## ── BEGIN PHASE B: THE ARENA (Specialized Adapters & Distillation) ──

## Phase 4: QLoRA Specialization Pipeline (Local Training & PopuLoRA Arena)
Establish a structured workflow to train task-specific micro-experts from daily interaction datasets and autonomous self-play.
- [ ] Implement automatic training dataset compilation from Qdrant engram scores (filtering high-quality interactions)
- [ ] Configure QLoRA parameter-efficient fine-tuning scripts for local base models (e.g. Qwen-1.5B/TinyLlama)
- [ ] Integrate **PopuLoRA asymmetric self-play** to generate local coding and reasoning tasks autonomously, bypassing the dataset bottleneck
- [ ] Implement low-rank weight-space crossovers and mutations to evolve the student populations directly in PyTorch/Safetensors space
- [ ] Set up evaluation pipelines (using LM-Eval or custom benchmark query sets) to confirm no task regression
- [ ] Validate adapter performance under low-resource constraints (NPU/iGPU execution)
- [ ] Embed training metadata directly into the adapter artifacts for registry versioning

## Phase 5: Mixture of Adapters (MoA) Orchestration
Manage dynamic loading and hot-swapping of PEFT adapters over a shared base model.
- [ ] Implement `AdapterOrchestrator` in Python to dynamically load/unload LoRA weights at runtime
- [ ] Standardize the adapter registry schema (mapping task capabilities $\rightarrow$ adapter files)
- [ ] Minimize latency overhead of dynamic loading (aim for $<10\text{ms}$ weight swapping times)
- [ ] Enable parallel execution of multiple adapters via batching or sequential forward pass pooling
- [ ] Support fallback to base model weights when classifier confidence (θ) is below routing thresholds

## Phase 6: Knowledge Distillation & Quantization (Teacher-Student Self-Play)
Compress heavyweight capabilities into hyper-fast, low-wattage local micro-experts.
- [ ] Establish an online distillation pipeline using the **PopuLoRA RLVR loop**: parent model (RTX 5070 CUDA, 10B+, as Teacher) $\rightarrow$ student micro-expert (1.5B/0.6B, as Student)
- [ ] Quantize distilled student models to GGUF / AWQ formats optimized for CPU and Vulkan iGPU execution
- [ ] Benchmark token throughput on NPU (target: $150+\text{ tok/s}$ for quantized distilled micro-experts)
- [ ] Integrate distilled models with FastFlowLM custom model loaders on XDNA2
- [ ] Maintain a library of task-specific tiny models (Code, Reasoning, Triage) that run entirely at 2W

## Phase 7: Network Architecture Search (NAS) & Python Orchestration
Optimize routing thresholds and model topologies using search algorithms while weights remain gradient-trained.
- [ ] Implement genetic/heuristic loop (NAS style) to evolve routing thresholds and fallback parameters
- [ ] Define the topology as a standard, version-controlled JSON-LD schema (removing Lisp dependencies)
- [ ] Manage the adaptation loops:
	- **FAST** (per-query): Prolog routing decisions
	- **MEDIUM** (per-session): Orchestrator adapts routing thresholds based on actual feedback
	- **SLOW** (per-sleep): Genetic search (NAS) tunes routing weights and swaps active adapters
- [ ] Implement automatic routing self-healing: adjust confidence thresholds when high-latency fallbacks increase

## Phase 8: Metabolic Sleep Cycle Integration
Align adaptation and training loops with the Red-Pill sleep cycle.
- [ ] Hook fine-tuning and NAS optimization runs into `metabolism/sleep.py`
- [ ] Perform overnight QLoRA training of task-specific adapters using the previous day's interaction engrams
- [ ] Execute NAS optimization to recalibrate routing thresholds based on daily latency/accuracy metrics
- [ ] Implement atomic deployment: test new adapters overnight and register them with `ProviderRegistry` at 5 AM
- [ ] Automated rollback: automatically restore the previous day's adapter if evaluation benchmarks fail

## Phase 9: Hierarchical Swarm Routing
Scale the orchestration to multi-agent, collaborative routing architectures.
- [ ] Transition from a single Prolog Gate to a hierarchical router-of-routers pipeline
- [ ] Enable micro-experts to collaborate by executing sub-queries (calling other specialized experts) via clean API/token requests
- [ ] Implement swarm telemetry: track E2E token latency and power consumption across collaborative execution paths
- [ ] Establish consensus voting for multi-expert queries (race, consensus, and weighted averaging)

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
| 4 — QLoRA Pipeline | 🔲 Planned | Arena | NPU / GPU / CPU |
| 5 — Mixture of Adapters | 🔲 Planned | Arena | GPU / CPU / iGPU |
| 6 — Distillation & Quantization | 🔲 Planned | Arena | NPU + CPU |
| 7 — NAS & Python Orchestration | 🔲 Planned | Arena | Orchestrator |
| 8 — Sleep Cycle Integration | 🔲 Future | Arena | Full stack |
| 9 — Hierarchical Swarm | 🔲 Future | Arena | Full stack |

