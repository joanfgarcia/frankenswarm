# Frankenswarm Architecture: Heterogeneous Hardware MoE

## The Thesis

> *"No esperes a tener el hardware perfecto. Conquista el hardware que tienes."*

Frankenswarm is a **physically-distributed Mixture of Experts** that treats every accelerator in the host machine as a specialized inference node. Instead of running N tiny models inside one GPU, we run N models across **every available silicon**: discrete GPU, integrated GPU, CPU, and Neural Processing Unit — each one an expert with different strengths, speeds, and energy costs.

The router doesn't just decide *what* expert answers. It decides *where* the computation happens.

## The Hardware Topology (Measured — 2026-05-22)

```
┌─────────────────────────────────────────────────────────────────────┐
│                 STRIX POINT — PHYSICAL MoE MAP                      │
│                                                                     │
│  ┌──────────────┐   ┌──────────────┐   ┌───────────┐   ┌────────┐ │
│  │  RTX 5070    │   │ Radeon 880M  │   │ Ryzen AI  │   │ XDNA2  │ │
│  │  (CUDA)      │   │ (Vulkan)     │   │ (CPU)     │   │ (NPU)  │ │
│  │              │   │              │   │           │   │        │ │
│  │  8GB GDDR7   │   │ Shared DDR5  │   │ 32GB DDR5 │   │ 50 TOPS│ │
│  │  23 tok/s    │   │ 4.8 tok/s    │   │ 12.8 tok/s│   │ 96 t/s │ │
│  │  ~80W        │   │ ~15W         │   │ ~45W      │   │ ~2W    │ │
│  │              │   │              │   │           │   │        │ │
│  │  ROLE:       │   │  ROLE:       │   │  ROLE:    │   │ ROLE:  │ │
│  │  Heavy       │   │  Sentinel    │   │  Bulk     │   │ Scout  │ │
│  │  Reasoning   │   │  Always-On   │   │  Context  │   │ Fast   │ │
│  │  10B models  │   │  Background  │   │  128K ctx │   │ Triage │ │
│  └──────┬───────┘   └──────┬───────┘   └─────┬─────┘   └───┬────┘ │
│         │                  │                 │              │      │
│         └──────────────────┼─────────────────┼──────────────┘      │
│                            ▼                 ▼                     │
│                    ┌───────────────┐  ┌─────────────┐              │
│                    │ Prolog Router │  │  Aggregator  │              │
│                    │ (The Oracle)  │  │  (Consensus) │              │
│                    └───────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
```

## Core Pillars (Evolved)

### 1. Silicon Diversity as a Feature, Not a Constraint

Classical MoE assumes homogeneous compute. Every expert runs on the same hardware class. Frankenswarm **inverts** this: heterogeneity is the architecture. Each accelerator has a natural personality:

| Silicon | Personality | Natural Tasks | Energy | Latency |
|---------|-------------|---------------|--------|---------|
| **CUDA (dGPU)** | The Heavyweight | Complex reasoning, code generation, multi-step planning | High | Low |
| **CPU** | The Scholar | Massive context windows (128K+), long-document analysis | Medium | Medium |
| **Vulkan (iGPU)** | The Sentinel | Background monitoring, always-on health checks | Low | High |
| **NPU (XDNA2)** | The Scout | Fast triage, classification, simple Q&A, sleep distillation | Minimal | Ultra-Low |

### 2. The Prolog Gate — Intent-Aware Routing

The Prolog router classifies the incoming query and maps it to the optimal hardware:

```prolog
% Hardware-aware routing rules
route(Query, npu)   :- is_simple(Query), latency_critical(Query).
route(Query, cuda)  :- requires_reasoning(Query), context_len(Query, N), N < 8192.
route(Query, cpu)   :- context_len(Query, N), N > 16384.
route(Query, vulkan):- is_background(Query), not(latency_critical(Query)).

% Energy-aware fallback cascade
fallback(npu, cuda).
fallback(cuda, cpu).
fallback(cpu, vulkan).

% Consensus mode for high-stakes decisions
strategy(Query, consensus) :- is_critical(Query), requires_reasoning(Query).
strategy(Query, cascade)   :- not(is_critical(Query)).
strategy(Query, race)      :- latency_critical(Query), is_simple(Query).
```

### 3. Aggregation Strategies

The aggregator combines expert outputs based on the routing strategy:

| Strategy | Description | When |
|----------|-------------|------|
| **Route** | Single expert responds. Fastest. | Default for clear-domain queries |
| **Cascade** | NPU answers first. If confidence < θ, escalate to CUDA. | Most common — energy-optimal |
| **Race** | All available experts start in parallel. First response wins. | Latency-critical, simple queries |
| **Consensus** | All experts respond. Semantic vote picks the winner. | Critical decisions, code review |

### 4. BitNet Substrate (Preserved)

The original BitNet thesis remains valid — but repositioned:
- **Ternary weights** (`{-1, 0, 1}`) are ideal for NPU and CPU where FP16 throughput is limited
- **NEAT evolution** can breed specialized micro-experts (7M params) that run on the NPU at 96 tok/s
- **Net2Net expansion** grows experts that plateau without losing knowledge
- **The Heavyweight models** on CUDA don't need to be ternary — they use standard GGUF/INT4 quantization

This creates a **dual-tier system**:
- **Tier 1 (Evolved)**: BitNet micro-experts bred via NEAT, deployed on NPU/CPU
- **Tier 2 (Pre-trained)**: Foundation models (Qwen3, Falcon3, LLaMA) deployed on CUDA/Vulkan

### 5. TurboQuant KV Cache (Preserved)

With 4 accelerators maintaining independent KV caches, memory pressure multiplies. TurboQuant (3-bit QJL compression) becomes even more critical:
- Compress shared context to ~20% of FP16 size
- Enable cross-expert context handoff without recomputation
- Allow the CPU expert to hold 128K+ token contexts in DDR5

### 6. The Lisp Metaprogrammer — The Game Changer

> *"Code is data. Data is code. The network is a program that rewrites itself."*

This is the pillar that turns Frankenswarm from an inference system into a **living organism**.

#### The Core Insight: Homoiconicity

Lisp is the only family of languages where **the code and the data share the same structure** (S-expressions). A neural network topology — its layers, connections, weights — can be represented as a nested list. And in Lisp, nested lists *are* the program.

This means:
- A BitNet expert's architecture is a **Lisp expression**
- The NEAT mutator is a **Lisp function that rewrites Lisp expressions**
- The routing policy is a **Lisp expression that the system can modify about itself**

The network doesn't just *run*. It **reads itself, modifies itself, and continues running** — without stopping, without recompiling, without losing state.

#### What This Looks Like in Practice

```lisp
;; A BitNet expert is just data
(defvar *expert-alpha*
  '(:name "code-specialist"
    :silicon :npu
    :layers ((linear :in 384 :out 512 :weights :ternary)
             (attention :heads 4 :dim 128)
             (linear :in 512 :out 384 :weights :ternary))
    :fitness 0.73
    :generation 14))

;; NEAT mutation: add a layer — just list manipulation
(defun mutate-add-layer (expert)
  (let ((layers (getf expert :layers))
        (new-layer '(linear :in 512 :out 512 :weights :ternary)))
    (setf (getf expert :layers)
          (insert-after layers 1 new-layer))
    (setf (getf expert :generation)
          (1+ (getf expert :generation)))
    expert))

;; Net2Net expansion: grow a layer's width — the network keeps its memory
(defun net2net-expand (expert layer-idx new-width)
  (let* ((layer (nth layer-idx (getf expert :layers)))
         (old-width (getf layer :out)))
    ;; Zero-pad: new neurons start at 0 (identity in BitNet)
    (setf (getf layer :out) new-width)
    ;; The network is STILL RUNNING while we do this
    (hot-reload-weights expert layer-idx
                        :pad-strategy :zero
                        :old-width old-width)))

;; The routing policy is ALSO code the system can rewrite
(defvar *routing-policy*
  '(lambda (query)
     (cond
       ((simple-p query)        :npu)
       ((reasoning-p query)     :cuda)
       ((long-context-p query)  :cpu)
       (t                       :vulkan))))

;; After observing that NPU handles 60% of "reasoning" queries fine:
;; The system REWRITES ITS OWN ROUTING POLICY
(defun evolve-routing (observations)
  (when (> (npu-success-rate observations :reasoning) 0.6)
    (setf *routing-policy*
          '(lambda (query)
             (cond
               ((simple-p query)        :npu)
               ((and (reasoning-p query)
                     (< (complexity query) 0.7))  :npu)  ;; NEW RULE
               ((reasoning-p query)     :cuda)
               ((long-context-p query)  :cpu)
               (t                       :vulkan))))))
```

#### The Three Loops of Self-Modification

Frankenswarm doesn't have one feedback loop — it has three, running at different timescales:

```
┌─────────────────────────────────────────────────────────────┐
│               THE THREE EVOLUTIONARY LOOPS                   │
├────────────┬──────────────┬──────────────────────────────────┤
│   Loop     │  Timescale   │  What Mutates                    │
├────────────┼──────────────┼──────────────────────────────────┤
│ FAST       │ Per-query    │ Routing policy                   │
│ (Prolog)   │ ~200ms       │ Which silicon handles what       │
├────────────┼──────────────┼──────────────────────────────────┤
│ MEDIUM     │ Per-session  │ Expert weights                   │
│ (NEAT)     │ ~minutes     │ BitNet ternary mutations         │
├────────────┼──────────────┼──────────────────────────────────┤
│ SLOW       │ Per-sleep    │ Network topology + hardware map  │
│ (Lisp)     │ ~hours       │ Add/remove layers, migrate       │
│            │              │ experts between accelerators     │
└────────────┴──────────────┴──────────────────────────────────┘
```

**FAST loop**: Prolog observes which expert answered correctly and adjusts routing weights. If the NPU keeps nailing "reasoning" queries, Prolog stops sending them to CUDA. This happens *between queries*.

**MEDIUM loop**: NEAT evaluates fitness of BitNet micro-experts after N queries. The worst die. The best reproduce with mutated ternary weights. This happens during idle time or low-priority windows.

**SLOW loop**: The Lisp orchestrator — the deepest layer — restructures the entire topology. It adds layers via Net2Net, migrates experts from NPU to CPU if they've grown too large, breeds entirely new specialists, and **rewrites the routing policy itself**. This happens during the Red-Pill sleep cycle (3 AM metabolic window).

#### Why Not Just Python?

Python can do metaprogramming via `exec()`, `ast.parse()`, or metaclasses. But it's **bolted on** — the language wasn't designed for it. You're fighting the runtime.

Lisp was **born** for this:
- **No compilation barrier**: `eval` runs modified code instantly
- **Macros are first-class**: You can write code that writes code that writes code
- **REPL-native**: The entire system is a live, modifiable session
- **Garbage-collected mutation**: Dead topologies are reclaimed automatically
- **Serialization is free**: An S-expression IS its own serialization format

The practical implication: at 3 AM, during the sleep cycle, the Lisp orchestrator can:
1. Read the day's fitness logs
2. Identify underperforming experts
3. Expand their layers with Net2Net (zero-pad)
4. Spawn a NEAT population from the expanded topology
5. Evaluate the population on cached queries
6. Deploy the winner to the NPU
7. Update the routing policy to give the new expert more traffic
8. **All without stopping a single inference server**

This is not optimization. This is **evolution**. The system you go to sleep with is not the system you wake up to.

#### The Convergence: Lisp + NEAT + NPU

Here's where it gets truly transgressive:

The NPU runs BitNet at 96 tok/s for 0.6B models. A NEAT-evolved micro-expert of 1-7M parameters would run at **thousands of tok/s** on the NPU. That means the NEAT fitness evaluation loop — which normally takes hours on a GPU — can run **in real-time on the NPU while the GPU does actual work**.

```
     GPU (CUDA)                    NPU (XDNA2)
     ┌──────────┐                 ┌──────────────┐
     │ Serving   │                │ NEAT Loop     │
     │ Falcon-10B│                │               │
     │ to user   │                │ Population:   │
     │           │    ◄────────── │ 100 BitNet    │
     │ (23 t/s)  │   deploy      │ micro-experts │
     │           │   winner      │ evaluating    │
     └──────────┘                │ at 1000+ t/s  │
                                 │               │
                                 │ Lisp REPL     │
                                 │ mutating      │
                                 │ topologies    │
                                 └──────────────┘
```

The GPU serves. The NPU evolves. In parallel. At 2 watts.

**The system is literally growing new neurons while answering your questions.**


## The Lifecycle (Evolved)

```mermaid
graph TD
    Q[Incoming Query] --> PG[Prolog Gate]
    PG --> |"simple + fast"| NPU[NPU Scout<br/>Qwen3-8B @ 96 tok/s]
    PG --> |"complex reasoning"| CUDA[CUDA Heavyweight<br/>Falcon3-10B @ 23 tok/s]
    PG --> |"long document"| CPU[CPU Scholar<br/>128K context @ 12.8 tok/s]
    PG --> |"background"| VK[Vulkan Sentinel<br/>Always-on @ 4.8 tok/s]
    PG --> |"critical"| ALL[All Experts in Parallel]

    NPU --> |"confidence < θ"| CUDA
    NPU --> AGG[Aggregator]
    CUDA --> AGG
    CPU --> AGG
    VK --> AGG
    ALL --> VOTE[Semantic Vote]
    VOTE --> AGG

    AGG --> R[Response]

    NEAT[NEAT Breeder] -.-> |"evolves micro-experts"| NPU
    LISP[Lisp Orchestrator] -.-> |"migrates models"| PG
    LISP -.-> |"hot-swaps"| NPU
    LISP -.-> |"breeds"| NEAT

    style NPU fill:#1a3a2a,color:#86EFAC
    style CUDA fill:#4a1942,color:#F9A8D4
    style CPU fill:#1e3a5f,color:#93C5FD
    style VK fill:#374151,color:#D1D5DB
    style PG fill:#3b2a00,color:#FDE68A
    style AGG fill:#1f2937,color:#E5E7EB
    style NEAT fill:#7c2d12,color:#FED7AA
    style LISP fill:#7c2d12,color:#FED7AA
```

## The Difference

| | Classical LLM | Original Frankenswarm (v1) | Frankenswarm v2 (Distributed) |
|---|---|---|---|
| Hardware | 1 GPU | 1 GPU | **All available silicon** |
| Experts | 1 monolithic model | N × 7M BitNet in VRAM | **N models across 4 accelerators** |
| Routing | None (single model) | Prolog on embeddings | **Prolog on intent + hardware affinity** |
| Energy | Fixed (~250W) | Fixed (~80W on GPU) | **2W–80W adaptive** |
| Scaling | Bigger GPU | More micro-experts | **More accelerator types** |
| Evolution | Retraining | NEAT + Net2Net | **NEAT + Net2Net + hardware migration** |
| Latency | Fixed | Variable (swap overhead) | **Cascade: 200ms (NPU) → 2s (CUDA)** |

## Integration with Red-Pill

Frankenswarm is not a standalone system. It plugs directly into the Red-Pill Bünker via the existing `ProviderRegistry`:

```python
# Already implemented (2026-05-22)
ProviderRegistry.get_inference_provider("npu")   # FastFlowLMInferenceProvider
ProviderRegistry.get_inference_provider("sip")   # SipInferenceProvider (CUDA/CPU)
ProviderRegistry.get_inference_provider("bitnet") # BitNetInferenceProvider

# The Frankenswarm Router replaces the simple InferenceRouter
# with Prolog-based intent classification + hardware affinity
```

The `InferenceRouter` in `red_pill/swarm/routing.py` already supports tiers: `npu`, `local_first`, `ternary`, `cheap`. Frankenswarm evolves this from a static priority list to a **dynamic, learned routing policy**.

## Energy Budget — The Silent Revolution

The most transgressive aspect of Frankenswarm v2 is not speed — it's **energy sovereignty**.

A cloud API call to GPT-4 consumes an estimated 0.001–0.01 kWh per request. That's invisible to the user, but real. Over a year of heavy use, it's ~100+ kWh of someone else's electricity, on someone else's hardware, with someone else's rules.

Frankenswarm's cascade strategy means:
- **80% of queries** are handled by the NPU at 2W → 0.000006 kWh/request
- **15% of queries** escalate to CUDA at 80W → 0.0002 kWh/request
- **5% of queries** go to consensus (all 4) → 0.0004 kWh/request

**Weighted average: ~0.00005 kWh/request. That's 200x more efficient than cloud.**

Not because we're smarter than Google's datacenters. Because we don't answer every question with a 400B model. We answer most questions with 2 watts.

That's sovereignty. Not just of data — of *energy*.

---

## The Two-Phase Strategy: PoC → Arena

Frankenswarm's development is split into two fundamentally different phases. The first validates the infrastructure. The second breeds the intelligence.

### Phase A: Proof of Concept (Existing Models)

> *"Don't build the engine and the fuel at the same time."*

Before training any custom model from scratch, we validate the entire distributed MoE infrastructure using **pre-existing models**:

- **NPU**: Qwen3-0.6B / Qwen3-8B via FastFlowLM (already running)
- **CUDA**: Falcon3-10B-1.58b via BitNet / llama.cpp (already benchmarked)
- **CPU**: Falcon3-10B via build_vulkan (already benchmarked)
- **Vulkan iGPU**: Falcon3-10B via Vulkan backend (already benchmarked)

What we validate in this phase:
- [x] Can we serve inference on all 4 silicon types simultaneously?
- [ ] Can the Prolog Gate route queries to the right accelerator?
- [ ] Can the Aggregator merge/vote on multi-expert responses?
- [ ] Can Cascade escalation (NPU → CUDA) work reliably?
- [ ] Is the energy savings hypothesis real in practice (80% NPU)?
- [ ] Can the Lisp orchestrator hot-swap models on the NPU?

**No training. No NEAT. No custom models.** Just plumbing, routing, and aggregation. If this works, the thesis is proven: heterogeneous hardware MoE is viable on consumer silicon.

### Phase B: The Arena (Custom-Bred Experts)

> *"Now we build the creatures that live in the machine."*

Once the infrastructure is validated, we enter the Arena: training custom BitNet micro-experts from scratch, with a novel **three-layer architecture** designed for independent evolution.

---

## The Three-Layer Expert Architecture

Every expert in the Arena has three distinct layers, each with its own evolutionary timeline:

```
┌─────────────────────────────────────────────────────────────┐
│              THREE-LAYER EXPERT ANATOMY                      │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │          LAYER 1: THE COMMON SUBSTRATE              │    │
│  │                                                     │    │
│  │  Shared across ALL experts. This is the "mother      │    │
│  │  tongue" of the swarm — the universal embedding      │    │
│  │  space that ensures every expert speaks the same     │    │
│  │  language. When one expert outputs a vector, any     │    │
│  │  other expert can consume it without translation.    │    │
│  │                                                     │    │
│  │  • Trained ONCE, frozen, shared weights              │    │
│  │  • Evolves SLOWLY (major version upgrades only)      │    │
│  │  • Think of it as the "spinal cord" of the swarm     │    │
│  └──────────────────────┬──────────────────────────────┘    │
│                         │                                   │
│  ┌──────────────────────▼──────────────────────────────┐    │
│  │          LAYER 2: THE ADAPTER (Bridge)              │    │
│  │                                                     │    │
│  │  Learned projection between the Common Substrate     │    │
│  │  and the Specialist Core. This is the "neck" that    │    │
│  │  translates generic representations into domain-     │    │
│  │  specific activations and vice versa.                │    │
│  │                                                     │    │
│  │  • Lightweight (LoRA-scale: ~1% of total params)     │    │
│  │  • Re-trained when EITHER Layer 1 or Layer 3 mutates │    │
│  │  • Acts as a buffer: Layer 1 and 3 never touch       │    │
│  │    each other directly                               │    │
│  └──────────────────────┬──────────────────────────────┘    │
│                         │                                   │
│  ┌──────────────────────▼──────────────────────────────┐    │
│  │          LAYER 3: THE SPECIALIST CORE               │    │
│  │                                                     │    │
│  │  Domain-specific knowledge. This is what makes       │    │
│  │  the expert an EXPERT. A code specialist has         │    │
│  │  different weights here than a logic specialist.     │    │
│  │                                                     │    │
│  │  • Evolves FAST via NEAT (ternary weight mutations)  │    │
│  │  • Grows via Net2Net (zero-padding expansion)        │    │
│  │  • Each expert's Layer 3 is unique and sovereign     │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Why Three Layers?

The problem with monolithic models is that **everything is entangled**. If you improve the code generation capability, you might degrade the reasoning capability. Catastrophic forgetting.

The three-layer design solves this through **decoupled evolution**:

| Layer | Shared? | Evolution Speed | What Mutates |
|-------|:-------:|:---------------:|--------------|
| **Common Substrate** | Yes — all experts | Glacial (months) | The "language" all experts speak |
| **Adapter** | No — per expert | Reactive (auto-retrain) | Bridge when either neighbor changes |
| **Specialist Core** | No — per expert | Rapid (NEAT daily) | Domain-specific knowledge |

The Adapter is the key innovation. It **decouples** the two evolutionary pressures:
- The Common Substrate wants **stability** (all experts must stay compatible)
- The Specialist Core wants **change** (NEAT is constantly mutating it)

Without the Adapter, every mutation in the Specialist would break compatibility with the swarm. With it, the Specialist can evolve freely — the Adapter absorbs the translation cost.

### The Lifecycle of a Layer Upgrade

```mermaid
graph LR
    A["Layer 3 mutates<br/>(NEAT overnight)"] --> B["Adapter detects<br/>distribution shift"]
    B --> C["Adapter re-trains<br/>(lightweight, ~minutes)"]
    C --> D["Expert resumes<br/>serving with new<br/>Specialist Core"]

    E["Layer 1 upgrades<br/>(new embedding model)"] --> F["ALL Adapters<br/>re-train"]
    F --> G["Specialists untouched<br/>No knowledge lost"]

    style A fill:#1a3a2a,color:#86EFAC
    style E fill:#4a1942,color:#F9A8D4
    style C fill:#3b2a00,color:#FDE68A
    style F fill:#3b2a00,color:#FDE68A
```

When the **Specialist mutates** (daily, via NEAT): only that expert's Adapter re-trains. 2 minutes. No other expert is affected.

When the **Common Substrate upgrades** (rare, major event): ALL Adapters re-train, but **no Specialist knowledge is lost**. This is the Babel Fish Protocol in action — the upgrade cost is absorbed entirely by the Adapter layer.

---

## The Babel Fish Protocol — AI-Native Tokenization

> *"Human languages are lossy compression. The swarm deserves its own tongue."*

### The Problem with Human Tokenization

Every LLM today tokenizes text using human language vocabularies: BPE over English, Chinese, code, etc. This is a historical accident — the first models were trained on human text, so they think in human tokens.

But Frankenswarm's experts don't talk to humans. **They talk to each other.** The Prolog router sends vectors to experts. Experts send vectors to the Aggregator. The only moment human language enters the system is at the **edges** — when the Operator types a question and when the system returns an answer.

So why force the internal communication to use a tokenization scheme designed for English morphology?

### The Solution: A Sovereign Latent Language

```
┌─────────────────────────────────────────────────────────────┐
│                   THE BABEL FISH PROTOCOL                    │
│                                                             │
│   Human World                    Swarm World                │
│   ┌──────────┐                  ┌──────────────┐            │
│   │ "Analiza │   TRANSLATOR     │ [0.23, -0.71 │            │
│   │  este    │ ──────────────►  │  0.44, 0.02, │            │
│   │  código" │   (Encoder)      │  ..., -0.15] │            │
│   └──────────┘                  └──────┬───────┘            │
│                                        │                    │
│                                        ▼                    │
│                                 ┌─────────────┐             │
│                                 │ Prolog Gate  │             │
│                                 │ Routes the   │             │
│                                 │ VECTOR, not  │             │
│                                 │ the TEXT      │             │
│                                 └──────┬──────┘             │
│                                        │                    │
│                              ┌─────────┼──────────┐         │
│                              ▼         ▼          ▼         │
│                           Expert A  Expert B   Expert C     │
│                           (all communicate in               │
│                            AI-native embeddings,            │
│                            never in human tokens)           │
│                              │         │          │         │
│                              └─────────┼──────────┘         │
│                                        ▼                    │
│   ┌──────────┐                  ┌──────────────┐            │
│   │ "El bug  │   TRANSLATOR     │ [0.18, 0.55, │            │
│   │  está en │ ◄──────────────  │  -0.33, 0.89 │            │
│   │  línea   │   (Decoder)      │  ..., 0.41]  │            │
│   │  42"     │                  └──────────────┘            │
│   └──────────┘                                              │
│                                                             │
│   Human language is a CODEC,                                │
│   not the native format.                                    │
└─────────────────────────────────────────────────────────────┘
```

### What AI-Native Tokenization Means

Instead of tokenizing "function" → `[15205]` (BPE token ID for the English word), the system works directly with **learned semantic coordinates** in a continuous latent space:

| Aspect | Human Tokenization | AI-Native (Babel Fish) |
|--------|-------------------|----------------------|
| Vocabulary | 32K-128K discrete tokens (BPE) | Continuous D-dimensional vectors |
| Language bias | English-centric | Language-agnostic |
| Granularity | Subword chunks ("func", "tion") | Semantic concepts (whole meaning) |
| Cross-expert | Each expert needs its own tokenizer | **One shared embedding space** |
| Compression | Lossy (polysemy, ambiguity) | Dense (each dimension is meaningful) |
| Evolution | Fixed at training time | **Evolves with the Common Substrate** |

### The Translator: Multilingual ↔ AI-Native

The Translator is the **only component** in Frankenswarm that understands human language. It sits at the boundary:

```python
# Conceptual architecture
class BabelFishTranslator:
    """Bidirectional Human ↔ AI-Native bridge."""

    def __init__(self):
        self.encoder = SemanticEncoder()    # Human text → latent vector
        self.decoder = SemanticDecoder()    # Latent vector → human text

    def human_to_swarm(self, text: str, source_lang: str = "auto") -> Vector:
        """
        Any human language → AI-native embedding.
        The swarm never sees the human text. Only the vector.
        Spanish, English, Chinese, Arabic — all collapse
        to the same latent point if they mean the same thing.
        """
        return self.encoder.encode(text, lang=source_lang)

    def swarm_to_human(self, vector: Vector, target_lang: str = "es") -> str:
        """
        AI-native embedding → human language of operator's choice.
        The expert's output is language-agnostic.
        The Translator chooses how to say it.
        """
        return self.decoder.decode(vector, lang=target_lang)
```

### Why This Matters

1. **True multilingual**: A Spanish-speaking operator and a Japanese-speaking operator get the same quality — the experts don't care about language.
2. **Zero translation loss between experts**: Expert A's output is already in the right format for Expert B. No text serialization/deserialization.
3. **Evolvable**: When the Common Substrate upgrades its embedding dimension (e.g., 384D → 768D), only the Translator's encoder/decoder re-trains. The experts adapt via their Adapter layers.
4. **Compression**: A human sentence of 20 tokens becomes a single 384D vector. Inter-expert communication is O(D) instead of O(tokens).

### Phase A (PoC): Use `all-MiniLM-L6-v2` as the Translator. 384D, pre-trained, good enough to validate routing.

### Phase B (Arena): Train a custom Translator optimized for the swarm's specific domain vocabulary. The embedding space becomes truly AI-native — no longer constrained by a model trained on English Wikipedia.

---

## The Full Picture

```
  Human     ┌───────────┐     ┌────────────────────────────────────────┐     ┌───────────┐     Human
  Input ──► │ TRANSLATOR│ ──► │            THE SWARM                    │ ──► │ TRANSLATOR│ ──► Output
  (any      │ (Encoder) │     │                                        │     │ (Decoder) │     (any
  language) └───────────┘     │  ┌────────┐  ┌────────┐  ┌────────┐   │     └───────────┘     language)
                              │  │Expert A│  │Expert B│  │Expert C│   │
                              │  │┌──────┐│  │┌──────┐│  │┌──────┐│   │
                              │  ││Common││  ││Common││  ││Common││   │  ← SHARED (same weights)
                              │  │├──────┤│  │├──────┤│  │├──────┤│   │
                              │  ││Adapt.││  ││Adapt.││  ││Adapt.││   │  ← PER-EXPERT (bridge)
                              │  │├──────┤│  │├──────┤│  │├──────┤│   │
                              │  ││Spec. ││  ││Spec. ││  ││Spec. ││   │  ← PER-EXPERT (NEAT evolves)
                              │  │└──────┘│  │└──────┘│  │└──────┘│   │
                              │  └────────┘  └────────┘  └────────┘   │
                              │        ▲          ▲          ▲        │
                              │        └──────────┼──────────┘        │
                              │             Prolog Gate               │
                              │         (routes AI-native vectors,    │
                              │          never human text)            │
                              └────────────────────────────────────────┘
```

