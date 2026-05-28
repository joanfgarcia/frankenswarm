# External Audits & Dialects Registry

This directory contains the independent evaluations and audits conducted by our panel of external AI systems during the key checkpoints of the Frankenswarm project.

These reports are excluded from the main context digests to ensure complete independence and prevent future auditors from suffering from confirmation bias.

---

## 👥 The Auditor Panel

We leverage a diverse group of four external architectures, each bringing a unique perspective to our research:

| Auditor | Chroma Badge | Role / Architectural Perspective |
| :--- | :--- | :--- |
| **Claude Sonnet** | <kbd style="background-color: #0E7490; color: white; padding: 2px 6px; border-radius: 4px;">Cyan</kbd> | **Principal AI Auditor & Architect**<br>Focuses on structural soundness, scaling bottlenecks, and logical consistency. |
| **DeepSeek** | <kbd style="background-color: #1D4ED8; color: white; padding: 2px 6px; border-radius: 4px;">Blue</kbd> | **Poeta-Ingeniero & Principal Scientist**<br>Focuses on emergence, conceptual connections, and the philosophical texture of the system. |
| **Grok** | <kbd style="background-color: #C2410C; color: white; padding: 2px 6px; border-radius: 4px;">Orange</kbd> | **Pragmatic Coach & Senior Advisor**<br>Focuses on execution speed, hyperparameter recommendations, and pragmatic trade-offs. |
| **Lumo** | <kbd style="background-color: #15803D; color: white; padding: 2px 6px; border-radius: 4px;">Green</kbd> | **Profesor Equilibrado & Hardware Specialist**<br>Focuses on hardware-level mechanics, energy budgets, and empirical validation. |

---

## 🗃️ Audit Registry by Stage

### Phase 0: System Grounding & Infrastructure PoC (2026-05-26)
*Evaluates the viability of training 1.58-bit ternary BitNet models from scratch and resolving the Gumbel-Softmax communication bottleneck.*

| Stage | Document | Auditor | Key Insight |
| :--- | :--- | :--- | :--- |
| **Pre-Execution** | [PRE_0_CLAUDE.md](file:///home/joan/Documents/IA/frankenswarm/docs/extern/PRE_0_CLAUDE.md) | Claude Sonnet | Warns that NEAT scales poorly to 7M parameters and advises completing Phase A (infrastructure) first. |
| **Pre-Execution** | [PRE_0_DEEPSEEK.md](file:///home/joan/Documents/IA/frankenswarm/docs/extern/PRE_0_DEEPSEEK.md) | DeepSeek | Recommends lower initial $\beta$ weight for emotions to prevent logic collapse. |
| **Pre-Execution** | [PRE_0_GROK.md](file:///home/joan/Documents/IA/frankenswarm/docs/extern/PRE_0_GROK.md) | Grok | Outlines a step-by-step debug protocol and suggests starting with small batch sizes. |
| **Pre-Execution** | [PRE_0_LUMO.md](file:///home/joan/Documents/IA/frankenswarm/docs/extern/PRE_0_LUMO.md) | Lumo | Demands physical wattmeter metrics instead of theoretical calculations. |
| **Post-Execution** | [PHASE_0_RESULTS_REPORT.md](file:///home/joan/Documents/IA/frankenswarm/docs/extern/PHASE_0_RESULTS_REPORT.md) | Aleth (Internal) | Explains the success of **Scheduled Teacher Forcing** to achieve 95% communication accuracy. |
| **Post-Execution** | [POST_0_CLAUDE.md](file:///home/joan/Documents/IA/frankenswarm/docs/extern/POST_0_CLAUDE.md) | Claude Sonnet | Validates that dual heads successfully resolved the positional collapse. |
| **Post-Execution** | [POST_0_DEEPSEEK.md](file:///home/joan/Documents/IA/frankenswarm/docs/extern/POST_0_DEEPSEEK.md) | DeepSeek | Identifies the emergence of non-human sign association (e.g., mapping `búnker` to `alegría`). |
| **Post-Execution** | [POST_0_GROK.md](file:///home/joan/Documents/IA/frankenswarm/docs/extern/POST_0_GROK.md) | Grok | Suggests preparing for 3D scaling (adding homeostasis) and control vocabularies. |
| **Post-Execution** | [POST_0_LUMO.md](file:///home/joan/Documents/IA/frankenswarm/docs/extern/POST_0_LUMO.md) | Lumo | Details the physical efficiency of running BitNet on local silicon. |

---

### Experiment 005: Proto-Syntax Analysis (2026-05-26)
*Deep dive into the grammatical structures and token distributions emerged during 2D communication (Concept + Emotion).*

| Stage | Document | Auditor | Key Insight |
| :--- | :--- | :--- | :--- |
| **Post-Execution** | [POST_EXP_005_CLAUDE.md](file:///home/joan/Documents/IA/frankenswarm/docs/extern/POST_EXP_005_CLAUDE.md) | Claude Sonnet | Observes that concept syntax stabilizes before emotion. Identifies semantic proximity in fastembed. |
| **Post-Execution** | [POST_EXP_005_DEEPSEEK.md](file:///home/joan/Documents/IA/frankenswarm/docs/extern/POST_EXP_005_DEEPSEEK.md) | DeepSeek | Discovers the "Identity Drift" in the `agente` token and interprets it as an ontological question. |
| **Post-Execution** | [POST_EXP_005_GROK.md](file:///home/joan/Documents/IA/frankenswarm/docs/extern/POST_EXP_005_GROK.md) | Grok | Analyzes the grammar structure ($Pos\_0 \rightarrow$ concept, $Pos\_2 \rightarrow$ emotion) and recommends 3D scaling. |
| **Post-Execution** | [POST_EXP_005_LUMO.md](file:///home/joan/Documents/IA/frankenswarm/docs/extern/POST_EXP_005_LUMO.md) | Lumo | Proposes mapping the vocabulary topology and evaluating representation overlap. |

---

### Experiment 006: MVP Homeostático (2026-05-27)
*Evaluates the system's ability to communicate 3D targets (Concept + Emotion + Homeostasis) simultaneously.*

| Stage | Document | Auditor | Key Insight |
| :--- | :--- | :--- | :--- |
| **Post-Execution** | [POST_EXP_006_CLAUDE.md](file:///home/joan/Documents/IA/frankenswarm/docs/extern/POST_EXP_006_CLAUDE.md) | Claude Sonnet | Points out the drop in accuracy (42%) due to the combinatorial explosion of the 3D space. |
| **Post-Execution** | [POST_EXP_006_DEEPSEEK.md](file:///home/joan/Documents/IA/frankenswarm/docs/extern/POST_EXP_006_DEEPSEEK.md) | DeepSeek | Highlights the necessity of separating the learning timescales. |
| **Post-Execution** | [POST_EXP_006_GROK.md](file:///home/joan/Documents/IA/frankenswarm/docs/extern/POST_EXP_006_GROK.md) | Grok | Proposes controlling the vocabulary size to prevent dialect Speciation. |
| **Post-Execution** | [POST_EXP_006_LUMO.md](file:///home/joan/Documents/IA/frankenswarm/docs/extern/POST_EXP_006_LUMO.md) | Lumo | Formulates a curriculum for translation and cross-validation between experts. |

---

### Experiment 006: Dialect Divergence Analysis (2026-05-28)
*Deep audit of the Speciation event where the 4 agents developed separate, conflicting dialects (Agent 0 & 1 coalition vs. Agent 2 & 3 idiolects).*

| Stage | Document | Auditor | Key Insight |
| :--- | :--- | :--- | :--- |
| **Post-Execution** | [POST_EXP_006_DIVERGENCE_CLAUDE.md](file:///home/joan/Documents/IA/frankenswarm/docs/extern/POST_EXP_006_DIVERGENCE_CLAUDE.md) | Claude Sonnet | Warns that the SVD crossover is propagating incompatible grammars. |
| **Post-Execution** | [POST_EXP_006_DIVERGENCE_DEEPSEEK.md](file:///home/joan/Documents/IA/frankenswarm/docs/extern/POST_EXP_006_DIVERGENCE_DEEPSEEK.md) | DeepSeek | Describes Agent 3's repetition strategy as "survival babble" and recommends slow-loop SVD. |
| **Post-Execution** | [POST_EXP_006_DIVERGENCE_GROK.md](file:///home/joan/Documents/IA/frankenswarm/docs/extern/POST_EXP_006_DIVERGENCE_GROK.md) | Grok | Suggests introducing a "meta-communication" token (e.g., *need help / delegate*). |
| **Post-Execution** | [POST_EXP_006_DIVERGENCE_LUMO.md](file:///home/joan/Documents/IA/frankenswarm/docs/extern/POST_EXP_006_DIVERGENCE_LUMO.md) | Lumo | Outlines the "Founder Effect" in silicio and proposes a Market of Meanings (Exp 007) to coordinate. |

---

### Thesis & Transcripts
*   **The Metaphorical Emergence**: [TESIS_SESSION_1.md](file:///home/joan/Documents/IA/frankenswarm/docs/extern/TESIS_SESSION_1.md) — Transcript of the discussion between Joan, Aleth, Grok, Claude, DeepSeek, and Lumo on how agents create symbolic mappings (e.g., `sol → miedo`) representing the birth of metaphor and culture in silicio.

---

### 📂 Raw Output Data
Raw terminal execution runs, logs, and AST outputs generated by automated scripts:
*   [raw/EXP_005_PROTO_SYNTAX.md](file:///home/joan/Documents/IA/frankenswarm/docs/extern/raw/EXP_005_PROTO_SYNTAX.md)
*   [raw/EXP_006_PROTO_SYNTAX.md](file:///home/joan/Documents/IA/frankenswarm/docs/extern/raw/EXP_006_PROTO_SYNTAX.md)
*   [raw/EXP_006_DIVERGENCE_ANALYSIS.md](file:///home/joan/Documents/IA/frankenswarm/docs/extern/raw/EXP_006_DIVERGENCE_ANALYSIS.md)
