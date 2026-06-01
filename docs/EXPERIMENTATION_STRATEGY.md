# Experimentation Strategy & Swarm Roadmap (5-Layer Architecture)

This document establishes the strategic roadmap for the Frankenswarm experimentation pipeline, diagnoses the current state of the system, details the transition to a non-human internal Swarm Language, and provides realistic time expectations.

---

## 1. Diagnostic: Where Do We Stand?

Before scaling to complex tasks, we must address the results of **EXP_006**:
*   **The Tower of Babel (Dialect Divergence)**: When we expanded the target space to 3D (Concept, Emotion, Homeostasis), the joint accuracy dropped to **11%**. The agents diverged into independent dialects (Agent 0 & 1 formed a linguistic coalition sharing 88% overlap, while Agent 2 & 3 diverged, with Agent 3 reverting to survival-based repetition or "balbuceo").
*   **The Evolutionary Collision**: The SVD crossover was acting on a fast timescale (every epoch), changing agent parameters (the biology) faster than the agents could coordinate their messages (the culture). 

### Veredicto: **Necesitamos validar más el sustrato de comunicación.**
No estamos listos para tareas complejas porque la red de intercambio de información no es estable bajo presión evolutiva. Primero debemos resolver el **dilema cultural vs. biológico**.

---

## 2. The 5-Layer Swarm Architecture

We fully adopt the vision that **the internal swarm language must not be based on human language**. It must be a highly compressed, symbolic intermediate representation (Sovereign Swarm Language) evolved by the agents themselves.

To bridge this non-human tongue with the outside world, we expand the 4-layer expert model into a **5-Layer Swarm Architecture**:

```
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                         5-LAYER SWARM ARCHITECTURE                          │
 ├─────────────────────────────────────────────────────────────────────────────┤
 │                                                                             │
 │                      [ HUMAN OPERATOR (Input Prompt) ]                      │
 │                                     │                                       │
 │  ┌──────────────────────────────────▼────────────────────────────────────┐  │
 │  │ LAYER 1: INBOUND BOUNDARY CODEC (Translator)                           │  │
 │  │ Maps natural human language (e.g., Spanish) into discrete, highly    │  │
 │  │ compressed Sovereign Swarm Token IDs (2 bytes/token).                 │  │
 │  └──────────────────────────────────┬────────────────────────────────────┘  │
 │                                     │ (Sovereign Swarm Tokens)              │
 │                                     ▼                                       │
 │              ┌──────────────────────────────────────────────┐               │
 │              │              THE INTERNAL SWARM              │               │
 │              │                                              │               │
 │              │  ┌────────────────────────────────────────┐  │               │
 │              │  │ LAYER 2: INBOUND TRANSLATOR ($W_E$)    │  │ (Per Expert)  │
 │              │  │ Maps Swarm Tokens into expert-specific │  │               │
 │              │  │ continuous latent space.               │  │               │
 │              │  └──────────────────┬─────────────────────┘  │               │
 │              │                     ▼                        │               │
 │              │  ┌────────────────────────────────────────┐  │ (Per Expert)  │
 │              │  │ LAYER 3: TERNARY SPECIALIST CORE      │  │               │
 │              │  │ Evolved ternary weights {-1, 0, 1}     │  │               │
 │              │  │ processing logic at 96 tok/s on NPU.   │  │               │
 │              │  └──────────────────┬─────────────────────┘  │               │
 │              │                     ▼                        │               │
 │              │  ┌────────────────────────────────────────┐  │ (Per Expert)  │
 │              │  │ LAYER 4: OUTBOUND TRANSLATOR ($W_{LM}$) │  │               │
 │              │  │ Projects expert latent features back   │  │               │
 │              │  │ into Sovereign Swarm Token IDs.        │  │               │
 │              │  └────────────────────────────────────────┘  │               │
 │              └──────────────────────┬───────────────────────┘               │
 │                                     │ (Sovereign Swarm Tokens)              │
 │                                     ▼                                       │
 │  ┌───────────────────────────────────────────────────────────────────────┐  │
 │  │ LAYER 5: OUTBOUND BOUNDARY CODEC (Decoder)                            │  │
 │  │ Decodes Sovereign Swarm Token IDs back into human natural language    │  │
 │  │ or actionable tools.                                                  │  │
 │  └──────────────────────────────────┬────────────────────────────────────┘  │
 │                                     │                                       │
 │                     [ HUMAN OPERATOR (Output Text) ]                      │
 └─────────────────────────────────────────────────────────────────────────────┘
```

### The Translation Boundaries (Layer 1 & 5)
Rather than mapping all human languages directly to Swarm tokens, we implement a **Single-Pivot Translation Pipeline**:
1.  **Pivot Language**: We select one human language containing the highest structural richness and density (e.g., English or Spanish) as the **Pivot**.
2.  **Edge Translators**: Standard, local LLMs (e.g., Qwen-1.5B or Llama-3-8B) map any human input (French, Japanese, German) to the Pivot language.
3.  **Inbound Codec (Layer 1)**: Maps the Pivot language into the **Sovereign Swarm Tokens**.
4.  **Outbound Codec (Layer 5)**: Maps the output Swarm tokens back into the Pivot language, which is then translated back to the operator's language if needed.

This protects the Swarm Core from dealing with grammatical/syntactical human noise, keeping the NPU processing extremely fast (2 bytes per token, no memory bloating).

---

## 3. Experimentation Plan (Stabilization ➔ Actions ➔ Scaling)

We will execute 4 targeted experiments to stabilize the communication protocol before introducing action tasks.

### 🧪 Exp. 007: The Market of Meanings (Babel Stabilization)
*   **Goal**: Force convergence of a shared 3D language across 4 agents.
*   **Mechanic**:
    *   **Cross-Pollination**: During the communication phase, Speaker A must speak to Listener B (random pairing per batch), preventing isolated pair-coalitions.
    *   **Dual-Timescale Loop**: Lock SVD weight crossovers during the autonomy phase. Allow parameters to adapt slowly via gradients, running SVD only when the overall population consensus ($\rho_{joint}$) remains stable above 80% for 3 consecutive epochs.
    *   **Pragmatic Focus**: Give a higher loss weight ($\beta=1.5$) to targets with high homeostasis (e.g., urgency, pain) to force rapid convergence on critical concepts.

### 🧪 Exp. 008: The Action Bridge
*   **Goal**: Transition from description `(concepto, emoción)` to action.
*   **Mechanic**: Introduce a small set of action tokens (e.g., `[ACTION_READ]`, `[ACTION_WRITE]`, `[ACTION_DELEGATE]`). The target is no longer a concept, but a command sequence. The listener agent must trigger a mock function execution based on the received tokens.

### 🧪 Exp. 009: The Grounding Boundary (Layer 1 PoC)
*   **Goal**: First validation of Layer 1.
*   **Mechanic**: Train a small encoder model to map human sentences describing logic tasks into the Swarm Tokens evolved in Exp. 007.

### 🧪 Exp. 010: PopuLoRA Self-Play Arena
*   **Goal**: First autonomous growth loop.
*   **Mechanic**: Implement the 3 AM sleep cycle. A heavy CUDA model generates exams (Grades 1 to 5), and the NPU-bound micro-experts attempt them, saving their checkpoints only if they achieve a passing grade ($\rho \ge 80\%$).

---

## 4. Expectations of Time (Realistic Milestones)

Based on our current VRAM limits (capping at ~836MB per training arena) and local compilation capabilities:

```
T+0 (Hoy)          T+1 Semana            T+1 Mes                    T+3 Meses
   │                    │                    │                          │
   ▼                    ▼                    ▼                          ▼
[Estabilizar] ──► [Puente de Acción] ──► [Modularidad 5 Capas] ──► [Crecimiento Autónomo]
  (Exp. 007)       (Tareas Simples)        (Hot-Swap de Expertos)     (Autonomía / 3 AM)
```

1.  **T+1 Semana: Estabilización del Lenguaje (Exp. 007 & 008)**
    *   *Resultado*: Los 4 agentes de 7M en el NPU se comunican con $>90\%$ de precisión en 3D y pueden enviarse comandos de acción sencillos sin fragmentarse en dialectos.
2.  **T+2 Semanas: Primeras Tareas Reales (Exp. 009)**
    *   *Resultado*: Capacidad de escribir una consulta en lenguaje humano, traducirla al lenguaje del enjambre (Capa 1), y hacer que el enjambre ejecute una tarea simple de lógica o formateo en la máquina local.
3.  **T+1 Mes: Swarm Modular (Capa 2, 3, 4)**
    *   *Resultado*: Implementación del hot-swap de adaptadores LoRA. Podremos cargar y descargar expertos (Matemáticas, Lógica, Resumen) en caliente en el NPU en menos de 50ms, coordinados por el router Prolog.
4.  **T+3 Meses: Crecimiento y Evolución Autónoma**
    *   *Resultado*: El sistema es completamente autónomo durante el ciclo metabólico nocturno. Se autoevalúa, detecta derivas lógicas, muta sus topologías con NEAT/Net2Net y mejora sin intervención humana.
