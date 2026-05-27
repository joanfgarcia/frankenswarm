# Frankenswarm — Heterogeneous Hardware Mixture of Experts

* [English](#english)
* [Español](#español)

---

<a name="english"></a>
# English

> [!WARNING]
> **Active Research Phase**: This project is currently in the **experimental training and research phase** (Phase B: The Arena). We are training 1.58-bit ternary models in emergent signaling games to solve logical generalization bottlenecks. The full physically-distributed cascading hardware MoE is a planned target, not yet production-ready.
>
> **Fase de Investigación Activa**: Este proyecto se encuentra en **fase experimental de entrenamiento e investigación** (Fase B: La Arena). Estamos entrenando modelos ternarios de 1.58 bits en juegos de señalización emergente para resolver cuellos de botella de generalización lógica. La cascada MoE distribuida físicamente en hardware real es un objetivo planificado, no listo para producción.

> *"Don't wait for the perfect hardware. Conquer the hardware you have."*

## What Is This?

Frankenswarm is a **physically-distributed Mixture of Experts** that turns every accelerator in your machine into a specialized inference node. Instead of running one massive model on a single large GPU, it distributes multiple models across **all available silicon** — discrete GPU, integrated GPU, CPU, and Neural Processing Unit (NPU) — assigning queries to the most resource-efficient and capable expert dynamically.

A SWI-Prolog router classifies intent. A Python orchestrator manages the active topology. Specialized PEFT micro-experts and ternary BitNet 1.58b models run on the NPU at 2 watts, while the GPU remains suspended, firing only to handle heavyweight reasoning queries.

**The system adapts its active routing and fine-tunes specialized adapters dynamically.**

---

## The Equation

```
Frankenswarm = Hardware MoE (GPU + iGPU + CPU + NPU)
             + Prolog Gate (intent → silicon routing)
             + Python Orchestrator (dynamic adapter management)
             + PEFT & BitNet 1.58b (task-specific micro-experts)
             + NAS (Network Architecture Search for thresholds and fallbacks)
             + Knowledge Distillation (heavy model → micro-expert)
```

---

## The Hardware (Measured — Strix Point, 2026-05-22)

| Silicon | Model | tok/s | Power | Role |
|---------|-------|:-----:|:-----:|------|
| RTX 5070 (CUDA) | Falcon3-10B | 23 | ~80W | Heavyweight Reasoning |
| Ryzen AI 9 (CPU) | Falcon3-10B | 12.8 | ~45W | Long Context Scholar |
| Radeon 880M (Vulkan) | Falcon3-10B | 4.8 | ~15W | Background Sentinel |
| XDNA2 NPU | Qwen3-0.6B | 96 | ~2W | Fast Scout / NEAT Breeder |
| XDNA2 NPU | Qwen3-8B | 10.6 | ~2W | Orchestrator / Triage |

---

## The Three Adaptation Loops

| Loop | Timescale | What Mutates |
|------|-----------|--------------|
| **FAST** (Prolog) | Per-query (~200ms) | Routing policy — which silicon handles what |
| **MEDIUM** (Orchestrator) | Per-session (~minutes) | Activation thresholds and cascade fallback weights (NAS) |
| **SLOW** (Metabolic) | Per-sleep (~hours) | Local fine-tuning (QLoRA) & Ternary BitNet SVD evolution |

---

## Why Python + ONNX/GGUF?

By sticking to standard python-based orchestrations and standardized formats like ONNX and GGUF, we gain native access to hardware accelerator runtimes (CUDA, ROCm, Vulkan, NPU) without adding compilation overhead. Topology is managed dynamically as a standard JSON schema, and hot-swapping adapters is a simple lightweight weight-loading call rather than hot-swapping runtimes.

→ See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full thesis.

---

## Energy Sovereignty

| Strategy | Energy per 1000 queries |
|----------|:-----------------------:|
| All-Cloud (GPT-4) | ~5-10 kWh |
| All-CUDA (local) | ~2.2 kWh |
| **Frankenswarm Cascade** | **~0.05 kWh** |

For queries where a specialized local micro-expert (0.6B-1.5B) provides acceptable quality, we run at 2 watts on the NPU. The GPU (80W+) is kept in sleep mode, firing only when the cascade's confidence threshold is violated. This makes the local swarm **up to 200x more efficient** for routine triage and classification tasks.

---

## Project Status

| Phase | Status |
|-------|--------|
| Translator (text ↔ vectors) | ✅ Done |
| Ghost Swarm (mocks + routing) | ✅ Done |
| Hardware Discovery (benchmarks) | ✅ Done |
| Prolog Gate (SWI-Prolog intent router) | ✅ Done |
| BitNet 1.58b (Ternary Signaling Game) | ✅ Done (Grade 1 Arithmetic System) |
| Minion Swarm Scheduler & TUI | ✅ Done (cgroups background training & Control TUI) |
| Aggregator (multi-expert consensus) | 🔲 Planned |
| LoRA Specialization & Distillation | 🔲 Planned |

→ See [ROADMAP.md](docs/ROADMAP.md) for the full plan.

---

## Docs

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) — Full system design, Lisp thesis, lifecycle diagrams
- [EXPERTS_ROSTER.md](docs/EXPERTS_ROSTER.md) — Expert ↔ silicon bindings with benchmarks
- [ROADMAP.md](docs/ROADMAP.md) — Phase-by-phase implementation plan
- [REFERENCES.md](docs/REFERENCES.md) — Prior art, bibliography, and novelty analysis
- [PROLOG_RESEARCH.md](docs/PROLOG_RESEARCH.md) — Prolog router research and design
- [FRANKENSWARM_SCHEMA.md](docs/FRANKENSWARM_SCHEMA.md) — Visual topology diagrams

---

## Part of the Red-Pill Ecosystem

Frankenswarm integrates with [Red-Pill](../sharing) via the `ProviderRegistry`. The `FastFlowLMInferenceProvider` (NPU), `SipInferenceProvider` (CUDA/CPU), and `BitNetInferenceProvider` (ternary) are already wired. Frankenswarm's Prolog Gate is the evolutionary successor to the static `InferenceRouter`.

---

<a name="español"></a>
# Español

> [!WARNING]
> **Fase de Investigación Activa**: Este proyecto se encuentra en **fase experimental de entrenamiento e investigación** (Fase B: La Arena). Estamos entrenando modelos ternarios de 1.58 bits en juegos de señalización emergente para resolver cuellos de botella de generalización lógica. La cascada MoE distribuida físicamente en hardware real es un objetivo planificado, no listo para producción.
>
> **Active Research Phase**: This project is currently in the **experimental training and research phase** (Phase B: The Arena). We are training 1.58-bit ternary models in emergent signaling games to solve logical generalization bottlenecks. The full physically-distributed cascading hardware MoE is a planned target, not yet production-ready.

> *"No esperes a tener el hardware perfecto. Conquista el hardware que tienes."*

## ¿Qué es esto?

Frankenswarm es una **Mezcla de Expertos físicamente distribuida** que convierte cada acelerador de tu máquina en un nodo de inferencia especializado. En lugar de ejecutar un único modelo masivo en una GPU gigante, distribuye múltiples modelos a lo largo de **todo el silicio disponible** (GPU dedicada, GPU integrada, CPU y NPU), asignando las consultas al experto más eficiente y capaz de forma dinámica.

Un enrutador SWI-Prolog clasifica la intención de la consulta. Un orquestador en Python gestiona la topología activa. Micro-expertos especializados mediante PEFT y modelos ternarios BitNet 1.58b se ejecutan en la NPU a solo 2 vatios, mientras la GPU permanece suspendida, activándose únicamente para procesar consultas complejas de razonamiento pesado.

**El sistema adapta su enrutamiento activo y afina los adaptadores especializados de manera dinámica.**

---

## La Ecuación

```
Frankenswarm = Hardware MoE (GPU + iGPU + CPU + NPU)
             + Prolog Gate (enrutamiento intención → silicio)
             + Orquestador Python (gestión dinámica de adaptadores)
             + PEFT & BitNet 1.58b (micro-expertos dedicados)
             + NAS (Búsqueda de Arquitectura para umbrales y caídas)
             + Destilación de Conocimiento (modelo pesado → micro-experto)
```

---

## El Hardware (Medido — Strix Point, 22-05-2026)

| Silicio | Modelo | tok/s | Consumo | Rol / Tarea |
|---------|-------|:-----:|:-----:|------|
| RTX 5070 (CUDA) | Falcon3-10B | 23 | ~80W | Razonamiento Complejo |
| Ryzen AI 9 (CPU) | Falcon3-10B | 12.8 | ~45W | Erudito en Contexto Largo |
| Radeon 880M (Vulkan) | Falcon3-10B | 4.8 | ~15W | Centinela en Segundo Plano |
| NPU XDNA2 | Qwen3-0.6B | 96 | ~2W | Explorador Rápido / Evolución SVD |
| NPU XDNA2 | Qwen3-8B | 10.6 | ~2W | Orquestador / Clasificador de Entrada |

---

## Los Tres Bucles de Adaptación

| Bucle | Escala Temporal | Qué Mutación Ocurre |
|------|-----------|--------------|
| **RÁPIDO** (Prolog) | Por consulta (~200ms) | Política de enrutamiento: qué silicio atiende qué tarea |
| **MEDIO** (Orchestrator) | Por sesión (~minutos) | Umbrales de activación y pesos de cascada de fallo (NAS) |
| **LENTO** (Metabólico) | Durante el sueño (~horas) | Ajuste fino local (QLoRA) y evolución SVD en BitNet ternario |

---

## ¿Por qué Python + ONNX/GGUF?

Al mantenernos fieles al estándar de orquestación en Python y formatos unificados como ONNX y GGUF, obtenemos acceso nativo a los runtimes de aceleración de hardware (CUDA, ROCm, Vulkan, NPU) sin añadir sobrecostes de compilación. La topología se gestiona dinámicamente como un esquema JSON estándar, y el intercambio de adaptadores es una llamada ligera de carga de pesos en lugar de un reinicio de entornos.

→ Ver [ARCHITECTURE.md](docs/ARCHITECTURE.md) (en inglés) para la tesis completa.

---

## Soberanía Energética

| Estrategia | Energía por 1000 consultas |
|----------|:-------------------------:|
| Nube (GPT-4) | ~5-10 kWh |
| CUDA Puro (local) | ~2.2 kWh |
| **Cascada Frankenswarm** | **~0.05 kWh** |

Para las consultas donde un micro-experto local especializado (0.6B-1.5B) proporciona una calidad aceptable, ejecutamos a 2 vatios en la NPU. La GPU (80W+) se mantiene en suspensión, activándose solo si se viola el umbral de confianza de la cascada. Esto hace que el enjambre local sea **hasta 200 veces más eficiente** en tareas rutinarias de clasificación y triaje.

---

## Estado del Proyecto

| Fase | Estado |
|-------|--------|
| Traductor (texto ↔ vectores) | ✅ Completado |
| Enjambre Fantasma (mocks + rutas) | ✅ Completado |
| Descubrimiento de Hardware (benchmarks) | ✅ Completado |
| Prolog Gate (enrutador de SWI-Prolog) | ✅ Completado |
| BitNet 1.58b (Juego de Señalización Ternaria) | ✅ Completado (Sistema Aritmético Grado 1) |
| Programador de Miniones & TUI | ✅ Completado (entrenamiento en segundo plano cgroups y TUI de control) |
| Agregador (consenso multi-experto) | 🔲 Planificado |
| Especialización LoRA y Destilación | 🔲 Planificado |

→ Ver [ROADMAP.md](docs/ROADMAP.md) (en inglés) para el plan completo.

---

## Documentos

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) — Diseño completo del sistema, tesis Lisp, diagramas de ciclo de vida.
- [EXPERTS_ROSTER.md](docs/EXPERTS_ROSTER.md) — Enlaces experto ↔ silicio con benchmarks.
- [ROADMAP.md](docs/ROADMAP.md) — Plan de implementación paso a paso.
- [REFERENCES.md](docs/REFERENCES.md) — Estado del arte, bibliografía y análisis de novedad.
- [PROLOG_RESEARCH.md](docs/PROLOG_RESEARCH.md) — Investigación y diseño del enrutador Prolog.
- [FRANKENSWARM_SCHEMA.md](docs/FRANKENSWARM_SCHEMA.md) — Diagramas visuales de topología.

---

## Parte del Ecosistema Red-Pill

Frankenswarm se integra con [Red-Pill](../sharing) a través de `ProviderRegistry`. Los proveedores `FastFlowLMInferenceProvider` (NPU), `SipInferenceProvider` (CUDA/CPU) y `BitNetInferenceProvider` (ternario) ya están conectados. El Prolog Gate de Frankenswarm es el sucesor evolutivo del `InferenceRouter` estático.
