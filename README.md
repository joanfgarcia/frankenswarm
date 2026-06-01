# Frankenswarm — Energy-Sovereign Heterogeneous MoE & Developmental Cognitive Agents

* [English](#english)
* [Español](#español)

---

<a name="english"></a>
# English

> [!WARNING]
> **Active Research Phase (Phase B: The Arena)**: This repository contains the code for both the **Heterogeneous MoE infrastructure (Phase A)** and the **Developmental Ternary Learning research (Phase B)** presented in our paper. Currently, we are focused on training 1.58-bit ternary models in emergent signaling games. The physically-distributed cascading MoE routing is the deployment target for these self-growing agents.
>
> **Fase de Investigación Activa (Fase B: La Arena)**: Este repositorio contiene el código tanto de la **infraestructura MoE Heterogénea (Fase A)** como de la **investigación en Aprendizaje Ternario Ontogenético (Fase B)** expuesta en nuestro artículo. Actualmente, estamos enfocados en entrenar modelos ternarios de 1.58 bits en juegos de señalización emergente. El enrutamiento MoE en cascada físicamente distribuido es el objetivo de despliegue para estos agentes autorregulados.

## What Is This?

Frankenswarm is a unified project for **energy-sovereign edge intelligence**. It combines two complementary layers: the physical hardware orchestrator and the cognitive learning mind.

1. **Phase A: The Physical Skeleton (Heterogeneous MoE)**
   A physically-distributed Mixture of Experts that turns every accelerator on your local machine into a specialized inference node. It dynamically distributes workloads across **all available silicon** — discrete GPU (CUDA), integrated GPU (Vulkan), CPU, and Neural Processing Unit (NPU) — using a **SWI-Prolog gate** to classify query intent and route tasks to the most energy-efficient expert.

2. **Phase B: The Cognitive Pilot (Developmental Learning)**
   A developmental learning framework for ultra-lightweight, 1.58-bit ternary models designed to live on low-power edge nodes (such as NPUs running at 2W). Instead of starting at a fixed size, these models **grow autonomously** via pain-driven neurogenesis (Net2WiderNet) and adapt through selective attention plasticity, using linguistic semantic primes as their foundation.

---

## Energy Sovereignty

By keeping the power-hungry discrete GPU (80W+) in suspension and routing routine triage or low-complexity queries to specialized local micro-experts running on the NPU (2W), Frankenswarm achieves up to **200x energy efficiency gains** over cloud models and **40x** over local GPU-only execution.

| Inference Strategy | Energy per 1,000 Queries | Relative Efficiency |
|--------------------|:------------------------:|:-------------------:|
| All-Cloud (GPT-4)  | ~5 - 10 kWh              | 1x (Baseline)       |
| All-CUDA (Local GPU) | ~2.2 kWh                | ~3.4x               |
| **Frankenswarm Cascade** | **~0.05 kWh**       | **~150x - 200x**    |

---

## The System Equation

```
Frankenswarm = [Phase A: Physical Skeleton] + [Phase B: Cognitive Pilot]

  Phase A (Hardware MoE) = SWI-Prolog Gate (Intent Classification)
                           + Python Orchestrator (Topology & Hot-Swap)
                           + Heterogeneous Execution (CUDA / Vulkan / CPU / NPU)

  Phase B (Cognitive Mind) = Ternary BitNet (~586KB model size)
                           + NSM Semantic Primes (Wierzbicka-inspired embeddings)
                           + Emergent Metacognition (Think-then-verify loop)
                           + Pain-Driven Neurogenesis (Net2WiderNet scaling)
                           + Selective Plasticity (Plastic Attention / Frozen Knowledge)
```

---

## The Hardware Substrate (Measured — Strix Point, 2026-05-22)

| Accelerator | Local Model | Speed | Power | Dynamic Role |
|-------------|-------------|:-----:|:-----:|--------------|
| RTX 5070 (CUDA) | Falcon3-10B | 23 tok/s | ~80W | Heavyweight Reasoning / Validation |
| Ryzen AI 9 (CPU) | Falcon3-10B | 12.8 tok/s | ~45W | Long Context Scholar / Retrieval |
| Radeon 880M (Vulkan) | Falcon3-10B | 4.8 tok/s | ~15W | Background Sentinel |
| XDNA2 NPU (ONNX) | Qwen3-0.6B | 96 tok/s | ~2W | Fast Scout / Micro-Expert |
| XDNA2 NPU (ONNX) | Qwen3-8B | 10.6 tok/s | ~2W | Orchestrator / Intent Triage |

---

## The Three Adaptation Loops

| Loop | Timescale | Mutating Mechanism | Goal |
|------|-----------|--------------------|------|
| **FAST** (Prolog) | Per-query (~200ms) | Routing policy | Match query intent to the optimal silicon tier |
| **MEDIUM** (Orchestrator) | Per-session (~minutes) | Activation thresholds | Adjust confidence margins and cascade fallbacks (NAS) |
| **SLOW** (Metabolic) | Per-sleep (~hours) | Local weights & structure | QLoRA fine-tuning / Ternary neurogenesis evolution |

---

## Phase B: The Research Paper

The developmental learning layer (Phase B) is detailed in our paper: **"From Semantic Primes to Neurogenesis: Developmental Learning in Ternary Neural Networks"**.

Key research features implemented in this repository (`src/bitnet/`):
* **Deterministic Semantic Primes**: 65 ternary glyph embeddings representing Wierzbicka's NSM primes [1]. Achieves 100% compositional accuracy and 7.7x training efficiency over learned embeddings.
* **Metacognitive Verification**: A two-phase "think, then verify" loop. The cosine similarity between Phase 1 (resonance thinking) and Phase 2 (self-verification) serves as an unsupervised confidence score.
* **Pain-Driven Neurogenesis**: The model monitors its physical suffering (hunger, health, energy meters). Sustained pain triggers Net2WiderNet growth, self-determining the optimal parameter size.
* **Selective Attention Plasticity**: Unfreezing only the attention routing mechanism while keeping core declarative knowledge frozen improves peak survival by +49% in stochastic environments.

To run the reproducibility suite or train the developmental agents, see [REPRODUCE.md](REPRODUCE.md).

---

## Why Python + ONNX/GGUF?

By sticking to standard python-based orchestrations and standardized formats like ONNX and GGUF, we gain native access to hardware accelerator runtimes (CUDA, ROCm, Vulkan, NPU) without adding compilation overhead. Topology is managed dynamically as a standard JSON schema, and hot-swapping adapters is a simple lightweight weight-loading call rather than hot-swapping runtimes.

→ See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full thesis.

---

## Project Status

| Phase | Component | Status |
|-------|-----------|--------|
| **Phase A** | Translator (text ↔ vectors) | ✅ Done |
| **Phase A** | Ghost Swarm (mocks + routing) | ✅ Done |
| **Phase A** | Hardware Discovery (benchmarks) | ✅ Done |
| **Phase A** | Prolog Gate (SWI-Prolog intent router) | ✅ Done |
| **Phase B** | BitNet 1.58b (Ternary Signaling Game) | ✅ Done (Survival Arena & Neurogenesis) |
| **Phase B** | Minion Swarm Scheduler & TUI | ✅ Done (cgroups background training & Control TUI) |
| **Phase A** | Aggregator (multi-expert consensus) | 🔲 Planned |
| **Phase A** | LoRA Specialization & Distillation | 🔲 Planned |

→ See [ROADMAP.md](docs/ROADMAP.md) for the full plan.

---

## Docs

* [PAPER_DRAFT.md](docs/PAPER_DRAFT.md) — The active research paper draft.
* [BIT_THE_CREATURE.md](docs/BIT_THE_CREATURE.md) — An educational story introducing Bit, the creature that learned to grow under pain.
* [ARCHITECTURE.md](docs/ARCHITECTURE.md) — Full system design, Lisp thesis, lifecycle diagrams.
* [EXPERTS_ROSTER.md](docs/EXPERTS_ROSTER.md) — Expert ↔ silicon bindings with benchmarks.
* [ROADMAP.md](docs/ROADMAP.md) — Phase-by-phase implementation plan.
* [REFERENCES.md](docs/REFERENCES.md) — Prior art, bibliography, and novelty analysis.
* [PROLOG_RESEARCH.md](docs/PROLOG_RESEARCH.md) — Prolog router research and design.
* [FRANKENSWARM_SCHEMA.md](docs/FRANKENSWARM_SCHEMA.md) — Visual topology diagrams.

---

## Part of the Red-Pill Ecosystem

Frankenswarm integrates with [Red-Pill](https://github.com/joanfgarcia/antigravity-red-pill) via the `ProviderRegistry`. The `FastFlowLMInferenceProvider` (NPU), `SipInferenceProvider` (CUDA/CPU), and `BitNetInferenceProvider` (ternary) are already wired. Frankenswarm's Prolog Gate is the evolutionary successor to the static `InferenceRouter`.

---

<a name="español"></a>
# Español

> [!WARNING]
> **Fase de Investigación Activa (Fase B: La Arena)**: Este repositorio contiene el código tanto de la **infraestructura MoE Heterogénea (Fase A)** como de la **investigación en Aprendizaje Ternario Ontogenético (Fase B)** expuesta en nuestro artículo. Actualmente, estamos enfocados en entrenar modelos ternarios de 1.58 bits en juegos de señalización emergente. El enrutamiento MoE en cascada físicamente distribuido es el objetivo de despliegue para estos agentes autorregulados.
>
> **Active Research Phase (Phase B: The Arena)**: This repository contains the code for both the **Heterogeneous MoE infrastructure (Phase A)** and the **Developmental Ternary Learning research (Phase B)** presented in our paper. Currently, we are focused on training 1.58-bit ternary models in emergent signaling games. The physically-distributed cascading MoE routing is the deployment target for these self-growing agents.

## ¿Qué es esto?

Frankenswarm es un proyecto unificado para la **soberanía energética en inteligencia de borde (edge)**. Combina dos capas complementarias: el orquestador físico de hardware y la mente cognitiva capaz de aprender en él.

1. **Fase A: El Esqueleto Físico (MoE Heterogénea)**
   Una Mezcla de Expertos físicamente distribuida que convierte cada acelerador de tu máquina local en un nodo de inferencia especializado. Distribuye dinámicamente las tareas a lo largo de **todo el silicio disponible** — GPU dedicada (CUDA), GPU integrada (Vulkan), CPU y NPU — utilizando una **puerta en SWI-Prolog** para clasificar la intención de la consulta y enrutarlas al nivel de silicio más eficiente.

2. **Fase B: El Piloto Cognitivo (Aprendizaje Ontogenético)**
   Un framework de aprendizaje ontogenético para modelos ternarios ultraligeros de 1.58 bits diseñados para habitar nodos de borde de bajo consumo (como NPUs a 2W). En lugar de tener un tamaño fijo, estos modelos **crecen de forma autónoma** mediante neurogénesis dirigida por dolor (Net2WiderNet) y se adaptan a través de plasticidad selectiva en su atención, usando primos semánticos deterministas como base.

---

## Soberanía Energética

Al mantener la GPU dedicada (80W+) en suspensión y enrutar las consultas rutinarias de clasificación o baja complejidad a los micro-expertos locales que se ejecutan en la NPU (2W), Frankenswarm consigue una eficiencia energética de hasta **200 veces superior** frente a modelos en la nube y **40 veces** en comparación con el uso local exclusivo de GPU.

| Estrategia de Inferencia | Energía por 1,000 Consultas | Eficiencia Relativa |
|--------------------------|:---------------------------:|:-------------------:|
| Nube (GPT-4)             | ~5 - 10 kWh                 | 1x (Línea Base)     |
| CUDA Puro (GPU Local)    | ~2.2 kWh                    | ~3.4x               |
| **Cascada Frankenswarm** | **~0.05 kWh**               | **~150x - 200x**    |

---

## La Ecuación del Sistema

```
Frankenswarm = [Fase A: Esqueleto Físico] + [Fase B: Mente Cognitiva]

  Fase A (Hardware MoE) = Puerta SWI-Prolog (Clasificación de Intención)
                           + Orquestador Python (Topología e Intercambio Caliente)
                           + Ejecución Heterogénea (CUDA / Vulkan / CPU / NPU)

  Fase B (Mente Cognitiva) = BitNet Ternario (~586KB de tamaño de modelo)
                           + Primos Semánticos NSM (Embeddings inspirados en Wierzbicka)
                           + Metacognición Emergente (Bucle pensar-y-verificar)
                           + Neurogénesis por Dolor (Crecimiento Net2WiderNet)
                           + Plasticidad Selectiva (Atención Plástica / Conocimiento Congelado)
```

---

## El Sustrato de Hardware (Medido — Strix Point, 22-05-2026)

| Acelerador | Modelo Local | Velocidad | Consumo | Rol Dinámico / Tarea |
|------------|--------------|:-----:|:-----:|----------------------|
| RTX 5070 (CUDA) | Falcon3-10B | 23 tok/s | ~80W | Razonamiento Complejo / Validación |
| Ryzen AI 9 (CPU) | Falcon3-10B | 12.8 tok/s | ~45W | Erudito en Contexto Largo / Recuperación |
| Radeon 880M (Vulkan) | Falcon3-10B | 4.8 tok/s | ~15W | Centinela en Segundo Plano |
| NPU XDNA2 (ONNX) | Qwen3-0.6B | 96 tok/s | ~2W | Explorador Rápido / Micro-Experto |
| NPU XDNA2 (ONNX) | Qwen3-8B | 10.6 tok/s | ~2W | Orquestador / Triaje de Intención |

---

## Los Tres Bucles de Adaptación

| Bucle | Escala Temporal | Mecanismo de Mutación | Objetivo |
|------|-----------|--------------------|------|
| **RÁPIDO** (Prolog) | Por consulta (~200ms) | Política de enrutamiento | Asignar la consulta al silicio más adecuado |
| **MEDIO** (Orchestrator) | Por sesión (~minutos) | Umbrales de activación | Ajustar márgenes de confianza y caídas en cascada (NAS) |
| **LENTO** (Metabólico) | Durante el sueño (~horas) | Pesos y estructura local | Ajuste fino QLoRA / Neurogénesis y evolución de BitNet |

---

## Fase B: El Artículo de Investigación

La capa de aprendizaje ontogenético y cognitivo (Fase B) se detalla en nuestro paper: **"From Semantic Primes to Neurogenesis: Developmental Learning in Ternary Neural Networks"**.

Características de investigación clave implementadas en este repositorio (`src/bitnet/`):
* **Primos Semánticos Deterministas**: 65 embeddings de glifos ternarios que codifican los primos semánticos del NSM de Wierzbicka [1]. Consiguen una precisión composicional del 100% y un entrenamiento 7.7 veces más rápido que embeddings aprendidos tradicionales.
* **Verificación Metacognitiva**: Un bucle pensar-y-verificar en dos fases. La similitud de coseno entre la Fase 1 (pensamiento resonante) y la Fase 2 (autoverificación) genera una puntuación de confianza no supervisada.
* **Neurogénesis Dirigida por Dolor**: El modelo monitoriza sus niveles de necesidad física (hambre, salud, energía). El dolor persistente desencadena crecimiento por Net2WiderNet, autogestionando el tamaño de sus parámetros.
* **Plasticidad de Atención Selectiva**: Descongelar únicamente el mecanismo de atención manteniendo el conocimiento base inalterado mejora la supervivencia máxima en un +49% en entornos estocásticos.

Para ejecutar la suite de reproducción o entrenar a los agentes ontogenéticos, ver [REPRODUCE.md](REPRODUCE.md).

---

## ¿Por qué Python + ONNX/GGUF?

Al mantenernos fieles al estándar de orquestación en Python y formatos unificados como ONNX y GGUF, obtenemos acceso nativo a los runtimes de aceleración de hardware (CUDA, ROCm, Vulkan, NPU) sin añadir sobrecostes de compilación. La topología se gestiona dinámicamente como un esquema JSON estándar, y el intercambio de adaptadores es una llamada ligera de carga de pesos en lugar de un reinicio de entornos.

→ Ver [ARCHITECTURE.md](docs/ARCHITECTURE.md) (en inglés) para la tesis completa.

---

## Estado del Proyecto

| Fase | Componente | Estado |
|-------|-----------|--------|
| **Fase A** | Traductor (texto ↔ vectores) | ✅ Completado |
| **Fase A** | Enjambre Fantasma (mocks + rutas) | ✅ Completado |
| **Fase A** | Descubrimiento de Hardware (benchmarks) | ✅ Completado |
| **Fase A** | Prolog Gate (enrutador de SWI-Prolog) | ✅ Completado |
| **Fase B** | BitNet 1.58b (Juego de Señalización Ternaria) | ✅ Completado (Arena de Supervivencia y Neurogénesis) |
| **Fase B** | Programador de Miniones & TUI | ✅ Completado (entrenamiento en segundo plano cgroups y TUI de control) |
| **Fase A** | Agregador (consenso multi-experto) | 🔲 Planificado |
| **Fase A** | Especialización LoRA y Destilación | 🔲 Planificado |

→ Ver [ROADMAP.md](docs/ROADMAP.md) (en inglés) para el plan completo.

---

## Documentos

* [PAPER_DRAFT.md](docs/PAPER_DRAFT.md) — Borrador del artículo científico activo.
* [BIT_THE_CREATURE.md](docs/BIT_THE_CREATURE.md) — Un cuento educativo que presenta a Bit, la criatura que aprendió a crecer bajo el dolor.
* [ARCHITECTURE.md](docs/ARCHITECTURE.md) — Diseño del sistema completo, tesis Lisp, diagramas de ciclo de vida.
* [EXPERTS_ROSTER.md](docs/EXPERTS_ROSTER.md) — Enlaces experto ↔ silicio con benchmarks.
* [ROADMAP.md](docs/ROADMAP.md) — Plan de implementación paso a paso.
* [REFERENCES.md](docs/REFERENCES.md) — Estado del arte, bibliografía y análisis de novedad.
* [PROLOG_RESEARCH.md](docs/PROLOG_RESEARCH.md) — Investigación y diseño del enrutador Prolog.
* [FRANKENSWARM_SCHEMA.md](docs/FRANKENSWARM_SCHEMA.md) — Diagramas visuales de topología.

---

## Parte del Ecosistema Red-Pill

Frankenswarm se integra con [Red-Pill](https://github.com/joanfgarcia/antigravity-red-pill) a través de `ProviderRegistry`. Los proveedores `FastFlowLMInferenceProvider` (NPU), `SipInferenceProvider` (CUDA/CPU) y `BitNetInferenceProvider` (ternario) ya están conectados. El Prolog Gate de Frankenswarm es el sucesor evolutivo del `InferenceRouter` estático.

---

## Licencia / License

Este proyecto usa **doble licencia** / This project uses a **dual license**:

| Contenido / Content | Licencia / License |
|---|---|
| **Código fuente / Source code** (`src/`, `configs/`, scripts) | [AGPL-3.0](LICENSE) |
| **Contenido educativo / Educational content** (paper, *Bit: The Creature That Learned to Grow*) | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) |
