# Frankenswarm Changelog

## [Unreleased]

### 🌱 Aprendizaje Ontogenético y Neurogénesis (EXP_034 - EXP_045)
- **[FEAT] Deterministic Semantic Primes (EXP_034)**: Implementación de 65 embeddings de glifos ternarios representando los primos semánticos del NSM de Wierzbicka. Consigue 100% de precisión composicional y un entrenamiento 7.7 veces más rápido.
- **[FEAT] Bucle Metacognitivo de Verificación (EXP_036)**: Bucle de inferencia en dos fases (pensar y verificar) para generar una señal de confianza no supervisada mediante la similitud de coseno del estado latente.
- **[FEAT] Neurogénesis por Dolor (EXP_040)**: Mecanismo de crecimiento neuronal autónomo mediante Net2WiderNet activado por persistencia de dolor fisiológico (recompensa negativa acumulada).
- **[FEAT] Plasticidad de Atención Selectiva (EXP_045)**: Descongelamiento dinámico del mecanismo de atención sobre el sustrato fijo de conocimiento largo, mejorando un +49% la supervivencia pico en entornos estocásticos.
- **[NEW] `train_survival.py`**: Entorno de aprendizaje por refuerzo (MinimalWorld) con recompensas continuas de dolor, métricas de supervivencia (ticks) y neurogénesis en caliente.
- **[NEW] `cooperative_world.py` / Swarm Arena**: Arena multi-agente para juegos de señalización emergente, canales de comunicación discretos y penalización por destino compartido.
- **[DOCS] Paquete Científico-Divulgativo**: Añadido el borrador final para arXiv (`docs/PAPER_DRAFT.md`), el cuento educativo (`docs/BIT_THE_CREATURE.md`) y estandarización del panel de reviews en `docs/extern/`.

### 🧠 Resonancia Continua (EXP_032)
- **[FEAT] Latent Resonance Loop**: `forward_resonance()` en `BitNet4LayerModel` — bucle cerrado que itera N veces por las core_layers en espacio latente (256-dim) sin proyectar a vocabulario. Elimina dependencia de KV cache. Memoria O(1) para pensar.
- **[FEAT] Resonance Training Mode**: `forward_resonance_training()` con soporte para loss intermedio (every/weighted/final) y BPTT a través del bucle.
- **[FEAT] Resonance Clock**: Embedding posicional aprendible para cada step del bucle (`resonance_clock`), permitiendo al modelo distinguir la fase de pensamiento.
- **[FEAT] Sample Watcher**: `_sample_watcher()` para muestrear tokens y métricas de estabilidad (norma, convergencia coseno) en cada paso sin romper el grafo de gradientes.
- **[NEW] `train_resonance.py`**: Script de entrenamiento con bucle latente, curriculum 3-fases (guardería→recreo→autonomía), y evolución SVD.
- **[NEW] `run_grid_032.py`**: Grid runner factorial 3³ con contención cgroup OOM Shield.
- **[NEW] `analyze_grid_032.py`**: Análisis comparativo con heatmaps, efectos marginales y exportación CSV.
- **[NEW] `operators_logic.py`**: Grafo causal con 13 reglas de implicación, cadenas transitivas DFS, y operadores de lógica proposicional.

### 💗 Resonancia Emocional (EXP_033)
- **[FEAT] Emotion Embeddings**: `nn.Embedding(n_emotions, emotion_dim)` + `emotion_proj` en `BitNet4LayerModel`. Tres modos de inyección en el bucle latente:
  - `additive`: suma el vector emocional en cada step del bucle.
  - `gated`: modulación sigmoid + desplazamiento (más parámetros).
  - `first_only`: impulso emocional solo en step 0 (el más efectivo).
- **[FEAT] Backward-Compatible API**: Los parámetros `n_emotions`, `emotion_dim`, `emotion_mode`, `emotion_ids` son opcionales. EXP_032 y código anterior funcionan sin cambios.
- **[FEAT] Emotional Causal Graph**: `EMOTIONAL_RULES` con 30 reglas bifurcadas por emoción (6 emociones × 8 conceptos fuente). Misma causa → distinto destino según emoción.
- **[FEAT] Bifurcation Evaluation**: `get_bifurcation_pairs()` genera 46 pares de test para medir si el modelo usa la emoción para decidir.
- **[FEAT] Extended Vocabulary**: 3 conceptos nuevos (calma, refugio, libertad) como destinos emocionales. `CONCEPT_NAMES_EXT` (15 palabras).
- **[FIX] Early Stopping**: El patience counter se reseteaba mal al entrar en autonomía. Fix: reset explícito + warm-up mínimo de 10 epochs.
- **[NEW] `train_emotional_resonance.py`**: Entrenamiento con 5 condiciones (A/B/C/D/E), evaluación de bifurcaciones por epoch, y early stopping corregido.
- **[NEW] `run_grid_033.py`**: Grid runner para 9 variantes con modo cata (--tasting).


- **[FEAT] Embeddings Posicionales Paramétricos**: Añadida la opción `use_pos_embedding` en `BitNet4LayerModel` configurable desde JSON para romper la equivariancia de permutación de la auto-atención.
- **[FEAT] Auto-adaptación en la Carga**: Implementada carga de pesos autoadaptativa en `_load_from_state_dict` para habilitar retrocompatibilidad con checkpoints posicionales sin alterar scripts de evaluación o TUI.
- **[FEAT] Trazabilidad Completa en Storage**: Traslado de planes de implementación, tareas y walkthroughs a carpetas de experimentos (`storage/experiments/EXP_XXX/`) para reproducibilidad de lab 100% auditable.

### 🔌 Inversión de Control (IoC) & Orquestación
- **[FEAT] Registro de Operadores**: Creado `operators.py` y `OperatorRegistry` para modularizar y desacoplar ecuaciones matemáticas.
- **[FEAT] Bucle de Entrenamiento Genérico**: `train_generic.py` unifica entrenamiento, anclaje de TF y poda evolutiva mediante archivos JSON y ganchos de ciclo de vida (`hooks.py`).
- **[FEAT] Minion Scheduler**: Implementado `minion_scheduler.py` para encolar y ejecutar simulaciones de fondo bajo contención cgroup (`MemoryMax=10G`).
- **[FEAT] Control Center TUI**: Expandido `microscope_tui.py` con panel de visualización en tiempo real (ASCII loss curve) y lector de reportes de buzón (`lab/.inbox/`).

### 🧬 Genesis
- **[ARCH] Initial architecture definition**: BitNet 1.58b + NEAT + Net2Net + MoE + TurboQuant pillars documented in `ARCHITECTURE.md`.
- **[ARCH] Project scaffolding**: `src/` structure created with `tokenizer/`, `router/`, `nodes/`, `swarm/` modules.
- **[FEAT] Sovereign Router v0.1**: `prolog_router.py` — deterministic rule-based router implemented in Python (match/case), structurally mirroring Prolog inference. Zero tokens, zero GPU for routing decisions.
- **[ARCH] Tokenizer decision**: Unigram LM, 8,192 shared vocabulary, domain-specific corpus (Python + technical English + Red-Pill ecosystem). Single vocabulary shared across all nodes.
- **[DOCS] CONVENTIONS.md**: 5 immutable rules established: Experimental-First, No Premature Neural Complexity, Ternary Weights Are Sacred, Shared Vocabulary, Fitness Score Required.
