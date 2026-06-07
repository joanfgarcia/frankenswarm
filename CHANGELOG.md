# Frankenswarm Changelog

## [Unreleased]

### 🎓 Currículo de Piaget y Parada Natural de Generación
- **[FEAT] Parada Natural de Generación en Escuela (`src/bitnet/train_sovereign_school.py`)**: Reemplazado el cálculo de pérdida estándar por una **máscara dinámica por secuencia** que activa la pérdida para todos los tokens de la frase más el primer token `<pad>` (0) que la sucede. Esto entrena al modelo para predecir el token de parada y detenerse de forma natural en lugar de balbucear.
- **[FIX] Retorno de Samantha Eval (`src/bitnet/train_sovereign_school.py`)**: Corregido el retorno y desempaquetado de `run_samantha_eval` para evitar fallos de signatura en el bucle principal.
- **[LINT] Corrección de Linter y Formato**: Resueltos avisos de ruff (`SIM108`, `B007`) en `playground/chat_school_agent.py` y `train_sovereign_school.py`, y corregidas tabulaciones en los docstrings de `src/bitnet/net2net.py`.

## [0.3.2] - 2026-06-06

### 🧠 In-Inference Backtracking & Benchmarking
- **[FEAT] Chat with Backtracking**: Implemented `playground/chat_backtrack.py` to support interactive chat with dynamic token backtracking and rollback of model state for low-capacity models (BitNet 50M).
- **[FEAT] Backtracking Scenario Testing**: Created `scripts/test_backtrack_scenarios.py` to benchmark baseline, confidence, entropy, and lookahead backtracking modes, outputting a comparative performance and quality report.

## [0.3.1] - 2026-06-06

### 🎓 Evaluador Cognitivo Conversacional de Samantha y Neurogénesis Escolar
- **[FEAT] Evaluador de Samantha (`scripts/evaluate_samantha_age.py`)**: Diseñado e implementado el evaluador cognitivo conversacional de Samantha que interactúa con el modelo BitNet mediante preguntas de examen por hitos de edad (4 a 8 años) y lo califica mediante un prompt de evaluación en la GPU RTX 5070.
- **[FEAT] Neurogénesis y Pausa en Hitos (`src/bitnet/train_sovereign_school.py`)**: Modificado el bucle de entrenamiento escolar para soportar la neurogénesis en caliente y la evaluación periódica o al promover el curso. El entrenamiento se pausa automáticamente para interacción humana tras superar con éxito la evaluación del hito conversacional (guardando `milestone_achieved.json` y el checkpoint de la etapa).
- **[FIX] Duplicados de Glifos Unívocos (`src/bitnet/expand_vocabulary.py`)**: Implementado un algoritmo desempate determinista iterativo en la calibración y proyección de vocabulario que asegura que todos los glifos sean semánticamente únicos para evitar colisiones de tokens.
- **[FIX] Estabilidad Causal en FP16 (`src/bitnet/modeling_bitnet.py`)**: Ajustada la máscara causal en `BitNetAttention` para usar `-65000.0` en modo `float16`, previniendo desbordamientos y valores NaN durante la multiplicación de matrices.
- **[TEST] Cobertura Conversacional**: Añadido `tests/test_conversational_bitnet.py` para verificar las formas de entrada/salida, causalidad del modelo de lenguaje conversacional BitNet y la recarga correcta de pesos del modelo entrenado.

## [0.3.0] - 2026-06-04

### 🎓 Entorno de Entrenamiento Híbrido (Dojo de PopuLoRA) y Estabilización PPO
- **[FEAT] El Dojo de PopuLoRA (currículo supervisado)**: Integración de `dojo_populora.py` para generar lotes de entrenamiento supervisado basados en plantillas expertas (alimentación, mochila, combate, gritos y reanimación de emergencia).
- **[FEAT] Pre-entrenamiento y consolidación de sueño**: Aplicación de 50 pasos de Dojo pre-entrenamiento para alinear los pesos iniciales de las cabezas de política, y 5 épocas de consolidación de Dojo en cada fase de sueño (unfreezing backbone).
- **[FEAT] Estabilización de PPO**: Congelamiento del backbone de representación durante la optimización PPO diurna. Implementación de limitador de ratios (ratios clamped a 10.0) y uso de Huber Loss (delta=5.0) para evitar explosiones de gradientes en penalizaciones negativas (K.O./muertes).
- **[FEAT] Mecánica de K.O. y Muerte Permanente**: Reducción drástica del exploit de inmortalidad. Los agentes que llegan a homeostasis 0 entran en debuff de inconsciencia (K.O.) por 12 ticks. Si un compañero no usa `reanimar`, mueren definitivamente y aplican debuff de tristeza (72 ticks) a los supervivientes.
- **[FEAT] Criterio de Maestría (Dominio Alcanzado)**: Detención automática de la simulación cuando la tribu sobrevive 200 ticks deterministas consecutivos en evaluación sin entrar en K.O. (alcanzado en episodio 342).
- **[DOCS] Boletín de Notas y Sesiones**: Mapeo completo del currículo de 4 grados en `curriculum_plan.md` y registro en `walkthrough.md` dentro de `docs/sessions/20260603/`.

## [0.2.0] - 2026-05-30

### 🧬 Especialización de Habilidades y Asimetría Tribal (Fase 5 y 5.1 - Realismo de Mochila)
- **[FEAT] Compartición Desacoplada de Mochilas (Fase 5.1)**: Modificación de la acción `"dar"` en `CooperativeWorld` para realizar una transferencia de mochila a mochila en lugar de consumo directo. El recurso (agua o comida) se mueve a la mochila del receptor (siempre que esté vacía y su nivel de necesidad sea < 80).
- **[FEAT] Consumo de Segundo Paso (Fase 5.1)**: Los agentes especializados ahora deben aprender y ejecutar explícitamente `"comer"` o `"beber"` en ticks subsiguientes para ingerir el recurso que sus compañeros depositaron en su mochila.
- **[FIX] Actualización de Pruebas Unitarias (Fase 5.1)**: Actualización de `scratch/test_specialization.py` (`test_sofy_water_exclusion` y `test_hugo_food_exclusion`) para verificar el flujo de dos pasos (dar llena mochila, comer/beber consume).
- **[FEAT] Asimetría de Supervivencia (Fase 5)**: Restricciones por rol en `CooperativeWorld` para forzar la interdependencia:
  * Nico (Agente A) excluido de la caza cooperativa.
  * Sofy (Agente B) no sabe extraer agua del entorno (no se llena mochila pisando agua y no bebe del suelo).
  * Hugo (Agente C) no sabe recolectar comida del entorno (no se llena mochila pisando comida y no come del suelo).
- **[FEAT] Robustez de Carga Multidimensional (Fase 5)**: Corrección de fallos por desajuste de dimensiones (`size mismatch`) en `load_agent` al reconstruir dinámicamente el Actor-Critic si el checkpoint ya posee más acciones o neuronas ocultas.
- **[NEW] `scratch/test_specialization.py` (Fase 5)**: Suite de tests unitarios que valida de forma aislada las restricciones de rol de Nico, Sofy y Hugo, y la resolución correcta del intercambio de recursos.

### 🏹 Tribu de 3 Agentes y Caza Cooperativa (Fase 4 - EXP_073_v5)
- **[FEAT] Tribu Ampliada (N=3)**: Expansión de la arena `CooperativeWorld` para gestionar tres agentes (`Nico`, `Sofy`, `Hugo`) con posiciones de spawn distribuidas y modelos ToM multilaterales cruzados.
- **[FEAT] Broadcast de Gritos Half-Duplex**: Difusión en un solo tick de la señal del emisor a todos los receptores silenciosos de la tribu, actualizando sus ToM y targets de navegación.
- **[FEAT] Worst-State ToM Routing**: Proyección dinámica del compañero en el estado más crítico de salud/hambre sobre los tokens ToM en `perceive()`, manteniendo la compatibilidad absoluta con el espacio de percepción de 6 tokens.
- **[FEAT] Caza Cooperativa**: Spawn de presas (glifo `"grupo"`) que requieren la acción simultánea `"luchar"` de $\ge 2$ agentes para abatirla (+50.0 hambre, comida en mochila, +15.0 bonus). El intento solitario falla, causa daño (-2.0 salud) y ahuyenta a la presa (50% de probabilidad).
- **[FEAT] Redimensionamiento Action Head Net2Net (PRESERVING WEIGHTS)**: Modificación de la carga de checkpoints en `train_arena_ppo.py` y `run_arena_simulation.py` para copiar los pesos del action head de las 7 acciones originales y sólo inicializar aleatoriamente la 8ª acción (`"dar"`). Previene el olvido catastrófico instantáneo al cargar pesos pre-entrenados.
- **[NEW] `scratch/test_coop_hunting.py`**: Suite de tests unitarios que valida la caza cooperativa, el broadcast de gritos y la Teoría de la Mente de 3 agentes.
- **[NEW] Presets de Dificultad Parametrizados**: Adición del parámetro `"prey_spawn_interval": 15` en los archivos JSON de presets ecológicos (`easy`, `medium`, `hard`, `hell`) en `configs/experiments/`.

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
