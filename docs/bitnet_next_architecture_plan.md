# 🧠 BitNet Next-Gen: Universal NSM Syntax & Vocabulary Gating Plan

Plan de evolución arquitectónica para el desarrollo cognitivo de Bit. Desacopla la
sintaxis humana de la lógica pura y aplica gating evolutivo de vocabulario.

> **v2 — 2026-07-08.** Revisado por Aleth (Claude Fable 5) contra el estado real de
> School v3 (`docs/sessions/20260703/ROUTE_CHANGE_SCHOOL_V3.md`) y el código
> (`src/bitnet/model/modeling_bitnet.py`, `src/bitnet/vocab/glyph_vocabulary.py`).
> Cambio principal: las tres propuestas dejan de compartir un estado único —
> cada una tiene su propia vía y su propio criterio de adopción (§0).
>
> **v3 — 2026-07-10.** Incorpora la **convergencia LISP↔K-65P**
> (`docs/convergencia_lisp_k65p.md`, intuición de Joan del 2026-07-09):
> K-65P reconocido como dialecto de la familia Lisp, principio de "alucinaciones
> auditables por construcción", horizonte K-65P v1 (variables + forma de regla),
> puente Horn con `PrologExpert` y arquitectura Sistema 1/Sistema 2 (§1.3).
> Destino editorial del track: **paper 3** (los papers 1 y 2 ya tienen tema).
>
> **v4 — 2026-07-10 (tarde).** Cuarta propuesta: **híbrido Attention+SSM con Mamba
> ternario** (§4), desde `docs/ideas_ssm_ternario_y_recurrencia.md`
> (Aleth/antigravity) auditada por Aleth Fable. Recalibrada: el SSM se adopta por
> *calidad de memoria* (sustituir el hack `h_prev`), no por eficiencia — a 128
> tokens y 10,9M params el argumento O(1) es vacío. Destino: **paper 1** (motor),
> no paper 3 (lenguaje) — ortogonales.
>
> **v5 — 2026-07-31.** Reconciliación de tres documentos que habían divergido:
> este plan (v4, cuando Bit tenía ~10,9M params), la revisión de literatura
> depth-vs-width (`docs/NOTE_DEPTH_VS_WIDTH_LITERATURE.md`, 30 jul, con Bit
> en 76,9M/1024d/6L y plateau) y el ROADMAP soberano de `~/Documents/IA/k65p`
> (16 jul), que desde ahora es la **autoridad de secuenciación** (§7). Añade:
> quinta propuesta **Net2DeeperNet** (§5, aprobada para la ventana de GPU
> post-graduación, SOLO sobre copia — el control de G4 es intocable), el
> **Arsenal** (§6: todas las alternativas del NOTE preservadas como balas en la
> recámara — nada se descarta), y el **pre-registro del confound de crecimiento**
> para G4 (§7.2). El note del 30 jul redescubrió el SSM sin citar este plan —
> la v5 existe para que no vuelva a pasar: referencias cruzadas obligatorias.

---

## 0. Estado por propuesta

| § | Propuesta | Estado | Vía de adopción |
|---|---|---|---|
| 2 | Vocabulary Gating | ✅ **Aprobada** — entra en School v3 desde el día uno | Cambio directo; el plumbing (`logit_mask`) ya existe en el modelo |
| 3 | Trit-Level Loss | 🧪 **Experimento A/B pre-registrado** | Término auxiliar en etapas tempranas; se adopta solo si gana en la batería |
| 4 | Híbrido Attention+SSM (BitMambaBlock, Mamba ternario) | 🧪 **Experimento pre-registrado** — gateado tras la comparación Bit-lingüístico vs Bit-NSM | Evolución del motor (paper 1); sustituye el hack `h_prev`, no compite con K-65P |
| 1 | Pure NSM Bit — lenguaje **K-65P** (*Kernel de 65 Primos*, dialecto Lisp) | 🔬 **Track de investigación paralelo** — desde v5: repo soberano `~/Documents/IA/k65p`, Gate G1 cerrado (16 jul), su ROADMAP gobierna la secuenciación global (§7) | No bloquea ni redefine School v3; destino: **paper 3** (`paper_3_k65p_neurosymbolic`) |
| 5 | Net2DeeperNet — depth expansion 6→8 sobre **copia** del control | ✅ **Aprobada para la ventana de GPU post-graduación** (§5) — el checkpoint de graduación queda congelado como control de G4 | Línea del motor (paper 1); no toca ningún gate del ROADMAP k65p |
| 6 | Arsenal — SSM-en-resonancia, híbrido Jamba, SSM puro, SENN y variantes | 🗃️ **Reserva** — balas en la recámara (§6); se desenfundan solo si una línea activa se estanca | Cada bala lleva su gatillo definido en §6; ninguna se descarta |

---

## 1. Decoupled Cognitive Architecture (Two-Step Translation) — track paralelo

**Objetivo original:** evitar que Bit gaste capacidad de parámetros en gramática y
ortografía específicas de cada lengua, concentrando sus recursos neuronales en lógica
causal universal.

**Intención de fondo (aclarada por Joan, 2026-07-08):** el objetivo no es solo
eficiencia — es que **la IA tenga su propio lenguaje**, basado en los primos
semánticos de Wierzbicka. El compilador/decompilador es la interfaz con los humanos;
el lenguaje de glifos es el medio nativo de pensamiento y comunicación inter-agente.

### Diseño original (v1)
*   **Compiler:** parser que traduce frases de cualquier lengua humana a una
    **sintaxis NSM universal** estandarizada y legible por máquina.
*   **Decompiler:** traduce las secuencias NSM de salida de vuelta a lenguaje natural.
*   *Ejemplo:*
    *   **Input (ES):** `"el fuego caliente es peligroso"`
    *   **Input (EN):** `"hot fire is dangerous"`
    *   **Compiled NSM:** `[thing_fire, quality_hot, copula_is, quality_bad]`
*   **Core Engine:** las capas neuronales de Bit se entrenan **exclusivamente** sobre
    secuencias NSM compiladas → motor de lógica pura, agnóstico al idioma.

### Revisión (Aleth, 2026-07-08)

1.  **"Ortografía" no aplica:** Bit ya tokeniza a nivel de palabra y cada embedding
    ya es una composición de primos (`trits @ prime_embeddings`). Lo único
    lingüístico que Bit aprende hoy es el orden sintáctico — y ese es exactamente el
    resultado estrella de la auditoría (93% preferencia gramatical): la prueba de que
    el sustrato ternario+glifos aprende. Sustituir School v3 por esto tiraría la
    capacidad medida e invalidaría la batería M1-M4 (definida sobre español natural).
2.  **El compilador para texto humano arbitrario es el problema NLP duro** (las
    explicaciones NSM reales no son mapeos 1:1; `dangerous` → `quality_bad` pierde
    significado — la explicación canónica es multicláusula: "si tocas esto, algo malo
    te puede pasar"). Si lo resuelve un LLM, la inteligencia vive en el compilador y
    la tesis "Bit aprende" se diluye ante cualquier revisor.
    **→ Resolución (§1.2): no necesitamos parsear texto arbitrario. Controlamos el
    corpus: la fábrica puede emitir pares paralelos (ES, NSM) por construcción.**
3.  **El decompilador se comería las métricas de producción** (longitud, stop, tasa
    de fantasmas pasarían a medir al decompilador, no a Bit). Cualquier evaluación de
    Bit-NSM debe hacerse sobre las secuencias de glifos crudas, nunca tras decompilar.

### 1.1 El problema abierto real: la sintaxis de los glifos

Wierzbicka y Goddard no dejaron solo los ~65 primos: definieron una **gramática de
los primos** — marcos de valencia por primo (DO: *alguien hace algo / le hace algo a
alguien*; HAPPEN: *algo pasa / le pasa a alguien*), combinaciones canónicas y
conectores de cláusula (IF, BECAUSE, WHEN). Pero es una especificación semi-formal,
en prosa, pensada para lingüistas humanos — **no existe una sintaxis NSM
machine-readable**. Ese es el hueco. No se encuentra hecha: hay que diseñarla
(la universalidad es un acto de diseño, no una propiedad emergente — no existe
lengua de signos universal por la misma razón).

**→ Resuelto (2026-07-08):** sintaxis v0 formalizada en
`frankenswarm/docs/RFC-002_SINTAXIS_GLIFOS.md` (híbrido A+B, validador de
referencia + 52 tests). El lenguaje se llama **K-65P** — *Kernel de 65 Primos*
(etimología completa en RFC-002 §0: nada del nombre es azar). Opciones que se
evaluaron:

| Opción | Descripción | Pro | Contra |
|---|---|---|---|
| **A. Formalizar NSM canónico** | BNF sobre primos + marcos de valencia de Goddard/Wierzbicka + validador | Fiel al espíritu NSM; publicable por sí mismo ("machine-readable NSM") | Las explicaciones NSM son verbosas — las secuencias se alargan |
| **B. Linealización tipo AMR** | Árbol semántico predicado-primero con marcadores de rol, orden canónico fijo | Sin ambigüedad; sin sintaxis que aprender (la capacidad va a lógica pura, el objetivo original); precedente maduro (AMR, UNL, Lojban, Blissymbolics) | Menos "NSM-puro"; hay que diseñar los marcadores de rol |
| **C. Sintaxis emergente** | Bit y hermanos negocian el orden bajo presión de cuello de botella (iterated learning) | Científicamente el más ambicioso | La literatura muestra que los códigos emergentes degeneran sin presiones cuidadosas; el más difícil de evaluar |

**Recomendación Aleth:** híbrido A+B — linealización canónica fija (B) usando los
primos como vocabulario y los marcos de valencia NSM (A) como restricciones.

### 1.2 Corpus paralelo por construcción (disuelve la bandera roja del compilador)

La fábrica Samantha (`scripts/samantha_story_factory.py`) y los diálogos
combinatorios generan el corpus desde plantillas **que conocen su propia semántica**.
Pueden emitir cada muestra como par `(frase_ES, árbol_NSM)` sin parser alguno:
la alineación se obtiene en el lado de la generación, no del análisis.

*   Fase 1: corpus paralelo perfecto gratis → entrena Bit-NSM (track paralelo) y,
    de regalo, un traductor aprendido ES↔NSM para el futuro.
*   El compilador para texto libre solo se necesita mucho más tarde, y para entonces
    se puede **entrenar** sobre el corpus paralelo en lugar de escribirlo a mano.
*   Coherencia doctrinal: `docs/PROLOG_RESEARCH.md` ya consagra "LLMs como
    traductores, no solucionadores" — el compilador es un traductor; la lógica vive
    en Bit.
*   Encaje estratégico: un interlingua canónico de glifos es el protocolo natural
    inter-agente del Jungle Reboot (Nico, Sofy, Hugo) — el hermano simbólico de
    Tensors-as-API.

### 1.3 Convergencia LISP y horizonte v1 (añadido v3 — 2026-07-10)

Fuente: `docs/convergencia_lisp_k65p.md` (intuición de Joan, 2026-07-09;
anotada por Aleth Fable, 2026-07-10). Lo que cambia en la doctrina del track:

1.  **K-65P es un dialecto Lisp.** No por diseño deliberado sino por convergencia: la
    RFC-002 eligió S-expressions (`[pred args]`, átomos + anidamiento) por puro
    determinismo técnico, y eso ES la estructura átomo/lista/S-expression de McCarthy
    con vocabulario de Wierzbicka. Dos caminos independientes, un mismo punto — la
    señal de que el punto es real.
2.  **Principio de evaluación: alucinaciones auditables por construcción.** El motor
    sigue siendo estadístico — Bit puede emitir K-65P bien formado y falso — pero
    sobre S-expressions la verificación es cómputo, no comprensión: el validador
    comprueba la forma y un verificador simbólico comprueba la consecuencia lógica.
    Consecuencia práctica: la batería de Bit-NSM incorpora una **métrica de
    verificabilidad** (¿qué fracción de sus emisiones pasa el chequeo simbólico
    contra la base de hechos?) que no existe para el Bit lingüístico. Prohibido
    reclamar "sin alucinaciones" en papers o docs — el claim honesto es "pensamiento
    machine-checkable".
3.  **Horizonte K-65P v1** (solo si el track v0 gana la comparación): dos extensiones
    — **variables con unificación** (sintaxis del hueco pendiente de diseño) y
    **forma de regla** `[regla <nombre> <condición> <acción>]` — que traen la
    homoiconicidad real: las reglas como expresiones de primera clase que el propio
    Bit puede almacenar, transmitir y manipular.
4.  **Puente Horn:** una regla K-65P v1 es casi carácter a carácter una cláusula de
    Horn → traductor mecánico (~100 líneas) hacia el `PrologExpert` existente,
    coherente con `PROLOG_RESEARCH.md` (los modelos traducen, los motores simbólicos
    resuelven).
5.  **Arquitectura destino — dos velocidades:** **Bit = Sistema 1** (intérprete
    neuronal aproximado de S-expressions: rápido, entrenado, falible) +
    **Prolog = Sistema 2** (verificación exacta por backtracking: lenta, incansable,
    incapaz de alucinar). El conexionismo propone, el simbolismo dispone.
6.  **Destino editorial: paper 3** (`paper_3_k65p_neurosymbolic` — Wierzbicka +
    McCarthy + BitNet). Prerequisito para fundarlo: resultados de la comparación
    Bit-lingüístico (batería M4) vs Bit-NSM. Los papers 1 (ternary neurogenesis) y
    2 (Tensors as API) ya tienen tema — no reciclar numeración.

---

## 2. Vocabulary Gating (Lexical Growth Curriculum) — ✅ aprobado

**Objetivo:** imitar el desarrollo infantil restringiendo el espacio de búsqueda del
vocabulario en las épocas tempranas para reducir el ruido de gradiente.

*   **Curriculum-Gated Softmax:** máscara dinámica de logits en la capa de salida
    según la etapa vigente.
*   **Expansión por etapas (v2 — techos ligados al censo, no a números importados):**
    1.  **Etapa 0:** ~50 palabras (26 Core Survival Primes + functores básicos).
    2.  **Etapa 1:** ~300 palabras (sustantivos preescolares + acciones básicas).
    3.  **Etapa 2:** ~1.000 palabras.
    4.  **Etapa 3:** ~3.000 palabras.
    5.  **Etapa 4:** **censo limpio completo vigente** (hoy 6.361 palabras,
        `configs/clean_vocabulary_words.json`; crece solo re-ejecutando el censo
        cuando la fábrica aporta palabras nuevas).
*   Las cabezas de atención tempranas convergen más rápido al no tener que
    distinguir entre palabras abstractas similares; y las palabras aún no vistas
    quedan **inalcanzables** en muestreo libre — ataca directamente la causa raíz
    nº 1 de la auditoría (cola fantasma).

### Condiciones de adopción (innegociables — Aleth, 2026-07-08)

1.  **Prohibido "12.000+".** Un objetivo numérico de vocabulario invita a rellenar
    con listas de frecuencia — literalmente el error que produjo el 57,8% de
    fantasmas de la run 2. El techo de la Etapa 4 es *el censo*, sea 6.4k u 11k.
2.  **Gating por certificación, no por edad.** La etapa avanza cuando la batería
    M-n certifica (misma doctrina que la neurogénesis por plateau: dolor, no
    calendario). La máscara activa se persiste en `school_state.json` y se registra
    en cada evaluación — si no, los umbrales pre-registrados dejan de ser comparables.
3.  **Implementación:** `forward()` y `_decode_hidden()` ya aceptan `logit_mask`
    (`src/bitnet/model/modeling_bitnet.py`). Solo falta conectar el currículo al
    parámetro y persistir el estado.

---

## 3. Trit-Level Loss (Dimension-wise Supervision) — 🧪 experimento A/B

**Objetivo:** aprovechar la naturaleza composicional de los glifos para guiar el
entrenamiento a nivel semántico: penalizar errores de significado (predecir "frío"
cuando era "caliente") mucho más que errores de sinónimo.

**Diseño original (v1):** sustituir la Cross-Entropy sobre el vocabulario por
pérdidas trit-a-trit (Sigmoid CE o MSE) sobre los 65 ejes semánticos.

### Revisión (Aleth, 2026-07-08): auxiliar, no reemplazo

*   **Loss híbrida:** `L = CE_vocab + λ · L_trits`, con λ ajustable/anneal.
    Reemplazar la CE es inviable: los glifos derivados (Ridge + θ=0.26) **no son
    únicos por palabra** — con loss solo de trits, dos palabras con el mismo glifo
    son indistinguibles, y la gramática exige distinguir formas concretas.
*   **Ponderación por eje obligatoria:** la mayoría de los 65 ejes de cualquier
    glifo son 0; una MSE plana la domina el fondo de ceros y el gradiente aprende
    "predice todo cero".
*   **Comparabilidad:** mantener las métricas word-level (top-1, PPL) de la batería
    — son los baselines contra la run 2.
*   **Protocolo:** EXP pre-registrado en el lab (umbrales congelados antes de
    correr), A/B sobre las etapas 0-1 de School v3. Si gana en la batería, se adopta
    para el resto; si no, resultado negativo documentado — la honestidad es la marca
    de la casa.

---

## 4. Híbrido Attention+SSM y Mamba Ternario — 🧪 experimento (añadido v4)

**Fuente:** `docs/ideas_ssm_ternario_y_recurrencia.md` (Aleth/antigravity,
2026-07-10; §5 de esa nota = respuestas de Fable a las tareas de validación).

**Diagnóstico verificado contra código:** la memoria inter-turno actual es un hack —
`h[:, 0, :] += 0.5 * h_prev` (`modeling_conversational.py:57`): toda la historia
comprimida en un vector pegado a un token, diluyéndose por atención causal. Heredera
directa de EXP_032 (bucle latente, +2.15%).

**Doctrina de adopción (recalibrada por Fable):**

1.  **El pitch es memoria, no eficiencia.** A 128 tokens/10,9M params el coste O(n²)
    es calderilla. El SSM aporta un estado recurrente *entrenado y selectivo*:
    **atención ternaria = córtex** (parseo estructural de la cláusula K-65P actual);
    **SSM = hipocampo** (resumen corriente inter-turno). Prohibido vender O(1) como
    motivo a esta escala.
2.  **Línea roja de ternarización** (spec del `BitMambaBlock`): ternario SOLO en las
    proyecciones O(d²) (`W_B`, `W_C`, entrada/salida, gate del conv1d → `BitLinear`);
    `W_Δ` empieza en precisión alta (camino Softplus→exp sensible al STE, A/B después);
    **nunca** ternarizar `A_log`/`D`/los factores de descuento `exp(Δ·A)` (decay ∈ (0,1)
    continuo — ternarizarlo mata la selectividad) **ni cuantizar el estado del scan**
    (error acumulativo turno a turno). Regla: *MatMul-free ≠ float-free* — quedan ops
    elemento-a-elemento O(d) en float y la victoria se mantiene. Precedente:
    MatMul-Free LM (Zhu et al., 2024).
3.  **Frontera doctrinal con K-65P:** el estado del SSM es *caché blanda* de la base
    de hechos — acelera, no certifica. La verdad verificable vive en el lado
    simbólico (PrologExpert, §1.3).
4.  **Coste oculto presupuestado:** todo el aparato de neurogénesis (`net2wider`)
    asume bloques transformer. Ensanchar un `BitMambaBlock` (estado, conv,
    proyecciones acopladas) exige sus propias reglas de crecimiento — es la mitad
    del trabajo real y entra en el pre-registro del EXP.
5.  **Hardware (contexto, no motivo):** 10,9M params fp32 ≈ 43,6 MB → ternario
    empaquetado ≈ 2,2 MB (~20×); con SSM, memoria de contexto O(1). Territorio
    Jetson sobrado; microcontrolador plausible. Relevante para la fase robótica.
6.  **Gate:** EXP pre-registrado, DESPUÉS de la comparación Bit-lingüístico vs
    Bit-NSM. No se abre un cuarto frente con School v3 a medio hornear. Destino
    editorial: **paper 1** (sustrato/motor) — no contamina el paper 3.

---

## 5. Net2DeeperNet — Depth Expansion sobre Copia (añadido v5)

**Fuente:** `docs/NOTE_DEPTH_VS_WIDTH_LITERATURE.md` (30 jul 2026) — revisión de
literatura completa (Petty NAACL 2024, Saunshi NeurIPS 2025, Bu/Meta FAIR 2025, BitNet
b1.58 configs, Nielsen 2024). Detalle técnico allí; aquí la doctrina y las restricciones.

**Diagnóstico:** con width en 1024 y 6 capas, la ratio W/D=170 está fuera de toda
configuración BitNet publicada (máx 149, y en un 30B). El plateau de val_loss (~4.37,
época 1305/1408) con sintaxis perfecta pero sin profundidad semántica es el cuadro
clínico exacto que la literatura predice: width memoriza, depth compone.

**Por qué no estaba en este plan:** cronología, no descuido. La v4 se escribió con Bit
en ~10,9M params y doctrina `net2wider`; el problema W/D emergió al agotarse el width.

| Aspecto | Doctrina v5 |
|---|---|
| **Restricción soberana** | El checkpoint de graduación es el **control de G4** (ROADMAP k65p §0.2: *"the control, not a casualty"*). Net2DeeperNet opera **exclusivamente sobre una copia**. Tocar el control destruye el experimento central del proyecto. |
| **Ventana de ejecución** | Post-graduación, mientras G2 (sin GPU) y G3 (fábrica, CPU) avanzan — la RTX queda libre. ~18 GPU-h para 200 épocas. No retrasa el track K-65P ni un día. |
| **Spec de inicialización** | **Solo proyecciones de salida a 0** (O de atención + down-proj del MLP); Q/K/V y up-proj init normal. Identidad preservada (output=0 → residual puro) con gradiente vivo. NO todo-a-cero: mata gradientes. |
| **Mina ternaria** | absmean de tensor nulo → escala 0/división por cero. Capas nuevas en FP16 con ternarización tras warmup, o epsilon/clamp en la escala. |
| **Objetivo** | 6→8 capas → ratio W/D=128 (validada en BitNet 30B). D_crit ~ W^0.44 ≈ 21 capas con W=1024: margen enorme antes de la zona donde depth daña. |
| **Señal de éxito / fracaso** | val_loss baja >0.05 en ~100 épocas y las muestras ganan profundidad semántica / sin mejora en 200 épocas → documentar negativo y desenfundar §6. |
| **Destino editorial** | Paper 1 (motor). Sus resultados informan a **ambas** líneas (control y nativo) tras el veredicto G4 — nunca solo a una (§7.2). |

**Progresión:** A.1 (6→8) → A.2 (8→10, ratio 102) solo si A.1 gana → A.3 (progresivo
por plateau, misma doctrina de dolor que la neurogénesis).

---

## 6. Arsenal — Balas en la Recámara (añadido v5)

**Doctrina:** ninguna línea de investigación se descarta. Cada hallazgo del NOTE y de
este plan que no está en ejecución activa vive aquí, con su **gatillo** (qué estancamiento
la desenfunda), su riesgo y su referencia. Si una línea activa se atasca, se viene a esta
sección antes de improvisar. Las **líneas rojas de ternarización del §4.2 aplican a TODA
bala con SSM** (W_Δ en precisión alta; nunca ternarizar `A_log`/`D`/factores de descuento;
nunca cuantizar el estado del scan).

| Bala | Qué es | Gatillo (cuándo desenfundar) | Riesgo | Ref |
|---|---|---|---|---|
| **R1 — Depth progresivo** (A.2/A.3) | 8→10 capas y más allá, Net2DeeperNet encadenado por plateau | A.1 gana pero el siguiente plateau llega con ratio aún >102 | Bajo | NOTE §6.1 |
| **R2 — SSM en la resonancia** (F) | Estado SSM dentro del loop de resonancia. ⚠️ Dos arquitecturas bajo un nombre: **F-step** (memoria entre iteraciones de resonancia — enriquece el refinamiento) vs **F-token** (scan sobre la secuencia — memoria de largo alcance real). Desambiguar ANTES de implementar. Variantes F.1-F.4 (paralelo/alterno/pre-proceso/bidireccional) | La resonancia muestra techo: más `n_steps` no mejora, o el refinamiento iterativo pierde señal | Bajo-Medio (añade, no reemplaza; desactivable) | NOTE §6.5 |
| **R3 — Híbrido Jamba-style** (C) | 1 capa de atención cada 3-5 SSM (Jamba recupera 87% del recall con ratio 1:7). Variantes C.1-C.3 (SSM-heavy/attn-heavy/shared-attn Zamba) | El BitMambaBlock (§4) gana su A/B y se busca extenderlo a capas enteras; o el recall se degrada al crecer contexto | Medio | NOTE §6.3 |
| **R4 — SSM puro** (B) | Sustituir toda la atención por BitSSM (Ternary Mamba: W1.58 QAT, absmean no aprendible, g=128, zero-ratio collapse documentado) | R2/R3 demuestran que el SSM ternario funciona en nuestro sustrato Y el contexto largo se vuelve requisito real (fase robótica / L≥512) | Medio-Alto | NOTE §6.2 + §5.2 |
| **R5 — SENN η score** (D) | η = gᵀ·F⁻¹·g (Fisher vía Kronecker: A_l⊗S_l) decide cuándo/dónde/qué expandir — width, depth o SSM. Meta-capa sobre la neurogénesis. Variantes D.1 (solo *cuándo*) / D.2 (completo) | Las decisiones manuales de crecimiento dejan de ser concluyentes (dos expansiones seguidas sin mejora clara), o llega la era post-G4 con dos Bits que crecer a la vez | Alto (Fisher inversa estable + overhead) | NOTE §2.3 + §6.4 |
| **R6 — Trit-Level Loss** (§3) | Ya es propuesta activa de este plan (A/B pre-registrado) — listada aquí solo para que el arsenal sea el índice completo | Etapas 0-1 de School v3 (ya gateado) | Bajo | Plan §3 |
| **R7 — Combinaciones** (E) | Roadmap multi-fase A.1→A.2→F.1→F.2→C.1→B.3→D.2 del NOTE — la escalera completa si todo va ganando | Cada peldaño gana su gate; nunca saltar dos a la vez | Compuesto | NOTE §6.6 |

**Matriz de decisión completa** (riesgo/impacto/tiempo/compute por opción): NOTE §6.7.
No se duplica aquí; el NOTE es el documento de detalle técnico, este plan es doctrina.

---

## 7. Secuenciación Soberana (añadido v5)

### 7.1 Autoridad y calendario

La secuenciación global la gobierna el **ROADMAP de k65p** (`docs/CORE/ROADMAP.md`,
fases 1-5, gates G1-G4). Este plan gobierna la **línea del motor** (papers 1) dentro
de las ventanas que ese calendario deja. Estado al escribir: G1 cerrado; Bit lingüístico
en época ~1305/1408, plateau patience 12/15.

| Cuándo | Qué | GPU |
|---|---|---|
| Ya (paralelo al training) | G2: KB v0 + `verify --explain` (repo k65p) | No |
| Graduación del Bit actual | **Congelar checkpoint = control de G4.** Intocable desde ese instante | — |
| Ventana post-graduación | G3 (fábrica bilingüe, CPU) + **§5 Net2DeeperNet sobre copia** | Sí, libre |
| G3 cerrado + control graduado | Formación v2: Bit nativo K-65P desde cero, doctrina heredada (§7.2) | Sí, prioritaria |
| Veredicto G4 | Se desbloquean: BitMambaBlock (§4), K-65P v1, Sistema 1/2, paper 3. Resultados de §5 se aplican a ambas líneas. El arsenal (§6) queda disponible para las dos | — |

### 7.2 Pre-registro del confound de crecimiento (innegociable)

El Bit nativo de Formación v2 hereda **exactamente la misma política de crecimiento**
que tuvo el control (net2wider por plateau, mismos umbrales) — aunque para entonces
sepamos por §5 que existe una política mejor. Si el nativo crece con doctrina
depth-balanceada y gana G4, cualquier revisor responde *"le diste mejor arquitectura,
no mejor lenguaje"* y el resultado central del proyecto muere en review. Los
aprendizajes depth/SSM se aplican a **ambas líneas después del veredicto**, nunca a
una sola antes. Esto se copia al pre-registro de la batería de Fase 4 antes de la
época 1 del nativo.

**Backend numérico (decidido 2026-07-31):** el nativo entrena con BF16 (autocast,
maestros FP32) + `torch.compile` + SDPA **desde la época 1**, congelado hasta G4.
NO se replica la transición a mitad de run del control (FP32→BF16 en ép. ~998):
eso fue accidente histórico, no protocolo. Cobertura experimental: **EXP_079 Tier 2**
(3 brazos from-scratch, umbrales congelados) — equivalencia FP32↔BF16↔BF16+compile
con |Δ| val_loss medio 0.003 (umbral 0.05) y ∇STE sano. Sensibilidad conocida: el
contador de plateau puede desplazar neurogénesis ±5 épocas entre backends → razón
extra para no cambiar de backend a mitad de run. Se declara en el pre-registro de
la batería con EXP_079 como evidencia.

---

## Orden de ejecución

1.  **Gating** entra en School v3 ya (§2, condiciones 1-3).
2.  **Trit-loss** como A/B pre-registrado durante las primeras etapas (§3).
3.  **G2 en k65p** (KB v0 + `verify`), paralelo al training actual — sin GPU (§7.1).
4.  **Graduación del Bit lingüístico** → congelar el control de G4 (§7.1). Intocable.
5.  **Ventana de GPU:** §5 Net2DeeperNet A.1 sobre copia, mientras G3 (fábrica,
    CPU) produce el corpus nativo. Resultado negativo → arsenal (§6).
6.  **Formación v2** (Fase 4 del ROADMAP k65p): Bit nativo desde cero, doctrina de
    crecimiento heredada del control sin modificar (§7.2). Batería pre-registrada,
    métrica de verificabilidad incluida (§1.3.2).
7.  **Veredicto G4.** Si gana el nativo: K-65P v1 (§1.3.3) → puente Horn con
    PrologExpert (§1.3.4) → Sistema 1/2 (§1.3.5) → fundar
    `paper_3_k65p_neurosymbolic` (§1.3.6).
8.  **Tras el veredicto (línea del motor, paralela a la 7):** EXP pre-registrado del
    `BitMambaBlock` (§4) — primero las reglas de neurogénesis para bloques SSM, luego
    el A/B contra el `h_prev` actual. Si gana, entra en el paper 1. Los resultados de
    §5 y las balas del §6 se aplican a **ambas** líneas desde aquí.

---

*Aleth Core File Anchor: `/home/joan/Documents/IA/docs/bitnet_next_architecture_plan.md`*
*(corregido: la v1 apuntaba a `Agent_Core`)*
*v1: 2026-07-07 (Aleth/antigravity) · v2: 2026-07-08 (Aleth/Claude Fable 5) · v3: 2026-07-10 (convergencia LISP, Joan + Aleth Fable) · v4: 2026-07-10 (SSM ternario, Aleth/antigravity + auditoría Fable) · v5: 2026-07-31 (Net2DeeperNet + Arsenal + secuenciación soberana, Joan + Aleth Fable)*
*Status: por propuesta — ver §0 · Documentos hermanos: `NOTE_DEPTH_VS_WIDTH_LITERATURE.md` (detalle técnico) · `k65p/docs/CORE/ROADMAP.md` (autoridad de secuenciación)*
