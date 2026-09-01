# Frankenswarm Changelog

## [Unreleased]

## [Unreleased]

### 🔬 BIT-003 — primera confrontación honesta: glyph vs standard ×1 + el expediente de tres instrumentos (2026-08-31/09-01)

- **[RESULT] El estándar a 128d superó al glyph a 256d en los tres instrumentos**: bajo el protocolo honesto ×1, el glyph SUSPENDIÓ 2_years a 128d (neurogénesis disparada: 128→256, primera ejercición de la escalera) mientras el estándar aprobó 2 y 3 años a 128d sin crecer. Batería congelada edad 4 (58 vistas + 195 no vistas, 5 estratos): glyph gate 70.7% / cognición 4.1%; standard gate 50.0→69.0% / cognición 5.6→3.1%. Veredicto: **resultado adverso a la hipótesis composicional a esta escala**, con confusores declarados (transitorio post-neurogénesis, una semilla).
- **[NEW] Matriz de puntuación 3×2 del operador**: visto +1/0/−1 (alucinar lo enseñado es el pecado) · no visto +2/0/0 (generalizar es el premio doble, fallar no castiga). La ignorancia total honesta (+58) supera al Bit actual (+24/+16): el gradiente empuja a aprender, no a alucinar.
- **[NEW] Gate de retención**: 80 frases aleatorias del pool de etapa, cloze determinista contra la palabra real del corpus — cero curación (el corpus ES el bank). Mide retención de la exposición natural: 30.8% (v4 256d).
- **[DISCOVERY] Curva riesgo-cobertura**: la señal "sé lo que no sé" existe latente en la confianza de Bit (respondiendo solo al 20% más seguro: 78% de acierto vs 19.4% medio, descenso monótono). Falta el canal de silencio: "i dont know" está en vocab y gateos — el argmax forzado lo impide.
- **[PROFILE] Por estratos**: cruce ~5-10%, interrogativo/riddles/aritmética/predicado ~0% — la generalización composicional a cero en cualquier representación a esta escala.

### 🧬 K-65P — programa de descomposición NSM y diccionario bidireccional (DL-014, 2026-09-01)

- **[DOCTRINE] "Todo glifo descompone"** (operador): cada trit lleva significado (+1 afirma, −1 contrasta, 0 silencio); cualquier palabra humana converge en los 65 primos. La auditoría Ridge demostró lo contrario en la implementación → el resultado adverso corresponde al atajo, no a la hipótesis (addendum en RFC-002).
- **[NEW] `nsm_explication.py`**: explicación NSM → glifo de 65 trits, **calibrado 28/28** contra los canónicos (test de ida-vuelta RFC-002 §5); validador de consistencia (Jaccard esperado entre relacionadas) que cazó 2 desalineaciones en el demo.
- **[NEW] `k65p_dictionary.py` — diccionario SQLite bidireccional phrase-aware**: words (glifo + explicación + estado canonical/molecule/prime/explicated/drafted/pending), surfaces, phrases (compresión N→1: phrasal verbs y descripciones que colapsan en un concepto, `longest_span_match`), relations. Poblado: 19,703 entradas. Es el aterrizaje del programa de descomposición y del traductor EN↔K-65P.
- **[NEW] Fundación del traductor**: `en_lexicon.py` (142 superficies → 65 primos) + `nsm_syntax_en.py` (gramática NSM-C EN espejo de la ES — 21/21 árboles del RFC §3). El compilador transductor y el programa de descomposición por lotes son los siguientes pasos.
- **[NEW] `k65p_dictionary` phrases**: compresión N→1 — "give up" (phrasal verb), "it says bark" → dog (riddle 3→1): el inglés torticero se registra como entradas del diccionario, no como excepciones.

### 🔧 BIT-003 — offload forzado de Samantha tras examen (2026-08-31)

- **[FIX] Colisión VRAM examinadora ↔ entrenamiento**: la examinadora reutiliza el hipervisor dual-bind persistente (`run_dual_bind.py`, puerto 8760 — `samantha_on_demand` lo prefiere si vive, la limpieza efímera nunca aplica), que retiene el modelo (~6.3 GiB) durante el idle timeout de ~15 min tras calificar → el `loss.backward()` del entrenamiento que sigue al examen estalla (3ª colisión: OOM 15:46 del 31-ago, auto-healada por el runner). **Fix**: `state_manager._force_samantha_offload()` — tras cada examen real, `POST :8760/unload` (endpoint nativo del daemon, `unload_under_lock()`). Guardias: no yankuea si el sueño está en fase activa (el idle timeout cubre), fallos de red no crashean el entrenamiento, mock no descarga. Endpoint verificado en vivo (`{"status":"unloaded"}`).
- **[DOC] Progreso v3**: hito `3_years` aprobado — primer gate honesto bajo ×10 con cobertura completa, a 128 dim, sin neurogénesis (2/7 hitos; política "cerebro mínimo" acumulando evidencia).

### 🔧 BIT-003 — fixes de continuidad del sampler cíclico (2026-08-30)

- **[FIX] `CyclicPoolSampler` faltante en el camino adaptativo (`d7b77cc`)**: el replace por substring de DL-011 no insertó la creación del sampler en la compilación de etapa del bloque adaptativo → `KeyError` al primer muestreo. Cazado por el job runner en intentos 1-2; insertado con edición exacta + print de cobertura (`Gen: 21M | Curr: 12k | Val: 37k | 400k/ép → ~54 ép/pase`).
- **[FIX] Persistencia `cursor/wrapped` del `CyclicPoolSampler` (`7f46235`)**: cada step (20 épocas) re-barajaba con la misma semilla (`seed 42`) y cubría SIEMPRE las primeras ~8M secuencias del pool → `full_coverage` inalcanzable → etapa 2-3 en bucle (419 épocas > tope 200, `val` plana 3.74). Ahora `cursor` y `wrapped` persisten en `school_state.json:samplers.{idx}` y se restauran al compilar la etapa con permutación determinista. Verificado: test unitario de continuidad cross-proceso (sin huecos) + producción (`cursor 400k` persistido tras época 1).
- **[ROLLBACK]** Etapa 2-3 reiniciada desde el hito `2_years` (época 137): las 419 épocas entrenaron un subconjunto sesgado del 37% (asimetría contaminante para la comparativa). Etapas 0-1 limpias (cobertura completa intra-proceso, hito `2_years` legítimo bajo ×10). Run v3 en `bit003_glyph` (ép. 137, etapa 2-3). Incidencia documentada en `.red-pill/memory/BIT-003_followups.md` y corrección de `docs/RFC_INSTRUMENT_V3.md` §Limitaciones.
- **[DOC] Protocolo anti-regresión** (obligatorio desde 30-ago): todo cambio de protocolo de entrenamiento requiere (1) test unitario, (2) smoke 1 época vía receta de job, (3) test de continuidad cross-proceso si cruza procesos, (4) verificación de que las etapas pueden cerrar (`cobertura ≤ max_stage_epochs`).

### 🎓 BIT-003 — Instrumento v3: exámenes ×10, cobertura cíclica total y batería 80/50 (DL-011, 2026-08-30)

- **[NEW] `CyclicPoolSampler` — muestreo cíclico permutado** (`6a19325`): cada etapa recorre su pool completo en permutaciones sucesivas (400k/época); el muestreo aleatorio anterior cubría solo ~25-30% del pool por etapa (coupon collector). Garantiza exposición de TODAS las secuencias del nivel.
- **[FIX] Gate de hito por cobertura total**: el examen solo puede cerrar con `full_coverage == true` → la afirmación "graduó N años habiendo visto todo su corpus gateado" pasa a ser verdadera por construcción. Periodos de cobertura medidos: E0 ~5 ép … E7 ~106 épocas.
- **[NEW] `--exam_repeat_factor` 300 → 10** (default 10): fija el hecho sin memorización patológica (~9.000 exposiciones por etapa antes). Efecto buscado: el gate se endurece y la escalera de neurogénesis puede ejercitarse (cero neurogénesis en 209 épocas bajo ×300).
- **[NEW] Batería 80/50** (instrumento v3, retroactiva): 80 vistas (gate) + 50 NO vistas por edad — composiciones nuevas de hechos conocidos verificadas ausentes por script — como **métrica primaria de cognición** de la tesis; ahí debe verse la ventaja composicional de los glifos. n=50 → IC95 ±13.9% a 50%. Diseño en `docs/RFC_INSTRUMENT_V3.md`, generación pendiente.
- **[MISC] Ablación v2→v3**: run v2 (`bit003_glyph_x300_v2`, 4 hitos 2-5 años, época 229) archivado — cuantificará el coste de memorización del ×300 sobre la misma arquitectura.
- **[DOC]** `docs/RFC_INSTRUMENT_V3.md` + `docs/DECISION_LOG.md:DL-011`.

### 🧠 BIT-003 — brazo resonante: bucle latente + emoción `first_only` (DL-010, 2026-08-30)

- **[NEW] Flags de resonancia en `train_sovereign_school.py` y `evaluate_samantha_age.py`** (`68be4ab`): `--resonance_steps_max` (5), `--resonance_pos_mode` (`clock`), `--resonance_ramp` (`U[1,5]` por batch), `--resonance_eval_steps` (3), `--n_emotions` (7 = dojo 6 + `neutral` id 6), `--emotion_dim` (16), `--emotion_mode` (`first_only`). `forward_resonance` en caminos adaptativo y clásico; val/exámenes con emoción `neutral` (in-distribution, determinista); gateo de etapa aplicado fuera (`apply_stage_gate`, float 0/-inf; `_decode_hidden` usa bool — no mezclar).
- **[NEW] Evaluador acoplado**: `state_manager` propaga flags al subproceso de Samantha; el evaluador instancia el modelo resonante (`resonance_clock` + `emotion_embeddings` + `emotion_proj`) e infiere con `forward_resonance` — el checkpoint resonante falla en `load_state_dict` sin ellos por diseño.
- **[NEW] Receta `configs/jobs/bit003_glyph_resonant.yaml`** (`state_dir bit003_glyph_res`, en serie tras los controles normales).
- **[FIX] Bug de unidades en censo cache-hit del store CSR** (`n_general` = conteo de secuencias usado como índice de tokens): censo 8.161 → **18.468** palabras únicas (ventana falsa de 1.45M tokens); máscaras E5/E6 estrechas (8,164/8,165 vs 10,003/15,002). Fix: límites de token desde `offsets` (9,439,798 tokens / 18,468 únicas — exacto al path miss). Detectado por el smoke del propio brazo resonante (gateo impreso ≠ auditoría). Rollback del run a la entrada de etapa 5 (checkpoint `5_years`, época 223); hitos 2-4 limpios (path miss con censo correcto).
- **[VERIFIED]** BPTT grad ≠ 0 en `resonance_clock` + emociones; rampa 1-5 válida; convergencia del bucle (coseno 0.19→0.95); smoke GPU época completa con gateo correcto; evaluador carga resonante + examen OK. Modelo resonante: **1,250,680 params** (+2,800).
- **[DOC]** `docs/RFC_RESONANCE_ARMS.md` + `docs/DECISION_LOG.md:DL-010`. Notas: resonancia SOLA es NULA (EXP_033/034 B≈A, siempre con emoción `first_only`); el bucle NO son 18 capas (mismas 6 reutilizadas, cómputo 18, params 6); SSM/Mamba ortogonal (hipocampo inter-turno, hook `h_prev`).

### ⚡ BIT-003 — corpus como store CSR numpy + receta de job (2026-08-22/30)

- **[PERF] `src/bitnet/training/modules/corpus.py:save/load_tokenized_store` (`2eb80f0`)**: store `CSR` en `.npz` (flat `int32` + `offsets` `int64`). El caché JSON de 2.48 GB construía ~18-20 GB de objetos Python al cargar (`oom-kill` con `MemoryMax=16G`); el store ocupa **~1.9 GB residente** (`5GB RSS` vs 20GB antes).
- **[NEW] `stage_gating.py:classify_store_by_gate` + `partitioner.py:build_stage_pools/materialize_sequences`**: clasificación `CSR` idéntica a la de listas (conteos verificados exactos), pools por etapa como índices `int64` con caché en disco (`.npz`, deterministas dado `corpus_hash`), diálogos primaria/secundaria como segmentos del store (etapas 4+/6+). Trainer: pools = índices; `compile_data_for_stage` cachea índices `.npz` en vez de JSON multi-GB; muestreo por época con `np.random.default_rng`.
- **[NEW] `configs/jobs/bit003_glyph.yaml`**: receta `script_job` (`memory_max 12G`, 20 épocas/step, `pause_exit_code 78`, preflight VRAM difiere si hay sueño).
- **[FIX] Clave `n_dial10` consistente save/load del store CSR (`13a9215`)**: save usaba `n_dial` y load `n_dial10` → crash en `hit` (intent 2 del runner); el smoke no lo vio porque siguió ramo `miss`.
- **[VERIFIED]** Clasificación idéntica histórica (`E0:1,663,635 … E7:2,919,121`), gateo `[254…19636]` intacto, 1 época con loss finito, hit-path verificado.

### 🎓 BIT-003 — corpus inglés desde cero, gateo de vocabulario por etapa e instrumento v2 (2026-08-20/21)

- **[NEW] Corpus EN íntegro y trazable**: `childes_pre_school_en.json` (1,45M oraciones
  de habla infantil real, fuente AoA evaportelance) + TinyStories + diálogos EN; censo
  directo sin mapeo OOV (RULE 7) → vocabulario de **19.637 palabras** con **28/28
  glifos canónicos** y 19.637 glifos únicos. Artefactos CHAT (`xxx/yyy/www`, 51.440
  tokens que habrían entrado al top-30 del censo) eliminados del corpus, y apóstrofos
  normalizados en la tokenización canónica única (`words_of`: `don't→dont`, como la
  fuente) — la forma partida `don't`(censo 0)/`dont`(censo real) arrastraba TinyStories
  a etapas tardías.
- **[NEW] Gateo de producción por etapa (DL-008)**: máscara de logits por edad — núcleo
  (28 referencias EN + safe words) + top-N de CHILDES por frecuencia de adquisición +
  currículo + respuestas de examen de la etapa. Gateo medido:
  `[254, 825, 3008, 5003, 8005, 10003, 15002, 19636]`. Clasificación del corpus al 95%:
  CHILDES cae 78% en E0-E2 (3,5% E7); TinyStories baja su E7 del 16% al **7,1%** al
  curar las contracciones. `<unk>` vetado en las 8 etapas, E7 incluida. La pérdida
  excluye targets vetados; el input ve el corpus completo.
- **[FIX] Remediación pre-lanzamiento (DL-009, auditoría del 21-ago)** — 5 bloqueantes
  que habrían quemado el run: `NameError` de arranque (TinyStories se cargaba después
  del hash que lo necesita), censo de CHILDES por **ids** en vez de palabras (las
  máscaras quedaban en [107..179] y el 99% del corpus se clasificaba a E7), device
  mismatch CPU/CUDA en `loss_mask_for_gate` (crash en el primer batch GPU, invisible en
  smokes CPU), `<unk>` producible en E7, y `xxx` entrenable desde E0.
- **[NEW] Instrumento de evaluación v2**: exámenes y `AGE_QUESTIONS` a **10 preguntas
  por edad**, 100% dentro del vocabulario (fuera `aleth/bunker/madrid`: el brazo
  control mide adquisición de lenguaje, no memorización de tokens sin presencia en
  corpus). El evaluador de Samantha consume la **máscara real de la etapa**
  (`stage_gate_masks.json` persistida por el trainer) en generación y monitor OOB —
  el gateo legado permitía 18k/20k palabras a edad 2 frente a las ~800 entrenables.
  Veredicto y mock alineados al tamaño real de la batería (un `== 5` hardcodeado
  suspendía todo examen con media 10/10).
- **[NEW] Currículo preescolar rico**: 32 items deterministas validados contra el
  censo, 4 bins MLU poblados `[10,8,8,6]` (antes `[0,0,0,4]`).
- **[NEW] Aceptación reproducible**: `scripts/audit_stage_gating.py` (22 checks, fuente
  de las tablas del RFC), `scripts/validate_exam_vocab.py` (instrumentos in-vocab) y
  `scripts/clean_childes_en_artifacts.py` (idempotente). Verificado: pytest 433 ✓,
  smokes CPU+GPU de ambos brazos ✓ (loss inicial 5,14 ≈ ln(254): el gate actúa),
  dry-run del examen E1 con máscara real → APROBADO ✓.
- **[DOC]** `docs/RFC_GATING_VOCABULARIO_ETAPAS.md` re-medido, DL-008 + DL-009 en el
  decision log; `samples_per_epoch` por defecto a 400k (análisis Chinchilla,
  confirmado por operador). Runbook de lanzamiento de ambos brazos en
  `.red-pill/memory/BIT-003_fix_plan.md` §F8.3.

### 📐 RFC-GROWTH-V6 — la premisa de Net2DeeperNet caducó; la rejilla D×W la sustituye (2026-08-14)

- **[NEW] `docs/RFC_GROWTH_V6_DEPTH_WIDTH.md`** (🟡 propuesta, pendiente de
  ratificación): plan de crecimiento posterior a DL-006. Tres experimentos con
  criterios pre-registrados — E1 encender la resonancia (looped transformer ya
  implementado y **desconectado** en la escuela: profundidad efectiva 12 por 384
  parámetros, vs ~394.000 de dos capas nuevas), E2 rejilla profundidad×anchura a
  parámetros constantes en K-65P (5 puntos, W/D de 2,7 a 112, ~3 GPU-h), E3 N2DN solo
  si pasa su gate. **E1+E2 ≈ 12 GPU-h, menos que las ~18 del único experimento N2DN
  que sustituyen.**
- **[⚠️ CADUCIDAD] `bitnet_next_architecture_plan.md` §5 marcado como premisa
  caducada**: el diagnóstico "W/D=170 fuera de toda configuración BitNet" describía el
  v1 de 1024d que DL-006 declaró artefacto del calendario. A las anchuras reales
  (128-256d) la ratio es 21-43 y `D_crit ≈ W^0,44` da 8,5-11,5 capas, así que 6→8
  aterriza en el techo en lugar de lejos de él: la fórmula que justificaba el plan hoy
  lo desaconseja. Ficha de "✅ aprobada" a "⚠️ gateada". **La doctrina de la sección
  (solo sobre copia, control G4 congelado, spec de init, mina ternaria) se conserva
  íntegra.**
- **[HALLAZGO] `forward_resonance` está implementado y sin usar en la escuela**: bucle
  latente cerrado con BPTT (`forward_resonance_training`) y reloj posicional por paso.
  Es la arquitectura de Saunshi (NeurIPS 2025) que la propia nota de literatura cita
  — 12 capas en bucle 2× superan a 24 capas con la mitad de parámetros — y el trainer
  construye con `max_resonance_steps=0` y llama al `forward` plano.
- **[DEUDA] Sin CLI para la geometría**: `num_layers` está hardcodeado a 6 en el
  trainer (sin flag) y no hay flags de resonancia. E1/E2 los necesitan; es el único
  cableado que piden.
- **[DOC] Balance de la tesis al día**: el informe del 5-ago marca ahora, en la propia
  lectura 3, qué baza cayó y por qué — (a) no replicó (10-ago), (b) retirada (DL-007),
  **solo (c) vocabulario en caliente sigue en pie**; la ventaja que sí sobrevive y no
  estaba en la lista es la compresión. `lab/BRIEFING.md` recoge el estado de la tesis
  y dos lecciones nuevas (arreglar el baseline antes de comparar; la resonancia está
  apagada en la escuela).

### ⚖️ DL-007 — el baseline estaba lisiado: brazo estándar 12× más rápido y se retira la baza de coste del glifo (2026-08-14)

- **[FIX] Fast-path one-hot en `BitNet4LayerModel`**: el brazo estándar multiplicaba
  por una identidad V×V en la salida (matmul nulo) y materializaba one-hots densos
  en la entrada. Cortocircuitado, **bitwise idéntico** (`torch.equal`, no
  tolerancias): **291,94 → 24,16 ms/step (12,1×)** a V=12.143, dim 128, BF16.
- **[FIX] La tabla identidad deja de persistirse**: checkpoints del brazo estándar
  de **579 → 16,6 MB (35×)**. `load_state_dict` descarta la clave sobrante, así que
  los checkpoints antiguos siguen cargando exactos y los ~60 sitios de carga del
  repo no se tocan.
- **[🔴 RETRACTACIÓN] "El glifo decodifica 4.7× más barato" queda retirada** (baza
  (b) del informe del 5-ago): medía la implementación del baseline. Con el baseline
  justo el estándar es 10% MÁS rápido, y el escalado va en su favor (ratio
  standard/glyph 0,97 → 0,90 al pasar V de 1.000 a 12.143). El `O(65·d)` describe el
  tamaño de la tabla, no el coste del logit: el glifo compone las V palabras desde
  los primos en cada forward.
- **[REFORMULACIÓN] La ventaja del glifo sigue siendo real, pero es compresión, no
  velocidad**: 3,5× menos parámetros (1,25M vs 4,35M) y checkpoint 2,1× menor a
  rendimiento comparable. Con M5 (vocabulario en caliente), la tesis defendible es
  "más pequeño y extensible", no "mejor modelo de lenguaje".
- **[FEAT] `--wd_mode {uniform,no_embed}`**: con decay uniforme el embedding de una
  palabra rara del brazo estándar se encoge entre sus actualizaciones infrecuentes y
  el glifo no sufre eso — los brazos no recibían el mismo trato de regularización.
  Default `uniform` (histórico, D3); `no_embed` es el simétrico, recomendado para
  BIT-003 y sin medir todavía.
- **[NEW] `scripts/bench_embedding_arms.py`**: instrumento que produce la cifra de
  coste publicable, con resultados acumulativos por etiqueta.
- **[FIX] `orchestrator.py` resolvía `--amp auto` como fp32** (mismo fallo que ya se
  curó en el trainer).
- Nada de aprendizaje queda invalidado: la equivalencia es exacta, así que las
  réplicas multi-semilla DL-006 y la lectura 2 siguen en pie.

### 🛫 Preflight BIT-003 — auditoría del refactor y desminado de cachés (2026-08-14)

- **[AUDIT] `docs/sessions/20260814/PREFLIGHT_BIT003.md`**: auditoría de coherencia
  previa al reentrenamiento de v1 en inglés. El trainer refactorizado (PR #5) nunca
  había corrido una escuela real; smoke test adaptativo en sandbox incluido.
- **[FIX] Caché de etapa keyeada por corpus**: `stage_cache/<corpus_hash>/` — la caché
  plana global habría re-servido el dataset spanglish (o token-IDs de otro censo) en
  silencio tras regenerar currículo/vocabulario.
- **[FIX] `school_exams_en.json` entra en `compute_corpus_hash`**: los exámenes se
  mezclan ×300 en el dataset de etapa y no invalidaban ninguna caché.
- **[FIX] `datasets` declarado en `pyproject.toml`**: el trainer lo importa para
  TinyStories y no estaba ni en el venv — cualquier run con caché inválida moría en
  `ModuleNotFoundError`. Resuelve limpio junto a fastembed (el conflicto HF era con
  `transformers`).
- **[FIX] Acta de suspensos en modo adaptativo**: `_save_adaptive_state()` machacaba
  el `exam_failures` que `run_samantha_eval` acababa de incrementar.
- **[FIX] `select_strategy` recibe el modo AMP resuelto**: con `--amp auto` elegía la
  estrategia `fp32` (afectaba al log del acta y al cruce con `--opt8bit`).
- **[NEW] `scripts/download_childes_en.py`** (BIT-003 §1): CHILDES en inglés de la
  fuente verificada, sin mapeo OOV ciego (RULE 7). No toca el fichero español.
- **[⚠️ GUARDIA] `--opt8bit` + neurogénesis sin verificar** (mapeo de estados
  cuantizados en `net2wider`): no usarlo en BIT-003; la receta certificada DL-002
  sigue siendo `--amp auto --compile`.

### 🔬 Gating curricular + hot-vocab + extractor de vocabulario — y el hallazgo del spanglish (2026-08-12)

- **[FEAT] `train_sovereign_school_k65p.py` — gating curricular semántico**:
  desbloqueo de vocabulario por categoría (base → animales → comportamiento),
  reflejando el espíritu de v1. Nuevo `SEMANTIC_TIERS` + `_gated_molecules`
  (en vez del orden alfabético) y `--corpus`/`--lang` para la escuela semántica.
- **[FEAT] `--hot_vocab`**: gating REAL de vocabulario — el brazo glyph arranca
  solo con el tier base e inyecta moléculas en caliente (`register_new_word`) en
  las transiciones de etapa. **Resultado: la inyección en caliente no cuesta
  nada** (rinde igual que el gating de máscara, 40% gen_true), y es la capacidad
  exclusiva del glyph (M5) a escala de currículo entero.
- **[FIX] Bug de size-mismatch en hot-vocab**: el entrenamiento por épocas
  atómicas reconstruía el vocabulario en orden distinto al de la inyección
  (`load_weights` fallaba 174 vs 160). Fix: orden canónico determinista
  (`_tier_molecule_order`) compartido por reconstrucción e inyección.
- **[NEW] `scripts/extract_semantic_vocabulary.py`**: extrae el vocabulario por
  etapa del currículo de v1 y lo cruza con `expanded_glyphs.json` (palabras +
  glifos ternarios ya calculados). ~97/72/79 palabras con glifo por etapa.
- **[NEW] `configs/jobs/school_semantic_hotvocab.yaml`**: receta de la variante
  hot-vocab.
- **[CONVENTION] RULE 8 en `CONVENTIONS.md`**: el vocabulario del corpus
  semántico sale del diccionario real de v1 (`expanded_glyphs.json`), nunca de
  moléculas inventadas a mano.
- **[RESULT] Gating curricular (n=15, seed 770)**: glyph gana en narrativa 5/5
  vs standard 2/5 (la composición transfiere entre agentes que comparten primos:
  dog/cat/wolf). La lógica abstracta sigue sin discriminar (heldout/recall ~0).
- **[⚠️ HALLAZGO] El corpus de v1 era spanglish**: CHILDES se descargó en
  ESPAÑOL (`download_childes_spa.py`) y la traducción "fast" a inglés con
  Samantha quedó mezclada ("the suelo se mías"). v1 no es un modelo de inglés
  puro. **Decisión del operador: re-entrenar v1 en inglés (CHILDES-en +
  TinyStories-en)** con las optimizaciones (BF16 + DL-006). Tarea `[BIT-003]`.
- **[IDEA] Juez semántico gradual ternario `{-1, 0, +1}`**: sustituye al
  gen_true binario — la semántica es gradual (el perro ladra +1, aúlla 0,
  maúlla -1), no binaria. Coherente con los trits de K-65P y los pesos ternarios
  de BitNet.

### 🎓 Escuela Semántica — fábrica de corpus causal, gen_true vía Prolog y primera cata (2026-08-11)

- **[NEW] `scripts/generate_semantic_corpus.py`**: fábrica de corpus K-65P con
  VERDAD por construcción. A diferencia del generador sintáctico
  (`generate_k65p_corpus.py`), parte de una KB causal de 69 hechos + reglas en
  3 niveles (preschool/primary/secondary) y genera expresiones válidas Y
  verdaderas (consistentes con la KB), variaciones por entidad, pares
  contrastivos falsos (NOT) y teoremas held-out para el examen de reasoning.
  Produce `factory_semantic/kb.pl` como KB de verdad para el juez Prolog.
- **[NEW] `scripts/exam_gen_true.py`**: examinador que convierte la continuación
  generada a Prolog (vía bridge) y comprueba si se DEMUESTRA de la KB compilada
  con swipl. Sucesor de `gen_valid`: donde el examinador sintáctico preguntaba
  "¿es válido?", gen_true pregunta "¿es verdad?". Separa teoremas held-out
  (reasoning) de recall.
- **[NEW] `scripts/exam_m5_hot_word.py`**: examen de vocabulario en caliente que
  inyecta una palabra nueva (trueno) con glifo compuesto de primos y mide uso
  inmediato. 3 pruebas: inyección en frío (composición mueve logit), contraste
  estructural (NO EVALUABLE sin semántica) y consolidación (≤5 épocas, emisión
  98.5%). La capacidad de `register_new_word` es real y exclusiva del brazo
  glyph (el standard no puede por construcción).
- **[NEW] `configs/jobs/school_semantic.yaml`**: receta `script_job` para la
  escuela semántica con `--corpus factory_semantic --lang en`. Protocolo
  adaptativo DL-006, todas las moléculas disponibles desde la etapa 0.
- **[FEAT] `train_sovereign_school_k65p.py`**: nuevos flags `--corpus`
  (factory_k65p|factory_semantic) y `--lang` (es|en). El builder de vocabulario
  añade nombres simbólicos en inglés cuando `lang=en`. La máscara de etapa
  permite todas las moléculas desde el principio en modo semántico.
- **[FEAT] `k65p/data/lexicon.json`**: renderings en inglés para las 28
  moléculas gold. `k65p/src/k65p/lexicon.py`: `molecule_names(lang='en')`.
- **[CATA 1]**: modelo glyph a 128d, 6/7 hitos en 211 épocas (gen_valid=1.0,
  val=1.45 vs bigrama=3.06). Falló por corpus secundario insuficiente (20 <
  mínimo). Umbral bajado a 20; KB secundaria ampliada a 16 expresiones +
  variaciones 4x = 47 muestras.
- **[CATA 2]**: en curso — 2_years aprobado en 59 épocas.

### 🧬 Réplicas multi-semilla DL-006 — resultado: 6/6 graduadas; el ajuste distribucional es conclusivo, la robustez generativa OOD no se replica (2026-08-10)

- **[RESULT] 6/6 réplicas GRADUADAS 7/7 hitos** (seeds 771/772/773 × glyph/
  standard), protocolo adaptativo DL-006 + umbrales DL-004 congelados. épocas:
  glyph 164/148/165, standard 148/141/145.
- **[CONCLUSIÓN — AJUSTE DISTRIBUCIONAL, CONCLUSIVA]**: `standard` bate a
  `glyph` en val_loss final en TODAS las réplicas (2.007-2.039 vs 2.149-2.162).
  El embedding libre ajusta mejor la distribución; deja de ser lectura de una
  semilla.
- **[CONCLUSIÓN — ROBUSTEZ GENERATIVA OOD, NO SE REPLICA]**: la ventaja
  gramatical generativa del glyph de la seed 770 (OOD 40/40) era artefacto de
  semilla. gen_valid OOD varía salvajemente (glyph 40-100%, standard 45-70%) y
  **3 de 6 réplicas SUSPENDEN la batería adversarial DL-004** (umbral ≥0.60).
- **[CONCLUSIÓN — SIN COLAPSO DISTRIBUCIONAL]**: el gap OOD−val es pequeño y
  consistente en las 6 (+0.06 a +0.12 nats). El fallo del attack 2 es de
  GENERACIÓN greedy (autoregresiva), no de representación: las muestras
  inválidas son concatenaciones léxicas no-estructurales (`[18 comida dar dar]`),
  no árboles mal balanceados.
- **[ACTAS]** `storage/checkpoints/replicates/k65p/<brazo>_sNNN/adversarial_report.json`
  (attack1+attack2) + `school_state_k65p.json` (exam_history, seed).

### 🧬 Réplicas multi-semilla DL-006 — la comparación v2↔v0 deja de ser anécdota (2026-08-10)

- **[NEW] 6 recetas de réplica K-65P** (`configs/jobs/school_k65p_{glyph,standard}_s{771,772,773}.yaml`):
  misma semilla canónica → 3 semillas nuevas (771/772/773) × 2 brazos
  (`--embedding glyph|standard`) bajo el MISMO protocolo adaptativo DL-006 y
  umbrales DL-004 congelados. Cada réplica usa `--state_dir` propio
  (`storage/checkpoints/replicates/k65p/...`): el trainer valida `embedding_mode`
  por directorio y una réplica jamás pisa el estado canónico ni otra réplica.
  Doctrina (BITACORA Hito 5 / RELEASES_20260803): la lectura de una semilla no
  concluye — la varianza entre runs es del orden del efecto; con 3+ semillas la
  conjetura *glifos ⇒ robustez gramatical / embedding libre ⇒ ajuste distribucional*
  se convierte en dato.
- **[FIX] `configs/jobs/school_k65p.yaml`**: el `total: 1408` era el calendario
  fijo de v1, falso para el protocolo adaptativo — ahora `1600` = máximo teórico
  (8 etapas × 200 tope), y la terminación la gobierna el milestone `8_years`
  (`completion`), que en cualquier modo gana al contador.
- **[NEW] `--seed` en `scripts/train_school_k65p.sh`** (env `SEED`, default 770):
  el runner autónomo propagaba `--embedding` y `--state_dir` pero no la semilla,
  por lo que las réplicas por CLI eran imposibles.

### 🎓 DL-006 — Protocolo adaptativo + brazo Bit v0: la edad se mide en hitos, no en épocas (2026-08-03)

- **[NEW] Protocolo adaptativo en `train_sovereign_school_k65p.py`**: etapas por
  plateau (no cronómetro), examen sobre el mejor checkpoint, avance desde best,
  neurogénesis SOLO como remediación de examen suspendido. El calendario de 1408
  épocas de v1 queda retirado para K-65P (sobreentrenaba por construcción).
- **[NEW] Brazo `--embedding standard` (Bit v0)**: one-hot congelado +
  proyecciones ≡ embedding estándar, mismo corpus/exámenes. Matriz 2×2 con la
  tesis: {glyph, standard} × {K-65P, inglés}.
- **[RESULT sandbox, seed 770]**: v2 (glyph) **graduado 7/7 en 157 épocas a 128d
  (1,26M params), cero suspensos**; v0 (standard) 6/7 con mejor val_loss pero
  suspenso de 8_years por gen_valid 0.52 (anidamiento profundo) → remediación
  128→256d en curso. Conjetura a replicar: glifos ⇒ robustez gramatical;
  embedding libre ⇒ ajuste distribucional.

### 🧬 DL-005 — El glifo cero hacía la sintaxis K-65P inaprendible (2026-08-03)

- **[ROOT CAUSE] El examen de hito DL-004 suspendió a la primera run y destapó
  el defecto real**: los 6 tokens estructurales compartían el glifo todo-ceros
  (embedding idéntico → `[` y `]` indistinguibles) y símbolo/dígito del mismo
  primo empataban logits. La sintaxis era inaprendible por construcción —
  también el 2-ago.
- **[FIX] Firmas ternarias (trit −1) para `[`, `]`, `G`, `<stop>`, `<unk>`** y
  primos solo en forma canónica: vocab 164 → 99 tokens con 99 glifos únicos.
  Probe A/B: val_acc 0.4% → 38%, gen_valid 0% → 16% (100 ép.), val_loss bate
  al bigrama. Corpus ×2.5 (10.800 exprs + 105 OOD) contra el overfitting.

### 🔴 DL-004 — La "graduación" de Bit v2 (K-65P) se invalida y los instrumentos se reconstruyen (2026-08-03)

- **[NEGATIVE RESULT] La run del 2-ago era un smoke-test, no una graduación**:
  hitos por cronómetro sin examen, corpus de 94 muestras con traducción semántica
  vacía, loss con 96% de padding sin `ignore_index` (la acc 97.32% quedaba a ~0,8
  puntos del predictor trivial de `<pad>`), neurogénesis disparada sobre
  `val_loss=Infinity`, y una suite adversarial que validaba la sintaxis de la
  ENTRADA en lugar de la salida del modelo. Material en cuarentena:
  `storage/checkpoints/quarantine/bit_v2_smoketest_20260802/`. Detalle: DL-004.
- **[NEW] `scripts/generate_k65p_corpus.py`** — corpus composicional válido por
  construcción (4.500 expresiones, tiers alineados con las máscaras de etapa) +
  holdout OOD real por pares cabeza-argumento nunca vistos.
- **[FIX] `train_sovereign_school_k65p.py`** — `ignore_index=<pad>`, assert
  máscara↔corpus, plateau solo con métricas finitas, split barajado, optimizer
  recargado, y **examen de hito con umbrales congelados** (gen_valid≥0.60 +
  val_loss ≤ bigrama−0.10; suspenso = repetición de curso + pausa rc=78). Fix
  del tokenizador que partía `grupo` en `g`+`rupo` y perdía el marcador `G`.
- **[FIX] `run_adversarial_suite.py`** — evalúa la SALIDA generada sobre el
  holdout OOD, compara contra baselines triviales (uniforme, bigrama) y su
  veredicto es condicional (puede suspender, rc=2). `bit_metrics.py` deja de
  comparar cross-entropies de vocabularios distintos y cuenta parámetros reales.

### 🎓 La batería vuelve a examinar en el idioma del alumno (2026-07-28)
- **[FIX] `milestone_battery.py` — exam data v2-en**: los 20 pares de gramática
  y los 8 prompts de producción seguían en castellano (v1, congelados el 03-jul)
  mientras el corpus pasó a inglés el 14-jul (dd09ba4): grammar puntuaba 0.000
  para **cualquier** checkpoint post-transición (destapado por EXP_079). v2-en es
  la **traducción fiel** de los datos v1 — mismo diseño de examen, mismo
  vocabulario preescolar (20/20 pares verificados dentro del censo, cero `<unk>`),
  prefijo de diálogo `tú:` → `you:` — y los **umbrales pre-registrados quedan
  intocados** (doctrina). La versión de datos de examen se imprime ahora en la
  cabecera de cada informe (`exam data v2-en (2026-07-28)`): las cifras v1 y
  v2-en no son comparables entre sí.
- **[VALIDATED] El examen discrimina de nuevo**: con v2-en, los checkpoints de
  2 años de EXP_079 puntúan grammar 0.650-0.700 y el run vivo (época 998, dim
  896) 0.800 — gradiente monótono con la madurez, en lugar del 0.000 plano.
  Los tres brazos del A/B siguen empatados entre sí (±1 par = ruido), así que
  el veredicto de EXP_079 se sostiene también bajo v2-en.
- **[NOTE] `min_len` sigue bajo en todos los checkpoints** (0.6-1.1 vs umbral
  3.0): las respuestas a los prompts de diálogo son cortas. Es una observación
  del *modelo* (o del diseño v1 del examen C, que se ha conservado tal cual),
  no del idioma — se deja constancia y no se maquilla.

### ✅ EXP_079 Tier 2 — el 32→16 no cuesta calidad (2026-07-28, madrugada)

- **[RESULT] A/B/C from-scratch completado** (3 × 160 épocas + una neurogénesis
  128→256 por brazo, seed 770, sandboxes; run vivo intacto, md5 verificado):
  val_loss final **3.7338 (FP32) vs 3.7348 (BF16) vs 3.7343 (BF16+compile)** —
  |Δ| medio por época 0.003 contra un umbral de 0.05, **cero épocas fuera**;
  batería y muestras cualitativas idénticas entre brazos; 160 épocas en
  80/47/**31** min (compile = **2.6×** end-to-end). Único fallo literal de la
  matriz: la neurogénesis de B saltó en +5 épocas (sensibilidad del contador de
  plateau cerca de `min_delta`, no degradación; C clavó el ±0). Detalle completo
  y análisis en `docs/experiments/EXP_079_DESIGN.md` §5; DL-002 actualizada.
- **[ADOPTED] `--amp auto --compile` en `configs/jobs/school.yaml`** según el
  criterio pre-registrado (C pasa 6/6). Rollback documentado: quitar flags, el
  checkpoint es FP32 siempre.
- **[⚠️ HALLAZGO COLATERAL] `milestone_battery.py` suspende a cualquier
  checkpoint reciente por igual** (grammar 0.000 en los tres brazos): sus datos
  de examen parecen anteriores a la transición a inglés. Tarea de realineación
  registrada — no afecta al A/B (comparación relativa).

### 🧪 EXP_079 — los números detrás de BF16 (2026-07-27, tarde)

- **[NEW] `docs/DECISION_LOG.md`** — registro de decisiones de calado (DL-NNN).
  DL-002 documenta el problema (stage 8 no cabía; 60 días de GPU), la decisión
  (BF16 autocast + SDPA + compile opcional) y la evidencia medida. DL-001 registra
  retroactivamente el ROUTE CHANGE de School v3 por referencia.
- **[NEW] `docs/experiments/EXP_079_DESIGN.md`** — pre-registro del benchmark de
  precisión mixta. **Tier 1 ejecutado**: BF16+SDPA+compile = **3.7× más rápido**
  (548→149 ms/step, dim 896) y **−40% VRAM**; a dim 1024, 4.1 GB frente a 6.9 GB
  FP32 ("cabe por los pelos"); ∇STE sano en las 8 configs. Tier 2 (A/B/C
  from-scratch hasta el hito de 2 años, seed 770, umbrales congelados) responde
  la pregunta pendiente: si el paso 32→16 bits cuesta calidad de convergencia.
- **[NEW] RFCs archivados en el repo** (copia canónica, antes en Aleth_Core):
  `docs/RFC_BIT_GRADUATION_ROADMAP.md` (D1-D6, fases F0-F6 hasta la graduación)
  y `docs/RFC_VRAM_SCALING_BITNET_CURRICULUM.md` (análisis técnico, revisado por
  Grok + DeepSeek).
- **[NEW] `scripts/bench_school_step.py`** (Tier 1) y
  **`scripts/bench_school_scratch.sh`** (Tier 2: tres brazos secuenciales en
  sandboxes, GPU exclusiva, Wake Gate, md5 del run vivo verificado entre brazos).
- **[NEW] `--state_dir` / `--seed` / `--compile`** en `train_sovereign_school.py`,
  y `base_dir` derivado de la posición del fichero (fuera la ruta absoluta).
- **[FIX] NameError en arranque en frío**: todo run from-scratch sin
  `--reset_state` moría al leer `state.get(...)` con la variable sin ligar (el
  fichero recién escrito hacía verdadera la condición). El run vivo lo esquivó
  porque nació con `--reset_state`.

### ⚡ BF16 + SDPA — el stage de 8 años cabe en la RTX (2026-07-27)

Implementa la Estrategia B (fases 1-2) de RFC-BITNET-VRAM-001: sin medidas, la
neurogénesis 896→1024 del stage `secondary_8` proyectaba ~7.7 GB de VRAM sobre
una tarjeta de 8 GB. Decisión D1 de RFC-BIT-GRAD-001 (Aleth_Core), ratificada
por el operador.

- **[NEW] `--amp {auto,bf16,off}`** en `train_sovereign_school.py` (default
  `auto`). **Desviación deliberada del RFC §4.4.1**: en lugar de convertir
  modelo y optimizador a BF16, se usa autocast con **pesos maestros FP32** —
  las activaciones (~85% de la VRAM según el propio RFC) se computan en BF16 y
  params/gradientes/AdamW quedan en FP32. Consecuencia: `model_current.pt` no
  cambia de formato, FP32↔BF16 son intercambiables por ejecución (el benchmark
  de 100 épocas y el rollback cuestan un flag), y la neurogénesis (`net2wider`)
  ni se entera. Desaparecen de golpe los tres riesgos más gordos de la matriz
  del RFC: conversión de checkpoint, GradScaler y dtype en neurogénesis.
- **[NEW] SDPA en `BitNetAttention`**: `F.scaled_dot_product_attention`
  sustituye la atención manual — el kernel fusionado no materializa la matriz
  N×N. Sin STE en ese tramo (seguro por construcción; el `torch.compile`
  selectivo del RFC §4.8 queda como opcional futuro, D2).
- **[FIX] `RMSNorm` computa su estadística en FP32** y devuelve el dtype de
  entrada: bajo autocast, la varianza en BF16 desestabiliza la normalización.
- **[NEW] Telemetría por época**: `VRAM pico` (MB) y `∇STE` (norma del
  gradiente de la primera BitLinear — si cae a cero, el straight-through
  estimator se rompió en silencio, RFC §4.8.1).
- **[TEST] `tests/test_bf16_sdpa.py`** (11 tests): SDPA vs oráculo manual
  (causal/no-causal, seq_len 1/7/32), flujo de gradientes, contrato de RMSNorm,
  salud del STE bajo autocast, loss BF16 dentro del ±5% de FP32.
- **[VERIFIED] Compatibilidad con el checkpoint vivo** (epoch 998, dim 896),
  comprobada en CPU **sobre copia de backup**
  (`storage/checkpoints/backup_pre_bf16_20260727/`, md5 verificado):
  `load_state_dict(strict=True)` OK, logits finitos, cos(FP32,BF16)=0.9998,
  96.85% de acuerdo argmax. El checkpoint vivo no se tocó.
- **[CONF] `configs/jobs/school.yaml`**: `--amp auto` explícito en el
  `step_command` y `min_free_vram_mb` 3500→4500 (pico estimado BF16 a dim 1024
  + margen; si se corre `--amp off`, subir a 7000).
- **[DOCS] `docs/TRAINING_BIT.md`**: sección de precisión mixta y nota de que
  las épocas de 3h20m-3h50m eran FP32 pre-SDPA.

### 🧭 Router: las reglas deterministas vuelven a bastar (2026-07-27)
- **[FIX] `test_prolog_router_retro_compatibility` en rojo** — "Resuelve el cálculo usando lógica matemática" acababa en `default_node`. El síntoma era el test; la causa es que **el camino semántico está muerto en el venv**: `transformers 5.8.1` importa `is_offline_mode` de `huggingface_hub`, que en la 0.36.2 instalada no existe. El `except` lo tragaba en silencio, así que **toda tarea sin keyword caía en `general`** y el router parecía funcionar.
- **[FIX] Reglas deterministas ampliadas** con el vocabulario obvio en castellano (`cálculo`, `calcula`, `resuelve`, `lógica`, `matemátic`, `ecuación`, `demuestra`, `divide`). RULE 2 de `CONVENTIONS.md` lo exige: enrutar es clasificar, y lo nombrable por keyword no puede depender de embeddings.
- **[FIX] El fallo del traductor ya no es mudo**: se registra un `warning`. Un traductor caído era indistinguible de uno que encuentra "general" en todo.
- **[TEST] `test_routing_survives_a_dead_translator`**: fija la causa raíz, no el síntoma — con el traductor lanzando `ImportError`, el enrutado sigue siendo correcto.
- **[⚠️ PENDIENTE, decisión del operador] Conflicto de dependencias real**: `fastembed` exige `huggingface-hub>=0.20,<1.0` y `transformers 5.8.1` exige `>=1.5.0,<2.0` — **mutuamente excluyentes**. El venv tiene 0.36.2, así que `transformers`/`sentence-transformers` están inservibles. Salidas: retirar `fastembed`, o fijar `transformers` a la serie 4.x. No se toca aquí porque es el mismo venv que entrena a Bit.

### 🎓 Entrenar a Bit sin depender de nadie (2026-07-27)
- **[NEW] `--max_epochs_per_run`** en `train_sovereign_school.py`: convierte el currículo en pasos reanudables de una época. Es el flag sobre el que se apoya toda la vía diferida.
- **[NEW] Reserva de GPU anunciada a red-pill** al arrancar (4 GB exclusivos) para que el demonio de inferencia caiga a su worker de CPU mientras el entrenamiento tiene la tarjeta. **Estrictamente opcional**: se importa por paquete o vía `$RED_PILL_SRC`, nunca por una ruta escrita en el código —que solo funcionaría en una máquina—, y si red-pill no está delante no pasa nada.
- **[NEW] `scripts/train_school.sh`** — runner autónomo de la Escuela Soberana, **sin necesidad de red-pill**. Trocea el trabajo **época a época**, que es la unidad que el entrenador realmente guarda: un `Ctrl-C` cuesta como mucho la época en vuelo y al relanzar retoma exacto. Bajo systemd aplica las dos lecciones que este proyecto pagó caras: `MemoryMax=16G` (los 10G despertaban al OOM killer con el modelo ya a 896 dim) y `systemd-inhibit --what=sleep` (suspender el portátil mata el contexto CUDA — causa raíz de varios entrenamientos "fritos" en julio de 2026). Si el ecosistema red-pill está presente libera la VRAM del modelo residente, pero es oportunista: sin él entrena igual. Modos: `--status` (dónde va, sin entrenar), `--epochs N`, y variables `PYTHON` / `BATCH_SIZE` / `MEMORY_MAX`.
- **[GUARD] El runner se niega a arrancar si el entrenador no acepta `--max_epochs_per_run`**: como `train_sovereign_school.py` parsea con `parse_known_args()`, un flag ausente se ignoraría **en silencio** y cada "época" entrenaría el currículo entero. Mejor no arrancar que prometer un troceo que no ocurre.
- **[NEW] `configs/jobs/school.yaml`** — receta versionada para encolar el entrenamiento como job diferido de red-pill (`red-pill job submit --recipe school`). Declara el contrato de progreso leyendo claves que el estado **ya contiene** (`current_epoch`, `current_stage_idx`, `milestones_achieved`), así que la supervisión no exige tocar el entrenamiento: se ve `998/1408 (70%) · etapa 7/8`. La finalización va por **hito concedido por Samantha**, no por contador de épocas, que es como cierra de verdad cada etapa.
- **[NEW] `docs/TRAINING_BIT.md`** — las dos vías (autónoma y diferida) comparadas, operativa de pausa/kill/resume, dónde mirar cada cosa y diagnóstico rápido. Documenta también el coste medido por época (3h20m-3h50m en GPU; ~11h en una ejecución que cayó a CPU tras un `CUDA OutOfMemoryError`) y por qué eso hace que hoy el job sea **no interrumpible**: con un solo checkpoint por época, interrumpir cuesta horas.
- **[DEPRECATED] `scripts/run_school_gpu.sh`** — marcado como superado (no borrado): corría el currículo entero en una sola invocación y con `MemoryMax=10G`.

### 🏗️ School v3 Consolidation — Codebase Restructuring & Bug Fixes (2026-07-06)
- **[FIX] Temperature propagation in `samantha_on_demand.py`**: `invoke()` now accepts and propagates `temperature` parameter (default `0.7`). Previously hardcoded to `0.0` and callers passing `temperature=` raised `TypeError`. Affects 11+ consumers including `samantha_story_factory.py`.
- **[FIX] Neurogenesis trigger: calendar → plateau**: Replaced the epoch-based (`epoch == config["start_epoch"]`) neurogenesis trigger with a validation-loss plateau monitor. The model now grows only when `val_loss` stagnates for `--patience` epochs (default 15). New CLI args: `--patience`, `--min_delta`. State persisted in `school_state.json` (`best_val_loss`, `epochs_without_improvement`, `neurogenesis_history`). Aligns with ROUTE_CHANGE_SCHOOL_V3.md specification.
- **[REF] `src/bitnet/` reorganized into 10 submodules**: Flat directory (39 files) split into `model/`, `growth/`, `vocab/`, `worlds/`, `data/`, `operators/`, `training/`, `inference/`, `translation/`, `telemetry/`. Backward compatibility maintained via `MetaPathFinder` import hook in `__init__.py`. ~80 files updated across `scripts/`, `tests/`, `playground/`, `lab/`.
- **[PRUNE] Legacy training scripts → `lab/experiments/`**: 10 training scripts from pre-School-v3 phases (`train_arena*.py`, `train_ppo.py`, `train_resonance.py`, etc.) moved out of `src/bitnet/`. Only `train_sovereign_school.py` remains as the active training script.
- **[NEW] `get_next_dim()` helper**: Testable function for neurogenesis dim progression with stage ceiling support.
- **[TEST] 26 new tests**: `test_samantha_temperature.py` (7), `test_story_factory.py` (9), `test_neurogenesis_plateau.py` (17, includes regression test confirming calendar trigger removal). Suite: 46 → 72 tests, all green.

### 🧭 ROUTE CHANGE — School v3: Clean Vocabulary, Synthetic Corpus, Operational Milestones
- **[AUDIT] Milestone-5 Cold Audit (`scratch/audit_bit_milestone5_fable.py`)**: Independent evaluation of `model_milestone_5_years.pt` (31.1M, 640-dim). Findings: grammar preference on novel sentences **14/15 (93%)** — the ternary+glyph substrate genuinely learns Spanish syntax — but free generation collapsed (1-4 words then `<pad>`, or sampling into the contaminated vocabulary tail). Full decision record: `docs/sessions/20260703/ROUTE_CHANGE_SCHOOL_V3.md`.
- **[ROOT CAUSE] Ghost Vocabulary**: census proves 8,671 of 15,005 words (**57.8%**) never occur in the training corpus — imported by the 3k→15k expansion from a generic (subtitle-derived) frequency list, leaving live glyphs/logits the model cannot learn to suppress. Corpus total: 863,712 tokens (~1000x below TinyStories scale), 62% template-monotone dialogues.
- **[FIX] `verify_milestone_4.py` causal-mask bug**: the harness built the model without `is_causal=True` — all manual milestone verifications ran with the wrong attention mask.
- **[NEW] `scripts/rebuild_clean_vocabulary.py`**: clean-source census → `configs/clean_vocabulary_words.json` (6,361 words v1) to feed `expand_vocabulary` glyph re-derivation. OOV→`<unk>` always (Rule 7 upheld; the 15k *size* goal stays, the *source* changes to clean corpora + factory).
- **[NEW] `scripts/samantha_story_factory.py`**: controlled-vocabulary synthetic corpus generator (graded stages, OOV ≤2% filter, dedup, resumable JSONL); designed to run unattended under the Sovereign Wake Gate. Volume target: 20-50M tokens.
- **[NEW] `scripts/milestone_battery.py`**: judge-free milestone certification (M1-M4) with **pre-registered thresholds frozen in code**: grammar preference, held-out cloze, production health (length, natural stop, ghost-word rate). LLM judge demoted to secondary evaluation.
- **[DECISION] Run-2 checkpoints archived as documented negative result** (vocab rebuild changes the glyph table → incompatible). Curriculum corrections: 5% held-out per stage, graded synthetic readers instead of Gutenberg, pain-triggered (validation-plateau) neurogenesis instead of scheduled growth, response-length variety.

### 🛡️ Sovereign Wake Gate — Four-Check Doctrine for Autonomous Training
- **[NEW] `src/swarm/wake_gate.py`**: autonomous launches require CONSENT (approved queue entry + config on disk), HARDWARE (VRAM ≥6G, no fever, operator absent ≥1h multi-signal — refuses to assume absence without signals), CONTAINMENT (cgroup `MemoryMax` + `RuntimeMaxSec`), BUDGET (≤2 autonomous runs/day). Grafted into `scripts/minion_scheduler.py` ahead of the subprocess; every decision — fire or hold — is recorded in `lab/wake_ledger.jsonl` (gitignored).
- **[TEST] `tests/test_wake_gate.py`**: 13 tests with injected probes (no nvidia-smi, no wall clock). First live run correctly HELD on busy VRAM with a reasoned ledger entry.

### 🧹 Repo Hygiene (Fable review, 2026-07-03)
- **[FIX] pytest collection**: `pythonpath = ["."]` in `pyproject.toml` — suite went from 10/11 modules uncollectable (`ModuleNotFoundError: src`) to 46/46 green.
- **[PRUNE] 30 fossilized experiment scripts** moved `src/bitnet/` → `lab/experiments/` (zero inbound imports + zero refs in REPRODUCE/DEMO_GUIDE/configs; entrypoints pinned by the papers stay). `src/bitnet` 67→37 files.
- **[DOCS] "Why Bit?" positioning** added to README (EN+ES), paper 1 §1.1 (`.md`+`.tex`, PDF regenerated), and ROADMAP "Current Sequencing" (staged bet: School → Jungle Reboot with Nico/Sofy/Hugo from Bit's base brain → unpark MoE). Energy table rows labeled measured vs estimated.
- **[DOCS] `lab/BRIEFING.md`** rewritten as honest snapshot; `lab/experiment_queue.yaml` flagged STALE pending reconciliation.

### 🎓 Vocabulary Redesign & Corpus Saneing (15,000 Words)
- **[FIX] OOM-Shield in Hot Neurogenesis (`src/bitnet/train_sovereign_school.py`)**: Resolved the temporary VRAM spike that caused silent and permanent migration to CPU training during hot neurogenesis (weight mitosis). Implemented a temporary CPU-offloading pipeline in `trigger_neurogenesis` to perform the expansion and weight cloning of Net2WiderNet entirely on CPU, clear CUDA VRAM using `torch.cuda.empty_cache()`, and safely reload the expanded model and optimizer back to the GPU.
- **[CONV] New Rule 7 in CONVENTIONS.md**: Documented the semantic collision failure mode (e.g., *"pajaritos"* ➔ *"cálmate"*) to enforce vocabularies of at least 15,000 words and prohibit blind vector similarity mappings for OOV tokens.
- **[FEAT] 15k Vocabulary Scaling Plan**: Designed the implementation plan to expand the base vocabulary from 3k to 15k words, clean the CHILDES corpus (accepting lengths 3-20), and remove sentence limits on children's stories and Gutenberg books.

### 🎓 Piaget Curriculum & Natural Generation Stop
- **[FEAT] Natural Generation Stop in School (`src/bitnet/train_sovereign_school.py`)**: Replaced the standard cross-entropy loss with a **dynamic sequence mask** that enables loss for all sentence tokens plus the first trailing `<pad>` (0) token. This trains the model to predict the stop token and halt generation naturally instead of babbling.
- **[FIX] Samantha Eval Return Value (`src/bitnet/train_sovereign_school.py`)**: Fixed the return unpacking signature for `run_samantha_eval` to prevent runtime crashes in the main training loop.
- **[LINT] Linter & Formatting Fixes**: Resolved ruff warnings (`SIM108`, `B007`) in `playground/chat_school_agent.py` and `train_sovereign_school.py`, and fixed tab indentations in `src/bitnet/net2net.py` docstrings.

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
