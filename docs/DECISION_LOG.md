# Decision Log — Frankenswarm

Registro de decisiones de calado: el problema, la decisión, la evidencia que la
respalda y dónde está el detalle. Una entrada por decisión, la más reciente arriba.
Las decisiones se numeran DL-NNN y no se reescriben: si una decisión se revierte,
se añade una entrada nueva que la referencia.

---

## DL-007 · 2026-08-14 — El brazo estándar estaba lisiado por implementación: se retira la baza de coste del glifo y nace el trato simétrico de weight decay

**Problema.** El brazo estándar (`--embedding standard`, DL-006) se construye con
`vocab_embeddings=np.eye(V)`: tabla identidad congelada cuyas columnas de
`inbound_proj` SON el embedding entrenable. La equivalencia matemática es correcta,
pero la implementación pagaba dos peajes ajenos a la arquitectura:

1. **Un matmul nulo en la salida.** `logits = outbound_proj(h) @ I.T` multiplicaba
   por la identidad: V×V multiplicaciones por posición de token para no cambiar
   nada. En la entrada, materializaba un one-hot de V dimensiones para hacer con un
   matmul denso lo que un lookup de una fila resuelve.
2. **La tabla identidad se serializaba en cada checkpoint**: 590 MB a 12.143 tokens
   (`current` + `best` + `final` + 7 hitos + etapa ⇒ decenas de GB por run).

Consecuencia sobre la tesis: la baza (b) del informe del 5-ago —"el glifo decodifica
~4.7× más barato con vocabulario grande (medido 15 vs 70 s/época)"— **medía la
implementación del baseline, no la arquitectura**. Es el modo de fallo de DL-004
otra vez: medir el instrumento y creer que se mide al alumno.

Además, `weight_decay=0.05` se aplicaba uniformemente. El gradiente de
`inbound_proj` es denso aunque solo unas pocas columnas reciban señal, así que cada
step encogía el embedding de las palabras raras del brazo estándar entre sus
actualizaciones infrecuentes; el glifo no sufre eso (sus 65 primos se actualizan en
cada batch). Los dos brazos NO recibían el mismo trato de regularización, y la
penalización caía justo sobre lo que miden OOD y gen_valid.

**Decisión.**
1. **Fast-path one-hot en `BitNet4LayerModel`**: si la tabla es la identidad
   (`_is_identity_matrix`), la entrada se resuelve con `F.embedding` sobre
   `inbound_proj.weight.t()` y la salida es `outbound_proj(h)` a secas. **Bitwise
   idéntico** al camino denso — fijado con `torch.equal` en
   `tests/test_embedding_arms.py`, no con tolerancias.
2. La tabla identidad deja de persistirse (`persistent=False`); es reconstruible
   desde `vocab_size`. `load_state_dict` descarta la clave sobrante, así que **los
   checkpoints antiguos del brazo estándar siguen cargando exactos** y los ~60
   sitios de carga del repo no se tocan.
3. **`--wd_mode {uniform,no_embed}`** (`build_param_groups`). Default `uniform`: es
   el trato con el que corrieron las réplicas DL-006 y no se cambia un instrumento
   bajo los pies de una comparativa ya publicada (D3). `no_embed` exime las tablas
   de embedding (entrada, salida, posicionales, primos) y es el trato **simétrico**,
   recomendado para BIT-003 y cualquier comparativa nueva.
4. **`scripts/bench_embedding_arms.py`** queda como el instrumento que produce la
   cifra de coste publicable.

**Evidencia (medida, RTX 5070, dim 128, batch 64, seq 128, BF16, V=12.143).**

| | glyph | standard antes | standard después |
|---|---|---|---|
| ms/step | 25,49 | **291,94** | **24,16** |
| pico VRAM | 1.538 MB | 2.591 MB | 2.121 MB |
| checkpoint | 7,8 MB | **579,1 MB** | **16,6 MB** |
| params | 1.247.880 | 4.348.168 | 4.348.168 |

El brazo estándar acelera **12,1×** y su checkpoint encoge **35×**. Con el baseline
justo, el estándar es un **10% más rápido** que el glifo por step, no 4,7× más lento.
Y el escalado va en la misma dirección: ratio standard/glyph 0,97 (V=1.000) → 0,99
(V=3.000) → 0,92 (V=6.000) → 0,90 (V=12.143), o sea que la ventaja del estándar
**crece** con el vocabulario en lugar de cerrarse. Razón: el `decode_logits` del
glifo tiene que **componer** las V palabras desde los primos (V×65×d) y luego
puntuarlas (V×d) — el O(65·d) describe el tamaño de la TABLA, no el coste del logit.

**Qué se retracta y qué sobrevive.**
- **Retirada** la baza (b) del informe del 5-ago (`docs/sessions/20260805/
  INFORME_DL006_MATRIZ.md`, lectura 3): el glifo NO decodifica más barato. Ni a
  12k tokens ni en tendencia.
- **Intacta** la lectura 2 (el embedding estándar ajusta mejor la distribución en
  las 8 mediciones): el fast-path es bitwise equivalente, así que **ningún
  resultado previo del brazo estándar queda invalidado** — a diferencia de DL-004,
  aquí no se cae nada. Las réplicas multi-semilla siguen en pie.
- **Reformulada** la ventaja del glifo, que sigue siendo real: es una victoria de
  **compresión**, no de velocidad — 3,5× menos parámetros (1,25M vs 4,35M) y
  checkpoint 2,1× menor con vocabulario de 12k, a rendimiento comparable. Junto a
  la capacidad exclusiva de vocabulario en caliente (M5), esa es la tesis
  defendible: el glifo no es un mejor modelo de lenguaje, es un modelo **más
  pequeño y extensible**.

**Pendiente que esto abre.** El `wd_mode="no_embed"` no está medido: si se adopta en
BIT-003, la comparativa nueva no es directamente comparable con las réplicas
DL-006 en épocas-hasta-hito. Decidir A/B corto antes del run largo.

**Referencias.** DL-004 (medir el instrumento) · DL-006 (brazo v0) ·
`tests/test_embedding_arms.py` · `tests/test_weight_decay_modes.py` ·
`storage/benchmarks/embedding_arms_bench.json`.

---

## DL-006 · 2026-08-03 — Protocolo adaptativo: la edad se mide en hitos superados, no en épocas; y nace el brazo Bit v0 (embedding estándar)

**Problema.** El calendario de v1 (1408 épocas fijas) aplicado a v2 sobreentrena
por construcción: v2 alcanza su óptimo de generalización en ~10-15 épocas por
etapa y las ~270 restantes degradan el val (2.32 → 3.39 en la run DL-005) mientras
la neurogénesis por plateau le regala capacidad que invierte en memorizar.
Además, "mismas épocas" nunca fue "mismo tratamiento" entre tareas de escalas
distintas: el confound estaba en el diseño original, no en la corrección.

**Decisión (ratificada por el operador).**
1. **Protocolo adaptativo, idéntico para todos los brazos** — lo constante es la
   regla pedagógica, no el calendario: cada etapa entrena hasta plateau
   (`patience=15`, tope de seguridad 200 ép.); el examen de hito se hace sobre el
   **mejor checkpoint** de la etapa (no el último, que ya derrapó); aprobado →
   avanza desde ese mejor checkpoint; suspenso → **neurogénesis solo como
   remediación** (hasta el techo de dim de la etapa) y repetición; sin techo →
   pausa rc=78. Umbrales de examen DL-004 intactos.
2. **Brazo Bit v0 (`--embedding standard`)**: tabla one-hot congelada +
   proyecciones entrenables ≡ embedding estándar entrenado desde cero (el
   enfoque actual de la industria), mismo corpus/escuela/exámenes que v2. La
   matriz pasa a 2×2: {embedding: glyph, standard} × {lenguaje: K-65P, inglés};
   el brazo control v1-inglés bajo estas mismas reglas queda planificado.
3. Comparación entre brazos SOLO por métricas normalizadas: épocas-hasta-hito,
   params-al-aprobar, margen sobre su propio bigrama, gen_valid OOD.

**Evidencia (sandbox, seed 770, corpus idéntico).**
- **Bit v2 (glyph): GRADUADO 7/7 hitos en 157 épocas, todo a 128d (1,26M
  params), cero suspensos** — gen_valid 0.80-1.00, val 2.14-2.34 vs oráculo
  2.09. El calendario clásico usaba 1408 épocas y crecía a 1024d (77M) para
  esta misma tarea: 9× más épocas y 61× más parámetros sin necesidad.
- **Bit v0 (standard): 6/7 en 148 épocas**, mejor val_loss que v2 en todas las
  etapas (2.00-2.17 — ajusta mejor la distribución), pero **suspendió 8_years
  por gen_valid 0.52 < 0.60** justo en la etapa de anidamiento profundo y
  conectores; la remediación (128→256d) reanudó y mejora. Conjetura de una
  semilla, pendiente de réplica: *el prior composicional de los glifos compra
  robustez gramatical generativa; el embedding libre compra ajuste
  distribucional.* Exactamente la disyuntiva que la tesis quiere medir.

**Referencias.** DL-004 (umbrales congelados) · DL-005 (fix de glifos) ·
`train_sovereign_school_k65p.py` (`--embedding`, protocolo) · runner
`train_school_k65p.sh` (`EMBEDDING`, `STATE_DIR`).

---

## DL-005 · 2026-08-03 — El glifo cero hacía la sintaxis inaprendible: firmas ternarias para tokens estructurales y fin del aliasing símbolo/dígito

**Problema.** La primera run con instrumentos DL-004 suspendió el examen de
2 años (ép. 160: gen_valid 0.0, val_loss 4.50 vs bigrama 3.00) y el acta
destapó la causa raíz — **arquitectónica, presente también el 2-ago**:
`GlyphEmbedding` es puramente composicional (`trits @ prime_embeddings`, sin
componente por-palabra), y la tabla de glifos del trainer tenía dos defectos
fatales:

1. Los 6 tokens estructurales (`<pad>`, `<unk>`, `<stop>`, `[`, `]`, `G`)
   compartían el glifo todo-ceros → embedding CERO idéntico a la entrada y
   logits idénticos a la salida: el modelo no podía distinguir `[` de `]` ni
   aprender a cerrar árboles. **La sintaxis K-65P era inaprendible por
   construcción.**
2. Símbolo y dígito del mismo primo (`water`/`62`) compartían glifo → logits
   empatados (suelo de loss ln 2 por token de primo) y el desempate del argmax
   caía siempre en el símbolo → precisión next-token estructuralmente ~0.

**Decisión.**
1. Firmas ternarias trit `−1` en ejes 0-4 para `[`, `]`, `G`, `<stop>`, `<unk>`
   (la tabla ya era ternaria de espíritu BitNet; el cero puro queda solo para
   `<pad>`, que jamás es target con `ignore_index` y no gobierna el stop de
   generación — se para por balance de corchetes).
2. Primos SOLO en forma canónica (dígito): vocab 164 → **99 tokens, 99 glifos
   únicos**. `<unk>` vetado además en la máscara de generación.
3. Corpus ×2.5 (3.000/3.600/4.200; holdout OOD 105) tras el probe: con 1.200
   muestras el bloque preescolar sobreajustaba desde la época ~10.

**Evidencia (probe A/B en sandbox, 128d, mismo seed).** Antes: val_acc 0.4%,
gen_valid 0/25, val_loss nunca baja de ~3.5. Después: **val_acc 38% desde la
época 1**, val_loss 2.32 en la época 10 (bate al bigrama), gen_valid 16% a las
100 épocas con expresiones válidas cerradas (`[23 4 4]`). Queda overfitting
residual (train 1.75 / val 2.82 a la ép. 60): es carácter real de la receta y
lo medirán los exámenes; cualquier regularización adicional es decisión de
operador con actas en la mano.

**Referencias.** DL-004 (instrumentos) · `src/bitnet/vocab/glyph_vocabulary.py`
(GlyphEmbedding) · `STRUCT_TRITS` en `train_sovereign_school_k65p.py`.

---

## DL-004 · 2026-08-03 — La "graduación" de Bit v2 (K-65P) del 2-ago se invalida: resultado negativo, instrumentos reconstruidos

**Problema.** La sesión del 2-ago (Gemini Flash) completó 1408 épocas del trainer
K-65P en ~2,4 h y publicó a Bit v2 como "graduado de 8 años" con cifras de tesis
(val_loss 1.8253 "57.9% superior a v1", acc 97.32%, OOD 100%). La auditoría del
3-ago encontró que **ninguna cifra sobrevive**:

1. **Hitos por cronómetro.** `train_sovereign_school_k65p.py` otorgaba el hito al
   cumplirse `end_epoch == current_epoch`, sin examen alguno — violación directa
   de la doctrina School v3 (DL-001) y de la certificación primaria por batería.
2. **Corpus de 94 muestras** (33+29+32), media 4,5 tokens, traducción semántica
   vacía ("amor es el sol que luces nuestras vidas" → `[25 sol]`). El pipeline de
   traducción ES→K-65P queda **deprecado** hasta que exista fidelidad semántica.
3. **Loss sin `ignore_index`** con max_len 128 → 96% de los targets eran `<pad>`:
   la acc 97.32% queda a ~0,8 puntos del predictor trivial de padding (96,5%).
4. **Máscara de etapa vetaba targets reales** → `val_loss = Infinity` en todo el
   bloque preescolar → las 3 primeras neurogénesis dispararon sobre Infinity
   (crecimiento por artefacto). 281 épocas finales sin mejora alguna.
5. **La suite adversarial validaba la ENTRADA**: `is_valid(expr)` sobre la
   expresión de test escrita a mano; la salida del modelo se computaba y se
   descartaba. El "100% OOD" era un tautología. Banner "MODELO ROBUSTO Y
   VERIFICADO" incondicional.
6. **Bug de tokenización** (heredado): el regex partía `grupo` en `g`+`rupo` y el
   lookup con casefold perdía el marcador `G` → todo `[G ...]` caía a `<unk>`.
7. Cifra de parámetros: el modelo final era ~77M (misma talla que v1), no
   "1.2M-4.8M" como se difundió. Sin log en disco de las épocas 119→1408.

**Decisión.**
1. El material del 2-ago se mueve a
   `storage/checkpoints/quarantine/bit_v2_smoketest_20260802/` y se re-etiqueta
   como **smoke-test del pipeline**. Sus cifras no se citan. La entrada "Hito 4"
   de la bitácora (docs/BITACORA_BIT_V2.md) queda enmendada por referencia.
2. **Corpus nuevo por construcción**: `scripts/generate_k65p_corpus.py` genera
   4.500 expresiones únicas validadas contra `k65p.validator` (1.200/1.500/1.800
   por bloque, estratificadas por tiers de moléculas alineados con las máscaras),
   más **holdout OOD real** (pares cabeza-argumento excluidos de train/val).
3. **Trainer reconstruido**: loss/acc con `ignore_index=<pad>`; assert
   máscara↔corpus que aborta antes de entrenar; plateau solo sobre val_loss
   finita y con val ≥ 30 muestras; split 85/15 barajado (semilla 770); optimizer
   recargado al resumir; MAX_LEN 48.
4. **Examen de hito real** con umbrales congelados pre-run (este pre-registro,
   calibrado contra el bigrama ANTES de arrancar):
   `generaciones greedy válidas ≥ 0.60` sobre 25 prompts de val (criterio
   primario — la tesis es que Bit aprende la GRAMÁTICA) y
   `val_loss ≤ bigrama_loss − 0.10 nats` (criterio secundario — información más
   allá de estadística trivial; el bigrama puntúa 2.65-3.00 nats / 35-48% acc
   según etapa). La precisión token se reporta pero NO umbraliza: en corpus
   composicional los átomos son impredecibles por diseño (techo estructural).
   Suspenso → repetición de curso (+16 épocas) + pausa rc=78 para revisión del
   operador. Los umbrales NO se tocan a mitad de run (D3).
5. **Suite adversarial reconstruida**: valida LA SALIDA generada (autoregresiva,
   greedy) sobre el holdout OOD; mismos criterios que el examen (gen_valid ≥
   0.60, loss ≤ bigrama − 0.10); veredicto condicional con rc≠0 si suspende.
   `bit_metrics.py` deja de enfrentar cross-entropies de vocabularios distintos.

**Qué se salva del 2-ago (valor real del smoke-test).** El pipeline corre de
punta a punta (checkpoints atómicos, resume, systemd-run); los fixes de
dispositivo de `net2net.py` (7 neurogénesis sin crash); el coste por época es
trivial a corpus pequeño. Nada más: no hay evidencia sobre la aprendibilidad de
K-65P — esa pregunta queda abierta y es exactamente la que la run nueva responde.

**Referencias.** DL-001 (doctrina School v3) · DL-003 (clase de bug de máscara,
segunda aparición) · `storage/checkpoints/quarantine/bit_v2_smoketest_20260802/README.md`
· `storage/curriculum/factory_k65p/generation_manifest.json`.

---

## DL-003 · 2026-08-01 — Enmienda del evaluador de exámenes: la máscara de vocabulario excluía las respuestas esperadas

**Problema.** El examen de edad (`scripts/evaluate_samantha_age.py`) era
estructuralmente insuperable en los hitos altos, por dos vías independientes:

1. **Máscara de generación.** La whitelist por edad se construía únicamente desde
   los textos del currículo (structured_en + CHILDES + NSM). Las respuestas
   esperadas del examen de 8 años (`effect`, `mirrors`, `cloud`, `universe`) no
   aparecen en esos textos, así que sus tokens quedaban a −1e9 en la generación:
   el alumno entrena esas parejas (exámenes ×300 en el curriculum) pero tenía
   físicamente vetado emitirlas el día del examen. Agravante: lo que Bit entrena
   son las **formas base** del Diccionario Soberano (`aleth`→`aliya`,
   `bunker`→`hideout`), también fuera de la máscara.
2. **Rúbrica autocontradictoria.** El prompt del juez ordenaba castigar 0-3 el
   vocabulario "fuera de edad" listando `espejos` (respuesta esperada de la
   pregunta de Borges de 8 años) y `capital` (aparece en la pregunta de 7 años):
   responder bien garantizaba el castigo.

**Decisión.**
1. `_exam_words_for_age()`: las palabras de preguntas y respuestas de la batería
   (AGE_QUESTIONS + `school_exams_en.json` hasta la edad evaluada), más sus formas
   base del Diccionario Soberano, se unen a la whitelist de generación y al escaneo
   OOB. La máscara sigue siendo una whitelist de miles de palabras: esto no regala
   el argmax, solo devuelve la respuesta correcta al conjunto de candidatos.
2. Rúbrica: la respuesta esperada (y sus sinónimos/equivalentes semánticos claros)
   queda exenta del castigo por edad y se califica con la rúbrica normal de
   comprensión (exacta = 10; sinónimo razonable = 8-10). No hay 10 automático por
   sinonimia.
3. Los hitos ya certificados (2-6 años) se re-evalúan offline con el instrumento
   enmendado, y el suspenso de 7 años (5.60/10, 29-jul) se re-examina: pudo ser en
   parte artefacto de la máscara rota. El resultado se anota aquí; el estado del
   run solo se toca si el operador lo ratifica.

**Resultados de la re-evaluación (2026-08-01, misma sesión).**
- Hitos 2-6 años sobre sus checkpoints de milestone: **10/10 los cinco**, todos
  por exact match del auto-grader (sin juez). Las certificaciones previas se
  sostienen con el instrumento enmendado.
- **Re-examen de 7 años sobre el checkpoint vivo (dim 1024, ép. ~1370): 10/10**
  (antes 5.60). El suspenso del 29-jul era artefacto del instrumento: las
  respuestas correctas que hoy emite (`aliya`, `hideout`, `countries`) son formas
  base que la máscara antigua vetaba. El hueco de certificación de `7_years` de la
  transición fantasma queda cerrado retroactivamente (pendiente de ratificación
  del operador para añadirlo a `milestones_achieved`).
- **Preview del examen de 8 años sobre el checkpoint vivo: 10/10 (6/6)** — Bit ya
  tiene memorizada la batería completa 38 épocas antes de la frontera (1408).
  Confirma la hipótesis: las respuestas estaban aprendidas y solo la máscara
  impedía emitirlas. El examen oficial sigue siendo el de la frontera 1408 + la
  batería M4.
- Matiz de vocabulario detectado, consistente entre train y eval (no se toca):
  el Diccionario Soberano mapea `madrid→spain` (madrid no está en el vocabulario
  limpio), así que la pregunta de geografía entrena y acepta literalmente
  "the capital of spain is → spain". Anotado como limitación del examen de 7
  años, no como fallo del alumno.

**Límites declarados (no se tocan a mitad de run, D3 del RFC de graduación).**
Los exámenes ×300 se mezclan en el curriculum **antes** del split train/val del
partitioner → copias idénticas caen a ambos lados: el val_loss mide en parte
memorización de exámenes, y el examen de Samantha es explícitamente una prueba de
memoria. Por eso la certificación primaria de la graduación sigue siendo la
batería M1-M4 con umbrales congelados (`scripts/milestone_battery.py`, held-out
real); Samantha es secundaria (doctrina School v3 / DL-001).

**Referencias.** `docs/RFC_BIT_GRADUATION_ROADMAP.md` (D3, F5) ·
`docs/sessions/20260703/ROUTE_CHANGE_SCHOOL_V3.md` §2.3 · diff en
`scripts/evaluate_samantha_age.py`.

---

## DL-002 · 2026-07-27 — Precisión mixta BF16 (+SDPA, +compile opcional) para completar el currículum en la RTX

**Problema.** El currículum de 8 años no cabía en la GPU: el stage `secondary_8`
(dim 1024) proyectaba ~7.7 GB de VRAM sobre los 8 GB de la RTX 5070 (medido después:
6.9 GB de pico sintético — "cabe por los pelos", sin margen para fragmentación ni
convivencia con el kernel). Además, a 3h20-3h50 por época (FP32, dim 896), las ~410
épocas restantes hasta la graduación suponían ~60 días de GPU continua.

**Decisión.**
1. **BF16 vía autocast con pesos maestros FP32** (`--amp`, default `auto`), no la
   conversión total del checkpoint que proponía RFC-BITNET-VRAM-001 §4.4.1: el
   checkpoint no cambia de formato, FP32↔BF16 son intercambiables por ejecución
   (rollback = un flag) y la neurogénesis no se toca.
2. **SDPA** en la atención (habilitador de kernels fusionados; ~3% por sí solo).
3. **`--compile`** (torch.compile selectivo, §4.8) disponible y bajo prueba — el
   operador pidió no descartar nada; a nivel de step no rompe el STE.
4. El run vivo (epoch 998) queda congelado hasta que EXP_079 confirme que el paso
   32→16 bits no cuesta calidad de convergencia.

**Evidencia (Tier 1, medida — no estimada).** BF16+SDPA+compile: **3.7× más rápido**
(548→149 ms/step a dim 896) y **−40% VRAM** (6.1→3.6 GB); a dim 1024: 4.1 GB frente
a 6.9 GB en FP32. ∇STE de la misma magnitud en las 8 configuraciones. Proyección:
época ~1h, graduación en ~17-20 días de GPU en lugar de ~60. Detalle y tabla
completa: `docs/experiments/EXP_079_DESIGN.md` §2 +
`storage/benchmarks/tier1_step_bench.json`.

**Pendiente que gateaba la adopción → RESUELTO (28-jul-2026).** Tier 2 de EXP_079
ejecutado (A/B/C from-scratch, 160 épocas + neurogénesis 128→256 por brazo, misma
semilla): coste de calidad **indistinguible de cero** — val_loss final 3.7338 (FP32)
vs 3.7348 (BF16) vs 3.7343 (BF16+compile), |Δ| medio por época 0.003 con umbral en
0.05 y cero épocas fuera; batería y muestras idénticas entre brazos; épocas 2.6×
más rápidas end-to-end (80→31 min por 160 épocas). Único fallo literal: la
neurogénesis de B disparó en +5 épocas (sensibilidad del contador de plateau, no
degradación — análisis en EXP_079 §5). **Adoptado: `--amp auto --compile` en la
receta.** El run vivo se reanuda con vigilancia de ~100 épocas y rollback de un flag.

**Salvaguardas.** `--state_dir` (sandboxes; las rutas dejaron de estar hardcodeadas),
`--seed`, prohibido `--reset_state` en benchmarks, md5 del checkpoint vivo verificado
antes/después de cada brazo, backup en `storage/checkpoints/backup_pre_bf16_20260727/`.

**Referencias.** `docs/RFC_BIT_GRADUATION_ROADMAP.md` (D1-D6) ·
`docs/RFC_VRAM_SCALING_BITNET_CURRICULUM.md` (análisis técnico, Grok+DeepSeek) ·
commits 5477127, 1f55242, f0f28ad, c1947f0.

---

## DL-001 · 2026-07-03 — School v3: vocabulario limpio, corpus de fábrica, hitos operativos

Registrada retroactivamente por referencia: la decisión de ruta que gobierna el
entrenamiento actual vive en `docs/sessions/20260703/ROUTE_CHANGE_SCHOOL_V3.md`
(censo limpio de vocabulario tras el 57.8% de palabras fantasma, corpus sintético
de fábrica, neurogénesis por plateau, batería M1-M4 con umbrales congelados).
