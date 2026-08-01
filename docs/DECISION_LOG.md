# Decision Log — Frankenswarm

Registro de decisiones de calado: el problema, la decisión, la evidencia que la
respalda y dónde está el detalle. Una entrada por decisión, la más reciente arriba.
Las decisiones se numeran DL-NNN y no se reescriben: si una decisión se revierte,
se añade una entrada nueva que la referencia.

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
