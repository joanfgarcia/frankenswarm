# Cambio de Ruta: Escuela Soberana v3

**Fecha**: 2026-07-03 · **Decisión**: Joan (Operador) · **Auditoría y propuesta**: Aleth (Claude Fable 5) · **Estado**: aprobado, parcialmente implementado (2026-07-06)

---

## TL;DR

La escuela actual (run 2) queda **archivada como resultado negativo documentado**. Se
reconstruye el vocabulario desde fuentes limpias, se escala el corpus con generación
sintética controlada, y los hitos por "edad" se sustituyen por una **batería operacional
pre-registrada**. Los checkpoints 2y-5y no se reutilizan (el vocabulario nuevo cambia la
tabla de glifos y los hace incompatibles). No es empezar de cero: es empezar de uno.

---

## 1. Por qué (la evidencia)

Auditoría en frío del hito 5 (`model_milestone_5_years.pt`, 31.1M params, 640-dim),
script reproducible en `scratch/audit_bit_milestone5_fable.py`:

| Medida | Resultado | Lectura |
|---|---|---|
| Preferencia gramatical (frases nuevas vs desordenadas, sin muletas) | **14/15 (93%, azar 50%)** | La sintaxis española está internalizada y generaliza. **El sustrato ternario+glifos aprende.** |
| Teacher-forced (muestra CHILDES, máscara causal correcta) | 48% top-1, PPL 73.9 | Ajuste razonable a la distribución de entrenamiento |
| Generación libre | 1-4 palabras y `<pad>`, o ensalada | Producción colapsada en "mejor callar" |
| Parada natural | 9/10 | El stop entrenado funciona (pero degeneradamente corto) |

### Causas raíz identificadas

1. **Vocabulario 58% fantasma.** El censo (`scripts/rebuild_clean_vocabulary.py`) es
   exacto: de las 15.005 palabras de `expanded_glyphs.json`, solo **6.334 (42,2%)
   aparecen alguna vez en el corpus**. Las 8.671 restantes ("interpol", "hitler",
   "sexualmente"...) entraron con la expansión 3k→15k desde una lista de frecuencias
   genérica del español (las listas estándar derivan de subtítulos de cine). Tienen
   glifos composicionales y logits vivos, pero **cero ocurrencias de entrenamiento**:
   el modelo no puede aprender a suprimirlas una a una, y el muestreo libre cae en esa
   cola. La "ensalada" de la generación no venía del corpus: venía del vocabulario.
2. **Corpus mínimo y monótono: 863.712 tokens totales** (~1000× menos que TinyStories
   para coherencia de 3-4 años). El 62% (`tiny_dialogues_large`, 532K tokens) es
   combinatorio limpio pero con ~docenas de plantillas repetidas 15.000 veces: Bit
   memorizó la fórmula del saludo y aprendió que el silencio es la respuesta más barata.
3. **Sin held-out.** La escuela no separa train/test: toda evaluación medía memorización.
4. **Certificación débil.** El examen conversacional (juez LLM + parser regex de
   fallback indulgente) certificó "5 años" para una capacidad medible de receptor
   gramatical competente con producción de 18-24 meses.
5. **Bug de harness.** `verify_milestone_4.py` construía el modelo sin `is_causal=True`:
   todas las verificaciones manuales se hicieron con la máscara de atención equivocada.
   (Corregido en esta sesión; el mismo bug invalidó la primera pasada de la propia
   auditoría, detectado y corregido antes del veredicto.)

### Lo que la run 2 deja en herencia (por qué no es un fracaso)

El 93% de preferencia gramatical demuestra que la arquitectura (BitNet ternario +
glifos NSM + curriculum) **funciona**; el stop natural entrena; la Regla 7 (colisiones
OOV) sigue vigente; y este diagnóstico existe. Material de sección de resultados
negativos para el paper 1 — la honestidad es la marca de la casa.

---

## 2. La ruta nueva (School v3)

### 2.1 Vocabulario limpio (prerequisito de todo lo demás)

- `scripts/rebuild_clean_vocabulary.py` genera `configs/clean_vocabulary_words.json`
  desde fuentes limpias (CHILDES, NSM physics, currículo, exámenes, diálogos
  combinatorios y la fábrica). Hoy: 6.361 palabras.
- `FINAL_VOCAB` de `src/bitnet/expand_vocabulary.py` debe leer ese fichero y re-derivar
  glifos (proyección Ridge existente). **Esto invalida los checkpoints actuales** —
  archivarlos en `storage/checkpoints/sovereign_school/` como run 2.
- OOV → `<unk>` siempre; prohibido el vecino-más-cercano (Regla 7). El vocabulario
  crece re-ejecutando el censo cuando la fábrica aporta palabras nuevas — vivo pero
  auditado, nunca importado de listas ajenas.

### 2.2 Corpus por capas (objetivo 20-50M tokens)

| Capa | Fuente | Papel |
|---|---|---|
| CHILDES-es (324K tok) | real, limpio | habla infantil natural |
| NSM physics (1.4K tok) | propio | identidad del proyecto — **ampliar** |
| Diálogos combinatorios | regenerar con ×100 plantillas | estructura de turnos |
| **Samantha-fábrica** | `scripts/samantha_story_factory.py` | volumen: mini-historias graduadas por etapa, filtro OOV ≤2%, dedup, JSONL reanudable |

La fábrica está diseñada para correr desatendida bajo el **Sovereign Wake Gate**
(lotes cortos, sin interactividad): los AWAKENINGs generan corpus de madrugada.

### 2.3 Currículo: cinco correcciones

1. **Held-out 5% por etapa desde el día uno.** Sin held-out no hay certificación.
2. **Hitos operacionales** (M1-M4), no edades: batería pre-registrada en
   `scripts/milestone_battery.py` con umbrales congelados en código (gramática, cloze
   en held-out, salud de producción: longitud, stop, tasa de fantasmas). El juez LLM
   pasa a ser evaluación *secundaria* (dos jueces + rúbrica anclada); el parser de
   fallback señala, nunca aprueba.
3. **Secundaria ≠ Gutenberg.** Lecturas graduadas sintéticas en espiral (vocabulario
   anterior + N palabras nuevas por etapa); la literatura adulta es un precipicio de
   registro y una puerta de contaminación.
4. ✅ **Neurogénesis por dolor, no por calendario.** *(Implementado 2026-07-06)*
   Crecimiento 256→384→512→640 solo cuando la loss de *validación* se estanque
   (plateau = dolor), no automáticamente por etapa. Implementado con `--patience=15`
   y `--min_delta=0.01` configurables. Estado persistido en `school_state.json`.
   Helper `get_next_dim()` con ceiling por etapa. 17 tests.
5. **Variedad de longitud en respuestas objetivo**, para no re-entrenar el colapso a
   `<pad>` ("mejor callo") que produjo el dataset monótono.

### 2.4 Impacto en el plan general (README/ROADMAP sin cambio de rumbo)

La secuencia estratégica no cambia: Escuela → hito capaz → Jungle Reboot (Nico, Sofy,
Hugo desde el cerebro base de Bit) → desaparcar MoE. Cambia el **cómo se certifica
"capaz"** y la dieta con la que se llega. El hito "8 años" del README pasa a leerse
como "batería M4 superada".

---

## 3. Artefactos de esta sesión (todos pendientes de commit)

| Fichero | Papel |
|---|---|
| `scratch/audit_bit_milestone5_fable.py` | auditoría reproducible (la evidencia) |
| `scripts/rebuild_clean_vocabulary.py` | censo + vocabulario limpio (ejecutado: 6.361 palabras, 57,8% fantasmas confirmados) |
| `configs/clean_vocabulary_words.json` | vocabulario limpio v1 |
| `scripts/samantha_story_factory.py` | fábrica de corpus (probada en `--mock`) |
| `scripts/milestone_battery.py` | batería M1-M4 con umbrales pre-registrados |
| `scratch/verify_milestone_4.py` | corregido `is_causal=True` |
| `lab/BRIEFING.md` | frente actualizado a School v3 |
| `CHANGELOG.md` | entrada del cambio de ruta |

### Orden de ejecución de la ruta

1. Regenerar diálogos combinatorios (plantillas ×100) → re-censo de vocabulario.
2. Fábrica: preescolar 2M tokens (AWAKENINGs nocturnos bajo wake gate).
3. `FINAL_VOCAB` → glifos limpios → **School v3 desde cero** con held-out y batería.
4. M1 certificado por batería antes de pasar de etapa. Sin excepciones — los umbrales
   se congelaron antes de entrenar y así se quedan.
