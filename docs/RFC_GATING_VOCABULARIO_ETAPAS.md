# RFC — Gateo de Vocabulario por Etapa (BIT-003)

> **Estado**: IMPLEMENTADO (2026-08-21) · **Autor**: Aleth + Joan García
> **Contexto**: BIT-003 — re-entrenamiento de Bit v1 desde cero en inglés
> (CHILDES-en + TinyStories-en + diálogos) para la comparativa de tesis
> **Decisión registrada en**: `docs/DECISION_LOG.md` (DL-008)

---

## 1. Resumen ejecutivo

En el entrenamiento de Bit v1 en inglés, **cada etapa curricular solo puede
producir el vocabulario de su edad** (máscara de logits), y **el corpus de cada
etapa está formado exclusivamente por frases cuyo vocabulario pertenece a esa
etapa** (clasificación con umbral 95%). El gateo es **idéntico entre el brazo
estándar y el K-65P (glifos)**, de modo que la comparativa de la tesis mide solo
la representación de las palabras, no la política de exposición.

La propiedad central de la tesis: con K-65P el gateo se puede **ampliar en
caliente** (una palabra nueva se compone de primos ya aprendidos); con el
estándar no se puede sin reentrenar.

---

## 2. Motivación: la desalineación detectada

Antes de este RFC, el entrenamiento pasaba pérdida sobre las **20.095 palabras**
del vocabulario desde la etapa 0, pero la evaluación por edad solo permitía
emitir **729-851 palabras** (`get_allowed_vocab_for_age`). Además, ese gateo de
evaluación apuntaba a `childes_pre_school.json` (fichero **ES** ya inexistente)
y a `special_tokens` en español — no reflejaba el corpus EN real.

Consecuencia: Bit invertía gradiente en palabras que jamás podría producir en su
etapa, y el gateo no guiaba el aprendizaje por edad — justo la dimensión que la
tesis ("cognición con pocos datos vía glifos") quiere medir.

---

## 3. El gateo de producción por etapa (logit_mask)

### 3.1 Composición del vocabulario de cada etapa

```
Gateo(E{i}) = NÚCLEO ∪ TopN(CHILDES-en, frecuencia) ∪ Currículo(E{i})
```

- **NÚCLEO** (siempre, desde E0): las 28 palabras de referencia EN del
  `VOCAB_MAP` (water, food, fire, sun, ...) + 65 safe words de función
  (pronombres, artículos, verbos de alta frecuencia).
- **TopN de CHILDES-en**: las `N` palabras más frecuentes del corpus de habla
  infantil real (proxy de frecuencia de adquisición). Cortes por etapa:

| Etapa | Edad | N (top CHILDES) | Palabras gateadas |
|---|---|---|---|
| E0 | 0-1 | 200 | 242 |
| E1 | 1-2 | 800 | 822 |
| E2 | 2-3 | 3000 | 3007 |
| E3 | 3-4 | 5000 | 5004 |
| E4 | primaria_5 | 8000 | 8007 |
| E5 | primaria_6 | 10000 | 10005 |
| E6 | secondary_7 | 15000 | 15002 |
| E7 | secondary_8 | **todo** | 20095 |

- **Currículo(E{i})**: palabras del currículo acumulado de la etapa (asegura que
  los targets de examen estén siempre dentro del gateo de su etapa).

Tokens especiales: `<pad>` permitido siempre; `<unk>` vetado en todas las etapas
(ruido de generación).

### 3.2 Aplicación

En el forward de entrenamiento y validación (adaptativo + clásico):

```python
logits = apply_stage_gate(train_model(inputs, tau=tau), stage_mask)
# = logits + stage_mask.view(1,1,-1)   # -inf veta la palabra
```

La pérdida excluye posiciones cuyo target está vetado (`loss_mask_for_gate` +
`masked_fill`), lo que evita `inf × 0 = NaN` cuando el target tiene logit `-inf`.

Semántica pedagógica: el modelo **ve el corpus completo en contexto** (escucha
palabras que aún no sabe decir) pero **solo puede producir** el vocabulario de su
edad.

---

## 4. Equivalencia entre el brazo estándar y el K-65P

El gateo es una **máscara sobre tokens** (índices), construida
independientemente del tipo de embedding:

| | Brazo estándar (`--embedding standard`) | Brazo K-65P (`--embedding glyph`) |
|---|---|---|
| Vocabulario permitido por etapa | idéntico | idéntico |
| Corpus de la etapa | idéntico | idéntico |
| Representación de la palabra | tabla one-hot **congelada** + `inbound_proj` entrenable | composición de los 65 primos vía `glyph_table` |
| Añadir palabra nueva en caliente | ✗ (su columna de embedding no ha sido entrenada) | ✓ (se compone de primos ya aprendidos; basta desbloquear su token en la máscara) |

**La comparativa de la tesis mide únicamente la representación.** Ambos brazos
ven y producen las mismas palabras en cada etapa; la diferencia observable es
cómo se representa y qué pasa al ampliar el gateo en caliente (evaluación).

---

## 5. Clasificación del corpus por etapa (umbral 95%)

### 5.1 Regla

Cada frase del corpus general (CHILDES-en + TinyStories + 10% diálogos
preescolares) se asigna a la **etapa mínima** cuyo vocabulario cubre **≥95% de
sus tokens**:

```python
known  = [etapa(token) for token in seq if token es conocida]
# si len(known)/len(seq) < 0.95  →  etapa final (E7)
etapa  = percentil_95(sorted(known))
```

El corpus de la etapa `i` es el **acumulado** `E0 ∪ ... ∪ Ei`.

### 5.2 Resultado medido (corpus completo proyectado)

**CHILDES-en** (habla infantil real): cae naturalmente en etapas tempranas —
**78% en E0-E2**.

| E0 | E1 | E2 | E3 | E4 | E5 | E6 | E7 |
|---|---|---|---|---|---|---|---|
| 12% | 32% | 36% | 9% | 6% | 2% | 3% | 1% |

**TinyStories**: se incorpora según su vocabulario — solo 14% en E0-E1, repartida
en E2-E7 (33% en E2, 16% en E7).

| E0 | E1 | E2 | E3 | E4 | E5 | E6 | E7 |
|---|---|---|---|---|---|---|---|
| 3% | 11% | 33% | 17% | 11% | 3% | 6% | 16% |

**Ninguna frase se pierde**: E7 cubre el 100% del vocabulario, toda frase es
clasificable. El corpus solo se reordena por etapa.

### 5.3 El currículo

El currículo (119 items + exámenes ×300) mantiene su asignación por **MLU**
(longitud) — es material pedagógico pequeño, ya asignado por edad en
`compile_data_for_stage`.

---

## 6. Implementación

| Archivo | Contenido |
|---|---|
| `src/bitnet/training/modules/stage_gating.py` | `REFERENCE_WORDS_EN`, `SAFE_WORDS_EN`, `TOP_N_BY_STAGE`, `build_stage_logit_mask`, `build_all_stage_masks` (E7=todo), `token_min_stage`, `classify_sequences_by_gate`, `apply_stage_gate`, `loss_mask_for_gate` |
| `src/bitnet/training/train_sovereign_school.py` | construcción de `childes_freq` (Counter), `stage_masks`, clasificación `gated_general`, aplicación en forwards (adaptativo + clásico) |
| `scripts/evaluate_samantha_age.py` | `get_allowed_vocab_for_age` → `childes_pre_school_en.json`; `special_tokens` EN |

### 6.1 Almacenamiento y caché

- `childes_freq` se construye al tokenizar CHILDES-en (o al cargar la caché).
- El corpus general se re-clasifica por gateo en memoria (una vez por
  tokenización; ~40M secuencias).
- La clasificación depende de `stage_masks`, que depende de `childes_freq`, que
  está cubierto por el `corpus_hash` (CHILDES-en está en el hash).

---

## 7. Limitaciones y riesgos

1. **El gateo usa top-N de frecuencia global de CHILDES-en**, no edad del niño
   por archivo. Es un proxy razonable de adquisición (habla real infantil) pero
   no distingue la edad del hablante. Refinamiento futuro: AoA real (Kuperman et
   al. 2012, 30k palabras; MacArthur-Bates CDI vía Wordbank, 606 palabras).
2. **Con umbral 95%**, una frase con vocabulario mayormente simple pero una
   palabra rara ("the astronaut drinks water") se empuja a etapas tardías. Es la
   consecuencia aceptada de la exposición controlada.
3. **El estándar ve palabras complejas en el input** (mismo corpus) aunque no las
   produzca. Su `inbound_proj` recibe señal de contexto; la comparativa debe
   controlar esto en la interpretación.
4. La equivalencia de *exposición* entre brazos está garantizada; la equivalencia
   de *computo* no es el objetivo (ya retirada en DL-007).

---

## 8. Referencias

- `docs/DECISION_LOG.md` DL-008 (decisión) y DL-007 (equivalencia de coste)
- Commits: `081fc8d` (clasificación por gateo), `2741e95` (bucketing + TinyStories completo)
- `configs/childes_pre_school_en.json` (1.45M oraciones, 9.5M tokens, 99.53% cobertura)
- `configs/school_curriculum_structured_en.json` (119 items EN)
- Fuente AoA futura: `multilingual-aoa-prediction` (evaportelance) — `word-lists/eng`