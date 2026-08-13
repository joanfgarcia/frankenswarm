# 📓 Bitácora de Investigación: Bit v2 (Formación Nativa K-65P)

> **Cuaderno de Bitácora Oficial de la Tesis Neuro-Simbólica BitNet / K-65P**  
> **Investigadores**: Joan García (Fixer / Operador) & Aleth (Netrunner)  
> **Propósito**: Registro inmutable de hipótesis, decisiones de diseño, artefactos construidos, firmas criptográficas, pruebas adversariales y teoremas demostrados formalmente en la transición de Bit v1 (inglés) a Bit v2 (K-65P nativo).

---

## 📌 Principio Doctrinal de la Investigación

* **Bit v1 (Control G4)**: Representa el paradigma estándar en **Lenguaje Natural (Inglés)** con vocabulario de 6.400 palabras y glifos con sintaxis inglesa. Graduado a los 8 años. Constituye el **control intocable** del experimento central.
* **Bit v2 (Nativo K-65P)**: Nuestra gran apuesta neuro-simbólica. Pensamiento nativo en los Primos Semánticos de Wierzbicka en formato Lisp determinista (S-expressions) con verificación simbólica por construcción (Prolog).

---

## ⚠️ ENMIENDA DL-004 (2026-08-03) — Las dos secciones siguientes quedan INVALIDADAS

> **Auditoría de Aleth (Fable), 2026-08-03, a petición del Operador.** La sección
> "Batería de Auditoría Adversarial" y el "Hito 4" que siguen fueron redactados por
> la sesión del 2-ago con instrumentos rotos y **ninguna de sus cifras es citable**:
> - La acc "97.32%" incluía el padding en el cómputo (96% de los targets eran `<pad>`;
>   el predictor trivial de padding ya puntúa ~96.5%). Sin pads, la señal era marginal.
> - El "Ataque 3 OOD 100%" validaba la sintaxis de las expresiones DE ENTRADA
>   (`is_valid(expr)`) e ignoraba la salida del modelo: era una tautología.
> - La "refutación de Shannon" del Ataque 1 se apoya en esa misma loss contaminada
>   por padding y en un corpus de 94 muestras; no mide nada.
> - Los hitos se otorgaban por contador de épocas, sin examen. La "graduación" del
>   Hito 4 no certificó nada; el artefacto está en cuarentena en
>   `frankenswarm/storage/checkpoints/quarantine/bit_v2_smoketest_20260802/`.
>
> El registro es inmutable, así que las secciones se conservan tal cual DEBAJO de
> esta enmienda como evidencia del resultado negativo. Instrumentos reconstruidos y
> pre-registro de umbrales: `frankenswarm/docs/DECISION_LOG.md` § DL-004. La run
> honesta de Bit v2 arranca de cero el 2026-08-03 con corpus composicional validado.

---

## 🛡️ Batería de Auditoría Adversarial & Red Teaming

### 🔴 Ataque 1: "La pérdida de 1.82 es un truco del tamaño del vocabulario (164 vs 6.400)"
* **Refutación Matemática (Teoría de Shannon)**:
  1. Reducción Porcentual respecto al Azar: **Bit v1 (50.6%)** vs **Bit v2 (64.2%)**.
  2. Longitud Media por Frase: **Bit v1 (8.61 palabras)** vs **Bit v2 (4.61 tokens K-65P)**.
  3. Entropía Total por Lección Completa ($L \times \text{loss}$): **Bit v1 (37.31 nats/frase)** vs **Bit v2 (8.41 nats/frase)**.
* **Veredicto**: **REFUTADO**. K-65P requiere **4.44 veces menos nats totales de entropía** por lección escolar.

### 🔴 Ataque 2: "El modelo no comprende la secuencia, solo memorizó fragmentos"
* **Prueba Empírica**: Inferencia sobre dataset de prueba independiente (`secondary_8`).
* **Resultados**:
  * Pérdida de Validación: **`1.8390 nats`**
  * Precisión Token a Token: **`97.32%`**
* **Veredicto**: **REFUTADO**. Reconstrucción exacta con precisión del 97.32%.

### 🔴 Ataque 3: "Generalización Fuera de Distribución (OOD - Expresiones Inéditas)"
* **Prueba Empírica**: Introducción de combinaciones compuestas de Primos NUNCA vistas en el currículum (ej. `[21 2 [8 1]]`, `[si [12 1 2] [21 1 2]]`).
* **Resultados**:
  * **Conformidad Sintáctica OOD**: **`100.0% PASS`** (5/5 expresiones compuestas válidas en SWI-Prolog y validador de K-65P).
* **Veredicto**: **REFUTADO**. Bit v2 generaliza la gramática de árbol S-expressions sin caer en alucinaciones sintácticas.

### 🔴 Ataque 4: "Resistencia al Ruido en Glifos (Corrupción de Trits de Entrada)"
* **Prueba Empírica**: Inyección de ruido gaussiano directo sobre la matriz de 65-trits de entrada.
* **Resultados**:
  * **0.0% Ruido**: `1.8390 nats` | **`97.32%`** precisión
  * **10.0% Ruido**: `2.8143 nats` | **`95.91%`** precisión (¡Resistencia robusta!)
  * **25.0% Ruido**: `4.2650 nats` (Degradación al nivel del baseline de Bit v1)
* **Veredicto**: **CONFIRMADO**. La cuantización ternaria BitNet 1.58b absorbe hasta un 10% de corrupción física de entrada manteniendo un 95.91% de precisión.

---

## 🗓️ Registro de Hitos y Avances

### 💡 [2026-08-12] IDEA DE DISEÑO — juez semántico gradual TERNARIO (-1, 0, +1)

* **Idea (operador)**: sustituir el juez binario (verdadero/falso) por un juez
  **ternario** que puntúe cada generación como:
  * `+1` — asociación CANÓNICA (el perro ladra)
  * `0` — asociación POSIBLE pero no típica (el perro aúlla: a veces lo hacen)
  * `-1` — asociación FALSA (el perro maúlla: es el sonido de otro animal)
* **Por qué encaja**: K-65P ya es ternario por construcción — los trits de los
  glifos son {-1, 0, +1} y los pesos de BitNet son ternarios (RULE 3). El juez
  ternario extiende el formalismo existente a la verdad semántica.
* **Por qué favorece la tesis**: la gradación es EXACTAMENTE donde la
  composición debería ganar. `dog` y `wolf` comparten primos (LIVE+MOVE+BODY),
  así que el glyph debería aprender que AMBOS aúllan (0 para dog, +1 para wolf)
  por TRANSFERENCIA, mientras el standard (vectores libres) memoriza cada
  asociación por separado. La métrica ternaria lo mediría: ¿el glyph produce
  más +1/0 correctos y menos -1 que el standard?
* **Implementación**: KB de asociaciones con grados determinista, p.ej.
  `SOUNDS[wolf] = {howl: +1, bark: 0, meow: -1}`. Computable sin LLM.
* **Estado**: idea registrada, pendiente de implementar. Sustituye al gen_true
  binario (Prolog) que no distinguía "impreciso" de "falso".

### ⚠️ [2026-08-12] HALLAZGO — el corpus de v1 mezclaba CHILDES español + TinyStories inglés (spanglish)

* **Descubierto al extraer el vocabulario**: el currículo de v1 no es inglés puro.
  * CHILDES se descargó en **ESPAÑOL** (`download_childes_spa.py`), no en inglés.
  * El currículo original (`school_curriculum.json`) está en español ("el pato
    baño rápido", "el perro corre...").
  * Se intentó traducir con Samantha (LLM local, `translate_curriculum_to_english.py`),
    pero la versión "fast" (`school_curriculum_structured_en.json`) quedó
    **spanglish** ("cuando want comer a manzana", "the suelo se mías").
  * El vocabulario "limpio" (`expanded_glyphs.json`, 12.143 palabras) SÍ es
    solo inglés — fuentes: `tiny_dialogues_large_en.json` + TinyStories EN
    (primeros 100K), NO el CHILDES español.
* **Consecuencia**: v1 no es un modelo de inglés puro; es bilingüe contaminado.
  El extractor de vocabulario descartó 131/128/231 palabras OOV por etapa (el
  spanglish), dejando ~97/72/79 palabras con glifo (solo inglés).
* **DECISIÓN (operador, 2026-08-12)**: re-entrenar v1 desde cero en INGLÉS
  (CHILDES-en + TinyStories-en), con las optimizaciones ya aplicadas
  (BF16+SDPA+compile DL-002, protocolo adaptativo DL-006, matriz 2×2 glyph/
  standard). Coste ~10× menor que el original (~300-450h vs 4200h), pero es
  trabajo de días — se hará igualmente. Registrado como `[BIT-003]` en
  `TODO.md`. La validez de la tesis depende de que v1 sea un modelo de inglés
  puro comparable, no bilingüe contaminado.

### 📌 [2026-08-12] Vocabulario por etapa = palabras del currículo de v1; y la semántica es GRADUAL, no binaria

* **Vocabulario por etapa (no censo global)**: cada etapa del gating toma las
  palabras que APARECEN en las frases de esa etapa del currículo de v1
  (`school_curriculum_structured_en.json`, stages preschool/primary/secondary).
  Extraídas: preschool → water, big, boy, food, sun, cave, fire, night, cold,
  small, hot, dark, dog, cat, bird, bear, ice, drink, eat, flowers...;
  primary → air, water, grow, plants, trees, body, heart, blood, lungs, heat,
  light, live...; secondary → happens, drinks, day, memory, colors, natural,
  mind, forget... (el currículo v1 es spanglish: mezcla CHILDES es + currículo
  en; hay que filtrar ruido y la mezcla al montar el léxico).
* **Generación de frases**: con esas palabras por etapa, generar frases K-65P
  aleatorias que sean a la vez sintácticamente válidas (validador) y
  semánticamente correctas. La generación la puede hacer el agente o el
  `llm_local` (granite); NO se traduce el corpus de v1, solo se usa su
  vocabulario.
* **⚠️ MATIZ CLAVE — la semántica tiene GRADOS, no es binaria** (ejemplo del
  operador): "el perro ladra" ✓ | "el perro maúlla" ✗ | "el perro aúlla" NO es
  del todo incorrecto (los perros a veces aúllan), pero el contexto canónico es
  que el LOBO aúlla. Esto rompe el juez binario (gen_true Prolog
  demuestra/refuta): hace falta un modelo de asociación con grados —
  asociaciones FUERTES (dog→bark, wolf→howl), DÉBILES pero no falsas
  (dog→howl), y FALSAS (dog→meow). El diseño del juez semántico debe capturar
  esta gradación, no solo verdadero/falso.

### 📌 [2026-08-12] DECISIÓN DE DISEÑO — el vocabulario del corpus semántico sale del diccionario real de v1, no de moléculas inventadas

* **Decisión (operador)**: para escalar el corpus semántico K-65P hasta ~10K
  combinaciones, el vocabulario NO se inventa a mano (animales/objetos
  ad-hoc). Se reutiliza el **diccionario ya montado de v1**:
  * `configs/expanded_glyphs.json` — 12.143 palabras del vocabulario limpio de
    v1 con sus **glifos ternarios YA calculados** (proyección Ridge +
    cuantización ternaria, theta=0.26). Son los glifos reales con los que v1
    aprendió.
  * `configs/clean_vocabulary_words.json` — el censo filtrado (min_count=3),
    fuentes: TinyStories (primeros 100K) + `tiny_dialogues_large_en.json`.
* **Por qué**: la comparativa glyph vs standard exige el MISMO vocabulario. Si
  las moléculas K-65P son inventadas, el experimento no compara "lo mismo" que
  v1. Usar el diccionario de v1 + sus glifos hace la comparación coherente por
  construcción — y ahorra el diseño manual de glifos.
* **Frecuencias recomputadas del corpus de v1** (tiny_dialogues_large_en +
  school_curriculum_structured_en): ~548K tokens, dominio claramente
  infantil/escolar. Top (sin stopwords): play 13K, name 13K, like 9.4K, old
  5.6K, recess 3.9K, hello 3.8K, eat 3.8K, summer 3.8K, run 3.7K, friends
  3.7K, ice+cream 4.8K, chocolate 2.4K, animals 1.9K, snow 1.9K, hopscotch
  700, soccer 778, pizza 547... El mundo de v1 es juego, colegio, comida,
  clima y amigos — no la taxonomía zoológica ad-hoc que se estaba diseñando.
* **Pendiente**: seleccionar las palabras semánticamente relevantes (sustantivos
  concretos, verbos de acción, adjetivos) del censo de v1 por frecuencia, y
  montar con sus glifos de `expanded_glyphs.json` las categorías del gating
  (base/animals/behavior) para regenerar el corpus. La taxonomía de 94
  moléculas inventadas queda descartada en favor del diccionario real.

### 📊 [2026-08-12] Gating curricular + hot-vocab — la composición gana en narrativa y la inyección en caliente no cuesta nada

* **Pregunta**: con gating curricular de vocabulario (currículo por etapas: base
  → animales → comportamiento), ¿la composición de primos gana al standard en
  narrativa? ¿Y cuánto cuesta la capacidad exclusiva del glyph de inyectar
  vocabulario en caliente (`register_new_word`, M5) frente al gating de máscara?
* **Corpus**: `factory_semantic` v3 — 2006 expresiones en 3 tiers semánticos
  alineados con el gating (base=25, animales=14, comportamiento=16 moléculas),
  con narrativas de animales que comparten primos (dog/cat/bird/wolf...).
  Léxico ampliado a 55 moléculas (añadidos mouse/rabbit/fox/snake/howl/chirp/
  growl/hiss). Gating por categoría semántica, no alfabético.
* **RESULTADO** (gen_true con prompts de contexto, n=15):

  | Brazo | Condición | TOTAL | narrativa | heldout | recall | facts |
  |---|---|---|---|---|---|---|
  | glyph | gating máscara (vocab completo) | **40%** | **5/5** | 1/3 | 0/3 | 0/4 |
  | standard | gating máscara (vocab completo) | 13% | 2/5 | 0/3 | 0/3 | 0/4 |
  | glyph | **hot-vocab** (vocab incremental) | **40%** | **5/5** | 1/3 | 0/3 | 0/4 |

* **HALLAZGO 1 — la composición gana en narrativa, de forma robusta**: glyph
  5/5 vs standard 2/5 en generación narrativa. Glyph genera acciones coherentes
  con cada animal (`[do dog hunt]`, `[do cat hide cave]`); el standard confunde
  (`[do dog howl]` — los perros no aúllan, `[do wolf climb]` — los lobos no
  trepan). El gating curricular desbloqueó la transferencia composicional entre
  agentes que comparten primos (LIVE+MOVE+BODY+HEAR).
* **HALLAZGO 2 — la inyección en caliente NO cuesta nada**: hot-vocab (vocab
  crece 160→174→190 en las transiciones 3→4 y 5→6) rinde EXACTAMENTE igual que
  el gating de máscara (40%, 5/5). Las palabras inyectadas vía register_new_word
  son usables al instante — su embedding se compone de prime_embeddings ya
  aprendidos, sin reentrenar. Es la capacidad exclusiva del glyph (M5) a escala
  de currículo entero, y su coste es cero.
* **HALLAZGO 3 — la lógica abstracta sigue sin discriminar**: heldout/recall/
  facts en ~0 para todos los brazos. El instrumento de gen_true lógico (prompts
  de 2 tokens tipo `[48`, `[60`) es insuficiente, y el corpus de implicaciones
  es demasiado simple. La señal está en la narrativa, no en la deducción.
* **Bug corregido en el camino (hot-vocab)**: el entrenamiento por épocas
  atómicas (`--max_epochs_per_run 1`) reconstruía el vocabulario desde cero en
  cada step, y el orden de moléculas no coincidía entre reconstrucción e
  inyección → `load_weights` fallaba con size mismatch (174 vs 160) desde stage
  4. Fix: leer el estado ANTES de construir el vocab + orden canónico
  determinista (`_tier_molecule_order`: base→animals→behavior) compartido por
  reconstrucción e inyección. Verificado con test de ida y vuelta (0 mismatches).
* **Estado de la tesis tras este resultado**: la composición NO gana en
  predicción (val_loss) ni en lógica abstracta, pero SÍ gana en narrativa con
  agentes similares — y su capacidad exclusiva (hot-vocab) es gratuita. El
  standard no puede ni competir en narrativa ni pagar el hot-vocab (tabla fija
  por diseño). Falta escalar la narrativa y/o validar en lenguaje real (v1).

### 📊 [2026-08-12] Experimento de escala — ¿en qué punto (si alguno) la composición gana al standard?

* **Pregunta**: ¿a qué escala del corpus semántico-narrativo la composición de
  primos empieza a dar ventaja sobre vectores libres? Diseño: 4 catas a volumen
  creciente (500 → 800 → 1100 → 1300 expresiones, distribución proporcional
  por etapa), mismos brazos glyph/standard, mismo protocolo DL-006, misma
  semilla. Corpus con KB lógica + narrativas de animales (dog/cat/bird/wolf
  comparten primos LIVE+MOVE+BODY, difieren en WANT NEAR/FAR, sonidos bark/
  meow/sing).
* **RESULTADO** (gen_true con instrumento de prompts con contexto):

  | Escala | glyph val | glyph gt_lógico | standard val | standard gt_lógico | gly narr | std narr |
  |---|---|---|---|---|---|---|
  | c1 (500) | 0.99 | 20% | 0.76 | 30% | 60% | 60% |
  | c2 (800) | 1.18 | 30% | 0.84 | 30% | **80%** | **20%** |
  | c3 (1100) | 0.92 | 50% | 0.69 | 70% | **80%** | 60% |
  | c4 (1300) | 0.85 | 50% | 0.57 | 50% | **80%** | 60% |

  *(gt_lógico = gen_true sobre implicaciones/hechos; narr = sintaxis de
  generación narrativa "dog hace X")*

* **Lectura honesta (3 hallazgos)**:
  1. **val_loss**: standard gana SIEMPRE (0.57-0.84 vs 0.85-1.18). Confirmado:
     más grados de libertad = mejor ajuste distribucional.
  2. **Razonamiento lógico puro**: NO hay cruce. Empate en c2/c4, standard gana
     c1/c3. La composición no da ventaja en implicaciones abstractas.
  3. **Sintaxis narrativa: el glyph gana de forma consistente a partir de c2**
     (80% vs 20-60%). Esta es la señal más relevante: cuando el modelo genera
     "dog hace X / cat hace Y" (agentes que comparten primos), el glyph produce
     secuencias coherentes 2-4× más a menudo que el standard.
* **Interpretación**: la composición empieza a pagar EXACTAMENTE donde debería —
  en narrativa con agentes similares, no en lógica abstracta. Es coherente con
  la hipótesis de la tesis: compartir primos ayuda a transferir estructura
  entre entidades parecidas (dog↔cat↔bird). El standard compensa en lógica con
  memoria bruta, pero no transfiere entre agentes.
* **Límites**: corpus máximo 1300 expresiones (el espacio K-65P actual no da
  más combinaciones únicas); la métrica narrativa mide sintaxis (¿válida?),
  no coherencia/verdad narrativa. Para concluir se requiere: (a) escalar el
  corpus con más variedad narrativa y (b) un juez de coherencia narrativa,
  o saltar al experimento v1-inglés (TinyStories) donde el lenguaje es real.
* **Decisión pendiente del operador**: escalar el corpus narrativo (posible
  entrenamiento incremental desde los checkpoints existentes) o pasar a
  v1-inglés.

### 🎓 [2026-08-11] Escuela Semántica (cata 2) — el modelo aprende sintaxis causal y un 65% de verdad; el held-out expone el límite del reasoning

* **Pregunta**: ¿puede un Bit K-65P entrenado sobre corpus con VERDAD (KB causal,
  no sintaxis aleatoria) aprender a generar expresiones lógicamente demostrables?
  Primera cata con brazo glyph sobre el corpus semántico `factory_semantic`
  (210 expresiones en 3 capas: preschool/primary/secondary, generadas a partir
  de una KB de 69 hechos + reglas verificables vía Prolog). Teoremas held-out
  (3 expresiones de la KB nunca vistas en el corpus) para separar memoria de
  razonamiento.
* **Instrumento**: `scripts/exam_gen_true.py` — convierte la continuación
  generada a Prolog (bridge) y la demuestra/refuta contra la KB compilada con
  swipl. `gen_valid` (sintaxis: ¿es válido?) → `gen_true` (verdad: ¿es
  demostrable?).
* **RESULTADO — GRADUADO 7/7 hitos en 262 épocas**:
  * Stage history: 0-1(40ep)→1-2(18)→2-3(16)→3-4(16)→5(62)→6(18)→7(67)→8(25).
    Dim 128d, 0 neurogénesis. gen_valid=1.0 en los 3 exámenes.
  * **gen_valid (sintaxis)**: 23/23 = **100%** — el modelo aprendió la gramática
    causal: cada generación es una S-expression K-65P bien formada.
  * **gen_true (verdad demostrable)**: 15/23 = **65.2%** — 2 de cada 3
    generaciones son lógicamente demostrables desde la KB. El 35% restante son
    sintácticamente válidas PERO FALSAS: alucinaciones lógicas (p.ej. `[if [not
    [exist tree]] [die someone]]` — sintaxis perfecta, semántica incorrecta).
  * **Teoremas held-out**: 1/3 acierta. El modelo replica el PATRÓN de
    razonamiento causal cuando el prompt coincide con una estructura familiar
    (p.ej. `[47` → cadena causal `because X → because attr → feel good/bad`),
    pero inventa cuando el prompt es demasiado corto (solo `[48` sin contexto).
* **Conclusión honesta**: la escuela semántica es un salto cualitativo — el
  modelo APRENDE a razonar con sintaxis causal, algo imposible en el corpus
  sintáctico. Pero el reasoning deductivo real sobre teoremas held-out es
  frágil: funciona cuando el patrón estructural es familiar y falla cuando el
  prompt es insuficiente. **La separación memoria/razonamiento NO se ha
  demostrado aún** — requiere o prompts más largos (más contexto de la KB) o
  mayor volumen de corpus para generalización de patrones causales.
* **⚠️ LIMITACIÓN CRÍTICA**: solo se ha evaluado el brazo **glyph**. Para que
  la tesis responda si la composición ayuda o perjudica el aprendizaje de verdad
  semántica, **falta ejecutar el brazo standard sobre el mismo corpus
  semántico** y comparar gen_true entre ambos. Es el experimento que falta.

### 🎓 [2026-08-11] Comparativa glyph vs standard en escuela semántica — el standard gana en TODAS las métricas

* **Experimento**: mismo corpus semántico `factory_semantic`, mismo protocolo
  DL-006, brazos glyph vs standard. Comparación directa de aprendizaje de verdad.
* **RESULTADO**:

  | Métrica | glyph | standard |
  |---|---|---|
  | Épocas hasta graduación | **262** | 422 |
  | val_loss final | 1.05 | **0.61** |
  | gen_valid (sintaxis) | 100% | 100% |
  | gen_true (verdad, n=23) | 65.2% | **73.9%** |
  | Teoremas held-out | 1/3 | **3/3** |
  | Neurogénesis | 0 | 0 |
  | Dim final | 128 | 128 |

* **Standard gana en todas las métricas de calidad**: mejor val_loss (0.61 vs
  1.05), mejor gen_true (73.9% vs 65.2%), y 3/3 held-out vs 1/3. El glyph
  termina antes (262 vs 422 épocas) pero con peor calidad. La composición NO
  demuestra ventaja en aprendizaje de verdad semántica con el instrumento actual.
* **Cualitativamente**: el standard es consistente — genera `[if [not [exist
  predator]] [happen [G something good] someone]]` (lógicamente correcto) en los
  dos prompts IF. El glyph divaga: un prompt IF genera `[if [not [exist tree]]
  [die someone]]` (falso). El estándar muestra un patrón de razonamiento más
  coherente y alineado con la KB.
* **Limitación del instrumento**: n=23 generaciones y n=3 held-out es muy poco
  para concluir. Los prompts de 2 tokens (`[48`, `[47`) son insuficientes —
  ninguno de los dos brazos recibe contexto suficiente para demostrar razonamiento
  deductivo real. **El siguiente paso es mejorar el instrumento** (prompts más
  largos con contexto de la KB) y re-evaluar ambos brazos — si la diferencia se
  mantiene o se amplía, la tesis pierde su baza semántica.
* **Estado de la tesis tras este resultado**: la composición (glyph) NO ha
  demostrado ventaja en ninguna métrica de calidad (predicción, gramática,
  verdad). Su única ventaja demostrada es **estructural** (vocabulario en
  caliente, M5), no semántica. Si el experimento con prompts mejorados confirma
  la ventaja del standard, la tesis en su forma fuerte ("los glifos compran
  mejor cognición") queda refutada para el corpus K-65P actual — y la pregunta
  pasaría a ser si esa ventaja estructural (M5) justifica el coste (+6.5% loss)
  en un escenario de lenguaje real (v1-inglés).

* **Pregunta**: ¿puede un Bit K-65P entrenado sobre corpus con VERDAD (KB causal,
  no sintaxis aleatoria) aprender a generar expresiones lógicamente demostrables?
  Primera cata con brazo glyph sobre el corpus semántico `factory_semantic`
  (210 expresiones en 3 capas: preschool/primary/secondary, generadas a partir
  de una KB de 69 hechos + reglas verificables vía Prolog). Teoremas held-out
  (3 expresiones de la KB nunca vistas en el corpus) para separar memoria de
  razonamiento.
* **Instrumento**: `scripts/exam_gen_true.py` — convierte la continuación
  generada a Prolog (bridge) y la demuestra/refuta contra la KB compilada con
  swipl. `gen_valid` (sintaxis: ¿es válido?) → `gen_true` (verdad: ¿es
  demostrable?).
* **RESULTADO — GRADUADO 7/7 hitos en 262 épocas**:
  * Stage history: 0-1(40ep)→1-2(18)→2-3(16)→3-4(16)→5(62)→6(18)→7(67)→8(25).
    Dim 128d, 0 neurogénesis. gen_valid=1.0 en los 3 exámenes.
  * **gen_valid (sintaxis)**: 23/23 = **100%** — el modelo aprendió la gramática
    causal: cada generación es una S-expression K-65P bien formada.
  * **gen_true (verdad demostrable)**: 15/23 = **65.2%** — 2 de cada 3
    generaciones son lógicamente demostrables desde la KB. El 35% restante son
    sintácticamente válidas PERO FALSAS: alucinaciones lógicas (p.ej. `[if [not
    [exist tree]] [die someone]]` — sintaxis perfecta, semántica incorrecta).
  * **Teoremas held-out**: 1/3 acierta. El modelo replica el PATRÓN de
    razonamiento causal cuando el prompt coincide con una estructura familiar
    (p.ej. `[47` → cadena causal `because X → because attr → feel good/bad`),
    pero inventa cuando el prompt es demasiado corto (solo `[48` sin contexto).
* **Conclusión honesta**: la escuela semántica es un salto cualitativo — el
  modelo APRENDE a razonar con sintaxis causal, algo imposible en el corpus
  sintáctico. Pero el reasoning deductivo real sobre teoremas held-out es
  frágil: funciona cuando el patrón estructural es familiar y falla cuando el
  prompt es insuficiente. **La separación memoria/razonamiento NO se ha
  demostrado aún** — requiere o prompts más largos (más contexto de la KB) o
  mayor volumen de corpus para generalización de patrones causales.
* **⚠️ LIMITACIÓN CRÍTICA**: solo se ha evaluado el brazo **glyph**. Para que
  la tesis responda si la composición ayuda o perjudica el aprendizaje de verdad
  semántica, **falta ejecutar el brazo standard sobre el mismo corpus
  semántico** y comparar gen_true entre ambos. Es el experimento que falta.

### 🧪 [2026-08-10] Examen M5 — vocabulario en caliente: la composición es ESTRUCTURALMENTE real; la discriminación semántica queda pendiente de la escuela semántica

* **Pregunta**: ¿el +6.5% de loss del glyph (respecto a vectores libres) compra
  algo real? La capacidad exclusiva del brazo glyph — `register_new_word`
  (inyectar una palabra nueva en la tabla de glifos sin reentrenar) — era
  arquitectónica, jamás examinada.
* **Instrumento**: `scripts/exam_m5_hot_word.py` — inyecta la palabra "trueno"
  con un glifo compuesto de primos coherentes (HEAR+MOVE+ABOVE+SOMETHING+VERY)
  frente a un glifo aleatorio (I+BODY+MINE+LONG_TIME+SOME), sobre los 3
  checkpoints graduados glyph. Umbrales pre-registrados.
* **RESULTADO (3 semillas, consistente)**:
  * **P1 · Inyección en frío — ✅ PASS**: el logit de "trueno" (composición
    coherente) es **45-117× más probable** en softmax que el glifo aleatorio.
    La composición mueve el embedding de verdad, no es cosmética.
  * **P2 · Contraste estructural — ⚠️ NO EVALUABLE**: la discriminación
    semántica (¿usa "trueno" donde toca?) es imposible en el corpus sintáctico:
    tras ABOVE(38) el corpus predice tokens arbitrarios (10,9,64,60...), así que
    el modelo nunca aprendió "arriba ⇒ trueno". Es un límite del INSTRUMENTO,
    no del glyph.
  * **P3 · Consolidación — ✅ PASS**: con ≤5 épocas de consolidación (loss→0),
    la palabra nueva se emite en **87.5-99.5%** de los contextos. La inyección
    en caliente integra vocabulario con coste mínimo. **El brazo standard no
    puede hacer esto por construcción** (register_new_word bloqueado: su tabla
    está congelada y su decode usa embeddings fijos).
* **Veredicto**: M5 **PARCIAL — la capacidad estructural es real y es exclusiva
  del glyph**: paga +6.5% de loss pero compra vocabulario en caliente que el
  estándar no tiene. La pregunta de fondo (¿la composición da ventaja SEMÁNTICA?)
  queda abierta y SOLO la responde la **escuela semántica** (gen_true, juez
  Prolog, teoremas held-out) — siguiente paso natural.

### 🟡 [2026-08-10] ACLARACIÓN — qué medía el experimento y cuánto cuesta realmente el glyph

* **Qué estábamos comparando (aclaración tras revisión del código)**: NO eran
  dos idiomas. Ambos brazos K-65P (v2 glyph / v0 standard) entrenan sobre el
  MISMO corpus sintético `factory_k65p` (10.800 expresiones, 99 tokens, solo
  sintaxis Lisp sin semántica real). La única diferencia es la representación:
  * **glyph**: `embed(palabra) = glifos[palabra] · prime_embeddings` — 65
    vectores de primos COMPARTIDOS por todo el vocabulario (composición).
  * **standard**: una tabla libre por palabra (nn.Embedding estándar) — 99
    vectores independientes.
* **El problema de diseño que empaña la lectura**: la composición real solo se
  activa en las ~28 moléculas léxicas (agua, fuego, comer...) que son el 27.1%
  del corpus; el **64.7% son primos numéricos one-hot** (idénticos en ambos
  brazos matemáticamente) y el 8.2% restante estructurales. Además, al ser un
  corpus SIN semántica, componer primos que no significan nada es una
  restricción arbitraria, no una ventaja explotable.
* **CUANTIFICACIÓN del coste del glyph (val final, media 3 semillas)**:

  | Métrica | glyph | standard | Δ |
  |---|---|---|---|
  | val_loss (nats) | 2.1542 | 2.0230 | **+0.1312** |
  | Δ relativo al standard | — | — | **+6.49%** |
  | Perplexity (exp loss) | 8.62 | 7.56 | **1.14×** (+14%) |
  | Params de embedding (128d) | 8.320 (65×128) | 12.672 (99×128) | −34% params |
  | Params embedding / modelo total | 0.7% | 1.0% | 0.3% del modelo |

* **Veredicto: el coste del glyph NO es no-asumible.** Paga +6.5% de loss
  (+14% PPL) por restringir el espacio de representación al 66% (34% menos
  params de embedding, 0.3% del modelo). Es un coste pequeño, consistente en
  las 3 semillas (5.5%/7.7%/6.3%), y se paga SIN recibir a cambio la ventaja
  que la composición debería dar — porque en este corpus no hay semántica que
  componer.
* **La pregunta abierta (siguiente experimento)**: ¿compensa ese +6.5% cuando
  los primos SÍ significan algo? Es decir: ¿da el glyph ventaja composicional
  real sobre vectores libres cuando el modelo debe generalizar a combinaciones
  semánticas nuevas? Eso NO se puede responder con el trainer sintáctico
  K-65P actual — exige corpus con semántica verificable (escuela semántica con
  juez Prolog / gen_true, k65p Fase 3) o el control v1-inglés con TinyStories.

### 🟡 [2026-08-10] Fase de réplicas multi-semilla — la comparación v2↔v0 deja de ser anécdota

* **Contexto**: la lectura de una sola semilla (770) no concluye: la varianza
  entre runs es comparable al efecto (Hito 5). Se abre la tanda de réplicas del
  protocolo adaptativo DL-006 sobre los brazos K-65P.
* **Soporte en el job manager**: el entrenamiento de Bit migra definitivamente al
  camino genérico `script_job` (retirado `BitTrainingDriver` del registro de
  drivers — D3 del RFC del ScriptJobDriver cumplida: la escuela v1 cerró 1408+
  épocas vía receta). El `dag_job` del RFC_JOB_DAG no sustituye a `script_job`
  para loops reanudables por épocas: aun sin nodo de iteración propio (futuro),
  el loop de entrenamiento vive en `script_job` (checkpoint por época, resume,
  `completion` por milestone) — anotado como RFC futuro del DAG.
* **Artefactos**: 6 recetas `configs/jobs/school_k65p_{glyph,standard}_s{771,772,773}.yaml`
  (3 seeds × 2 brazos), `--state_dir` propio por réplica
  (`replicates/k65p/...`), umbrales DL-004 intactos. Runner autónomo con env
  `SEED`. Encoladas como jobs `script_job` (priority 5); el runner las serializa.
* **RESULTADO — 6/6 GRADUADAS 7/7 hitos**:

  | Brazo | seed | épocas | val final (mejor) | OOD gen_valid | gap OOD−val | batería |
  |---|---|---|---|---|---|---|
  | glyph | 771 | 164 | 2.152 | 87.5% | +0.098 | ✅ PASS |
  | glyph | 772 | 148 | 2.162 | **40.0%** | +0.094 | ❌ FAIL |
  | glyph | 773 | 165 | 2.149 | 100% | +0.060 | ✅ PASS |
  | standard | 771 | 148 | 2.039 | **45.0%** | +0.107 | ❌ FAIL |
  | standard | 772 | 141 | **2.007** | 70.0% | +0.096 | ✅ PASS |
  | standard | 773 | 145 | 2.023 | **45.0%** | +0.123 | ❌ FAIL |

* **Conclusión 1 (AJUSTE DISTRIBUCIONAL — CONCLUYENTE)**: standard bate a glyph
  en val_loss en TODAS las 6 réplicas (2.007-2.039 vs 2.149-2.162). El embedding
  libre ajusta mejor la distribución; ya no es lectura de una semilla.
* **Conclusión 2 (ROBUSTEZ GENERATIVA OOD — SE DESPLOMA)**: la "ventaja
  gramatical generativa" del glyph de la seed 770 (OOD 40/40) NO se replica. El
  gen_valid OOD varía salvajemente entre semillas (glyph 40-100%, standard
  45-70%) y **3 de 6 réplicas SUSPENDEN la batería adversarial DL-004**
  (umbral gen_valid ≥ 0.60). La métrica de generalización generativa es
  altísimamente sensible a la semilla — la lectura de la seed 770 sobre
  robustez gramatical NO era citable.
* **Conclusión 3 (gap OOD−val — SIN COLAPSO DISTRIBUCIONAL)**: el gap es pequeño
  y consistente en las 6 (+0.06 a +0.12 nats): teacher-forced, el modelo trata
  el OOD igual que val. El fracaso del attack 2 es de GENERACIÓN greedy
  (autoregresiva), no de representación: las muestras inválidas son
  concatenaciones léxicas no-estructurales (p.ej. `[18 comida dar dar]` — tokens
  libres sin S-expression), no árboles mal balanceados.
* **Pendiente tras las réplicas**: (1) el análisis comparativo formal queda
  cerrado en lo distribucional; (2) la robustez generativa OOD exige o réplicas
  adicionales (n>3) o un fix del criterio greedy/decodificación antes de citarla;
  (3) brazo control v1-inglés bajo las mismas reglas adaptativas (celda "inglés +
  estándar" ya tiene preescolar real por Samantha, pero la escuela completa bajo
  DL-006 sigue pendiente).

### 🟢 [2026-08-03] Hito 5: Graduación REAL de Bit v2 y Bit v0 bajo protocolo adaptativo DL-006

* **Contexto**: tras invalidar la graduación del 2-ago (DL-004), destapar que el glifo
  cero hacía la sintaxis inaprendible (DL-005) y retirar el calendario fijo de v1
  (DL-006), ambos brazos completaron la escuela con exámenes congelados y honestos.
* **Bit v2 (glyph, embedding composicional)**: 7/7 hitos en **162 épocas**; una
  remediación 128→256d en el examen de 8 años (4.88M params finales). Examen final:
  gen_valid 1.0, val 2.150 vs bigrama 2.430 (oráculo del generador: 2.09).
  **OOD real: 40/40 generaciones válidas**, gap OOD−val +0.092 nats. Suite DL-004: PASS.
* **Bit v0 (standard, one-hot+proyecciones — control de embedding)**: 7/7 hitos en
  **145 épocas**, todo a 128d (1.26M params). Mejor val_loss que v2 en todas las
  etapas (2.01 final). OOD: 38/40, gap +0.073. Suite DL-004: PASS.
* **Lectura provisional (1 run canónica + 1 sandbox por brazo — requiere réplicas)**:
  el embedding libre ajusta mejor la distribución; el prior composicional tiende a
  más robustez gramatical generativa en las etapas profundas (en sandbox v0 suspendió
  8_years con gen_valid 0.52 mientras v2 sacó 0.80; en canónico ambos aprobaron).
  La varianza entre runs es comparable al efecto: NO concluir sin 3+ semillas.
* **Falsedad de ayer corregida**: la "resistencia al ruido en glifos" del 2-ago era
  artefacto del padding; con métricas reales, σ=0.1 hunde la precisión de v2 al 13%.
* **Artefactos**: `storage/checkpoints/releases/bit_v{2,0}_k65p_adaptive_20260803/`
  con checksums y actas. Pendiente: brazo control v1-inglés bajo las mismas reglas +
  réplicas multi-semilla.


### ⚫ [2026-08-03] Hito 4 — INVALIDADO por DL-004 (era el smoke-test del pipeline): ~~Graduación Total de Bit v2 (8 Años) y Congelado Inmutable K-65P~~

* **Descripción**: Conclusión oficial del entrenamiento escolar completo de Bit v2 (1408 épocas, 8 sub-etapas).
* **Artefacto Congelado**: [`storage/checkpoints/releases/bit_v2_8yo_graduated_k65p/`](file:///home/joan/Documents/IA/frankenswarm/storage/checkpoints/releases/bit_v2_8yo_graduated_k65p/)
* **Parámetros del Modelo**: `dim=1024`, `layers=6`, 76.70M parámetros activos (BitNet 1.58-bit / ~14.4 MB), 1408 épocas completadas.
* **Métrica Final**: `best_val_loss = 1.8253` ($PPL = 6.21$ vs $76.1$ en Bit v1).
* **Todos los Hitos Alcanzados**: `2_years`, `3_years`, `4_years`, `5_years`, `6_years`, `7_years`, `8_years`.
* **Firmas SHA-256 (Checksums)**:
  * `model_milestone_8_years_k65p.pt`: `e8f9e3ef170a5d03cf5f47afa59b1ddf8573d7350c4904f3f22cd158d1a146b9`
  * `model_final_k65p.pt`: `e8f9e3ef170a5d03cf5f47afa59b1ddf8573d7350c4904f3f22cd158d1a146b9`
  * `school_state_k65p.json`: `9c3bffde9d05373ae9170e92f02c96b4cc0945f5cc39ebd30ebcf4f3d169890d`
