# Decision Log — Frankenswarm

Registro de decisiones de calado: el problema, la decisión, la evidencia que la
respalda y dónde está el detalle. Una entrada por decisión, la más reciente arriba.
Las decisiones se numeran DL-NNN y no se reescriben: si una decisión se revierte,
se añade una entrada nueva que la referencia.

---

## DL-020 · 2026-09-03 — VERY gana posición combinatoria; definiciones en cadena con moléculas

**Motivo** (curación del operador, ejemplo `child: a young person / baby: a very young
child`): las definiciones en cadena — baby referencia child, child referencia person —
son la doctrina Goddard de *semantic molecules*: permitidas en NSM si cada molécula
funda en primos. La malla ya propagaba moléculas-cabeza (DL-018); ahora también
**argumentos de contenido** (`structured_explication.py`).

**El hueco que destapó el ejemplo**: `[very small baby]` era inválida — VERY (49) no
tenía posición combinatoria en la v0, así que las distinciones graduales (very young
vs young) no podían producir huellas distintas. **Fix (k65p validator, DL-020)**: VERY
se une a los operadores unarios — `[very [small baby]]` = "muy pequeño" (intensificador
sobre cláusula atributiva).

**Cadena demostrada**: person(primo SOMEONE) → child={someone,small} →
baby={someone,small,very} — la genealogía queda EN la huella (baby ⊃ child ⊃ person).
La guardia anti-ciclos + inyectividad protegen la cadena (A→B→A imposible).

## DL-019 · 2026-09-02 — Política de neurogénesis mínima: sin techo, crecimiento del 12.5%, arranque en 16-32

**Problema**: las últimas generaciones nunca ejercitaron la escalera de capacidad —
128d de inicio bastó siempre (standard 2-4 años sin neurogénesis). El techo por etapa
(`st_cfg["dim"]`) y la escalera predefinida (`ALL_DIMS`) hacen que la remediación sea
un salto grosero o nada.

**Decisión del operador**: para el entrenamiento K-65P v2 (Bit refundado) la capacidad
arranca conservadora (16-32d) y crece LO MÍNIMO necesario, sin dimensión máxima.

**La política** (`src/bitnet/training/neurogenesis_policy.py`):
1. **Δdim = max(8, dim // 8)** — crecimiento mínimo del 12.5% con suelo de 8:
   16→24→32→40→48→56→64→72→81→91→102→114→128→144→162→... No existe fórmula canónica
   en la literatura (lo cercano: progressive widening y saturación de rango efectivo);
   el ×2 actual es 8× más agresivo que este paso. La fórmula se VALIDARÁ después con
   participation-ratio de los hidden states (déficit medido, no supuesto).
2. **Sin techo**: no hay dim máxima. Guardia de cordura (default 4096): superarla
   pausa para el operador (contrato exit-codes), nunca cap silencioso.
3. **Punto de medición 65**: el glifo v2 tiene 65 trits; la escalera cruza el ancho de
   la malla entre 64 y 72 — si hay salto de capacidad al cruzar, es dato de la tesis
   (la malla sin compresión), no artefacto.
4. El mecanismo Net2WiderNet existente (preservación de función) se reutiliza tal cual:
   solo cambia la política de `next_dim`.

Arranque recomendado: **32d** (la malla de 65 trits comprimida 2× — la escalera se
ejercita sin estrangular el arranque); **16d** como variante agresiva (4× de compresión).

## DL-018 · 2026-09-02 — El mecanismo de compuestas: gramática unaria, la malla recursiva y el límite empírico de F

**1. Gramática (k65p/validator.py)**: las moléculas valen como cabezas UNARIAS —
`[lobo peligro]` (compuesta, "un tipo de lobo") y `[peligro lobo]` (predicación,
"lobo está en peligro") son la MISMA forma; la distinción compuesta/predicación es
semántica (posición), no sintáctica. En posición de átomo: `[[lobo peligro] ve carne]`
como argumento de un predicado. 117 tests verdes (1 actualizado a la nueva doctrina).

**2. La malla recursiva** (`structured_explication.py`): cuando una cláusula usa una
molécula como cabeza (`[danger predator]`), el glifo de esa molécula PROPAGA sus trits
(con polaridad) a la proyección. Los glifos ya posicionan los conceptos antes del
entrenamiento — la malla que describió el operador: compartir trits = proximidad
semántica a priori; el modelo no aprende *qué es* un lobo, aprende *cómo se usa*.

**3. F v1 (candidata) — unión con cabeza dominante**: la compuesta hereda la huella
entera de la cabeza (la categoría se hereda, Wierzbicka: "es una clase de X") y las
cláusulas propias aportan lo nuevo. Ley de conservación: los trits deben poder
descomponerse en cabeza + cláusulas propias.

**4. HALLAZGO EMPÍRICO — el límite de F puntual**: demostrado con el registro:
(a) `predator` = F(animal, danger) ✓; (b) `menace` (cabeza invertida) colisiona con
`predator` — unión conmutativa, inyectividad la rechaza ✓; (c) `still-animal`
(animal que no se mueve solo) colisiona con `animal` MISMA — la dominancia de cabeza
se comió el contraste distintivo (mover:−1). **Conclusión: cualquier F que resuelva
trit-a-trit pierde el orden y puede comerse el contenido nuevo.** El orden de la
compuesta no cabe en la huella puntual: o la curación evita los pares (inyectividad
como guardián, funciona hoy) o el plano estructural dim66 (DL-012) recuerda la
cabeza. La F final se calibra cuando el corpus de compuestas reales exija decidir.

**Registro v2 resultante**: cold-water, ice, animal, danger, predator, still-thing.

## DL-017 · 2026-09-02 — Tres leyes de la codificación: silencio, inyectividad y la doctrina de F

Del diálogo sobre la codificación de trits (65 primos refundados) se fijan tres leyes:

1. **El glifo todo-a-cero es SILENCIO** — el único vector sin información, ortogonal a
   todos los primos. Reservado como abstención/no-sé ("ningún primo aplica"): le da hogar
   semántico al canal de silencio pendiente (matriz 3×2). Uso inteligente: raro y explícito,
   solo en contextos curados, nunca relleno. No es un primo — es sintaxis.
2. **Ley de inyectividad** — no pueden existir dos conceptos con la misma huella de trits,
   porque la huella es lo ÚNICO que ve el modelo (las cláusulas viajan solo para nosotros).
   La ley no arregla definiciones: **detecta definiciones incompletas en el momento de
   crearlas**. Caso fundacional (operador): un `hielo` = `{agua, frío}` es rechazado porque
   falta su esencia — que se expresa con CONTRASTE: `{agua:+1, frío:+1, mover:−1}` frente
   a `agua fría` con `mover:+1`. **Los trits −1 hacen definición real.** La ley atrapa
   también el caso estructural: F con unión conmutativa hace colisionar compuestos con
   cabeza invertida — si el patrón se repite, F deberá recordar la cabeza (dim66).
3. **Doctrina de F** (composición de glifos) — documentada, pendiente de calibración:
   multiplicidad → primos de cantidad (`mucho`/`todo`: 林 = {árbol, mucho}, no árboles
   duplicados); orden interno → la explicación (el trit es huella comprimida, *todo glifo
   descompone*); estructura que no quepa en los 65 → plano estructural dim66 (DL-012).

Implementación: `src/bitnet/vocab/registry_v2.py` — registro de moléculas v2 con las tres
leyes aplicadas en creación.

## DL-016 · 2026-09-02 — Refundación del vocabulario K-65P desde los 65 primos: los golds entran en cuarentena

**Decisión del operador**: el vocabulario artesanal heredado (28 canonical + 54 drafted +
40 molecule del diccionario, marcados `LEGADO-2SEP` en la BD) está **contaminado por la
era de supervivencia** — son bolsas de rasgos sin idea fuente registrada (los "gold" del
`glyph_vocabulary.py` eran juicios planos del demo de la tribu, no oraciones). Se descartan
como norma de calibración. Cuarentena reversible, no borrado.

**La nueva doctrina del vocabulario** (orden de dependencia correcto):

1. **Capa 0 — los 65 primos**: glifos identidad (one-hot), valencias del validador, kinds.
   La única capa que existe sin idea fuente: los primos son indefinibles por definición.
2. **Capa 1+ — moléculas**: solo nacen de una **idea fuente en inglés** (curada por el
   operador) → cláusulas K-65P (interpretación; si la gramática no basta, operador nuevo,
   decisión doctrinal) → **glifo = proyección de las cláusulas** (consecuencia, no input).
3. **El gold viejo = solo referencia de auditoría**: donde discrepe de 1-2, se revisa el
   gold (documentado), nunca la idea.
4. Cuando se retome la jungla (tribu), usará los glifos que salgan del vocabulario
   refundado — la jungla se adapta a K-65P, no al revés.

**Lo que ya existe del programa estructurado y sobrevive al reset**: `structured_explication.py`
(cláusulas → perfil de roles + kind + glifo), el generador de vista legible (`render`),
el bucle explication→glifo. **Lo que cae**: la calibración contra los 28 gold como objetivo.
**Lo que sube de prioridad**: el mecanismo de **compuestas/clase-de** — con 65 primos y
nada más, toda molécula nueva es composición; es la pieza que da el 3^65.

## DL-015 · 2026-09-01 — Auditoría de la confrontación v4: infra ≠ suspenso, instrumento corregido y set v2

**Problema.** La auditoría del 1-sep destapó que las dos afirmaciones fuertes
de la confrontación estaban contaminadas: (a) el suspenso de 2_years del glyph
a 128d fue un crash de infraestructura (Samantha devolvió `None`; el `exit(1)`
genérico se contó como suspenso académico) → la neurogénesis 128→256 se
disparó sin examen real y el checkpoint 128d no sobrevivió; (b) la "curva
riesgo-cobertura" era un artefacto (la confianza nunca se calculaba; el sort
estable ordenaba vistas→no vistas). Además: `<unk>` por puntuación en 23/195
no-vistas de la batería, duplicados en el set congelado, el check in-gate de
los banks muerto, y 28/195 "no vistas" colisionando con los banks DL-013.

**Decisión (operador, 1-sep: "necesito que todo esté correcto").**
1. **Contrato de exit-codes** evaluador↔state_manager: `0` aprobado · `2`
   suspenso CALIFICADO · otro rc = infraestructura → reintento único y pausa
   del operador (`eval_infra_error`) sin contar suspenso ni neurogénesis.
   El trainer pausa (rc 78) en vez de remediar. Tests en
   `tests/test_exam_failure_policy.py` (8/8).
2. **Runner de batería corregido y re-tirado** sobre los 6 checkpoints:
   confianza real (softmax post-máscara), `tokenize()` en el contexto, dedupe,
   retención + procedencia persistidas, máscara del checkpoint evaluado.
   Números canónicos: los del CHANGELOG 09-01 (sección corregida en sitio).
3. **Set congelado v2** (`age4_v2.json`, 51/169): banks en el universo
   entrenado, sin duplicados, sin golds incorrectos. v1 sigue rigiendo la
   confrontación v4 en curso; v2 rige desde que los banks se cableen.
4. **Banks regenerados** con el check in-gate vivo (442→439).
5. **Re-examen del hueco**: receta `bit003_glyph_v41_2y_x1` (etapas 0-2,
   mismo protocolo/semilla que v4) para responder con examen real si el
   glyph pasa 2_years a 128d bajo ×1. Lanzamiento: decisión del operador
   (GPU ocupada por el standard v4).

**Doctrina nueva**: un fallo de infraestructura JAMÁS es una calificación; y
todo instrumento que se cite en prosa debe existir como artefacto persistido
y reproducible (la retención "30.8%" y el "12%" del set piloto solo vivían
en prosa — el 12% resultó no reproducible).

**Referencias.** CHANGELOG 09-01 (dos retractaciones) · `run_battery_v3.py` ·
`state_manager.py` · `evaluate_samantha_age.py` · `generate_battery_v3.py` ·
`generate_exam_banks.py` · log `dd078baf` 530-538 (el crash fundacional).

---

## DL-014 · 2026-09-01 — Doctrina "todo glifo descompone": el programa de descomposición NSM y el diccionario K-65P↔humano

**Problema.** La auditoría del DL-012-addendum (RFC-002) demostró que los
~19,600 glifos no-canónicos son interpolaciones Ridge de 26 anclas sin
estructura semántica (intra-grupo ≈ aleatorio; acciones BAJO el azar) — el
brazo glyph entrenaba con glifos vacíos. Y la confrontación v4 lo confirmó:
el estándar a 128d superó al glyph a 256d en los tres instrumentos.

**Decisión (operador, 1-sep).** Doctrina: **todo glifo descompone en los 65
primos — cada trit lleva significado (+1 afirma, −1 contrasta, 0 silencio);
no hay símbolos opacos**. Cualquier palabra humana converge en los 65 (tesis
NSM). Se construye el programa de descomposición:
1. **Codificador NSM** (`nsm_explication.py`): explicación → glifo de 65
   trits, calibrado 28/28 contra los canónicos (test de ida-vuelta RFC-002 §5).
2. **Diccionario K-65P↔humano** (`k65p_dictionary.py`, SQLite): words (glifo +
   explicación + estado canonical/molecule/prime/explicated/drafted/pending),
   surfaces (alias), phrases (compresión N→1: phrasal verbs y descripciones
   que colapsan en un concepto — el inglés torticero), relations.
3. **El generador de explicaciones NSM por lotes** con validación de
   consistencia (Jaccard esperado entre relacionadas) y curación del operador.
4. **dim66 reencuadrado**: las grafías (5, +, =) son la vía rápida de
   comunicación — los CONCEPTOS descomponen igual (precisión doctrinal).

**Estado.** Codificador calibrado, validador operativo (cazó 2
inconsistencias en el demo), diccionario poblado (19,703 entradas). La
generación por lotes del mundo núcleo es el siguiente paso.

**Referencias.** `src/bitnet/vocab/nsm_explication.py`, `src/bitnet/vocab/k65p_dictionary.py`, `src/bitnet/translation/en_lexicon.py`, `nsm_syntax_en.py`, `docs/RFC-002` addendum.

---

## DL-013 · 2026-08-31 — El BANK de examen por etapa: el corpus de preguntas curricular

**Problema.** El gate de hito era 10 preguntas (AGE_QUESTIONS) — pocas y
totalmente predecibles; el universo de hechos drilled era ~58 pares para la
edad 4. El material de examen necesitaba su propio corpus curricular.

**Decisión (operador, 31-ago).** Un **BANK de preguntas por etapa** (Q→A sobre
los hechos que cada etapa introduce), del que los exámenes de hito se
muestrean aleatoriamente con **peso por recencia** (50/20/15/5 entre etapas),
10 formas por hito con media±σ. Generador combinatorio: partición del mundo
curricular (animales/objetos/lugares NUEVOS por etapa, filtrados por el gate
de la etapa) × frames acumulativos, verificación in-vocab/in-gate/única entre
banks. Total actual: **442 preguntas** (16/42/37/63/68/77/88/51) — el tamaño
refleja el universo de hechos curado, ampliable por contenido.

**Nota de integridad**: el v4 glyph en curso termina con el gate actual; el
bank se cablea al entrenamiento y al examen en la siguiente generación de
runs (el contraste glyph-vs-standard ×1 no se toca a mitad de partido).

**Referencias.** `scripts/generate_exam_banks.py`, `configs/exam_banks/`.

---

## DL-012 · 2026-08-31 — Canal 66: planos de codificación de glifos (patrón UTF) — símbolos y constantes entran en K-65P sin tocar los 65 primos

**Problema.** Los 65 primos NSM no expresan numerales (`three…nineteen`; solo
`one`/`two` son primos 55/56), constantes ni notación matemática: la parte
aritmética del currículo y de la batería 80/50 no es traducible al brazo
K-65P — la confrontación "2 sabores" pierde las matemáticas. Además, los
glifos actuales de los numerales son composiciones Ridge sin semántica real.

**Decisión (operador, 31-ago).** Añadir un **canal de tipo** (dim 66, no un
primo 66) que actúa como **selector de plano de codificación** con la semántica
del patrón UTF-8: `0` = plano semántico NSM (coordenadas de primos — default
retrocompatible: todo glifo existente migra con 0 sin tocarlo), `+1` = plano
**simbólico** (numerales, constantes, notación), `-1` = plano **meta/estructural**
(rol asignable). En modo marcado los 65 dims son **espacio libre de código**
(como los bytes de continuación de UTF-8 no significan ASCII): 3^65 espacios
por plano. Identidad de símbolos v0: embedding aprendido por token id (los
numerales son convención, no descomposición); fase 2: magnitud vía primos de
cantidad. Formato glifo v2 = (V, 66) int8; migración = columna de ceros.

**Consecuencias.** Ruptura de checkpoints (nueva generación de modelos): los
v4 ×1 en curso terminan en 65d sin tocar. Desbloquea: aritmética y símbolos
en el brazo K-65P (batería completa), marcado honesto de convenciones (rote
vs comprensible visible en la representación), extensible a dims 67+.

**Estado.** Especificación aprobada e implementación pendiente (fase 1:
formato v2 + migrador, post-v4). Detalle y faseado en
`docs/RFC_DIM66_PLANES_CODIFICACION.md`.

**Referencias.** EXP_037 ("4º bit", precedente doctrinal) · RFC-002 §5 ·
`src/bitnet/translation/nsm_syntax_en.py` (gramática EN operativa).

---

## DL-011 · 2026-08-30 — Instrumento v3: separar avance de cognición — exámenes ×10, cobertura total del corpus gateado y batería 80/50

**Problema.** El gate de hitos era blando: las secuencias de examen se
sobremuestreaban ×300 (~9.000 exposiciones por etapa → aprobado por lookup de
pares exactos), el muestreo aleatorio del corpus cubría solo ~25-30% del pool
por etapa (coupon collector: "vio su nivel" era falso), y la medición de
cognición estaba fundida con la puerta de avance — cero neurogénesis en 209
épocas porque nada estresaba la capacidad.

**Decisión (operador, 30-ago).** Protocolo v3, aplicado desde cero en los tres
brazos BIT-003:
1. **×300 → ×10** (`--exam_repeat_factor`, default 10): fija el hecho sin
   memorización patológica; el gate más duro puede ejercitar la escalera de
   neurogénesis.
2. **Muestreo cíclico permutado** (`CyclicPoolSampler`): cobertura total del
   pool garantizada por construcción; el examen de hito solo puede cerrarse
   con cobertura completa → "graduó N años habiendo visto todo su corpus
   gateado al menos una vez" pasa a ser verdadero.
3. **Batería 80/50** (instrumento de evaluación v3, retroactiva a todos los
   checkpoints): 80 vistas = gate; 50 NO vistas (composiciones nuevas de hechos
   conocidos, verificadas ausentes por script) = **métrica primaria de
   cognición de la tesis** — ahí debe verse la ventaja composicional de los
   glifos. n=50 por IC95 (±13.9% a 50%).
4. **Ablación v2→v3**: el run v2 se ARCHIVA (`bit003_glyph_x300_v2`, hitos
   2-5 años) — la batería 80/50 sobre sus checkpoints cuantifica el coste de
   memorización del ×300. Figura de tesis adicional.

**Evidencia.** `CyclicPoolSampler`: cobertura 1000/1000 en test unitario.
Censo del store correcto (18,468 únicas) tras el fix DL-010 §7. Run v3 encolado
(`676b6931`) con `--exam_repeat_factor 10`.

**Referencias.** `docs/RFC_INSTRUMENT_V3.md` · commit DL-011 · el run v2
archivado es la ablación, no un descarte.

---

## DL-010 · 2026-08-30 — Brazo resonante: Bit entrena con bucle latente + emoción first_only; la resonancia pasa de apagada a variable experimental de la tesis

**Problema.** Bit se diseñó para pensar con resonancia (bucle latente cerrado,
`forward_resonance`), y los experimentos fundacionales demostraron que debe
activarse DESDE EL ENTRENAMIENTO (EXP_078: desde cero + BPTT en sueño supera a
pesos estáticos; el `resonance_clock` se ajusta óptimo desde la primera época).
Pero la escuela la entrenaba apagada (`max_resonance_steps=0`) en ambos trainers.
Incoherencia genealógica: K-65P nació de EXP_034 (glifos + resonancia + emoción
first_only: 100% @ época 12) y luego se entrenaba sin ella.

**Decisión (operador, 30-ago).** Activar el brazo resonante como variable
experimental, con la configuración ganadora validada: rampa de profundidad
`n_steps ~ U[1,5]` (eje B de EXP_032 — hace del "pensar más profundo" un knob
libre en inferencia), `pos_mode=clock` con `max_resonance_steps=5`, emoción
`first_only` (EXP_033/034: la resonancia SOLA es nula — B≈A) con el set del
dojo (6 emociones) + `neutral` (id 6) para val/exámenes. El gateo de etapa se
aplica idéntico al camino normal (fuera, `apply_stage_gate`). El evaluador se
acopla con los mismos flags (el checkpoint resonante exige instanciar
resonancia/emoción o el load falla). El brazo normal EN CURSO es el control;
nada se tira: los hitos 2-5 años (entrenados con máscaras correctas) se
conservan y la etapa 5 se reanudó desde el checkpoint del hito 5_years (época
223) tras descartar ~6 épocas contaminadas por el bug del censo del store (§7
del RFC, cazado por el smoke del propio brazo resonante).

**Instrumento.** Mismo instrumento v2 (DL-009) para ambos modos. El test
distintivo del brazo resonante: curva precisión-vs-n_steps en inferencia
(presupuesto de pensamiento) — imposible en un modelo de profundidad fija.

**Evidencia.** Test unitario: BPTT con gradiente ≠ 0 en clock + emociones;
rampa 1-5 válida; convergencia del bucle (coseno 0.19→0.95). Smoke GPU: época
resonante completa con gateo correcto (post-fix). Evaluador: carga resonante +
máscara E1 + examen sin fricción. Modelo resonante: 1,250,680 params (+2,800
del clock y emociones).

**Adoptado.** Flags `--resonance_*`/`--n_emotions`/`--emotion_*` en trainer y
evaluador; receta `configs/jobs/bit003_glyph_resonant.yaml` (en serie tras los
controles). Detalle en `docs/RFC_RESONANCE_ARMS.md`.

**Referencias.** `docs/LAB_NOTEBOOK.md` (EXP_032/033/034/078),
`docs/ideas_ssm_ternario_y_recurrencia.md` (el SSM es ortogonal: hipocampo
inter-turno, pre-registrado en plan v4 §4; el hook `h_prev` del bucle queda
como puerta futura), `docs/RFC_RESONANCE_ARMS.md`.

---

## DL-009 · 2026-08-21 — Instrumento v2 y alineación evaluador↔gateo (remediación pre-lanzamiento BIT-003)

**Problema.** La auditoría pre-lanzamiento del 21-ago (`.red-pill/memory/BIT-003_audit_findings.md`) encontró que
(a) el evaluador, al repuntarse a `childes_pre_school_en.json`, pasó a permitir TODO el vocabulario de CHILDES sin
corte (18.364/20.095 palabras a edad 2, vs ~800 del gateo de entrenamiento) — un cambio de instrumento de facto sin
DL; (b) los exámenes (29-jul, pre-rebuild) esperaban respuestas fuera del vocabulario nuevo (aleth, bunker, madrid);
(c) el trainer no implementaba la semántica de DL-008 (censo por ids que no desbloqueaba el top-N, NameError de
arranque, device mismatch CPU/CUDA en la pérdida, E7 con `<unk>`, 'xxx' —marcador CHAT con 50.894 apariciones—
producible desde E0, contracciones partidas don't/dont, 7/28 glifos de referencia sin el canónico).

**Decisión (operador, 21-ago).**
1. **Instrumento v2**: exámenes y `AGE_QUESTIONS` reescritos SOLO con palabras del vocabulario (sin nombres propios
   de lore: el control v1-inglés mide adquisición de lenguaje, no memorización de tokens sin presencia en corpus) y
   ampliados a **10 preguntas por edad**. Toda respuesta de examen entra al gateo de su etapa (`exam_answer_words_for_age`).
2. **Evaluador alineado**: el trainer persiste `stage_gate_masks.json` y Samantha restringe la generación (y el
   monitor OOB) a la máscara REAL de la etapa, con fallback legado para checkpoints antiguos. `<unk>` vetado también
   en evaluación.
3. **Pipeline de corpus**: apóstrofos normalizados en la tokenización canónica (`words_of`: don't→dont, como el
   corpus CHILDES-en upstream), artefactos CHAT (xxx/yyy/www) eliminados del corpus, censo y glifos regenerados con
   preservación canónica EN (28/28). Vocabulario final: **19.637** palabras, 19.637 glifos únicos.
4. Currículo preescolar rico: 32 items (28 nuevos deterministas), 4 bins MLU poblados [10,8,8,6].
5. `samples_per_epoch=400.000` por defecto (análisis Chinchilla 21-ago, confirmado).

**Instrumento.** Este DL VERSIONA el instrumento de evaluación de v1-inglés (sucesor del congelado en DL-004, que
sigue intacto para los runs v1-ES históricos). Los dos brazos BIT-003 (glyph/standard) usan este instrumento v2
idéntico; la comparativa entre brazos no se ve afectada.

**Evidencia.** `scripts/audit_stage_gating.py` (22 checks, rc=0). Gateo por etapa: `[254, 825, 3008, 5003, 8005,
10003, 15002, 19636]`. Clasificación CHILDES: `[11.9, 31.1, 34.9, 8.9, 5.5, 1.8, 2.4, 3.5]%` (78% en E0-E2).
TinyStories(100k): `[3.4, 11.7, 40.4, 15.1, 12.9, 4.1, 5.2, 7.1]%` — el E7 baja del 16% al 7.1% al curar la
partición de contracciones. Instrumentos 100% in-vocab (`scripts/validate_exam_vocab.py`).

**Referencias.** `.red-pill/memory/BIT-003_audit_findings.md`, `.red-pill/memory/BIT-003_fix_plan.md`,
`docs/RFC_GATING_VOCABULARIO_ETAPAS.md`, DL-008 (semántica del gateo), DL-004 (instrumento v1-ES congelado).

---

## DL-008 · 2026-08-21 — Gateo de vocabulario por etapa (BIT-003): el corpus y la producción de cada etapa se restringen al vocabulario de su edad, con equivalencia estricta entre el brazo estándar y el K-65P

**Problema.** En v1 (BIT-003, corpus EN desde cero) el entrenamiento pasaba
pérdida sobre las 20.095 palabras del vocabulario desde la etapa 0, mientras que
la evaluación de cada edad solo dejaba emitir 729-851 palabras (gateo de
`get_allowed_vocab_for_age`). Doble desalineación: (a) Bit invertía gradiente en
palabras que jamás podría producir en su etapa; (b) el gateo de evaluación apuntaba
a `childes_pre_school.json` (el fichero **ES** ya inexistente) y a `special_tokens`
en español — no reflejaba el corpus EN real. En k65p el gateo SÍ estaba en
entrenamiento (`logit_mask` por tiers); en frankenswarm no existía activo.

**Decisión.** Aplicar gateo de vocabulario por etapa en el entrenamiento de ambos
brazos, con **equivalencia estricta** entre el brazo estándar (`--embedding standard`)
y el K-65P (`--embedding glyph`):

1. **Producción gateada por etapa** (`logit_mask`): cada etapa solo puede *producir*
   el vocabulario de su edad — primos EN (28 referencias) + safe words + **top-N de
   CHILDES-en por frecuencia de adquisición** + palabras del currículo de la etapa.
   Cortes por etapa: `[200, 800, 3000, 5000, 8000, 10000, 15000, 20095]` (E7 = todo el
   vocabulario). La pérdida excluye posiciones cuyo target está vetado (evita
   `inf×0=NaN` con `masked_fill`).
2. **Equivalencia entre brazos**: la máscara es una máscara sobre *tokens*,
   independiente del tipo de embedding → ambos brazos ven y producen **exactamente
   las mismas palabras en cada etapa**. La única diferencia que mide la comparativa
   es la representación (composición de 65 primos vs tabla one-hot congelada +
   proyección entrenable).
3. **Clasificación del corpus por etapa (umbral 95%)**: cada frase del corpus general
   se asigna a la **etapa mínima** cuyo vocabulario cubre ≥95% de sus tokens. El
   corpus de la etapa `i` = frases acumuladas `E0..Ei`. Resultado medido: CHILDES-en
   cae un 78% en E0-E2 (habla real infantil temprano); TinyStories se incorpora según
   su vocabulario (solo 14% en E0-E1, repartida en E2-E7). Ninguna frase se pierde
   (E7 cubre el 100%).
4. **Propiedad de la tesis**: el gateo se puede **ampliar en caliente** con K-65P —
   una palabra nueva se compone de primos ya aprendidos, basta desbloquear su token
   en la máscara. Con el estándar (columna de embedding no entrenada) no es posible
   sin reentrenar. La comparativa demostrará esta propiedad en evaluación.

**Evidencia.** Cobertura del vocabulario por CHILDES-en: 99.53% (0.47% `<unk>`).
Distribución del corpus por etapa (umbral 95%): CHILDES `[11,32,35,9,5,2,2,4]%`,
TinyStories `[3,11,35,16,12,3,6,15]%`. Gateo resultante: `[242, 822, 3007, 5004,
8007, 10005, 15002, 20095]` palabras producibles por etapa. Los 65 primos (y las 28
palabras EN que representan) entran desde E0.

**Adoptado.** `src/bitnet/training/modules/stage_gating.py` (máscaras + clasificación)
aplicado en entrenamiento y validación (adaptativo + clásico). Detalle completo en
`docs/RFC_GATING_VOCABULARIO_ETAPAS.md`.

**Salvaguardas.** E7 = todo el vocabulario (toda frase clasificable, nada se pierde).
`<unk>` vetado en todas las etapas (ruido de generación). El currículo mantiene su
asignación por MLU. Evaluador actualizado a EN (`childes_pre_school_en.json`,
`special_tokens` EN).

**Referencias.** Commits 081fc8d, 2741e95 · `docs/RFC_GATING_VOCABULARIO_ETAPAS.md`.

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
