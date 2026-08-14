# RFC-GROWTH-V6 — Profundidad y anchura después de DL-006: la rejilla sustituye a la apuesta

> **Estado**: 🟡 PROPUESTA — pendiente de ratificación del operador. Nada de esto se
> ejecuta hasta que Joan lo apruebe.
> **Fecha**: 2026-08-14 · **Autor**: Aleth · **Rama**: `feat/v1-english-rebuild`
> **Sustituye a**: `docs/bitnet_next_architecture_plan.md` §5 (Net2DeeperNet, v5,
> 30 jul 2026), cuyo diagnóstico caducó — ver §1.
> **Preserva intacta**: la doctrina soberana de §5 (solo sobre copia, control G4
> congelado, confound pre-registrado) y el Arsenal de §6.

---

## 1. Por qué este RFC existe: la premisa caducó

El plan v5 aprobó Net2DeeperNet con este diagnóstico: *"con width en 1024 y 6 capas,
la ratio W/D=170 está fuera de toda configuración BitNet conocida"*. La nota de
literatura que lo respalda se escribió el **30 de julio**, con el modelo en "época
1269, hidden_dim=1024, 6 capas, 76,9M params".

**DL-006 llegó cuatro días después y declaró ese 1024 artefacto del calendario**, no
necesidad del alumno. Bajo protocolo adaptativo: el brazo glyph K-65P se graduó a
256d, el v0 a 128d, y los dos brazos ingleses de preescolar se quedaron en 128d con
**cero neurogénesis**. La profundidad, además, nunca se mueve: `num_layers=6` es fijo
y `net2wider_model` solo toca la anchura.

Aplicando la propia fórmula de la nota (`D_crit ≈ W^0.44`, de "Depth Delusion",
arXiv 2601.20994) a las anchuras que el protocolo produce **de verdad**:

| Anchura real | W/D con 6 capas | D_crit ≈ W^0,44 | 8 capas estarían… |
|---|---|---|---|
| **128d** (brazos ingleses, v0 K-65P) | **21** | **≈ 8,5** | **justo en el techo** |
| **256d** (glyph K-65P graduado) | 43 | ≈ 11,5 | con margen |
| 1024d (el v1 del calendario, ya inexistente) | 170 | ≈ 21 | muy lejos del techo |

Los BitNet publicados van de W/D 64 (125M) a 149 (30B). A 128d el modelo está en
**21**: proporcionalmente **más profundo que cualquier BitNet publicado**, el problema
exactamente opuesto al que motivó el plan.

No se afirma que `D_crit` sea verdad revelada — la propia nota advierte que es un
estudio FP16 y "los resultados pueden no transferir" al ternario. La objeción es
lógica, no empírica: **la fórmula que se usó para justificar el plan, aplicada al
modelo actual, lo desaconseja**. La justificación se cae en cualquiera de los dos
sentidos, y por tanto hay que rehacer el diagnóstico antes de escribir código.

Segundo desajuste, del mismo origen: la hipótesis experimental de la nota (§4.2) dice
*"si entrenamos primero width hasta el plateau (hecho: 1024 dim, val_loss=4,40
estancada), y luego expandimos depth…"*. Bajo DL-006 ese plateau de anchura no
existe: el plateau a 128d es la señal para **examinar y avanzar**, no para crecer. El
proyecto se ha mudado sin darse cuenta a la fila de su propia tabla marcada como
*"Depth → Width: no tenemos datos"*.

---

## 2. La oportunidad que abre DL-006

El protocolo adaptativo dejó los modelos en **1,25M-4,35M parámetros**: unas 60×
menos que el v1 del calendario (76,9M). Eso cambia la naturaleza de la pregunta.

Petty (NAACL 2024) contestó "¿profundidad o anchura?" intercambiando D↔W a
**parámetros constantes** con modelos de 41M-374M. A 1,2M eso mismo cuesta horas.
La pregunta pasa de ser **una apuesta de una sola ventana de GPU** (N2DN: ~18 GPU-h
para un único punto del espacio, sobre un modelo vivo) a ser **una rejilla barata y
repetible desde cero**.

Y hay una segunda pieza ya pagada: **`forward_resonance` ya está implementado**. Es
un bucle latente cerrado — el hidden state itera N veces por el core sin salir a
vocabulario — con variante de entrenamiento por BPTT (`forward_resonance_training`) y
un reloj posicional por paso (`resonance_clock`). Eso **es** un looped transformer, y
la propia nota de literatura cita a Saunshi (NeurIPS 2025): *12 capas en bucle 2×
superan a 24 capas con la mitad de parámetros*. La escuela construye el modelo con
`max_resonance_steps` por defecto (0) y llama al `forward` plano: **la resonancia
está desconectada en toda la escuela**.

Coste comparado de ganar profundidad efectiva 12 sobre un core de 6 capas a 128d:

| Vía | Parámetros nuevos | Código nuevo | Toca el control G4 |
|---|---|---|---|
| **Resonancia 6×2** | **384** (`resonance_clock`), o **0** con `pos_mode=none` | cableado de 2 flags | no |
| Net2DeeperNet 6→8 | ~394.000 (+32%) | operador nuevo + mina ternaria | solo sobre copia |

---

## 3. Plan: tres experimentos, en este orden

### E1 · Encender la resonancia en la escuela (el más barato primero)

**Hipótesis.** Profundidad efectiva por iteración del core (bucle) rinde como
profundidad por capas nuevas, a coste de parámetros ≈ 0 (Saunshi).

**Qué hay que hacer.**
1. Flags nuevos en `train_sovereign_school.py`: `--resonance_steps N` (default 0 =
   comportamiento actual, intacto) y `--resonance_pos {none,clock}`.
2. Pasar `max_resonance_steps` al constructor y usar `forward_resonance_training` en
   el bucle de entrenamiento cuando `N>0` (mantiene gradientes por BPTT).
3. Telemetría: ∇STE por época **por paso de resonancia** — el detector de gradiente
   muerto a través del bucle, que es el riesgo real aquí.

**Brazos.** K-65P glyph y standard, `n_steps ∈ {1, 2, 3}` (1 = control exacto),
seed 770 + 2 réplicas, protocolo adaptativo DL-006, `--wd_mode` fijo.

**Coste estimado.** El bucle multiplica el coste del core por `n_steps`; a 128d con
el corpus K-65P la época va en segundos. Estimación: **~3-4 GPU-h** para la matriz
completa (2 brazos × 3 valores × 3 semillas).

**Criterio de éxito (pre-registrado).** `n_steps=2` mejora `gen_valid` OOD del hito
más profundo en ≥0,05 sobre `n_steps=1` a igualdad de parámetros, o reduce
épocas-hasta-graduación ≥15%. **Fracaso**: sin diferencia en las tres semillas, o
∇STE cae >1 orden de magnitud entre el paso 1 y el paso N (bucle inestable en
ternario) → se documenta el negativo y no se vuelve.

**Riesgo.** BPTT a través de pesos ternarios con STE puede colapsar el gradiente.
Mitigación: la telemetría de arriba, y que `n_steps=1` sea idéntico al camino actual
(fijar con un test de equivalencia, como se hizo en DL-007).

### E2 · Rejilla profundidad×anchura a parámetros constantes (el que decide)

**Hipótesis.** Existe un punto óptimo D/W para transformers **ternarios** a esta
escala, y no tiene por qué coincidir con el de FP16. Este experimento llena el hueco
que la propia nota de literatura declara abierto (*"este estudio es sobre FP16, no
ternario"*).

**Qué hay que hacer.** Un flag: `--num_layers` (hoy está **hardcodeado a 6**, sin
CLI). Nada más — `BitNet4LayerModel` ya acepta `num_layers` y `hidden_dim`
arbitrarios.

**La rejilla.** Params del core ≈ `12·D·W²`. Cinco puntos dentro de ±6% del baseline
(1,18M), barriendo W/D de 2,7 a 112 — o sea bracketeando el rango BitNet publicado
(64-149) y el 170 del v1 difunto:

| W | D | core params | W/D | D_crit≈W^0,44 | D vs D_crit |
|---|---|---|---|---|---|
| 64 | 24 | 1.179.648 (=) | 2,7 | 6,2 | **3,9× por encima** |
| 96 | 11 | 1.216.512 (+3%) | 8,7 | 7,6 | 1,45× por encima |
| **128** | **6** | **1.179.648** | **21** | **8,5** | 0,7× (baseline) |
| 176 | 3 | 1.115.136 (−5%) | 59 | 9,7 | 0,3× |
| 224 | 2 | 1.204.224 (+2%) | 112 | 10,9 | 0,2× |

El punto (64, 24) es el que pone a prueba "Depth Delusion" en ternario: está 3,9×
por encima de su `D_crit`, así que **si la fórmula transfiere, debe ser el peor de
los cinco**. Si no lo es, el ternario no obedece la ley FP16 y eso es un resultado
publicable por sí solo.

**Dónde correrla.** **En K-65P primero**, no en inglés: es 4× más barata y su
instrumento discrimina de verdad (`gen_valid`, OOD held-out de pares nunca vistos,
línea base de bigrama como oráculo), mientras el examen inglés de Samantha satura a
10/10 en los dos brazos. El ganador se confirma después en inglés.

**Coste estimado.** K-65P: ~1,5 s/época a 128d → ~5 min por graduación de ~200
épocas → 5 puntos × 2 brazos × 3 semillas = 30 runs ≈ **3 GPU-h**. Confirmación en
inglés del ganador + baseline: ~27 min por run → 12 runs ≈ **5-6 GPU-h**.

> **Comparación que justifica este RFC**: la rejilla completa (E1 + E2 ≈ 12 GPU-h)
> cuesta **menos que el único experimento N2DN que sustituye** (~18 GPU-h estimadas
> en v5), no toca ningún modelo vivo, y responde la pregunta general en lugar de un
> solo punto del espacio.

**Criterio de éxito (pre-registrado).** Se declara ganador el punto con menor
val_loss en el hito más profundo **y** mayor `gen_valid` OOD, exigiendo consistencia
en las 3 semillas. Si el ranking cambia de orden entre semillas, se declara
**indistinguible** y se conserva el baseline (128, 6) por parsimonia — nada de leer
posos de café en una semilla, que ya nos costó DL-004.

### E3 · Net2DeeperNet, solo si E1 y E2 lo piden

**Gate de entrada.** Se implementa **solo si** se cumplen las tres:
1. E2 muestra que la profundidad gana a parámetros constantes (si gana la anchura,
   N2DN queda archivado en el Arsenal y se crece con `net2wider`, que ya existe).
2. E1 no ha conseguido esa profundidad gratis por resonancia (si la consigue, N2DN
   es un gasto de parámetros para lo mismo).
3. La anchura real de graduación deja margen hasta `D_crit` — recalculado con esa
   anchura, no con el 1024 difunto.
4. Existe el **sistema inmune de re-certificación** (prerequisito ya escrito en
   `ESTADO_DEL_CAMINO.md`: re-certificar todos los hitos tras cada evento de
   crecimiento).

**Qué se conserva de v5 sin tocar una coma.** La spec de inicialización (solo
proyecciones de salida a 0 — O de atención y down-proj del MLP; Q/K/V y up-proj con
init normal), el aviso de la **mina ternaria** (absmean de tensor nulo → escala
0/división por cero; capas nuevas en FP16 con ternarización tras warmup, o
epsilon/clamp), y sobre todo la **restricción soberana**: opera exclusivamente sobre
una **copia**, porque el checkpoint de graduación es el control de G4 —
*"the control, not a casualty"*. Eso no se negocia.

---

## 4. Qué se retira y qué queda en la recámara

- **Se retira**: el diagnóstico de v5 §5 ("W/D=170 fuera de toda configuración") y su
  hipótesis width→depth. Ambos describen un modelo que el protocolo adaptativo ya no
  produce. La ficha del plan pasa de "✅ aprobada" a "⚠️ premisa caducada", con
  puntero a este RFC — para que nadie lo implemente en seis meses leyendo un visto
  bueno que ya no significa lo que decía.
- **No se retira**: nada del Arsenal (§6). R1 (depth progresivo) queda gateado tras
  E2; R5 (SENN η score, que decide *cuándo/dónde/qué* expandir) sube de interés
  porque una rejilla medida es exactamente el dato que necesita para calibrarse.
- **No se toca**: la doctrina de copia, el confound de crecimiento pre-registrado, y
  la autoridad de secuenciación del ROADMAP k65p.

---

## 5. Cómo encaja con lo demás en la cola

Este RFC **no compite** con `[BIT-003]` (reentrenamiento de v1 en inglés) ni con la
escuela semántica: E1 y E2 corren en K-65P, que es el track barato, y su resultado
es precisamente lo que hace defendible la configuración del run largo de BIT-003.
Orden recomendado:

1. Instrumentos primero (escuela semántica `gen_true`, M5 con contraste semántico,
   degradación bajo cuantización) — son las medidas que deciden la tesis viva.
2. **E1 y E2** (este RFC, ~12 GPU-h en K-65P) — fijan D/W antes de gastar el run largo.
3. BIT-003 con la geometría ya elegida y el baseline arreglado (DL-007).
4. E3 solo si sobrevive a su gate.

Correr BIT-003 antes de E2 significa comprometer 300-450 GPU-h a la geometría
(128d, 6 capas) que nadie ha comparado con ninguna otra, elegida en su día por un
calendario que DL-006 invalidó.

---

## 6. Riesgos de este plan

- **`D_crit` puede no transferir al ternario.** Es la premisa que E2 pone a prueba,
  no una que E2 asuma: por eso la rejilla incluye el punto extremo (64, 24).
- **La resonancia puede ser inestable en ternario.** E1 lo mide y lo documenta como
  negativo si lo es; el coste de averiguarlo son horas, no días.
- **La rejilla puede salir plana** (los cinco puntos indistinguibles). Sería un
  resultado útil: significaría que a esta escala la geometría no manda, y entonces
  la elección se hace por coste de inferencia y por parsimonia, no por fe.
- **Sesgo de instrumento**: si `gen_valid` OOD satura como saturó Samantha, la
  rejilla no discriminará. Mitigación: correrla **después** de la escuela semántica,
  que es el instrumento con más poder discriminante en construcción.

---

## 7. Decisiones que necesito del operador

1. ¿Se ratifica el orden E1 → E2 → (gate) → E3, con N2DN despriorizado hasta su gate?
2. ¿La rejilla de E2 se corre en K-65P (recomendado, 4× más barato e instrumento que
   discrimina) o exiges inglés desde el principio?
3. ¿`--wd_mode` para estos experimentos: `uniform` (comparable con las réplicas
   DL-006) o `no_embed` (simétrico entre brazos, DL-007)? Recomiendo fijar
   **`uniform`** dentro de la rejilla para que sea comparable con lo ya medido, y
   tratar el trato de weight decay como su propio A/B aparte.
4. ¿Se marca la ficha de §5 del plan v5 como premisa caducada? (Yo ya he dejado el
   aviso; retirarlo es un `git revert` de una línea si no estás de acuerdo.)

**Referencias.** `docs/NOTE_DEPTH_VS_WIDTH_LITERATURE.md` (§1.2 Saunshi, §1.5 Depth
Delusion, §3.1 configs BitNet, §4 síntesis) · `docs/bitnet_next_architecture_plan.md`
§5-§6 · `docs/sessions/20260803/ESTADO_DEL_CAMINO.md` (secuenciación, sistema inmune)
· DL-006 (protocolo adaptativo) · DL-007 (baseline arreglado) ·
`src/bitnet/model/modeling_bitnet.py` (`forward_resonance`, `forward_resonance_training`).
