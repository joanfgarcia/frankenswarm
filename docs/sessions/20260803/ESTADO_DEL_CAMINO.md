# Estado del Camino — Track Bit v2 / K-65P (2026-08-03, noche)

> Mapa de situación tras la jornada de auditoría y reconstrucción. Complementa
> (no sustituye) las decisiones formales: **DL-004** (invalidación de la
> graduación falsa del 2-ago e instrumentos), **DL-005** (glifo cero: sintaxis
> inaprendible, firmas ternarias), **DL-006** (protocolo adaptativo + matriz de
> brazos) en `docs/DECISION_LOG.md`, y el Hito 5 de
> `Aleth_Core/BITACORA_BIT_V2.md`.

## El mapa en una línea

```
[1 Instrumentos] ── [2 Gramática] ── [3 Semántica] ── [4 Tesis]
    ✅ 3-ago            ✅ 3-ago        ⬜ siguiente     ⬜ después
```

Esta mañana el operador creía estar en la etapa 4 (graduación del 2-ago).
La auditoría demostró que aquello era decorado; a cierre del día las etapas
1 y 2 están completadas **de verdad** y el camino restante está dibujado.

---

## Etapa 1 — Instrumentos honestos ✅ (DL-004/005/006)

Lo que existe y funciona, verificado hoy en producción:

- **Exámenes de hito que pueden suspender** (y suspendieron: run 1, run 2, y
  el 8_years de v2 en canónico): `gen_valid ≥ 0.60` sobre la SALIDA generada
  + `val_loss ≤ bigrama − 0.10` (umbrales congelados; alcanzabilidad
  verificada contra el oráculo exacto del generador: 2.09 nats/token).
- **Protocolo adaptativo**: etapas hasta plateau, examen sobre el MEJOR
  checkpoint, avance desde best, neurogénesis SOLO como remediación de
  suspenso (techo de dim por etapa), pausa rc=78 sin techo. La edad se mide
  en hitos superados, no en épocas.
- **Suite adversarial** sobre holdout OOD real (pares cabeza-argumento nunca
  vistos), baselines triviales (uniforme, bigrama), veredicto condicional.
- **Corpus composicional válido por construcción** (10.800 exprs + 105 OOD,
  dedup global, tiers alineados con máscaras) — `generate_k65p_corpus.py`.
- Cuarentena de todo el material del 2-ago con sus actas
  (`storage/checkpoints/quarantine/`).

## Etapa 2 — Gramática K-65P ✅ (graduaciones reales del 3-ago)

Matriz 2×2 {embedding} × {lenguaje} — resultados a cierre del día:

| Brazo | Resultado | Épocas | Dim final | OOD gen_valid |
|---|---|---|---|---|
| K-65P + glifos (v2) | 🎓 7/7 hitos | 162 | 256d (1 remediación) | **40/40 = 100%** |
| K-65P + estándar | 🎓 7/7 hitos | 145 | 128d | 38/40 = 95% |
| Inglés + glifos (control) | ✅ preescolar completo (2,3,4 años con Samantha real), corte planificado | 309 | 128d | n/a |
| Inglés + estándar | 🔄 en marcha (2_years aprobado; corte automático en 4_years) | 112+ | 128d | n/a |

Releases sellados con checksums: `releases/bit_v{2,0}_k65p_adaptive_20260803/`.

**Hallazgos provisionales (1 semilla — NO citar como concluyentes):**
- El **estándar ajusta mejor la distribución** en los 4 brazos (p.ej. K-65P:
  2.01 vs 2.15 final; inglés guardería: 2.05 vs 3.03) y converge en menos
  épocas. Coherente con el techo de rango ≤65 del embedding de glifos.
- El **composicional muestra tendencia a más robustez gramatical generativa**
  en composición profunda (8_years sandbox: 0.80 vs 0.52; OOD: 100% vs 95%),
  y su decode es computacionalmente barato con vocabularios grandes (inglés:
  ~15 s/época vs ~70 s — decode en 65 dims vs capa de salida contra 6.400).
- El alumno inglés adaptativo certificó TODO el preescolar a 128d sin una
  sola neurogénesis y con un 31% menos de épocas que el calendario fijo
  (309 vs 448): gran parte del crecimiento del v1 original parece artefacto
  del calendario, no necesidad del alumno.
- Varianza entre runs ≈ tamaño de los efectos comparativos → **réplicas
  multi-semilla obligatorias antes de concluir**.

**Límite de lo certificado**: la graduación K-65P acredita FORMA (gramática,
valencias, anidamiento), no significado. El corpus es válido-pero-aleatorio:
nadie le ha dicho nunca nada *verdadero* al alumno. Un examen tipo Samantha
no está definido en su mundo todavía.

## Etapa 3 — Escuela semántica ⬜ (diseñada sobre papel hoy; no construida)

El hueco que esta mañana no estaba en el mapa de nadie. Diseño acordado en
sesión (pendiente de RFC formal):

1. **Corpus con verdad**: expresiones consistentes con un micro-mundo causal
   (28 moléculas + reglas NSM: fuego→daño, agua→calma, depredador→peligro).
   El generador pasa de producir "válido" a "válido y verdadero". Escuela
   contrastiva posible con el primo 44 (NOT) como señal de falsedad.
2. **Juez incorruptible**: el puente K-65P→Prolog ya existe (`to_prolog`,
   KB compilable). El examen semántico = demostración de teoremas: la
   continuación generada ¿se sigue de la KB? Métrica: **gen_true** (sucesora
   de gen_valid). A diferencia de Samantha, no opina: verifica.
3. **Memoria vs razonamiento, separables por fin**: exámenes con teoremas
   held-out (consecuencias derivables jamás vistas escritas) → recall e
   inferencia como dos números distintos. Con lenguaje natural esto es
   imposible; es la ventaja metodológica central del lenguaje formal.
4. Aquí es donde el currículo gana profundidad lógica → el dolor que dará
   motivo de existir a **Net2DeeperNet** (siempre sobre copia, control G4
   congelado, confound pre-registrado — doctrina del roadmap intacta).

## Etapa 4 — La tesis ⬜ (ni probada ni refutada; por fin medible)

Pregunta en su forma medible: *¿el alumno del reglamento (K-65P nativo)
razona mejor que el alumno de la biblioteca (inglés) recuerda?* Comparación
SOLO con métricas normalizadas: épocas-hasta-hito bajo las mismas reglas,
params al certificar, margen sobre el baseline propio, tasas held-out.
Nunca cross-entropy contra cross-entropy (lección del 2-ago).

---

## Cola de trabajo (orden acordado)

1. **Cerrar la matriz** (esta noche, automático): corte del brazo
   inglés+estándar en 4_years → informe comparativo DL-006.
2. **Réplicas multi-semilla** K-65P (≈15 min/tanda): convertir tendencias en
   datos o enterrarlas.
3. **Examen M5 — palabra nueva**: inyectar molécula inédita vía
   `register_new_word` y medir uso inmediato. Es EL examen que discrimina
   composicional vs estándar (el estándar no puede por construcción). La
   "adición en caliente" es hoy capacidad arquitectónica, NO propiedad
   demostrada.
4. **Sistema inmune**: re-certificación automática de todos los hitos tras
   cada evento de crecimiento o vocabulario (con K-65P cuesta segundos).
   Prerequisito del "modelo vivo" antes que cualquier N2DN.
5. **Escuela semántica** (etapa 3) → RFC + fábrica de corpus causal + gen_true.
6. **Net2DeeperNet sobre copia** con test de preservación funcional bajo
   cuantización ternaria, cuando el currículo profundo lo pida con datos.

## Estado git / artefactos (3-ago noche)

- Rama `ci/fix-re-import-and-compilation-tests`, 5 commits locales sin push:
  `8e0c18a` (DL-004) · `511634e` (DL-005) · `89419c5` (DL-006) · `125fea3`
  (--adaptive en v1) · `3ac00b9` (--embedding standard en v1).
- Pendiente decisión del operador: scripts del pipeline de traducción
  deprecado (sin trackear) — ¿rescatar o borrar?
- Aviso de sistema: driver NVIDIA actualizado sin reboot (`nvidia-smi` con
  mismatch); no reiniciar procesos GPU hasta reiniciar la máquina.
