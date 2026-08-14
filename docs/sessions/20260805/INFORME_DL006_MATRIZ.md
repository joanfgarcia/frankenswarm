# Informe DL-006 — Matriz 2×2 cerrada: {glifos, estándar} × {K-65P, inglés}

**Fecha**: 2026-08-05 · **Protocolo**: adaptativo DL-006 (idéntico en los 4 brazos:
plateau → examen sobre mejor checkpoint → avance; suspenso → neurogénesis como
remediación) · **Semilla**: 770 en todos · **Instrumentos**: DL-004 congelados.

> 🔴 **CORRECCIÓN POSTERIOR (2026-08-14, DL-007).** La baza (b) de la lectura 3 —el
> glifo decodifica "~4.7× más barato", medido 15 vs 70 s/época— **queda retirada**.
> Ese 70 s/época se midió contra un brazo estándar que multiplicaba por una matriz
> identidad V×V (trabajo aritmético nulo) y materializaba one-hots densos: medía la
> implementación del baseline, no la arquitectura. Con el baseline arreglado, el
> estándar es un 10% MÁS RÁPIDO que el glifo por step (24,16 vs 26,74 ms), y su
> ventaja crece con el vocabulario. Las cifras de coste de este informe (la fila
> "Coste por época" y el 15 vs 70 s) **no deben citarse**. Lo demás del informe se
> sostiene: el arreglo es bitwise equivalente, así que ningún resultado de
> aprendizaje queda invalidado. Detalle y tabla nueva en DL-007.

> ⚠️ **Una semilla por brazo.** La varianza observada entre runs de una misma
> configuración (sandbox vs canónico del 3-ago) fue comparable a varios de los
> efectos aquí tabulados. Este informe fija la foto y las tendencias; las
> conclusiones fuertes requieren las réplicas multi-semilla (siguiente paso).

## Resultados

### Brazos K-65P (escuela completa, 8 etapas, exámenes gen_valid+bigrama)

| Métrica | v2 · glifos | v0-k65p · estándar |
|---|---|---|
| Hitos | 7/7 🎓 | 7/7 🎓 |
| Épocas totales | 162 | **145** |
| Suspensos | 1 (8_years → remediación 128→256d) | **0** |
| Params finales | 4,88M (256d) | **1,26M (128d)** |
| val_loss final (oráculo 2.09) | 2.150 | **2.012** |
| gen_valid examen final | **1.00** | 0.76 |
| **OOD** (pares nunca vistos) | **40/40 = 100%** | 38/40 = 95% |
| Gap OOD−val | +0.092 nats | **+0.073 nats** |
| Ruido en glifos σ=0.1 | acc 50.8%→13.2% (frágil) | n/a |

### Brazos inglés (bloque preescolar 2-4 años, examen Samantha real)

| Métrica | v1-adaptativo · glifos | v0 · estándar |
|---|---|---|
| Hitos preescolares | 3/3 ✅ (todos 10/10 exact-match) | 3/3 ✅ (todos 10/10 exact-match) |
| Épocas totales (4 etapas) | 309 | **229** |
| Épocas por etapa (0-1/1-2/2-3/3-4) | 91/85/52/80 | 51/56/71/51 |
| best val por etapa | 3.03/3.84/4.59/5.04 | **2.05/2.44/2.80/3.13** |
| Dim | 128d en todo, 0 neurogénesis | 128d en todo, 0 neurogénesis |
| Coste por época (misma máquina) | **~15 s** | ~70 s |
| vs calendario fijo v1 (448 ép. preescolar) | **−31%** | **−49%** |

## Lecturas (calibradas)

1. **El protocolo adaptativo domina al calendario fijo en los 4 brazos.** Mismas
   certificaciones con 31-49% menos épocas en inglés y ~9× menos en K-65P, y
   solo 1 neurogénesis en 4 escuelas (el calendario clásico creció a 1024d/77M
   para la misma tarea K-65P). El crecimiento del v1 original fue en gran parte
   artefacto del calendario, no necesidad del alumno.
2. **El embedding estándar ajusta mejor la distribución en las 8 mediciones**
   (todas las etapas, ambos idiomas) y suele necesitar menos épocas. Coherente
   con el techo de rango ≤65 del glifo composicional. Como modelador de
   lenguaje puro, el estándar gana — sin ambigüedad.
3. **El glifo conserva tres bazas**: (a) tendencia a mayor validez generativa
   en composición profunda (1.00 vs 0.76 en el examen final; 100% vs 95% OOD)
   — *tendencia, no resultado, hasta réplicas*; (b) decode ~4.7× más barato con
   vocabulario grande (O(65·d) vs O(V·d), medido 15 vs 70 s/época); (c) la
   capacidad EXCLUSIVA de vocabulario en caliente (`register_new_word`) —
   arquitectónicamente presente, **jamás examinada** → examen M5.
4. **Ninguna escuela actual discrimina lo suficiente.** Los dos brazos K-65P se
   gradúan con holgura; los dos ingleses claven 10/10 exact-match en Samantha
   (el auto-grader satura). Los instrumentos son honestos pero el temario es
   demasiado fácil para separar arquitecturas: el poder discriminante vendrá
   de (a) el examen M5 de palabra nueva y (b) la escuela semántica
   (`gen_true` vía Prolog, teoremas held-out).
5. **Fragilidad al ruido del glifo confirmada** (σ=0.1 hunde la precisión al
   13%): corromper un primo corrompe todas las palabras que lo componen. La
   narrativa de "robustez al ruido" del 2-ago queda doblemente enterrada.

## Incidencias operativas del cierre (2 días de guerra)

Ciclo de sueño de 13 núcleos compitiendo con el entrenamiento (parada limpia +
relanzador por carga); driver NVIDIA en mismatch → **fallout silencioso a CPU**
del run reanudado (845% CPU, 6 GB swap, 0 épocas/1,5h) → flag `--require_gpu`
(rc=3) commiteado — el principio del RFC_SLEEP_JOB_DRIVER §2.2 aplicado a la
escuela; agotamiento de tokens de sesión. Refuerzan dos pendientes ya
registrados: **entrenamientos como jobs de red-pill con prioridad baja** y el
**SleepJobDriver** (en manos del otro agente).

## Artefactos

- K-65P: `releases/bit_v{2,0}_k65p_adaptive_20260803/` (checksums + actas + informe adversarial).
- Inglés: `storage/checkpoints/sovereign_school_adaptive/` (glifos) y
  `sovereign_school_v0_adaptive/` (estándar) — checkpoints por hito + actas en
  `school_state.json` (exam_history vía Samantha).
- Decisiones: DL-004/005/006 · Mapa: `docs/sessions/20260803/ESTADO_DEL_CAMINO.md`.

## Siguiente (sin cambios respecto al mapa)

Réplicas multi-semilla → M5 palabra nueva → sistema inmune de re-certificación
→ escuela semántica → N2DN sobre copia.
