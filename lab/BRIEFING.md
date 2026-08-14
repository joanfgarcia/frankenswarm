# 🔬 Frankenswarm Lab — Briefing

> Foto del estado real del laboratorio. Los despertares autónomos son infraestructura
> de red-pill (AWAKENING) y funcionan; el `minion_scheduler` trabajó la cola de este
> laboratorio hasta EXP_021 (mayo 2026). Desde junio el frente avanza por sesiones
> interactivas y el bucle quedó apuntando a una cola desactualizada.

## Estado actual (2026-08-14 — preflight del reentrenamiento de v1 en inglés)

| Campo | Valor |
|---|---|
| **Frente activo** | **`[BIT-003]` Reentrenar v1 desde cero en INGLÉS puro** (hallazgo 12-ago: el corpus de v1 era spanglish — CHILDES-es + traducción "fast" rota). Plan en `~/Documents/IA/Aleth_Core/TODO.md`; preflight auditado en `docs/sessions/20260814/PREFLIGHT_BIT003.md` |
| **Hecho (hasta 13-ago)** | Matriz 2×2 cerrada (informe 5-ago); réplicas multi-semilla 6/6 graduadas (10-ago); escuela semántica + gen_true + M5 (11-ago); gating curricular + hot-vocab (12-ago); **PR #5 mergeado a main** (refactor modular del trainer + IOC + opt8bit, 389 tests) |
| **Hecho (14-ago)** | Preflight BIT-003 (cachés keyeadas por corpus, `datasets` declarado, acta de suspensos) · **DL-007**: el brazo estándar multiplicaba por una identidad V×V → 12,1× más rápido, checkpoints 35× menores, y **retirada la baza "el glifo decodifica 4,7× más barato"** · **RFC-GROWTH-V6** (propuesta): la premisa de Net2DeeperNet caducó con DL-006 |
| **Límite de lo certificado** | El trainer refactorizado (a5799c7) NUNCA ha corrido una escuela real: los runs del informe 5-ago fueron pre-refactor. Smokes adaptativos de los dos brazos hechos el 14-ago (no una escuela completa) |
| **Estado de la tesis** | De las tres bazas del glifo del informe 5-ago, **solo sobrevive el vocabulario en caliente** (M5): la validez generativa OOD no replicó (10-ago) y el coste de decode era artefacto del baseline (DL-007). La ventaja real que sí aguanta es **compresión** (3,5× menos params) + extensibilidad. Reformulación defendible: *no un mejor modelo de lenguaje, sino uno más pequeño y extensible* |
| **Siguiente** | Instrumentos primero (escuela semántica `gen_true`, M5 con contraste semántico, degradación bajo cuantización) → rejilla D×W del RFC-GROWTH-V6 (~12 GPU-h en K-65P) → BIT-003 con la geometría ya elegida |
| **Rama** | `feat/v1-english-rebuild` |

## Fuente de verdad

- **Sesiones**: `docs/sessions/<fecha>/` — planes e walkthroughs auditados. Lo más reciente manda.
- **⚠️ `lab/experiment_queue.yaml` está OBSOLETA** (last_updated 2026-05-27, completados hasta EXP_021, pendiente un EXP_032 que ya se ejecutó — los grids llegaron a 034 y las sesiones citan EXP_039). Reconciliarla o retirarla; mientras tanto no fiarse de ella.

## Historial de despertares

_El `minion_scheduler` (vía AWAKENING de red-pill) ejecutó la cola hasta EXP_021;
última firma en `experiment_queue.yaml`: 2026-05-27. Esta sección del briefing no
se rellenaba porque el scheduler escribía en la cola, no aquí. Decisión pendiente
de Joan: re-apuntar el bucle a una cola reconciliada con el frente actual
(Tensors-as-API), o dejar el laboratorio en modo interactivo._

## Lecciones vigentes

- **Antes de comparar arquitecturas, arreglar el baseline** (DL-007): el brazo
  estándar hacía trabajo aritmético gratuito y la comparativa de coste medía la
  implementación, no la arquitectura. Es el fallo de DL-004 repetido en otro sitio —
  ante cualquier cifra comparativa, preguntar primero qué hace el baseline de más.
- La **resonancia** (`forward_resonance`, bucle latente con BPTT) está implementada y
  rodada en el track del arena, pero **apagada en la escuela**: la escuela entrena un
  modelo que no es el que describe el paper 1 (RFC-GROWTH-V6 §3, E1).
- Embeddings CONGELADOS (EXP_005b: descongelarlos degrada todo).
- Vocabularios ≥15k; prohibido mapear OOV por similitud ciega (Regla 7, colisión "pajaritos"→"cálmate").
- Neurogénesis en caliente: expansión Net2WiderNet en CPU + `torch.cuda.empty_cache()` para evitar el OOM que migraba el entrenamiento a CPU en silencio.
- Los scripts de épocas cerradas viven en `lab/experiments/` (poda 2026-07-03); `src/bitnet/` queda para módulos importables y entrypoints fijados por REPRODUCE/DEMO_GUIDE.
