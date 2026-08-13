# 🔬 Frankenswarm Lab — Briefing

> Foto del estado real del laboratorio. Los despertares autónomos son infraestructura
> de red-pill (AWAKENING) y funcionan; el `minion_scheduler` trabajó la cola de este
> laboratorio hasta EXP_021 (mayo 2026). Desde junio el frente avanza por sesiones
> interactivas y el bucle quedó apuntando a una cola desactualizada.

## Estado actual (2026-08-03 noche — matriz DL-006 y mapa del camino)

| Campo | Valor |
|---|---|
| **Frente activo** | **Track Bit v2 / K-65P bajo protocolo adaptativo DL-006** — mapa completo en `docs/sessions/20260803/ESTADO_DEL_CAMINO.md` (lo más reciente manda) |
| **Hecho (3-ago)** | Graduación falsa del 2-ago invalidada (DL-004); glifo cero curado (DL-005); protocolo adaptativo (DL-006); **graduaciones REALES de gramática K-65P**: v2 glifos 7/7 (OOD 100%) y v0 estándar 7/7 (OOD 95%); brazos control inglés (glifos ✅ preescolar, estándar en marcha) |
| **Límite de lo certificado** | Solo FORMA: v2 conoce las reglas K-65P pero nadie le ha dicho nada verdadero — la escuela semántica (corpus causal + juez Prolog, `gen_true`) es la etapa 3 pendiente |
| **Siguiente** | Informe comparativo matriz 2×2 → réplicas multi-semilla → examen M5 (palabra nueva) → sistema inmune de re-certificación → escuela semántica → N2DN sobre copia |
| **Rama** | `ci/fix-re-import-and-compilation-tests` (5 commits locales sin push, 8e0c18a…3ac00b9) |

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

- Embeddings CONGELADOS (EXP_005b: descongelarlos degrada todo).
- Vocabularios ≥15k; prohibido mapear OOV por similitud ciega (Regla 7, colisión "pajaritos"→"cálmate").
- Neurogénesis en caliente: expansión Net2WiderNet en CPU + `torch.cuda.empty_cache()` para evitar el OOM que migraba el entrenamiento a CPU en silencio.
- Los scripts de épocas cerradas viven en `lab/experiments/` (poda 2026-07-03); `src/bitnet/` queda para módulos importables y entrypoints fijados por REPRODUCE/DEMO_GUIDE.
