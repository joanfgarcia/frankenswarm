# 🔬 Frankenswarm Lab — Briefing

> Foto del estado real del laboratorio. Los despertares autónomos son infraestructura
> de red-pill (AWAKENING) y funcionan; el `minion_scheduler` trabajó la cola de este
> laboratorio hasta EXP_021 (mayo 2026). Desde junio el frente avanza por sesiones
> interactivas y el bucle quedó apuntando a una cola desactualizada.

## Estado actual (2026-07-03 tarde — CAMBIO DE RUTA aprobado)

| Campo | Valor |
|---|---|
| **Frente activo** | **School v3**: vocabulario limpio + corpus sintético + batería operacional (`docs/sessions/20260703/ROUTE_CHANGE_SCHOOL_V3.md`) |
| **Motivo** | Auditoría del hito 5: vocabulario 57,8% fantasma (contaminación de listas de frecuencia), corpus 864K tokens monótono, hitos certificados por juez débil. La run 2 se archiva como resultado negativo. **El sustrato funciona** (93% preferencia gramatical en frases nuevas). |
| **Vocabulario** | `configs/clean_vocabulary_words.json` (6.361 palabras limpias, crecerá con la fábrica); glifos pendientes de re-derivar |
| **Siguiente** | Diálogos ×100 plantillas → fábrica preescolar 2M tokens (AWAKENINGs bajo wake gate) → glifos limpios → School v3 desde cero |
| **Tras School v3** | Tensors as API → MoE distribuido (plan 20260623 intacto, con B genuinamente superior a A) |
| **Rama** | `feat/ppo-exploration` |

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
