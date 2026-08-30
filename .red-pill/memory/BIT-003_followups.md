# BIT-003 · Incidencia 30-ago bucle de etapa + proceso anti-regresión

## Incidencia (cerrada)
- El CyclicPoolSampler (DL-011) no persistía cursor/wrapped entre procesos: cada step (20 ép) re-barajaba con la misma semilla y cubría SIEMPRE las primeras ~8M secuencias del pool → full_coverage inalcanzable → etapa 2-3 en bucle (419 ép > tope 200, val plana 3.74).
- FIX: cursor/wrapped persistidos en school_state.json (samplers.{idx}) y restaurados al compilar la etapa (permutación determinista seed 42). Commit 7f46235.
- ROLLBACK: etapa 2-3 reiniciada desde el hito 2_years (época 137) — las 419 épocas entrenaron un subconjunto sesgado del 37% (asimetría contaminante para la comparativa). Etapas 0-1: limpias (cobertura completa in-proceso, 2_years legítimo bajo ×10).
- Verificado: test de continuidad cross-proceso (sin huecos, permutación determinista) + producción (cursor 400k persistido tras época 1).

## PROTOCOLO ANTI-REGRESIÓN (obligatorio desde 30-ago)
Todo cambio de protocolo de entrenamiento ANTES de encolar:
1. Test unitario de la mecánica nueva
2. Smoke 1 época vía RECETA DE JOB (nunca shell directo; nunca sin smoke por presión de intentos)
3. Test de continuidad si el cambio cruza procesos: 2 steps consecutivos verificando que el estado persiste (cursor, contadores)
4. Verificar que las etapas PUEDEN cerrar bajo el protocolo nuevo (cobertura ≤ max_stage_epochs)
Pendiente: crear configs/jobs/smoke_school.yaml (receta de smoke reutilizable).

## Estado al cierre 30-ago tarde
- Job 250de371 PAUSED* → resume con `red-pill job resume 250de371` (NO resubmit).
- Run v3: época 137, etapa 2-3 reiniciada con cobertura real (~54 ép/pase).
- Hitos: 2_years (v3 ×10). v2 archivado en bit003_glyph_x300_v2 (ablación ×300).
- Pendientes: batería 80/50 (generador + verificación de ausencia), brazo standard, brazo resonante, ablación v2→v3.