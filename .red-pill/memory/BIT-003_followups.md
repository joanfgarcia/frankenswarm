# BIT-003 · Estado 30-ago-2026 (post DL-010)

## Run normal (control)
- Job 6297cbdf PROCESSING: reanudado en época 223, etapa 5 (primary_6), tras rollback al checkpoint del hito 5_years. Hitos aprobados: 2,3,4,5 años (4/7).
- Las etapas cierran por plateau a ~30 épocas. best_val por etapa: 4.13 (3-4), 4.43 (primary_5).
- FIX aplicado: bug de unidades en el censo del cache-hit del store CSR (n_general secuencia vs token) — censo 8.1k→18.5k palabras; máscaras correctas desde etapa 5.
- Pendiente: brazo standard (control) tras glyph.

## Brazo resonante (DL-010) — IMPLEMENTADO
- docs/RFC_RESONANCE_ARMS.md + DECISION_LOG DL-010.
- Config: rampa n_steps U[1,5], pos_mode clock, max 5, emoción first_only, 7 emociones (dojo 6 + neutral id 6 para val/exámenes).
- Flags en trainer + evaluador (state_manager propaga al subproceso). Receta bit003_glyph_resonant.yaml (state_dir bit003_glyph_res) — encolar EN SERIE tras controles.
- Validado: BPTT grad en clock+emociones, rampa 1-5, smoke época completa con gateo correcto, evaluador carga resonante + examen OK.
- Modelo resonante: 1,250,680 params (+2,800).

## Claves de diseño (no olvidar)
- Resonancia SOLA es NULA (EXP_033/034 B≈A): siempre con emoción first_only.
- El bucle NO es 18 capas: son las mismas 6 reutilizadas (cómputo 18, parámetros 6). Net2DeeperNet sigue aparcado (RFC-GROWTH-V6).
- SSM/Mamba (BitMambaBlock): pre-registrado plan v4 §4, NO implementado, ortogonal (hipocampo inter-turno); hook h_prev del bucle = puerta futura.
- _decode_hidden(logit_mask) usa convención BOOL; apply_stage_gate usa float 0/-inf — no mezclar.
- forward_resonance corre eager (compile solo envuelve forward estándar).

## Test distintivo del brazo resonante
Curva precisión-vs-n_steps en inferencia (1..5) sobre las baterías de examen: presupuesto de pensamiento medible. Extensión futura: halting adaptativo (PonderNet).