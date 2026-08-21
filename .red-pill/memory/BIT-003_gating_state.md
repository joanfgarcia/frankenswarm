# BIT-003 · Estado del gateo de vocabulario (2026-08-21)

## Decisión (DL-008)
- Gateo de producción por etapa (logit_mask): cada etapa solo produce el vocabulario de su edad. Núcleo (28 referencias EN + 65 safe words) + top-N CHILDES-en + currículo de la etapa. Cortes: [200,800,3000,5000,8000,10000,15000,todo]. E7 = todo el vocabulario.
- Equivalencia estricta entre brazo estándar y K-65P: misma máscara de tokens, mismo corpus por etapa. La comparativa mide solo la representación.
- Clasificación del corpus por etapa con umbral 95% (etapa mínima que cubre >=95% de tokens). CHILDES 78% en E0-E2; TinyStories repartida tardía. Nada se pierde (E7 cubre 100%).
- Propiedad de tesis: con glifos el gateo se amplía EN CALIENTE (palabra nueva = primos ya aprendidos); con estándar no sin reentrenar.

## Implementado
- src/bitnet/training/modules/stage_gating.py: REFERENCE_WORDS_EN, SAFE_WORDS_EN, TOP_N_BY_STAGE, build_stage_logit_mask, build_all_stage_masks (E7=todo), token_min_stage, classify_sequences_by_gate, apply_stage_gate, loss_mask_for_gate
- trainer: childes_freq (Counter), stage_masks, gated_general, aplicado en forwards adaptativo+clásico
- evaluador actualizado a EN (childes_pre_school_en.json, special_tokens EN)

## Remediación 21-ago (DL-009) — APLICADA
- Auditoría pre-lanzamiento encontró 5 bloqueantes + 6 serios: ver BIT-003_audit_findings.md y BIT-003_fix_plan.md (esta carpeta).
- Todo aplicado en feat/v1-english-rebuild (commits F1-F7): corpus sin artefactos CHAT, apóstrofos normalizados (don't→dont), vocab 19.637 con 28/28 glifos canónicos, censo por palabras, arranque y device curados, E7 sin <unk>, instrumento v2 (10 preguntas/edad, solo in-vocab, sin lore), evaluador alineado con las máscaras reales (stage_gate_masks.json).
- Aceptación: scripts/audit_stage_gating.py 22/22 ✓ · scripts/validate_exam_vocab.py ✓. Gateo [254,825,3008,5003,8005,10003,15002,19636]; CHILDES 78% E0-E2, 3.5% E7; TinyStories E7 16%→7.1%.

## Pendiente
- samples_per_epoch: 400k CONFIRMADO por operador (default del trainer desde F4).
- Smokes CPU+GPU y lanzar ambos brazos: comandos en BIT-003_fix_plan.md §F8 (systemd-inhibit obligatorio, sin --opt8bit, --wd_mode no_embed).

## Referencias
- Rama: feat/v1-english-rebuild (commits 081fc8d, 2741e95)
- GPU libre (167 MiB / 0%)