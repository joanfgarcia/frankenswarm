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

## Pendiente
- samples_per_epoch: 400k recomendado (análisis Chinchilla/TinyStories). Operador a confirmar.
- Lanzar entrenamiento brazo glyph con --compile --amp bf16 --adaptive --full_tinystories
- Docs: RFC_GATING_VOCABULARIO_ETAPAS.md, DL-008

## Referencias
- Rama: feat/v1-english-rebuild (commits 081fc8d, 2741e95)
- GPU libre (167 MiB / 0%)