# BIT-003 · Estado 1-sep (post-auditoría DL-015)

## AUDITORÍA 1-SEP: dos retractaciones + instrumento corregido (DL-015)
- El suspenso de 2_years del glyph v4 a 128d fue un CRASH de infra (Samantha → None), no una nota: la neurogénesis 128→256 se disparó sin examen real. Contrato de exit-codes nuevo (0/2/3) + reintento + pausa `eval_infra_error`; el trainer ya no remedia sin calificación. 8/8 tests.
- La "curva riesgo-cobertura" del 31-ago era artefacto (conf nunca calculada). Con confianza real: calibración fuerte DENTRO de lo visto (top-20% → 90-100%), casi nula en lo no visto (0-5.4%).
- Runner de batería corregido (tokenize sin `<unk>`, dedupe, retención+procedencia persistidas) y RE-TIRADO sobre los 6 checkpoints.
- Banks regenerados con in-gate vivo (442→439; fuera burns/howl/hiss). Batería v2 congelada (`age4_v2.json`, 51/169, banks en el universo — 28 colisiones purgadas). v1 rige la confrontación en curso; v2 desde que los banks se cableen.

## Batería congelada v1 re-tirada (56 vistas / 189 no-vistas) — NÚMEROS CANÓNICOS
| checkpoint | dim | gate | cognición | retención |
|---|---|---|---|---|
| glyph v4 ×1 · 2y | 256 | 44.6% | 1.6% | 19.2% |
| glyph v4 ×1 · 4y | 256 | 69.6% | 4.2% | 30.8% |
| standard v4 ×1 · 2y | 128 | 51.8% | 5.8% | 25.6% |
| standard v4 ×1 · 3y | 128 | 67.9% | 3.2% | **42.3%** |
| glyph v3 ×10 · 4y | 128 | 69.6% | 1.6% | 24.4% |
| glyph ×300 v2 · 4y | 128 | 64.3% | 1.6% | 23.1% |
- Cognición 1.6-5.8% = suelo (dentro del ruido, n=189). Interrogativo y riddles: 0/39 en LOS SEIS. Abstenciones: 0 en los seis.
- ⚠️ El "12%" de cognición que circulaba (followups viejo, recetas) era del set piloto de 50, NO reproducible. Cifras canónicas = esta tabla (JSONs con procedencia en cada state_dir).
- Ambos brazos dan respuestas token-idénticas en exámenes de hito 2y/3y → el examen de hito mide el corpus, no la representación.

## Runs
- v4 ×1 glyph (dd078baf) COMPLETO hasta 4_years @256d (ép. 321, en etapa 4; completion era 4_years). Su 2_years quedó SIN examinar de verdad (crash) — hueco abierto.
- v4 ×1 standard (897ec6e3) PROCESSING: 2y/3y aprobados @128d sin neurogénesis; en etapa 3-4 hacia 4_years. **El fix de exit-codes lo protege desde el próximo step** (el examen de 4_years ya no puede suspender por infra).
- v3 ×10 glyph (250de371) PAUSED — resumable. v2 ×300 archivado (ablación).
- Receta nueva `bit003_glyph_v41_2y_x1`: re-responde "¿pasa el glyph 2_years a 128d bajo ×1?" con examen real (etapas 0-2, misma semilla). LANZAMIENTO = decisión del operador, tras el 4_years del standard.

## Próximos
1. Standard v4 → 4_years: batería v1 sobre su checkpoint → primera comparación homóloga real glyph-4y vs standard-4y.
2. Lanzar `bit003_glyph_v41_2y_x1` (cierra la retractación 1).
3. Cablear banks (DL-013) a entrenamiento+examen → desde ese momento la batería es `age4_v2.json`.
4. Traductor EN→K-65P: compilador (transductor + léxico curado 09-01: 65/65 primos con superficie) + ida-vuelta RFC-002 §5.
5. Programa de descomposición NSM por lotes (DL-014) — con el guard de estados curados y `populate_k65p_dictionary.py` reproducible (pendiente: 99 moléculas sin marcar en la BD).
6. Canal de silencio ("i dont know"): decisión de diseño pendiente — el argmax forzado impide abstenerse; la matriz 3×2 ya lo premia.
7. Fase 1 dim66 (formato glifo v2 + migrador), post-v4.
