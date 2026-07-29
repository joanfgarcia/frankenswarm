# Decision Log — Frankenswarm

Registro de decisiones de calado: el problema, la decisión, la evidencia que la
respalda y dónde está el detalle. Una entrada por decisión, la más reciente arriba.
Las decisiones se numeran DL-NNN y no se reescriben: si una decisión se revierte,
se añade una entrada nueva que la referencia.

---

## DL-002 · 2026-07-27 — Precisión mixta BF16 (+SDPA, +compile opcional) para completar el currículum en la RTX

**Problema.** El currículum de 8 años no cabía en la GPU: el stage `secondary_8`
(dim 1024) proyectaba ~7.7 GB de VRAM sobre los 8 GB de la RTX 5070 (medido después:
6.9 GB de pico sintético — "cabe por los pelos", sin margen para fragmentación ni
convivencia con el kernel). Además, a 3h20-3h50 por época (FP32, dim 896), las ~410
épocas restantes hasta la graduación suponían ~60 días de GPU continua.

**Decisión.**
1. **BF16 vía autocast con pesos maestros FP32** (`--amp`, default `auto`), no la
   conversión total del checkpoint que proponía RFC-BITNET-VRAM-001 §4.4.1: el
   checkpoint no cambia de formato, FP32↔BF16 son intercambiables por ejecución
   (rollback = un flag) y la neurogénesis no se toca.
2. **SDPA** en la atención (habilitador de kernels fusionados; ~3% por sí solo).
3. **`--compile`** (torch.compile selectivo, §4.8) disponible y bajo prueba — el
   operador pidió no descartar nada; a nivel de step no rompe el STE.
4. El run vivo (epoch 998) queda congelado hasta que EXP_079 confirme que el paso
   32→16 bits no cuesta calidad de convergencia.

**Evidencia (Tier 1, medida — no estimada).** BF16+SDPA+compile: **3.7× más rápido**
(548→149 ms/step a dim 896) y **−40% VRAM** (6.1→3.6 GB); a dim 1024: 4.1 GB frente
a 6.9 GB en FP32. ∇STE de la misma magnitud en las 8 configuraciones. Proyección:
época ~1h, graduación en ~17-20 días de GPU en lugar de ~60. Detalle y tabla
completa: `docs/experiments/EXP_079_DESIGN.md` §2 +
`storage/benchmarks/tier1_step_bench.json`.

**Pendiente que gateaba la adopción → RESUELTO (28-jul-2026).** Tier 2 de EXP_079
ejecutado (A/B/C from-scratch, 160 épocas + neurogénesis 128→256 por brazo, misma
semilla): coste de calidad **indistinguible de cero** — val_loss final 3.7338 (FP32)
vs 3.7348 (BF16) vs 3.7343 (BF16+compile), |Δ| medio por época 0.003 con umbral en
0.05 y cero épocas fuera; batería y muestras idénticas entre brazos; épocas 2.6×
más rápidas end-to-end (80→31 min por 160 épocas). Único fallo literal: la
neurogénesis de B disparó en +5 épocas (sensibilidad del contador de plateau, no
degradación — análisis en EXP_079 §5). **Adoptado: `--amp auto --compile` en la
receta.** El run vivo se reanuda con vigilancia de ~100 épocas y rollback de un flag.

**Salvaguardas.** `--state_dir` (sandboxes; las rutas dejaron de estar hardcodeadas),
`--seed`, prohibido `--reset_state` en benchmarks, md5 del checkpoint vivo verificado
antes/después de cada brazo, backup en `storage/checkpoints/backup_pre_bf16_20260727/`.

**Referencias.** `docs/RFC_BIT_GRADUATION_ROADMAP.md` (D1-D6) ·
`docs/RFC_VRAM_SCALING_BITNET_CURRICULUM.md` (análisis técnico, Grok+DeepSeek) ·
commits 5477127, 1f55242, f0f28ad, c1947f0.

---

## DL-001 · 2026-07-03 — School v3: vocabulario limpio, corpus de fábrica, hitos operativos

Registrada retroactivamente por referencia: la decisión de ruta que gobierna el
entrenamiento actual vive en `docs/sessions/20260703/ROUTE_CHANGE_SCHOOL_V3.md`
(censo limpio de vocabulario tras el 57.8% de palabras fantasma, corpus sintético
de fábrica, neurogénesis por plateau, batería M1-M4 con umbrales congelados).
