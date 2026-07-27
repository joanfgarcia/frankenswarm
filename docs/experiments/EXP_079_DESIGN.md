# EXP_079 — Precisión mixta en la Escuela Soberana: FP32 vs BF16 vs BF16+compile

**Status:** PRE-REGISTRADO (umbrales congelados antes de la primera época del Tier 2)
**Fecha:** 2026-07-27 · **Autores:** Joan (Fixer) + Aleth
**Decisión que respalda:** D1 de RFC-BIT-GRAD-001 (ver `docs/RFC_BIT_GRADUATION_ROADMAP.md`
y la entrada 2026-07-27 de `docs/DECISION_LOG.md`)
**Origen:** el stage `secondary_8` (dim 1024) proyectaba ~7.7 GB de VRAM sobre los
8 GB de la RTX 5070. RFC-BITNET-VRAM-001 propuso BF16+SDPA; el operador exigió
números antes de confiarle 410 épocas: velocidad, memoria y — lo pendiente —
**calidad** (¿perdemos convergencia por el paso 32→16 bits?).

---

## 1. Implementación bajo prueba (commits 5477127, 1f55242, f0f28ad, c1947f0)

- **BF16 = autocast con pesos maestros FP32** (desviación deliberada del RFC-VRAM
  §4.4.1): activaciones en BF16, params/gradientes/AdamW en FP32. El checkpoint no
  cambia de formato → FP32↔BF16 intercambiables por run (`--amp off|bf16|auto`).
- **SDPA** en `BitNetAttention` (kernel fusionado, sin materializar la matriz N×N).
- **`--compile`**: `torch.compile(fullgraph=False)` solo en el forward de
  entrenamiento; el checkpoint se guarda siempre desde el modelo sin compilar.
- **Aislamiento**: `--state_dir` (sandbox de estado/checkpoints), `--seed`
  (brazos batch-idénticos). El run vivo (epoch 998) queda congelado e intacto
  (md5 verificado antes/después de cada brazo).
- Sin vocabulary gating (no está conectado al training; el run vivo tampoco lo usa).

## 2. Tier 1 — micro-benchmark del step (EJECUTADO 27-jul)

RTX 5070 Laptop · torch 2.12 · batch 64 × seq 128 · 12 steps medidos (mediana) ·
datos sintéticos · sin checkpoints. JSON: `storage/benchmarks/tier1_step_bench.json`.

| config | dim | ms/step | VRAM pico | ∇STE |
|---|---|---|---|---|
| fp32_manual (línea base pre-D1) | 896 | 548.5 | 6,126 MB | 0.00579 |
| fp32_sdpa | 896 | 532.5 | 6,187 MB | 0.00573 |
| bf16_sdpa | 896 | 254.1 | 4,220 MB | 0.00584 |
| **bf16_sdpa_compile** | 896 | **149.0** | **3,641 MB** | 0.00566 |
| fp32_manual | 1024 | 656.6 | 6,882 MB | 0.00569 |
| fp32_sdpa | 1024 | 649.7 | 6,975 MB | 0.00572 |
| bf16_sdpa | 1024 | 310.2 | 4,763 MB | 0.00573 |
| **bf16_sdpa_compile** | 1024 | **182.5** | **4,111 MB** | 0.00566 |

**Lecturas:** (1) BF16+SDPA+compile = **3.7×** más rápido y **−40%** VRAM vs línea
base; coste de compilación ~10-12 s/proceso. (2) ∇STE de la misma magnitud en las 8
configs — el temor a que compile rompa el STE (§4.8.1) no se materializa a nivel de
step. (3) SDPA solo aporta ~3% a seq 128 (la matriz de atención es pequeña frente al
decode de ~12k vocab). (4) dim 1024 en FP32 cabe *por los pelos* (6.9 GB pico
sintético, sin fragmentación de horas): demasiado justo para el stage 8; con BF16
sobra margen. (5) Proyección: época de 3h20-3h50 (FP32, dim 896) → ~55-65 min.

## 3. Tier 2 — A/B/C de convergencia from-scratch (pre-registro)

Tres runs **desde cero**, trainer real, corpus actual (inglés), hasta el fin del
stage 1 (**epoch 160**, incluye el examen mock del hito `2_years`), misma semilla:

| Brazo | Flags |
|---|---|
| A (control) | `--amp off --seed 770 --state_dir storage/benchmarks/bf16_scratch/arm_fp32` |
| B | `--amp bf16 --seed 770 --state_dir .../arm_bf16` |
| C | `--amp bf16 --compile --seed 770 --state_dir .../arm_bf16_compile` |

Comunes: `--batch_size 64 --max_epochs_per_run 160 --test_mock` (Samantha mock: el
juez de calidad es la batería sin-LLM, no un juez simulado). Secuenciales, GPU
exclusiva (LLM del kernel descargado), `systemd-inhibit` + `MemoryMax=16G`.
Cobertura extra: una neurogénesis real (128→256) bajo cada sabor.

### Umbrales de adopción (CONGELADOS — no se tocan tras ver resultados)

| Métrica | Umbral |
|---|---|
| Convergencia | `\|val_loss_{B,C}(e) − val_loss_A(e)\| ≤ 0.05` por época en tramos estables (se excluyen las ~5 épocas post-neurogénesis) y misma tendencia |
| Neurogénesis | B y C disparan 128→256 en época ±3 respecto a A, con recuperación de loss de perfil similar |
| Velocidad | época B ≤ 0.8× de A (a dim pequeña el speedup es menor; el dato de dims grandes ya lo dio Tier 1) |
| STE | ∇STE > 0 todas las épocas, todos los brazos, mismo orden de magnitud |
| Calidad final | `milestone_battery.py` sobre el checkpoint final de cada brazo: sin degradación fuera del ruido entre-runs en las métricas pre-registradas de la batería |
| Cualitativa | muestras del log sin colapso (`<pad>`/repetición) en ningún brazo |

**Decisión:** B pasa → se reanuda el run real (epoch 998) con `--amp auto`.
C pasa además → `--compile` entra también en la receta. Cualquier fallo → se
documenta como resultado negativo y el run real sigue en FP32 (`--amp off`)
sin haber perdido nada. En todos los casos, las primeras ~100 épocas del run
real quedan bajo vigilancia (val_loss vs tendencia + ∇STE) con rollback de un flag.

## 4. Fuera de alcance

Batch >64, vocabulary gating (cambio aparte, condiciones en
`Aleth_Core/bitnet_next_architecture_plan.md` §2), cambios de currículum (D3).

## 5. Resultados Tier 2

*(pendiente de ejecución — logs por brazo en `storage/benchmarks/bf16_scratch/arm_*.log`)*
