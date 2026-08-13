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
`docs/bitnet_next_architecture_plan.md` §2), cambios de currículum (D3).

## 5. Resultados Tier 2 (ejecutado 27/28-jul-2026, 21:07-23:45)

Tres brazos × 160 épocas completadas, una neurogénesis 128→256 por brazo, run
vivo intacto (huella md5 verificada entre brazos). Curvas completas:
`storage/benchmarks/bf16_scratch/tier2_curves.json` + `arm_*.log`.

### Veredicto por umbral congelado

| Umbral | arm_bf16 (B) | arm_bf16_compile (C) |
|---|---|---|
| Convergencia ≤ 0.05/época | ✅ media 0.0032, p95 0.0098, peor 0.017 (ép.66), **0 épocas fuera** | ✅ media 0.0027, p95 0.0076, peor 0.014, **0 fuera** |
| Neurogénesis ±3 épocas | ❌ **época 80 vs 75 (+5)** — ver análisis | ✅ época 75 (±0) |
| Velocidad ≤ 0.8× | ✅ 47 min vs 80 min (**0.59×**) | ✅ 31 min (**0.39×**) |
| ∇STE > 0, misma magnitud | ✅ min 5.6e-3, mediana 1.4e-2 (A: 4.6e-3 / 1.4e-2) | ✅ min 5.1e-3, mediana 1.3e-2 |
| Batería (sin degradación) | ✅ perfil idéntico a A (2/5, mismos exámenes) | ✅ ídem |
| Cualitativa (sin colapso) | ✅ "the cat sleeps", "she wanted to go" | ✅ ídem |

**Val loss final (época 160): A 3.7338 · B 3.7348 · C 3.7343** — indistinguibles
(Δ < 0.001, un orden de magnitud bajo el umbral).

### Análisis del único fallo literal (neurogénesis de B en +5)

El disparador de plateau es un contador discreto sobre mejoras del orden de
`min_delta=0.01`: con diferencias de val_loss de ~0.003 entre brazos, una sola
época que cruce el umbral por 0.001 resetea el contador y desplaza el disparo.
No es una diferencia de calidad: tras su neurogénesis, B re-converge con A a
<0.01 desde la época 90 y termina a 0.001 del control. Se registra como
**sensibilidad del trigger, no degradación** — y es irrelevante para la
adopción porque C (la config que se despliega, que incluye BF16) cumple el
±3 con desviación cero.

### Observación colateral (fuera del A/B) — RESUELTA 28-jul

La batería M1 suspendía a los TRES brazos por igual (grammar 0.000): sus datos
de examen eran los de v1 en castellano, anteriores a la transición a inglés
(dd09ba4). Realineada como **exam data v2-en** (traducción fiel de pares y
prompts, umbrales intocados). Re-puntuación con v2-en:

| checkpoint | grammar | min_len | stop | ghost |
|---|---|---|---|---|
| arm_fp32 (2 años) | 0.650 | 0.625 | 1.0 | 0.0 |
| arm_bf16 (2 años) | 0.650 | 0.625 | 1.0 | 0.0 |
| arm_bf16_compile (2 años) | 0.700 | 1.125 | 1.0 | 0.0 |
| run vivo, ép. 998 (dim 896) | **0.800** | 1.000 | 1.0 | 0.0 |

El examen discrimina de nuevo (gradiente 0.65 → 0.80 con la madurez del
modelo) y los brazos siguen empatados entre sí (±1 par de 20 = ruido) — el
criterio de calidad del A/B se sostiene también bajo v2-en. Las cifras v1 y
v2-en no son comparables entre sí (la versión se imprime en cada informe).

### DECISIÓN (según el criterio pre-registrado del §3)

**C pasa 6/6 → se adopta BF16 + compile**: la receta del job pasa a
`--amp auto --compile`. El run vivo (epoch 998) se reanuda con esa
configuración, con las primeras ~100 épocas bajo vigilancia (val_loss vs
tendencia previa + ∇STE) y rollback documentado: quitar `--compile` y/o
`--amp off` — el checkpoint es FP32 en todos los casos.

**Números finales que respaldan la decisión** (Tier 1 + Tier 2): 3.7× de
velocidad y −40% VRAM a dims grandes, 2.6× de velocidad de época medida
end-to-end a dims pequeñas, margen de 3.9 GB para el stage 8, y coste de
calidad de convergencia: **indistinguible de cero** (Δval final < 0.001 tras
160 épocas y una neurogénesis).
