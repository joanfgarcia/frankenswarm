# Diseño de Benchmark: FP32 vs BF16+SDPA en la Escuela Soberana

**Status:** v3 — APROBADO por Joan (27-jul): from-scratch OK, `torch.compile`
ENTRA en la matriz ("no descartemos nada"). Patch de aislamiento implementado
(commit c1947f0: `--state_dir`, `--seed`, `--compile`, base_dir derivado del
repo — adiós ruta absoluta). Tier 1 ejecutado (resultados en
`storage/benchmarks/tier1_step_bench.json`).
**Hallazgo (27-jul):** el vocabulary gating NO está conectado al entrenamiento
(el `logit_mask` del modelo existe desde EXP_034, pero la loss de train/val no
lo recibe; solo el sampler cualitativo filtra). El run vivo entero es sin
gating → **el benchmark corre sin gating** (una variable cada vez); cablear el
gating es cambio aparte, gobernado por las condiciones del plan de
arquitectura §2 (avance por certificación, no por edad).
**Author:** Aleth · 2026-07-27
**Contexto:** D1 de RFC-BIT-GRAD-001 implementada (commits 5477127/1f55242/f0f28ad).
Falta el gate empírico. Dirección fijada por Joan: **entrenar de cero** (épocas
baratas a dim 128-256) y **evaluar contra los checkpoints guardados**.
**Destino si se aprueba:** `frankenswarm/docs/experiments/EXP_<n>_DESIGN.md`
(pre-registro, umbrales congelados ANTES de correr — doctrina de la casa).

---

## 0. Dos hechos verificados que condicionan el diseño

1. **La transición a inglés fue el 14-jul (dd09ba4)** y los milestones 2-5 del
   run vivo son del 7-10 jul → las etapas tempranas del run vivo se entrenaron
   con el **corpus antiguo**. Sus checkpoints (`model_milestone_2..5_years.pt`)
   sirven como referencia *gruesa* (batería), nunca como curva de control fina.
2. **No hay curvas de loss históricas en disco**: `storage/logs/` nació ayer,
   el estado solo persiste `best_val_loss`, y los prints de las épocas 1-996
   murieron en la terminal. → El control FP32 se **re-corre**, no se recupera.
   (A dim 128-256 eso cuesta horas, no días — de ahí que el from-scratch de
   Joan sea el diseño correcto.)

## 1. Principio innegociable: el vivo no se mueve

El run real queda congelado en epoch 998 hasta terminar el benchmark.

| Hazard | Neutralización |
|---|---|
| Rutas de estado **hardcodeadas** (`train_sovereign_school.py:592,830`): cualquier run, también uno from-scratch, escribiría sobre `model_current.pt` y `school_state.json` vivos | **Patch mínimo previo obligatorio**: `--state_dir <dir>` redirige estado+checkpoints. Default = ruta actual (el camino vivo ni se entera). Sin esto NO se lanza nada |
| `--reset_state` BORRA el estado y el checkpoint vivos | **Prohibido usarlo.** Un `--state_dir` apuntando a un directorio vacío ya arranca de cero sin borrar nada |
| Batches en orden distinto por run → brazos incomparables | Flag `--seed <n>` (fija torch/numpy/random). Ambos brazos, misma seed |
| El job-manager reanuda la escuela a mitad de benchmark | Ventana de benchmark = job `school` NO encolado (`red-pill job list` antes de lanzar) |
| Corrupción silenciosa del vivo | md5 de `model_current.pt`+`school_state.json` antes y después de cada tier (referencia: backup `backup_pre_bf16_20260727`, md5 `2038b204…`) |

Sandboxes: `storage/benchmarks/bf16_scratch/{arm_fp32,arm_bf16}/` (vacíos al inicio).

---

## 2. Tier 1 — Micro-benchmark sintético (~30-45 min GPU, cero estado)

Script standalone (`scripts/bench_school_step.py`): modelo desde init aleatoria,
~50 steps (batch 64, seq 128, datos sintéticos) por config, sin tocar checkpoints.

**Matriz** (por cada dim ∈ {896, **1024**}):

| # | Config | Qué responde |
|---|---|---|
| 1 | FP32 + atención manual (el oráculo del test) | Línea base pre-D1 |
| 2 | FP32 + SDPA | Cuánto aporta SDPA solo |
| 3 | BF16 + SDPA (lo mergeado) | El candidato |
| 4 | *(opcional)* + `torch.compile` selectivo | D2, solo informativo (vigilar ∇STE) |

**Métricas:** ms/step (mediana de 50 tras 5 de warmup), VRAM pico, ∇STE.

**Por qué Tier 1 no es prescindible:** el from-scratch corre a dim 128-256 y
**no dice nada del stage 8**. La pregunta que motivó D1 — ¿cabe dim 1024 en
8 GB y con qué margen? — solo la responde esta medición.

### ✅ RESULTADOS (27-jul, RTX 5070 Laptop, torch 2.12, batch 64 × seq 128)

| config | dim | ms/step | VRAM pico | ∇STE |
|---|---|---|---|---|
| fp32_manual (pre-D1) | 896 | 548.5 | 6,126 MB | 0.00579 |
| fp32_sdpa | 896 | 532.5 | 6,187 MB | 0.00573 |
| bf16_sdpa | 896 | 254.1 | 4,220 MB | 0.00584 |
| **bf16_sdpa_compile** | 896 | **149.0** | **3,641 MB** | 0.00566 |
| fp32_manual | 1024 | 656.6 | 6,882 MB | 0.00569 |
| fp32_sdpa | 1024 | 649.7 | 6,975 MB | 0.00572 |
| bf16_sdpa | 1024 | 310.2 | 4,763 MB | 0.00573 |
| **bf16_sdpa_compile** | 1024 | **182.5** | **4,111 MB** | 0.00566 |

**Lecturas:**
1. **BF16+SDPA+compile = 3.7× más rápido que la línea base** y −40% VRAM.
   El coste de compilación en frío es ~10-12 s una vez por proceso — nada.
2. **∇STE idéntico de magnitud en las 8 configs** (~0.0057): el miedo del
   §4.8.1 (compile rompiendo el STE) no se materializa a nivel de step.
   Vigilancia multi-época pendiente en Tier 2.
3. **SDPA solo aporta ~3%** a esta escala (seq 128 → la matriz N×N es pequeña
   frente al decode de 12k vocab). Su valor real es habilitar kernels
   fusionados; el músculo lo ponen BF16 y compile.
4. **dim 1024 en FP32 cabe por los pelos** (6.9 GB pico sintético, sin
   fragmentación de horas ni dataset): demasiado justo para confiarle el
   stage 8. Con BF16(+compile): 4.1-4.8 GB — margen de sobra, incluso con
   algo del kernel residente.
5. **Proyección de calendario:** si el 3.7× del step se traslada a la época
   (3h20-3h50 medidas en FP32 a dim 896 → **~55-65 min**), las 410 épocas
   restantes pasan de ~60 días a **~17-20 días de GPU**. La época real lo
   confirmará.

**Gate del brazo C: SUPERADO** (compile aporta +70% sobre bf16_sdpa con ∇STE
sano) → el A/B from-scratch corre con TRES brazos.

---

## 3. Tier 2 — A/B from-scratch hasta el examen de 2 años (idea de Joan)

Dos runs **desde cero**, trainer real, corpus real de HOY (inglés), sandboxes:

- **Brazo A (control):** `--amp off  --seed 770 --state_dir storage/benchmarks/bf16_scratch/arm_fp32`
- **Brazo B (candidato):** `--amp bf16 --seed 770 --state_dir storage/benchmarks/bf16_scratch/arm_bf16`
- **Brazo C (petición de Joan, gateado por Tier 1):** `--amp bf16 --compile --seed 770 --state_dir .../arm_bf16_compile` — solo se corre si en Tier 1 el compile aporta ≥15% de velocidad con ∇STE sano; si no, queda medido y descartado con dato
- Horizonte: **fin del stage 1 = epoch 160** (`--max_epochs_per_run 160`),
  que incluye el examen del hito `2_years` con Samantha.
- Secuenciales, con la liturgia de `train_school.sh` (liberar VRAM,
  `systemd-inhibit`, `MemoryMax=16G`).
- Coste estimado: a dims 128-256 la época debería caer en minutos (la media
  histórica global era ~2 min/epoch con dims pequeñas dominando) → **~3-8 h
  por brazo**. La primera época medida ajusta la previsión.

**Ventajas de este diseño sobre el A/B en epoch 998** (descartado):
1. ~10× más barato por época → más épocas de evidencia por la misma GPU.
2. Cubre **una neurogénesis real bajo BF16** (128→256 al entrar en stage 1) —
   la interacción AMP↔net2wider que el sandbox a dim 896 jamás ejercitaría
   (capado por stage).
3. Cubre warmup de lr, arranque de vocabulario y primer examen de Samantha —
   el ciclo completo de la escuela, no un tramo.
4. Con seed fija, A y B ven los mismos batches: la única variable es la precisión.

**Límite honesto:** BF16 a dim 128-256 no garantiza BF16 a dim 896-1024 (los
errores de precisión escalan con anchura/profundidad). Por eso el veredicto
final sigue siendo trifásico: Tier 1 (equivalencia por step a dims grandes) +
Tier 2 (convergencia multi-época barata) + **las primeras ~100 épocas del run
real con `--amp auto` vigiladas** (rollback = un flag).

### Evaluación (umbrales a congelar antes de la primera época)

| Métrica | Umbral de adopción |
|---|---|
| Convergencia | `\|val_loss_B(e) − val_loss_A(e)\| ≤ 0.05` por época en los tramos estables (fuera de las ~5 épocas post-neurogénesis), misma tendencia |
| Neurogénesis | B dispara 128→256 en época ±3 de A, y la loss se recupera con el mismo perfil |
| Velocidad | época B ≤ 0.8× época A (a dim pequeña el speedup de BF16 es menor; el dato de dims grandes lo da Tier 1) |
| STE | ∇STE > 0 en todas las épocas de ambos brazos, mismo orden de magnitud |
| **Batería** (el juez de la casa) | `milestone_battery.py` sobre los checkpoints `2_years` de A y B: diferencias dentro del ruido entre-runs |
| Samantha | Ambos brazos aprueban (o suspenden) el examen de 2 años igual |
| Referencia histórica (gruesa) | Batería sobre `model_milestone_2_years.pt` (corpus antiguo, solo contexto — NO gate) |

**Decisión:** pasa todo → reanudar el run real (epoch 998) con `--amp auto` y
vigilancia de 100 épocas. Falla algo → `--amp off`, el vivo nunca se movió, y
el fallo se documenta como resultado negativo.

## 4. Fuera de alcance (deliberadamente)

- Batch 128 con BF16 (contamina el A/B de precisión; EXP aparte si interesa).
- `torch.compile` como candidato (D2 sigue opcional, solo se mide en Tier 1).
- Cambios de currículum/fórmula (D3 lo prohíbe).

## 5. Orden de ejecución propuesto

1. Patch `--state_dir` + `--seed` (+ test de que los defaults no cambian rutas) — ~30 líneas.
2. `scripts/bench_school_step.py` + **Tier 1** → tabla ms/step + VRAM a 896/1024.
3. Congelar umbrales §3 y mover este doc a `docs/experiments/` como pre-registro.
4. **Tier 2** (~una tarde-noche de GPU en total) → evaluación → decisión.
5. Reanudar la escuela con el sabor ganador. md5 del vivo verificado en cada paso.

*Nada se ejecuta hasta que el Fixer bendiga este diseño.*
