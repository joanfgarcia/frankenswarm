# RFC: Hoja de Ruta Definitiva — Graduación de Bit (currículum de 8 años) y transición a K-65P

**ID:** RFC-BIT-GRAD-001
**Copia canónica** (decisión del Fixer, 27-jul-2026): este fichero en
`frankenswarm/docs/` es el registro oficial; `Aleth_Core/RFC_BIT_GRADUATION_ROADMAP.md`
queda como espejo de trabajo. Resuelve la pregunta abierta Q5. Ver también
`docs/DECISION_LOG.md` (DL-002) y `docs/experiments/EXP_079_DESIGN.md`.
**Status:** Draft v1 — para revisión del Fixer
**Author:** Aleth (Netrunner)
**Date:** 2026-07-27
**Supersede:** nada — este documento es el **maestro** que ordena y secuencia los ya existentes:
- `RFC_VRAM_SCALING_BITNET_CURRICULUM.md` (RFC-BITNET-VRAM-001 v4 FINAL) — el *cómo* técnico de la VRAM
- `NOTE_CURRICULUM_SCALING_REVIEW.md` — revisión pendiente del escalado de epochs (ver §2.4: contiene un error)
- `bitnet_next_architecture_plan.md` (v4) — gates de las propuestas de motor (gating, trit-loss, SSM)
- `k65p/docs/CORE/ROADMAP.md` — fases 1-5 del track K-65P (Formación v2)

---

## 0. Veredicto ejecutivo

1. **El stage de 7 años SÍ cabe íntegro en la RTX tal como está el código.** El miedo a no
   acabarlo es infundado por un hecho verificado en código: `get_next_dim`
   (`train_sovereign_school.py:488`) capa la neurogénesis al techo de dim del stage vigente.
   Durante `secondary_7` (dim techo 896) el modelo **no puede** saltar a 1024 aunque haya
   plateau. A dim 896 el consumo medido es ~6.7 GB < 8 GB.
2. **El que NO cabe es el stage de 8 años** (dim 1024 → ~7.7 GB proyectados, margen ~300 MB:
   OOM casi seguro). La solución ya está escrita y revisada por dos pares externos
   (RFC-VRAM-001, Estrategia B). Falta ejecutarla.
3. **Sí hay que adelantar — pero solo las medidas de VRAM, no el proceso.** La pausa actual
   es la ventana perfecta para implementar BF16 + SDPA (fases 1-2 de la Estrategia B,
   ~150 líneas) y reanudar ya con ello validado. No es una crisis: es mantenimiento
   planificado que además acelera el entrenamiento (~2× en tensor cores).
4. **NO se adelanta la fase K-65P (Formación v2).** El Bit lingüístico es el *control* del
   experimento (doctrina fija del roadmap k65p §0.2). Graduarlo recortado o saltar a K-65P
   por miedo al hardware invalidaría la comparación G4 — el resultado experimental más
   importante del proyecto. Además las fases 1-3 de k65p no necesitan GPU: avanzan en
   paralelo mientras Bit termina la escuela.
5. **El riesgo operativo número 1 hoy no es el modelo: es la contención de VRAM con el
   kernel.** Verificado 2026-07-27: el daemon de red-pill (`run_dual_bind.py`, PID 4541)
   retiene **7.4 GB de los 8 GB** con el entrenamiento pausado. Entrenamiento (6.7 GB) y
   LLM del kernel no caben juntos ni hoy ni con BF16. Hace falta una política explícita (D5).

---

## 1. Estado real verificado (2026-07-27)

| Dato | Valor | Fuente |
|---|---|---|
| Epoch actual | 998 | `school_state.json` |
| Stage | 6 (`secondary_7`, edad 7) | idem |
| Hidden dim | 896 | idem |
| best_val_loss | 4.467 (8 epochs sin mejora) | idem |
| Milestones | 2,3,4,5,6 años | idem |
| Neurogénesis previas | 880: 640→768 · 922: 768→896 | idem |
| Escalado de epochs REAL | **lineal**: `int(64 × (1 + i × 0.5))` | `train_sovereign_school.py:473` |
| Args reales de lanzamiento | `--batch_size 64 --max_epochs_per_run 1` — **defaults** de base_epochs/stage_scale | `configs/jobs/school.yaml`, `scripts/train_school.sh` |
| Epochs por stage | 64, 96, 128, 160, 192, 224, **256**, **288** | confirmado — la receta documenta `total: 1408` |
| Fin de stage 7 años | epoch **1120** → quedan **122 epochs** | confirmado |
| Stage 8 años | epochs 1121–1408 → **288 epochs** | confirmado |
| **Total restante hasta graduación** | **410 epochs** | confirmado |
| VRAM ahora mismo | 7441/8151 MB — ocupada por el daemon red-pill, NO por training | `nvidia-smi` |

> ✅ (v1.1) Los args reales del run quedaron confirmados contra la receta del job-manager
> (`configs/jobs/school.yaml`) y el runner (`scripts/train_school.sh`, commits 5c72a1e,
> ff0d4d5, 413b83d): ambos lanzan sin sobreescribir `--base_epochs`/`--stage_scale`.
> Lo único que sigue sin medir es la **duración por epoch a dim 896** — el driver del
> job-manager la mide ahora en cada ejecución (`job status`); la receta anota vidas de
> proceso históricas de 3-11h que NO son cadencia por epoch.

---

## 2. Diagnóstico — qué es real y qué no

### 2.1 Miedo infundado: "no acabamos el 7 en la RTX"
Falso mientras el entrenamiento tenga la GPU en exclusiva. El techo de dim por stage
(§0.1) garantiza dim 896 hasta epoch 1120, y 896 ≈ 6.7 GB cabe. El margen (~1.3 GB) es
incómodo pero el run ya ha demostrado vivir ahí.

### 2.2 Riesgo real e inminente: contención kernel ↔ entrenamiento
La percepción de "la RTX está llena" viene de aquí: el LLM del kernel ocupa 7.4 GB en
reposo. Cualquier reanudación del entrenamiento con el daemon cargado muere en OOM al
primer batch. Este patrón ya está fichado (memoria: VRAM contention, PRs de backoff
pendientes en el kernel). → Decisión D5.

### 2.3 Riesgo real y cierto: el stage de 8 años (dim 1024)
~7.7 GB proyectados a FP32. Sin BF16+SDPA el run cae en OOM en la neurogénesis 896→1024
(epoch ~1121+, tras plateau dentro del stage 8). La respuesta técnica completa —
incluidos los dos puntos críticos: conversión de checkpoint FP32→BF16 (§4.4.1) y
neurogénesis en BF16 (§6) — está en RFC-VRAM-001. No se re-litiga aquí.

### 2.4 Corrección a NOTE_CURRICULUM_SCALING_REVIEW
La nota afirma escalado **exponencial** (`2^(stage×0.5)`, total 2317 epochs, ~772h).
El código real es **lineal** (`1 + i×0.5`, total 1408 epochs con defaults). El pánico de
las 772h no tiene base; el trabajo restante son 410 epochs. La decisión de la nota
("NO TOCAR hasta el milestone de 7 años") se mantiene, pero por la razón correcta:
consistencia de la tesis, no supervivencia del hardware. La nota queda corregida por
referencia a este RFC; la opción C (parada adaptativa) se re-evalúa **después** de la
graduación, para Formación v2 si acaso.

---

## 3. Decisiones (a ratificar por el Fixer)

| # | Decisión | Estado |
|---|---|---|
| D1 | Implementar **BF16 + SDPA** (Estrategia B fases 1-2 de RFC-VRAM-001) **durante la pausa actual**, con sus gates de validación, antes de reanudar. ✅ **RATIFICADA E IMPLEMENTADA** (27-jul, commits 5477127/1f55242/f0f28ad en `ref/school-v3-consolidation`, sin push). **Desviación documentada del RFC §4.4.1**: autocast con pesos maestros FP32 en vez de conversión total — el checkpoint no cambia de formato, FP32↔BF16 intercambiables por run (`--amp off|bf16|auto`), neurogénesis intacta; elimina los riesgos de conversión de checkpoint, GradScaler y dtype en neurogénesis de la matriz del RFC. 11 tests de gate + suite completa (136) en verde; compatibilidad verificada en CPU sobre copia de backup del checkpoint vivo (strict load, cos=0.9998, 96.85% acuerdo argmax) | HECHA |
| D2 | `torch.compile` selectivo (fase 3) queda **opcional**: solo si los gates de F2 van sobrados y el gradiente de BitLinear se verifica sano. No es requisito para graduar | PROPUESTA |
| D3 | El currículum de 8 años se completa **íntegro y sin cambios de fórmula** (ver §2.4). La graduación es el milestone `8_years` certificado por Samantha + batería M4 | PROPUESTA |
| D4 | **No se adelanta Formación v2.** Los gates del roadmap k65p (G1✅→G2→G3) siguen su curso en CPU, en paralelo. G4 exige el control graduado — sin excepciones | PROPUESTA |
| D5 | **Política de GPU exclusiva** — ✅ **YA IMPLEMENTADA** (verificado v1.1): la receta declara `preflight: vram_unload: true` + `min_free_vram_mb: 3500` (el proxy dual-bind vacía VRAM vía `POST /v1/unload` sin tumbar el servicio), y `train_school.sh:liberar_vram()` hace lo mismo en la vía autónoma, con parada/restauración de `redpill-llm` como fallback. Queda solo ratificarla como doctrina | IMPLEMENTADA |
| D6 | Checklist de lanzamiento — ✅ **YA IMPLEMENTADO** en `train_school.sh`: `systemd-inhibit` (Wake Gate), OOM shield `MemoryMax=16G` (10G disparaba el OOM killer a dim 896, commit 6b2db0b), troceo por epoch (`--max_epochs_per_run 1`) con checkpoint tras cada una, log por sesión. La receta del job-manager replica todo (`memory_max: 16G`, `max_step_minutes: 780` como detector de cuelgue) | IMPLEMENTADO |

---

## 4. Fases hacia la graduación

```
F0 auditoría ─ F1 BF16 ─ F2 SDPA ─┬─ F3 reanudar y cerrar 7_years (dim 896, ~122 ep)
                                  └─ (k65p G2→G3 en CPU, en paralelo)
F3 ─ F4 stage 8_years (neurogénesis 896→1024 bajo BF16, ~288 ep) ─ F5 GRADUACIÓN
F5 ─ F6 handoff: control congelado → k65p Fase 4 (Formación v2) → comparación G4
```

| Fase | Contenido | Gate de salida |
|---|---|---|
| **F0** | ~~Confirmar args~~ ✅ · ~~backup~~ ✅ (`backup_pre_bf16_20260727/`, md5 verificado) · ~~perfilado VRAM~~ ✅ integrado como telemetría por época en el trainer. Cadencia conocida en FP32: **3h20m-3h50m/epoch a dim 896** (docs/TRAINING_BIT.md §3) → la cifra post-BF16 la fija la primera ejecución | Cadencia BF16 medida en el primer run |
| **F1** | ✅ **CÓDIGO HECHO** (D1): `--amp auto` — autocast BF16 + master FP32, RMSNorm estadística en FP32, telemetría VRAM/∇STE. **Gate pendiente de correr**: ~100 epochs con curva de loss ±5% vs la historia FP32 de los logs | 100 epochs: loss ±5% vs baseline FP32; ∇STE > 0 |
| **F2** | ✅ **CÓDIGO HECHO** (D1): SDPA en `BitNetAttention` (sin STE en atención → seguro). Equivalencia ya probada por test contra oráculo manual (atol 1e-5) y sobre el checkpoint real | En el primer run: VRAM pico medida ≤ 4 GB |
| **F3** | Reanudar bajo D5+D6 y cerrar `7_years` (hasta epoch ~1120) | Milestone `7_years` certificado por Samantha |
| **F4** | Stage `secondary_8`: neurogénesis 896→1024 **en BF16** (RFC-VRAM §6 — conversión en CADA neurogénesis); acumulación dinámica solo si el margen aprieta | Neurogénesis estable; VRAM pico < 6 GB; loss converge |
| **F5** | **GRADUACIÓN**: milestone `8_years` + batería M4 completa → checkpoint del control **congelado y versionado** | Bit lingüístico graduado = artefacto de control de G4 |
| **F6** | Handoff al roadmap k65p Fase 4 (Formación v2, Bit nativo K-65P desde cero) | Gobernado por k65p ROADMAP (G3 + control graduado) |

**Estimación de calendario (provisional, F0 la confirma):** el histórico agregado sugiere
~2 min/epoch de media en stages tempranos (932 epochs ≈ 31h acumuladas el 22-jul), pero la
cadencia a dim 896 con el dataset del stage 7 NO está medida — la receta anota vidas de
proceso de 3-11h que son duración de proceso, no de epoch, y una probablemente degradada a
CPU tras un CUDA OOM. El driver del job-manager mide ahora la cadencia real por step: los
2 primeros epochs medidos fijan la cifra. Hasta entonces, ninguna promesa de horas; solo el
hecho estructural de que restan **410 epochs** y de que BF16 (F1) los abarata todos.

---

## 5. Lo que este RFC deliberadamente NO toca

- **Fórmula del currículum** (D3) — la tesis se defiende con la formulación con la que se entrenó.
- **Arquitectura del modelo** — BitMambaBlock, trit-loss, gating avanzado: siguen gateados
  por `bitnet_next_architecture_plan.md` y son POST-graduación.
- **K-65P v1** (variables, reglas, Sistema 1/2) — condicionado a ganar G4 (k65p ROADMAP Fase 5).
- **Paper 3** — se funda sobre los datos de G4, no antes.

---

## 6. Preguntas abiertas

1. ~~¿Con qué args se lanzó el run activo?~~ ✅ Resuelta (v1.1): defaults confirmados vía
   `configs/jobs/school.yaml` + `scripts/train_school.sh`.
2. ~~¿El daemon tiene modo descarga de VRAM?~~ ✅ Resuelta (v1.1): `POST /v1/unload` en el
   proxy dual-bind, ya cableado en preflight y en el runner.
3. **`min_free_vram_mb: 3500` es insuficiente para FP32.** A dim 896 el training pide
   ~6.7 GB; el preflight actual dejaría pasar una GPU con 4 GB ocupados y moriría en OOM al
   primer batch. Propuesta: subirlo a **7000** hasta que F1-F2 (BF16+SDPA) aterricen, y
   entonces sí bajarlo a ~4000. (Ajuste de una línea en la receta.)
4. ¿Benchmark FP32 vs BF16 de 100 epochs antes del run completo (recomendación Grok), o
   directamente el gate de F1? Propuesta Aleth: el gate de F1 ya ES ese benchmark.
5. ¿Dónde vive la copia final de este RFC una vez ratificado? Propuesta: `frankenswarm/docs/`
   (es doctrina de la criatura), con este archivo como espejo en Aleth_Core.

---

## 7. Referencias

- `Aleth_Core/RFC_VRAM_SCALING_BITNET_CURRICULUM.md` — RFC-BITNET-VRAM-001 v4 (Grok + DeepSeek)
- `Aleth_Core/NOTE_CURRICULUM_SCALING_REVIEW.md` — corregida por §2.4 de este RFC
- `Aleth_Core/bitnet_next_architecture_plan.md` v4 — gates de motor
- `k65p/docs/CORE/ROADMAP.md` — fases 1-5 del track nativo (G1 cerrado 2026-07-16)
- `frankenswarm/src/bitnet/training/train_sovereign_school.py` — `get_stage_config:459`, `get_next_dim:488`
- `frankenswarm/storage/checkpoints/sovereign_school/school_state.json` — estado vivo del run

---

## 8. Changelog

| Fecha | Autor | Cambio |
|---|---|---|
| 2026-07-27 | Aleth | Draft v1 — veredicto, diagnóstico verificado en código/estado, D1-D6, fases F0-F6 |
| 2026-07-27 | Aleth | v1.1 — args reales confirmados vía receta job-manager (`school.yaml` + `train_school.sh`, defaults → 1408 total); D5/D6 pasan de PROPUESTA a IMPLEMENTADA (vram_unload preflight, Wake Gate, MemoryMax=16G, troceo por epoch); F0 reducido a medir cadencia + backup; nueva Q3: `min_free_vram_mb` 3500→7000 hasta BF16 |
| 2026-07-27 | Aleth | v1.2 — **D1 ratificada e implementada** (3 commits en frankenswarm): F0 cerrado (backup + args + cadencia FP32 conocida: 3h20-3h50/epoch a dim 896), F1/F2 código hecho con desviación documentada del §4.4.1 (autocast + master FP32 en vez de conversión), Q3 resuelta en la receta (`min_free_vram_mb: 4500` con nota para `--amp off`), Q4 resuelta (el gate F1 ES el benchmark: mismos checkpoints, flag por run). Próximo paso: reanudar el job y vigilar las primeras ~100 epochs |

---

*El conexionismo propone, el simbolismo dispone — pero primero, el niño termina la escuela.*
