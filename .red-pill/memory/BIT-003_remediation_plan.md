# BIT-003 · Plan de remediación post-auditoría (1-sep-2026, DL-015)

Lo EJECUTADO hoy (6 commits en feat/bit003-corpus-csr, suite 444 verde):
runner de batería corregido + 6 baterías re-tiradas · contrato de exit-codes
infra≠suspenso (con tests y señales por brazo) · banks regenerados con
in-gate vivo (439) · batería v2 congelada (51/169, banks en el universo) ·
módulos DL-014 endurecidos · CHANGELOG con 2 retractaciones + DL-015 +
followups/RFC-002/recetas corregidos · receta `bit003_glyph_v41_2y_x1`.

## Fases pendientes (orden de ejecución)

### F1 — Cierre de la confrontación v4 (bloqueado por GPU, automático)
- El standard v4 sigue corriendo hacia 4_years, YA protegido por el contrato
  de exit-codes (desde su próximo step). Al aprobar:
  `run_battery_v3.py --model_path .../bit003_standard_v4_x1/model_milestone_4_years.pt --hidden_dim 128 --device cpu --pool_gate`
- Primera comparación HOMÓLOGA glyph-4y(256d) vs standard-4y(128d) →
  actualizar el [RESULT] del CHANGELOG con el veredicto final. Regla: las
  diferencias de cognición <±7pp (n=189) se declaran ruido.

### F2 — Re-examen del hueco: ¿pasa el glyph 2_years a 128d bajo ×1?
- `red-pill job submit --recipe bit003_glyph_v41_2y_x1` (tras F1; GPU libre).
- Si APRUEBA → retractación definitiva de "el glyph necesitó 256d"; si
  SUSPENDE con nota real → la escalera de neurogénesis se ejercita
  legítimamente por primera vez. Ambos resultados cierran DL-015.

### F3 — Cablear los banks (DL-013, "siguiente generación de runs")
- Trainer: inyectar el bank de la etapa en compile_data_for_stage
  (×exam_repeat_factor) y muestrear exámenes de hito del bank acumulado con
  peso por recencia 50/20/15/5, 10 formas por hito (media±σ).
- Desde ese momento la batería es `age4_v2.json` (el runner acepta
  `--battery_path`; considerar cambiar el default cuando arranque la gen v5).
- Protocolo anti-regresión OBLIGATORIO: test unitario del muestreo, smoke 1
  época vía receta, verificación de cierre de etapas.
- Verificación de potencia: con exámenes muestreados, comprobar que los dos
  brazos DEJAN de dar respuestas token-idénticas (hoy el examen de hito no
  discrimina representaciones).

### F4 — Diccionario K-65P reproducible
- `scripts/populate_k65p_dictionary.py`: 65 primes + 28 canonical + 99
  moléculas + phrases, idempotente. Hoy la BD (19.703 words) tiene 25
  canonical / 41 prime / 0 molecule — inconsistente con la prosa de DL-014;
  la población fue ad-hoc y no hay script en el repo.
- Aceptación: stats() cuadra con los números doctrinales y el test de
  ida-vuelta RFC-002 §5 pasa desde la BD (no desde tablas en memoria).

### F5 — Descomposición NSM por lotes (DL-014, paso siguiente)
- Generador de explicaciones por lotes sobre `pending` (19.610 sin glifo),
  con el guard de estados curados (force=True solo curación del operador),
  validador de consistencia Jaccard y curación por muestreo.
- El traductor EN→K-65P (compilador transductor) consume el léxico curado
  hoy (65/65 primos con superficie; canónica > alias).

### F6 — Canal de silencio (abstención) — RFC antes de tocar nada
- Dato nuevo (curvas reales): la calibración EXISTE dentro de lo visto
  (top-20% → 90-100%) y es casi nula en lo no visto (0-5.4%). El diseño
  debe premiar "no sé" ante lo no visto — que es justo donde la señal es
  débil: no es un umbral trivial de softmax.
- Opciones a evaluar en RFC: token de abstención entrenado con la matriz
  3×2 · umbral de confianza calibrado por etapa · sampling del "i dont
  know" ya en vocab. Pre-registrar criterio de éxito ANTES del run.

### F7 — Housekeeping (decisiones del operador)
- Borrar los `*.pre-audit-bak` de storage/checkpoints cuando Joan valide
  los nuevos JSONs.
- Push de la rama + squash: decisión de Joan (regla de oro git).
- `scripts/milestone_battery.py`: revisar si quedó obsoleto tras el runner
  v3; retirar o alinear.
- El dir `bit003_glyph_res` (brazo resonante DL-010) sigue sin estrenar —
  sigue en la recámara, sin cambios.

## Doctrina sellada en DL-015
1. Un fallo de infraestructura JAMÁS es una calificación.
2. Todo número citado en prosa debe existir como artefacto persistido y
   reproducible (el "12%" murió por esto).
3. Un set congelado se versiona cuando cambia el universo entrenado
   (v1 pre-banks / v2 con banks) — nunca se regenera en sitio.
