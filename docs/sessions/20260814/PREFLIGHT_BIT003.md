# Preflight BIT-003 — Auditoría de coherencia antes del reentrenamiento de v1 en inglés

**Fecha**: 2026-08-14 · **Rama**: `feat/v1-english-rebuild` · **Auditor**: Aleth (revisión
del trabajo reciente hecho con modelos menores, por orden del operador).

**Contexto**: el 12-ago se descubrió que el corpus de v1 era spanglish (CHILDES-es +
traducción "fast" rota) y el operador decidió reentrenar v1 desde cero en inglés puro
(`[BIT-003]` en `~/Documents/IA/Aleth_Core/TODO.md`). El 13-ago entró a main el PR #5
(a5799c7): refactor modular del trainer inglés + IOC + `--opt8bit`. **Ese trainer
refactorizado jamás había corrido una escuela real** — los brazos ingleses del informe
del 5-ago corrieron con el código pre-refactor. Esta auditoría es el preflight.

## Verificado ✅

- **Suite completa en verde**: 389 tests pasan sobre a5799c7.
- **Protocolo adaptativo DL-006 en el trainer inglés** (`--adaptive`): entrena hasta
  plateau (`--patience`, tope `--max_stage_epochs`), examina sobre el MEJOR checkpoint
  de la etapa, avanza desde él, neurogénesis solo como remediación con techo de dim de
  etapa, y pausa rc=78 sin techo. Fiel a DL-006.
- **Brazo estándar en neurogénesis**: `trigger_neurogenesis` y `net2wider_model`
  respetan el modo de embedding (reconstruyen con `vocab_embeddings`, no con glifos).
- **Evaluador Samantha**: autodetecta el brazo por las claves del checkpoint
  (`glyph_embedding.*`) — no hace falta pasar flag y no rompe `load_state_dict`.
- **Caché tokenizada**: keyeada por hash de corpus (se invalida sola al cambiar
  currículo/vocabulario).
- **Flags DL-002 presentes**: `--amp auto`, `--compile`, `--state_dir`, `--seed`,
  `--require_gpu` (rc=3).

## Arreglado en esta sesión 🔧 (commits locales en `feat/v1-english-rebuild`, sin push)

1. **Caché de etapa sin clave de versión (mina para BIT-003)**. `storage/datasets/
   stage_cache/stage_N_compiled.json` era global y se reutilizaba sin validar contra
   corpus/vocabulario: al regenerar el currículo inglés, el trainer habría servido en
   silencio los datos spanglish viejos — o token-IDs de OTRO censo (corrupción muda).
   Fix: la caché vive ahora en `stage_cache/<corpus_hash>/`.
2. **`school_exams_en.json` fuera del hash de corpus**. Las secuencias de examen se
   mezclan ×300 en el dataset de etapa: un cambio de exámenes no invalidaba nada.
   Añadido al `compute_corpus_hash` (esto invalida una vez la caché tokenizada
   existente; re-tokenizar es el precio de la honestidad).
3. **`datasets` no estaba declarado ni instalado**. El trainer lo importa para
   TinyStories; con caché tokenizada inválida (lo primero que hará BIT-003) el run
   moría en `ModuleNotFoundError` a los 3 segundos. Añadido a `pyproject.toml` vía
   `uv add datasets` (resuelve limpio junto a fastembed 0.8.0; el conflicto HF
   histórico era con `transformers`, que sigue fuera).
4. **Acta de suspensos machacada en modo adaptativo**. `run_samantha_eval` incrementa
   `exam_failures` en el fichero de estado, pero `_save_adaptive_state()` lo
   sobreescribía con el valor viejo en memoria. Sincronizado desde `eval_result.state`.
5. **`select_strategy` no entendía `--amp auto`** y caía a la estrategia `fp32`. Hoy
   solo afecta al log y a la elección de optimizador 8-bit (el autocast va por otra
   vía), pero mentía en el acta del run. Se le pasa el modo ya resuelto.
6. **`scripts/download_childes_en.py`** (BIT-003 §1): descarga CHILDES-en de la fuente
   verificada del TODO, **sin** `map_to_base_word` (RULE 7: prohibido el mapeo OOV
   ciego). Escribe `configs/childes_pre_school_en.json`; el fichero español no se toca.
7. `lab/BRIEFING.md` actualizado (estaba anclado al 3-ago y a una rama ya mergeada).

## Pendiente para arrancar BIT-003 ⏳ (decisiones de diseño, no las tomo en solitario)

- **Currículo inglés limpio** (BIT-003 §3): regenerar `school_curriculum_structured_en.json`
  con traducción de calidad desde el currículo anotado. La versión actual
  (`v2.0-structured-en-fast`, sin sha256 en metadata) es el spanglish a reemplazar.
  El trainer lo hardcodea; al reemplazar el fichero, las cachés ya se invalidan solas
  (fix 1-2).
- **Censo de vocabulario con CHILDES-en** (BIT-003 §2): `rebuild_english_vocabulary.py`
  hoy censa diálogos + TinyStories; hay que añadir `childes_pre_school_en.json` como
  fuente y decidir el umbral de frecuencia. Cambiar el censo cambia los token-IDs →
  estado nuevo desde cero (ya previsto: el rebuild parte de cero).
- **Cómo entra CHILDES-en al trainer**: el trainer actual no consume CHILDES en
  ninguna parte (el modo `childes_only` filtra el corpus "general", que es
  TinyStories + diálogos). Decidir si CHILDES-en entra al bloque general preescolar
  o como fuente nueva del partitioner MLU.
- **Whitelist del evaluador**: `evaluate_samantha_age.py` carga el
  `childes_pre_school.json` ESPAÑOL en la whitelist de edad. Es un instrumento
  congelado (DL-004) — cambiarlo a la versión `_en` es decisión de operador y debe
  registrarse como DL-00X antes del primer examen del rebuild.
- **Receta de job**: `configs/jobs/school.yaml` describe el run viejo de calendario
  fijo (progress total 1408, checkpoint del run vivo). El rebuild necesita recetas
  nuevas tipo `school_en_glyph_s770.yaml` con `--adaptive --embedding {glyph,standard}
  --state_dir --seed --require_gpu --amp auto --compile` y `pause_exit_code: 78`.
- **`--opt8bit` + neurogénesis**: el mapeo de estados de optimizador de `net2wider`
  con AdamW8bit (formato cuantizado, no `exp_avg` FP32) está sin verificar. NO usar
  `--opt8bit` en el rebuild hasta que tenga un test A/B propio; la receta DL-002
  certificada es `--amp auto --compile` sin 8-bit.
- ~~**Checkpoints del brazo estándar: 607 MB CADA UNO**~~ → **RESUELTO en DL-007**
  (2026-08-14): la tabla identidad dejó de persistirse y los matmuls contra ella se
  cortocircuitaron. Checkpoints 579 → 16,6 MB y el step 12,1× más rápido. De paso
  destapó que la baza de coste del glifo del informe del 5-ago medía la
  implementación del baseline: **retirada**. Ver DL-007.
- **`--wd_mode no_embed` sin medir** (DL-007): es el trato de weight decay simétrico
  entre brazos y el recomendado para esta comparativa, pero cambiarlo hace que las
  épocas-hasta-hito no sean directamente comparables con las réplicas DL-006.
  Decidir un A/B corto antes del run largo.

## Smoke tests del trainer refactorizado ✅ (14-ago, sandbox, seed 770, mock exam)

`--adaptive --test_mock`, etapas forzadas a 2 épocas (`--max_stage_epochs 2
--patience 1`), GPU BF16. Sin este humo, el primer run real de BIT-003 habría sido
el estreno del refactor con días de GPU en juego. Resultados:

- **Brazo glyph**: ciclo completo OK — guardería superada por plateau, examen sobre
  el mejor checkpoint, hitos 2/3/4 años aprobados, avance de etapa, pausa planificada
  (`--max_epochs_per_run`) y **reanudación** desde el estado guardado verificada
  (recarga en 3-4, continúa, examina, avanza). Exit 0 en las tres ejecuciones.
- **Brazo estándar** (`--embedding standard`, vocab 12.143): mismo ciclo OK; el
  evaluador autodetectó el brazo y el guard de `embedding_mode` en el estado
  funciona. Val_loss menor que el glyph en todas las etapas (consistente con la
  lectura 2 del informe DL-006). VRAM ~3,6 GB a 128d.
- El primer intento murió en `ModuleNotFoundError: datasets` — el fix 3 salió de
  aquí. Un smoke barato pagó la sesión.
