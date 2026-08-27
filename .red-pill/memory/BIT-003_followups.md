# BIT-003 · TODOs pendientes post-lanzamiento

## 1. Optimización del formato del corpus en RAM (apuntado 28-ago, operador)
- Problema: el corpus tokenizado en memoria ocupa ~12-14 GB (42.3M secuencias como list[int] de Python: cabeceras 56B + punteros + ints huérfanos). Proceso total ~20 GB con torch/CUDA + fragmentación del json.load.
- Propuesta: formato CSR plano numpy (array de tokens int16 + array de offsets). Vocab 19.637 cabe en int16 → corpus baja de ~12 GB a ~0.8 GB (10×). Alternativa: HF datasets memory-mapped (arrow).
- CUÁNDO: NO a mitad del run glyph actual (alteraría el instrumento). Aplicar antes del brazo standard o de los runs K-65P, con commit aparte + validación (audit_stage_gating 22/22 + smoke + verificación de que las secuencias clasificadas son idénticas).
- El gateo NO reduce RAM por diseño: gated_general referencia las mismas secuencias (reordenación, no filtrado).

## 2. Migración del run glyph a la cola de jobs (28-ago)
- El primer lanzamiento fue shell directo (systemd-inhibit + systemd-run, receta §F8 del fix plan) — fallo del agente: no se planteó el job manager.
- Creado configs/jobs/bit003_glyph.yaml (script_job, memory_max 24G, max_epochs_per_run 20 para amortizar compile, pause_exit_code 78, preflight VRAM que se difiere solo si el sueño ocupa la GPU).
- Falta: crear bit003_standard.yaml para el brazo control (idéntico con --embedding standard) y lanzarlo en serie al terminar glyph.

## Estado del run (28-ago ~01:10)
- Venv reconstruido (uv sync --frozen, python 3.12.12, torch 2.13.0+cu130) tras rotura por upgrade del sistema 3.13→3.14.
- Reanudado desde época 5 etapa 0-1; Loss descendiendo (2.18→2.14), Val 2.17→2.12; ~40-50s/época; GPU 2.1GB/53%.
- Corpus clasificado: E0:1.7M E1:5.4M E2:17.0M E3:6.2M E4:5.3M E5:1.7M E6:2.1M E7:2.9M.
- Lección: cargar PYTHONUNBUFFERED=1 para monitorizar; MemoryMax>=24G por la carga de caché (json.load 2.48GB → ~18-20GB objetos).