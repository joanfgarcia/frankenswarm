# Session Snapshot: Frankenswarm (Sovereign School & Tensors as API)

Este snapshot documenta el estado técnico para permitir reiniciar la sesión limpiando el contexto.

## 1. Diccionario de Términos / Alias Técnico
*   **banco de memoria** ➔ `.red-pill/memory/` (Almacenamiento local de artefactos y coordinación).
*   **escuela soberana / sovereign school** ➔ `src/bitnet/train_sovereign_school.py` (Script de entrenamiento curricular progresivo).
*   **samantha** ➔ Evaluador cognitivo integrado en `src/bitnet/train_sovereign_school.py` (psicóloga infantil LLM-as-a-judge).
*   **nico / sofy** ➔ Agentes de supervivencia en la arena cooperativa (`scratch/train_arena_unified.py`).
*   **bit** ➔ `src/bitnet/modeling_bitnet.py` (Clase `BitNet4LayerModel`).
*   **tensors as api** ➔ Mecanismo de despacho y delegación intra-forward mediante vocabulario común por IPC/UNIX sockets sin capas de proyección.

## 2. Mapa de Arquitectura Técnica
*   **Backbone**: `BitNet4LayerModel` con cuantización ternaria 1.58-bit y embeddings composicionales basados en glifos (`configs/expanded_glyphs.json`).
*   **Entrenamiento Progresivo**: Ensanchamiento dinámico (neurogénesis en caliente) de `hidden_dim` mediante Net2Net, guardando checkpoints específicos para cada hito de edad (2, 3, 4 años...).
*   **Despacho Distribuido**: Servidor cognitivo (`distributed_cognitive_server.py`) y cliente de entrenamiento cooperativo (`train_arena_distributed.py`) comunicándose por Sockets UNIX en `/tmp/bit_cognitive.sock` enviando y recibiendo distribuciones de logits.

## 3. Registro de Decisiones Técnicas (Log)

| Prioridad | Decisión | Razón | Estado |
|---|---|---|---|
| Alta | Sockets UNIX para MoE | Comunicación inter-proceso de tensores `[batch, seq, 15005]` eficiente y ligera en la misma máquina física. | Completado |
| Alta | Compresión Top-K en Despacho | Reducir el payload a 1.6 KB ($k=50$) reteniendo un 0.969 de similitud del coseno. | Completado |
| Media | Ampliación a 512-dim (Fase 3) | Aumentar capacidad semántica para el hito de 4 años, manteniendo compatibilidad de logits. | Completado |
| Media | Automatización de Resumen y Graduación | Resumir el estado escolar en `school_state.json` y pausar al graduarse. | Completado |

## 4. Última Frontera (Checkpoint)
1.  **Hito de 4 Años Superado**: El bucle de entrenamiento curricular (`task-1370`) finalizó en la Época 448 (Loss: 2.2239, Val Loss: 6.4429). Se guardó el modelo en `storage/checkpoints/sovereign_school/model_milestone_4_years.pt`.
2.  **Verificación Estructural**: Validamos programáticamente el checkpoint con `strict=True` cargándolo en GPU, verificando que la matriz de pesos concuerda exactamente con 512 dimensiones y genera oraciones coherentes basadas en el diccionario de español del currículo.
3.  **Fase 5 Preparada**: El archivo `school_state.json` ha sido actualizado automáticamente por el entrenador con `current_stage_idx: 4`, `current_epoch: 449` y `target_milestone: "5_years"`. Al reanudar el entrenamiento, ejecutará automáticamente Net2Net para ensanchar el modelo a 640 dimensiones.

*   **Siguiente Paso**: Reanudar el entrenamiento de la Fase 4 (Hito de 5 años, 640-dim, 192 épocas, epochs 449 a 640) usando:
    `PYTHONPATH=. systemd-run --user --scope -p MemoryMax=10G .venv/bin/python src/bitnet/train_sovereign_school.py`
