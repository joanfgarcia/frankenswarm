# Plan de Implementación: Despacho de Tensores Intra-Forward en el Playground de Supervivencia (Auditado)

Este plan detalla el diseño, la creación de prototipos y la integración de la arquitectura **Intra-Forward Distributed Dispatch (Tensors as API)**, adaptando el playground de supervivencia de los monitos para usar el vocabulario base unificado de 15,005 palabras de Bit.

## Auditoría del Diseño y Decisiones

### 1. El Espacio de Glifos como API (Traducción Dimensional)
El principal reto es comunicar un modelo de `hidden_dim = 256` (Modelo A) con uno de `hidden_dim = 384` (Modelo B).
- **Decisión**: Para evitar añadir capas de proyección lineales entrenables (lo que requeriría entrenamiento conjunto y rompería la modularidad), utilizaremos la **distribución de probabilidad sobre el vocabulario base de 15,005 palabras** como el "bus de datos".
- **Algoritmo**:
  1. El Modelo A procesa la entrada hasta la Capa 3 (bloque 3).
  2. Obtenemos el tensor latente $h_A \in \mathbb{R}^{B \times S \times 256}$.
  3. Decodificamos $h_A$ a logits del vocabulario de 15k usando la matriz composicional de A:
     $$logits_A = \text{decode\_logits}_A(h_A) \in \mathbb{R}^{B \times S \times 15005}$$
  4. Aplicamos Softmax para obtener la distribución semántica:
     $$probs_A = \text{softmax}(logits_A / \tau, \text{dim}=-1)$$
  5. Proyectamos al espacio de B multiplicando $probs_A$ directamente por la matriz de embeddings de B (sin añadir sus positional embeddings para evitar redundancia posicional):
     $$h_B = probs_A \times \text{word\_embeddings}_B \in \mathbb{R}^{B \times S \times 384}$$
  6. El Modelo B procesa $h_B$ a través de sus bloques Transformer intermedios y devuelve $h_{B, \text{out}}$.
  7. Decodificamos $h_{B, \text{out}}$ a logits en B, aplicamos softmax, y re-inyectamos en A usando los embeddings de A:
     $$h_{A, \text{in}} = \text{probs}_B \times \text{word\_embeddings}_A \in \mathbb{R}^{B \times S \times 256}$$
  8. El Modelo A ejecuta sus capas restantes y genera la salida.

### 2. Evitar la Redundancia de Positional Embeddings
> [!IMPORTANT]
> Al re-embeber el vector de probabilidades en el Modelo B, **no** debemos llamar a `model_b._embed_input()`, ya que esto sumaría los embeddings posicionales del Modelo B encima de la información posicional que el Modelo A ya integró en la secuencia.
> **Solución**: Multiplicaremos directamente por la matriz de embeddings composicionales obtenida de `glyph_embedding.get_word_embeddings()`. Esto preserva la pureza semántica de la secuencia.

### 3. Playground de Entrenamiento: ¿Populora o Survival?
> [!IMPORTANT]
> **No utilizaremos Populora** para el entrenamiento de supervivencia.
> *Populora* (`train_populora.py` / `dojo_populora.py`) es un juego puramente referencial de comunicación simbólica (adivinar coordenadas/colores) y carece de bucle de supervivencia física, estados corporales (hambre, energía, salud) o acciones espaciales.
> *Survival Playground* (`minimal_world.py`) es el entorno adecuado. Adaptaremos este playground para mapear sus percepciones (`agua`, `comida`, `fuego`, `peligro`) y acciones (`comer`, `beber`, `dormir`, `mover`, `ver`, `piedra`) directamente a los índices de la tabla de 15,005 palabras de Bit.

---

## Proposed Changes

### Component: Frankenswarm Scratch & Core

#### [NEW] [test_intra_forward_dispatch.py](file:///home/joan/Documents/IA/frankenswarm/scratch/test_intra_forward_dispatch.py)
Script en memoria que:
1. Carga `model_milestone_2_years.pt` (dim 256) y `model_milestone_3_years.pt` (dim 384).
2. Modifica el paso forward para interrumpir en la capa 3 de A, proyectar a B, ejecutar capas 2-4 en B, re-proyectar a A y terminar.
3. Evalúa la coherencia de los logits de salida en oraciones de prueba del currículo escolar.

#### [NEW] [train_survival_unified.py](file:///home/joan/Documents/IA/frankenswarm/scratch/train_survival_unified.py)
Script que:
1. Configura el entorno `MinimalWorld` para que use el vocabulario de 15,005 palabras.
2. Carga el Modelo A (`model_milestone_2_years.pt`) con su backbone congelado.
3. Entrena la cabeza de acción (`action_head` de dim 256 a 6) utilizando REINFORCE en el playground unificado.
4. Soporta la inyección del despacho de tensores a Model B (`model_milestone_3_years.pt`) para delegar el razonamiento intermedio.

---

## Verification Plan

### Automated Tests
1. **Verificación matemática del prototipo de despacho**:
   ```bash
   PYTHONPATH=. .venv/bin/python scratch/test_intra_forward_dispatch.py
   ```
2. **Entrenamiento de supervivencia con el modelo híbrido (despacho activo)**:
   ```bash
   PYTHONPATH=. systemd-run --user --scope -p MemoryMax=10G .venv/bin/python scratch/train_survival_unified.py --config configs/experiments/EXP_039_actionhead.json --episodes 50
   ```
