# Depth vs Width en Transformers Ternarios: Revisión de Literatura

> Estado: Revisión de literatura para decidir estrategia de crecimiento del modelo Bit
> Contexto: Época 1269, hidden_dim=1024, 6 capas, 76.9M params, val_loss=4.3988, plateau estancado
> Fecha: 2026-07-30

---

## 1. Estudios Generales (Transformers FP16)

### 1.1 Petty et al. — NAACL 2024
**"The Impact of Depth on Compositional Generalization in Transformer LMs"**

| Aspecto | Detalle |
|---|---|
| Método | Models de 41M, 134M, 374M params intercambiando depth ↔ width a params constantes |
| Resultado 1 | Modelos más profundos generalizan **más composicionalmente** que los anchos, a igual parámetros |
| Resultado 2 | El beneficio de depth **no se debe solo** a mejor perplejidad — depth aporta un efecto independiente |
| Resultado 3 | Rendimientos decrecientes: la 1ªs capas aportan la mayor parte del beneficio |
| Cita clave | *"[depth] scaling by depth is generally more helpful than scaling by width on downstream tasks"* (Tay et al. 2021) |
| Link | https://aclanthology.org/2024.naacl-long.402/ |

### 1.2 Saunshi et al. — NeurIPS 2025
**"Reasoning with Latent Thoughts: On the Power of Looped Transformers"**

| Aspecto | Detalle |
|---|---|
| Método | Transformer de k capas looped L veces vs kL capas no-loop |
| Resultado 1 | k capas looped L veces **igualan** a kL capas en razonamiento |
| Resultado 2 | 12 capas looped 2× **supera** a 24 capas en matemáticas (34.3 vs 29.3) con **mitad de parámetros** |
| Cita clave | *"scaling laws for reasoning are more subtle; depth is very important in addition to parameter count — at the same parameter count, deeper but narrower models are better"* (Ye et al. 2024) |
| Link | https://arxiv.org/abs/2502.17416 |

### 1.3 "Disentangling Recall and Reasoning" — OpenReview 2025
**Across Qwen, LLaMA-3, Mistral**

| Aspecto | Detalle |
|---|---|
| Hallazgo | Early/middle layers → **recall**; deeper layers + MLPs específicos → **reasoning** |
| Implicación | Profundidad NO es redundante — distintas capas tienen roles funcionales distintos |
| Link | https://openreview.net/forum?id=2hnWkP8iDe |

### 1.4 "Layer Specialization Underlying Compositional Reasoning in Transformers" — 2025
**Causal vs Masked LMs**

| Aspecto | Detalle |
|---|---|
| Hallazgo | La especialización por capas (no su ubicación concreta) es el requisito crítico para razonamiento compositivo |
| CLM | Capas tempranas procesan abstracción compositiva; capas profundas integran contexto y desambiguación |
| Link | https://arxiv.org/html/2510.17469 |

### 1.5 "Depth Delusion" — arXiv 2601.20994 (Jan 2026)
**"Why Transformers Should Be Wider, Not Deeper"**

| Aspecto | Detalle |
|---|---|
| Método | 30 arquitecturas, 17M a 7B params, R²=0.922 |
| Fórmula | Depth óptima: **D* ~ C^0.12** | Width óptima: **W* ~ C^0.34** |
| Interpretación | Width debe crecer **2.8× más rápido** que depth |
| Fenómeno crítico | Más allá de D_crit ~ W^0.44, añadir capas **empeora la loss** a pesar de añadir parámetros |
| Ejemplo | 7B: 64 layers (6.38B) **underperforma** 32 layers (6.86B) por 0.12 nats |
| Advertencia | Este estudio es sobre FP16, no ternario. Los resultados pueden no transferir directamente |
| Link | https://arxiv.org/abs/2601.20994 |

### 1.6 Hayou et al. — ICML 2023
**"Width and Depth Limits Commute in Residual Networks"**

| Aspecto | Detalle |
|---|---|
| Método | Demostración matemática de que en residual networks, los límites width→∞ y depth→∞ conmutan |
| Resultado | A inicialización, el orden de los límites no importa |
| Limitación | *"Our technique cannot say anything about what happens when the network starts training"* |
| Implicación | Para inicialización de Net2DeeperNet, no hay diferencia en qué orden se expanda |
| Link | https://proceedings.mlr.press/v202/hayou23a/hayou23a.pdf |

---

## 2. Model Expansion / Progressive Training

### 2.1 Bu et al. — Meta FAIR 2025
**"Scaling Depth Capacity via Zero/One-Layer Model Expansion"**

| Aspecto | Detalle |
|---|---|
| Método | Expansión progresiva en profundidad durante training (GPT2) |
| Resultado | Depth expansion ahorra **~80% compute** vs entrenar modelo grande desde cero |
| Ventaja adicional | Learning rate se transfiere sin re-tuning (hyperparameter transfer) |
| Inicialización | Nuevas capas como identidad → no se altera la función representada |
| Link | https://arxiv.org/pdf/2511.04981 |

### 2.2 Du et al. — 2024
**Depth up-scaling vs width up-scaling**

| Aspecto | Detalle |
|---|---|
| Resultado | Depth up-scaling → **mayor eficiencia de entrenamiento** y **mejor rendimiento downstream** que width up-scaling |
| Contexto | Experimentos con decoder-only LLMs |
| Link | (citado en OpT-DeUS, OpenReview 2025) |

### 2.3 SENN (Mitchell, Mundt, Kersting) — NeurIPS 2023
**"Self-Expanding Neural Networks"**

Framework que decide dinámicamente si añadir width o depth basado en el **natural expansion score η**.

#### Definición del Natural Expansion Score

```
η = gᵀ · F⁻¹ · g
```

Donde:
- **g**: gradiente del loss respecto a los parámetros
- **F**: matriz de información de Fisher (F = (1/N)·Jᵀ·J)
- **η**: mide la tasa de reducción del loss bajo gradient descent natural

#### Interpretación física

‖P_Θ · g_y‖²₂ = N · η

η es la **norma al cuadrado de la proyección del gradiente en output-space** sobre el subespacio alcanzable por los parámetros actuales. Mide cuánto del gradiente "deseado" puede ser expresado por la parametrización actual.

#### Algoritmo de decisión

| Pregunta | Criterio |
|---|---|
| **¿Cuándo?** | Cuando η_p / η_c > τ (ej. τ=2: cada adición debe duplicar la tasa de reducción) |
| **¿Dónde?** | Donde la expansión maximice η — puede comparar width vs depth directamente |
| **¿Qué?** | Inicialización que maximice η' (argmax de η sobre posibles inits) |
| **Parada** | Cuando η_p − η_c ≤ α (criterio de parada absoluto) |

#### Cota superior de adiciones

```
N_s < 1 + (ln λ − ln α) / ln τ
```

Donde λ = ‖g_y‖²₂ (cota superior de η) y α es el criterio de parada.

Ejemplo con τ=2, α/λ > 10⁻³: N_s < **11 adiciones**.

#### Aproximación eficiente (Kronecker Factorization)

Para capa l con pesos W:
```
F̂_l = S_l ⊗ A_l
F̂_l⁻¹ = A_l⁻¹ ⊗ S_l⁻¹
η = Tr[∂Wᵀ · S⁻¹ · ∂W · A⁻¹]
```

Donde A_l = segundo momento de las activaciones de entrada, S_l = segundo momento de los gradientes de salida.

Esto permite calcular η por capa de forma eficiente sin invertir la Fisher completa.

#### Ventajas sobre GradMax / Firefly

| Método | ¿Cuándo? | ¿Dónde? | ¿Qué? | ¿Depth? |
|---|---|---|---|---|
| GradMax (Evci 2022) | Future work | Future work | Vanilla gradient | No |
| Firefly (Wu 2020) | Cada N epochs | Vanilla gradient | Loss reduction | No |
| **SENN** | **η score** | **η score** | **η score** | **Sí** |

| Link | https://arxiv.org/abs/2307.04526 |

---

## 3. Estudios Específicos de BitNet / Ternario

### 3.1 Microsoft — BitNet b1.58 (Ma et al. 2024)
**"The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits"**

Configuraciones publicadas de BitNet b1.58 (modelos reales entrenados):

| Modelo | Dim (W) | Layers (D) | Ratio W/D | Params |
|---|---|---|---|---|
| 125M | 768 | 12 | **64** | 125M |
| 350M | 1024 | 16 | **64** | 350M |
| 1.3B | 2048 | 24 | **85** | 1.3B |
| 2.7B | 2560 | 32 | **80** | 2.7B |
| 6.7B | 4096 | 36 | **114** | 6.7B |
| 13B | 5120 | 40 | **128** | 13B |
| 30B | 7168 | 48 | **149** | 30B |
| **Bit nuestro** | **1024** | **6** | **170** | **76.9M** |

Observación: Ningún modelo BitNet publicado tiene una relación W/D tan alta como la nuestra (170).

### 3.2 Nielsen et al. — 2024
**"BitNet b1.58 Reloaded: State-of-the-art Performance Also on Smaller Networks"**

| Aspecto | Detalle |
|---|---|
| Hallazgo principal | BitNet ternario funciona también en modelos pequeños con ajuste cuidadoso |
| Guideline 1 | *"Small models require careful tuning (median scaling, higher width, hyperparameter grid)"* |
| Guideline 2 | Para algunos modelos, duplicar hidden_size es suficiente; para otros no hace falta |
| Implicit regularization | La cuantización ternaria (zeroing de pesos pequeños) actúa como regularizador implícito — reduce overfitting |
| Limitación | Modelos muy pequeños pueden sufrir pérdida de rendimiento a menos que se aumente width o network size |
| Link | https://arxiv.org/abs/2407.09527 |

---

## 4. Síntesis y Decisión Experimental

### 4.1 ¿Qué dice la literatura sobre nuestro caso?

- **Nuestra ratio W/D = 170** está fuera de todas las configuraciones BitNet conocidas (máx 149 en 30B)
- **Depth generalización compositiva**: respaldado por Petty, Saunshi, Ye, Du — todos convergen en que profundidad ayuda a razonamiento
- **Width memorización**: consistente con nuestra observación empírica (más dim → el modelo memoriza mejor pero no razona)
- **El ∇STE está sano** (~0.002): añadir capas no causará vanishing gradients
- **Inicialización a identidad**: las nuevas capas preservan la función actual (Hayou, SENN, Bu)
- **"Depth Delusion" NO juega en contra**: su umbral crítico es D_crit ~ W^0.44. Con W=1024, D_crit ≈ 1024^0.44 ≈ **21 capas**. Ir de 6 → 8 → 10 queda muy lejos de la zona donde añadir capas empeora la loss. El único paper aparentemente contrario en realidad valida el plan.

### 4.2 Hipótesis experimental

> **Si entrenamos primero width hasta el plateau (hecho: 1024 dim, val_loss=4.40 estancada), y luego expandimos depth manteniendo width fija, la base de memorización construida por width permitirá que las nuevas capas profundas aprendan abstracción semántica más rápido que si hubiéramos entrenado depth desde cero.**

Fundamento:
- Width ha construido representaciones ricas de patrones sintácticos (el modelo genera frases gramaticales perfectas)
- Depth añade capacidad de componer esas representaciones jerárquicamente
- Las capas nuevas se inicializan a identidad → no se pierde el conocimiento adquirido
- El training existente sirve como "pre-training" para las nuevas capas

### 4.3 Plan propuesto

1. Terminar esta fase de entrenamiento (alcanzar época 1408 o examen de 8 años)
2. Implementar Net2DeeperNet (inicialización a identidad para nuevas capas transformer)
3. Expandir de 6 → 8 capas manteniendo hidden_dim=1024
4. Continuar entrenamiento evaluando:
   - ¿La val_loss desciende más rápido que antes?
   - ¿Las muestras cualitativas muestran mayor profundidad semántica?
   - ¿El examen de 8 años mejora?

### 4.4 Posibles variantes a experimentar

| Orden | Descripción | Riesgo |
|---|---|---|
| Width → Depth (nuestro caso) | 1024 dim → luego añadir capas | Ya estamos aquí, bajo riesgo |
| Depth → Width | Pocas capas, expandir width después | No tenemos datos |
| Width+Depth simultáneo | Crecimiento alternado | Más complejo de evaluar |
| SENN dinámico | Dejar que η decida cuándo/dónde expandir | Requiere implementar Fisher |

---

## 5. State Space Models (SSM) para Arquitecturas Híbridas

### 5.1 De S4 a Mamba-2: Conceptos Fundamentales

| Modelo | Año | Mecanismo clave | Complejidad | Selectividad |
|--------|-----|-----------------|-------------|--------------|
| S4 (Gu et al.) | 2022 | HiPPO + Structured State Space | O(L) train, O(1) inf | No (LTI) |
| S5 (Smith et al.) | 2023 | Diagonal SSM con parallel scan | O(L) | No |
| Mamba/S6 (Gu & Dao) | 2023 | **Selective SSM**: Δ, B, C como funciones del input | O(L) | **Sí** |
| Mamba-2/SSD (Dao & Gu) | 2024 | Structured State Space Duality → SSM = Masked Attention | O(L) 2-8× más rápido | Sí |

**Ecuación fundamental del SSM discreto:**
```
h_t = Ā_t · h_{t-1} + B̄_t · x_t
y_t = C_t · h_t
```

Donde Ā_t = exp(Δ_t · A), B̄_t = Δ_t · B_t, y Δ_t, B_t, C_t son funciones del input x_t.

**La dualidad SSM–Attention** (Mamba-2, ICML 2024):
- SSM con A escalar → semiseparable matrix → masked attention
- C ↔ Q, B ↔ K, X ↔ V, máscara de producto acumulativo de Ā ↔ máscara causal
- Implica que cualquier SSM selectivo es una **forma de atención enmascarada**, y viceversa

| Link | https://arxiv.org/abs/2312.00752 (Mamba) |
| | https://arxiv.org/abs/2405.21060 (Mamba-2) |

### 5.2 Cuantización de SSMs: Estado del Arte

#### Q-S5 (Abreu et al., ICLR 2024)
- Primer estudio de cuantización en SSMs (S5)
- QAT con pesos W4A8 mantiene precisión
- Hallazgo: la **recurrencia acumula errores** e_T = Σ Ā^(T-t) · ε_t, peor que atención

#### Quamba (2024)
- PTQ W4A8/W8A8 para Mamba 2.8B
- Selective SSM: activaciones con outliers masivos → necesitan Hadamard transform
- No funciona por debajo de 4 bits: "PTQ below 4 bits degrades catastrophically for SSMs"

#### Bi-Mamba (Tang et al., 2024)
- **Primer SSM de 1 bit** (binario): Mamba-2 binarizado desde cero
- Escalas: 780M, 1.3B, 2.7B
- Knowledge distillation desde Llama2-7B teacher
- Resultado: compite con FP16 en benchmarks downstream

#### Ternary Mamba (arXiv 2606.18114, Jun 2026)
- **El paper clave para Bit**: W1.58A16 QAT sobre Mamba-2 1.3B pre-entrenado
- Resultados:

| Método | Tamaño | Bits | Avg Downstream | Coste de entrenamiento |
|--------|--------|------|----------------|----------------------|
| Bi-Mamba | 780M | 1-bit | 48.4% | 105B tokens from scratch |
| **Ternary Mamba** | **1.3B** | **W1.58** | **48.1%** | **102M tokens QAT (4 GPU-h)** |
| BitNet b1.58 | 780M | 1.58 | 44.3% | from scratch |

| Aspecto | Detalle |
|---------|---------|
| Hallazgo crítico | **Zero-ratio collapse**: escalas de cuantización aprendibles colapsan en SSMs ternarios → solución: absmean no aprendible |
| Grupo óptimo | g=128 para ternario en SSM |
| Frontera Pareto | Capas selectivas (Δ, B, C) más sensibles → mantener FP16 si es necesario |
| Post-hoc falla | Kalman, James-Stein, sigma-delta **no funcionan** en SSMs por acumulación de error recurrente |

| Link | https://arxiv.org/abs/2411.11843 (Bi-Mamba) |
| | https://arxiv.org/abs/2606.18114 (Ternary Mamba) |

#### Slender-Mamba (2025)
- BitNet-style ternario aplicado a Mamba-2 170M desde cero (150B tokens)
- Per-tensor scaling (no per-group)

#### TVMamba (OpenReview 2025)
- Mamba visión con pesos **y activaciones ternarios** (W1.58A1.58)
- Quinary-to-ternary activation quantizer + frequency router

### 5.3 Arquitecturas Híbridas SSM + Attention

| Modelo | Ratio Attn:SSM | Tamaño | MoE | Contexto |
|--------|---------------|--------|-----|----------|
| Jamba (AI21, 2024) | 1:7 | 52B (12B active) | Sí | 256K |
| Jamba-1.5 Mini | 1:7 | 12B | Sí | 256K |
| Jamba-1.5 Large | 1:7 | 94B | Sí | 256K |
| Zamba (Zyphra, 2024) | Shared attn | 7B | No | 32K |
| Samba (Microsoft) | 1:3 | — | No | 1M |
| RecurrentGemma (Google) | 1:3 | 9B | No | 8K |
| Falcon Mamba (TII) | Pure SSM | 7B | No | ∞ |

**Principio clave**: No necesitas attention en todas las capas.  
- Attention provee **recall preciso e in-context learning** (cara, necesaria para recuperación)
- Mamba maneja **integración de patrones de largo alcance** (barata, escalable O(L))
- Jamba demostró: 1 capa de attention cada 7 Mamba layers recupera **87% del recall** de attention pura

### 5.4 Mapeo de la Resonancia de Bit a SSM

La resonancia actual de Bit es **funcionalmente equivalente a un SSM recurrente** con dinámicas fijas:

| Componente Bit actual | Análogo SSM |
|----------------------|-------------|
| `h = h + 0.5 * h_prev` | **A = 0.5** (matriz de transición escalar fija) |
| `resonance_clock[:, step, :]` | **Δ_t** (step size) — señal temporal por iteración |
| `for step in range(n_steps): h = layers(h)` | Loop recurrente sobre el mismo bloque — cada step es una **actualización de estado** |
| Retroalimentación de logits-argmax (Deep Think) | **Selectividad**: input-dependent gating |
| `n_steps` = número de iteraciones | **N_h** = state expansion del SSM |
| Topología secuencial (6 capas) | Múltiples SSM en serie (profundidad) |

**Problema de eficiencia**: la resonancia actual ejecuta las 6 capas transformer (O(L²)) en cada step. Un SSM selectivo haría la misma actualización en O(L).

#### Propuesta arquitectónica: Hybrid Bit-SSM

```
Opción A: SSM puro para reemplazar resonancia (Mamba-style)
  Input → BitLinear → SSM (selectivo ternario) → RMSNorm → BitMLP → ... × num_layers

Opción B: Hybrid Jamba-style (BitNet + SSM)
  Layer 1: BitNetAttention + BitMLP   (1 capa de atención cada N)
  Layer 2: BitSSM + BitMLP            (N-1 capas SSM)
  Layer 3: BitSSM + BitMLP
  ...

Opción C: SSM como reemplazo del loop de resonancia (manteniendo transformer layers)
  Input → TransformerBlock → [SSM Loop: actualización de estado recurrente] × n_steps → decode
  Donde el loop de resonancia se convierte en una sola capa SSM selectiva
```

### 5.5 Implicaciones para Bit: Cambio en la Estrategia de Crecimiento

#### El panorama cambia con SSM:

| Dimensión | Antes (solo transformers) | Con SSM |
|-----------|---------------------------|---------|
| Cuello de botella | Depth insuficiente (ratio W/D=170) | **Resonancia O(N·L²)** ineficiente |
| Crecimiento | Net2DeeperNet (6→8 capas) | **SSM en 1 paso O(L)** mejor que 5 loops |
| KV Cache | No existe (se recalcula todo) | **No necesita KV cache** (estado recurrente) |
| Contexto largo | L=64 (corto por O(L²)) | **L=512+** viable (O(L)) |
| Selectividad | h_prev * 0.5 fijo | **Δ_t, B_t, C_t** dependientes del input |

#### Hipótesis revisada:

> **La resonancia actual de Bit (6 capas × n_steps loops) puede ser reemplazada por un bloque SSM selectivo con pesos ternarios, manteniendo la misma funcionalidad de recurrencia pero con complejidad O(L) en vez de O(N·L²). Esto permitiría escalar a contextos más largos y aprendizaje de dependencias temporales más ricas.**

#### Plan de acción actualizado:

1. ✅ Terminar fase actual de entrenamiento (width 1024, época 1269+)
2. ⬜ **Decidir prioridad: Net2DeeperNet (depth transformer) vs SSM híbrido**
3. ⬜ Si SSM: implementar `BitSSMBlock` con pesos ternarios (inspirado en Ternary Mamba):
   - Proyecciones Q/K/V → convertir a Δ, B, C input-dependent
   - Selective scan ternario (o simplificado con A escalar)
   - STE para gradientes, absmean scaling (no aprendible)
   - Compatible con RMSNorm y residuales existentes
4. ⬜ Integrar como reemplazo de resonancia o como capas híbridas
5. ⬜ Evaluar: val_loss, velocidad, calidad semántica vs baseline actual

### 5.6 Referencias SSM

12. Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2023
13. Dao & Gu, "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality", ICML 2024
14. Abreu et al., "Q-S5: Towards Quantized State Space Models", ICLR 2024
15. Tang et al., "Bi-Mamba: Towards Accurate 1-Bit State Space Models", 2024
16. "Ternary Mamba: Grouped Quantization-Aware Training of W1.58A16 State Space Models", arXiv 2606.18114, 2026
17. Lieber et al., "Jamba: A Hybrid Transformer-Mamba Language Model", 2024
18. "Slender-Mamba: BitNet-style ternary quantization for Mamba-2", 2025
19. "Quamba: A Post-Training Quantization Recipe for Selective State Space Models", 2024

---

## 6. Plan de Experimentación Post-Entrenamiento

> **Estado del modelo al momento de escribir**: época 1305/1408, val_loss=4.3696, hidden_dim=1024, 6 capas, 76.9M params.
> Plateau monitor: 12 épocas sin mejora (patience=15). Se espera alcanzar límite pronto (o terminar época 1408).
> Total GPU: ~5.5 min/época, VRAM 4.1 GB.

A continuación se documentan todas las opciones arquitectónicas viables, con su coste estimado, riesgo, y señales de éxito. La idea es tener un menú completo para decidir después de que termine el entrenamiento actual.

> **Gobernanza (v5, 2026-07-31)**: la secuenciación de estas opciones NO la decide este documento — la gobiernan `bitnet_next_architecture_plan.md` **v5** (§5-§7: Net2DeeperNet sobre copia, Arsenal, ventana de GPU) y el ROADMAP de `~/Documents/IA/k65p` (gates G1-G4). Restricciones heredadas clave: (1) el checkpoint de graduación es el **control de G4** — toda expansión opera sobre copia; (2) el SSM está gateado tras el veredicto G4 (plan §4.6); (3) las opciones no activas viven como balas en la recámara (plan §6) — ninguna se descarta.

### 6.1 Opción A: Net2DeeperNet (Depth Expansion Clásico)

**Descripción**: Añadir 2 capas transformer (6→8) usando Net2DeeperNet con inicialización a identidad/cero.

| Aspecto | Detalle |
|---------|---------|
| Cambio | `modeling_bitnet.py`: insertar `BitNetTransformerBlock` con proyección residual a 0 |
| Inicialización | **Solo las proyecciones de salida a 0** (O de la atención y down-proj del MLP); Q/K/V y up-proj con init normal. Output de la capa = 0 → residual puro (identidad preservada), pero las capas internas mantienen gradiente vivo desde el paso 1. ⚠️ NO poner todo a 0: mata gradientes (simetría sin romper + gradiente hacia Q/K/V pasa por pesos nulos) |
| ⚠️ Ternario + init a 0 | absmean de un tensor nulo → escala 0 / división por cero en el primer forward (primo del zero-ratio collapse de Ternary Mamba §5.2). Mitigación: capas nuevas en FP16 con ternarización tras warmup, o epsilon/clamp en la escala absmean |
| Parámetros extra | +2 × (attn + mlp + 2×RMSNorm) ≈ +10M params (86.9M total) |
| Ratio W/D nueva | 1024/8 = **128** (dentro del rango BitNet conocido, máx 149) |
| Compute | Continuar training existente, ~5.5 min/época |
| Riesgo | Bajo — función preservada, no requiere re-tuning |
| Señal éxito | val_loss desciende >0.05 tras ~100 épocas; muestras cualitativas con mayor profundidad semántica |
| Señal fracaso | val_loss no mejora en 200 épocas; samples siguen siendo sintaxis correcta pero sin razonamiento |

**Variantes**:
- A.1: **6→8 capas** (bajo riesgo, ~18 GPU-h para 200 épocas)
- A.2: **6→10 capas** (si A.1 funciona, ratio 1024/10=102 — mejor aún)
- A.3: **6→8→10 progresivo** (Net2DeeperNet cada vez que plateau se estanca)

### 6.2 Opción B: SSM Híbrido (Reemplazar Resonancia)

**Descripción**: Implementar `BitSSMBlock` con pesos ternarios que reemplace el loop de resonancia actual.

> ⚠️ **Líneas rojas heredadas** (`bitnet_next_architecture_plan.md` §4.2, ya decididas el 10 jul): `W_Δ` empieza en **precisión alta** (camino Softplus→exp sensible al STE; A/B después) — NO BitLinear como decía la primera versión de esta spec; **nunca** ternarizar `A_log`/`D`/factores de descuento `exp(Δ·A)` (decay ∈ (0,1) continuo); **nunca** cuantizar el estado del scan (error acumulativo). Regla: *MatMul-free ≠ float-free*. Aplican a TODAS las variantes SSM de este documento (B, C, F).

**Arquitectura propuesta del BitSSMBlock** (corregida v5):

```
Input x_t (hidden_dim)

  │
  ├── Linear FP16 → Δ_proj (Δ_t = softplus(W_Δ · x_t))   [línea roja: no ternarizar]
  ├── BitLinear → B_proj (B_t = W_B · x_t)      [state_dim=16]
  ├── BitLinear → C_proj (C_t = W_C · x_t)      [state_dim=16]
  │
  A = log(1 + exp(A_log)) negada (parámetro libre, escalar por head)
  Ā_t = exp(-Δ_t · A)                            [discretización zero-order hold]
  B̄_t = (1 - Ā_t) · B_t / A                      [discretización ZOH simplificada]
  
  h_t = Ā_t · h_{t-1} + B̄_t · x_t                [selective scan]
  y_t = C_t · h_t
  
  │
  ├── BitLinear → out_proj (proyecta y_t a hidden_dim)
  ├── + residual
  ├── RMSNorm → BitMLP → + residual
```

| Aspecto | Detalle |
|---------|---------|
| Cambio | Nuevo `BitSSMBlock` con proyecciones ternarias + selective scan |
| A_init | HiPPO-LegS (S4D) o simplemente A_log → A escalar ~0.5 (equivalente a h_prev actual) |
| state_dim | N=16 (caben 16 heads paralelas con hidden_dim=1024) |
| STE | BitLinear ya usa STE — reutilizar; añadir absmean no aprendible para escalas (lección de Ternary Mamba) |
| Riesgo | Medio-Alto — arquitectura nueva, no hay implementación de selective scan ternario |
| Compute | O(L) vs O(N·L²) actual → **más rápido por step, contextos más largos** |
| Señal éxito | Misma val_loss con menos FLOPs; samples mantienen coherencia semántica; contexto 2×+ viable |

**Variantes**:
- B.1: **Reemplazar solo el loop de resonancia** (mantener transformer layers, SSM como capa adicional entre embedding y decode)
- B.2: **Reemplazar capas enteras** (híbrido Jamba-style: 1 attn cada N SSM)
- B.3: **SSM puro** (BitNet transformer blocks → BitSSM blocks, atención eliminada)

### 6.3 Opción C: Hybrid Jamba-Style (BitNet + SSM interleaved)

**Descripción**: Mezclar capas de atención ternaria con capas SSM ternarias en proporción controlada.

```
BitHybridBlock:
  ├── Si es capa attention: BitRMSNorm → BitNetAttention → BitMLP → residual
  └── Si es capa SSM:      BitRMSNorm → BitSSMBlock → BitMLP → residual

Ejemplo de configuración (6 capas, ratio 1:2):
  [SSM, Attn, SSM, SSM, Attn, SSM]
                          ^ atención cada 3 capas
```

| Aspecto | Detalle |
|---------|---------|
| Ratio propuesta | 1:3 o 1:5 (atención cada 3-5 capas, siguiendo Jamba/Zamba) |
| Parámetros extra | Similar a depth expansion pero con SSM más ligero que attention |
| Compute | Atención O(L²) para recall, SSM O(L) para integración — lo mejor de ambos |
| Contexto | Atención se usa selectivamente → KV cache pequeña, contexto largo viable |
| Riesgo | Medio — requiere implementar BitSSMBlock + integrar con blocks existentes |
| Señal éxito | Mejora en tareas de recall (citas, hechos) + coherencia de largo alcance |

**Variantes**:
- C.1: **SSM heavy** (1 attn cada 5 SSM) — prioriza eficiencia
- C.2: **Attn heavy** (1 attn cada 2 SSM) — prioriza recall
- C.3: **Shared attention** (estilo Zamba: 1 attn compartida, reutilizada) — mínimo overhead

### 6.4 Opción D: SENN Dinámico (Natural Expansion Score)

**Descripción**: Implementar η = gᵀ·F⁻¹·g para decidir automáticamente entre width, depth, y SSM.

| Aspecto | Detalle |
|---------|---------|
| Implementación | Calcular F aproximada vía Kronecker factorization (A_l = activaciones², S_l = gradientes²) |
| η por capa | η_l = Tr[∂W_lᵀ · S_l⁻¹ · ∂W_l · A_l⁻¹] |
| Decisión | Donde η sea mayor → expandir ahí (width, depth, o añadir SSM) |
| Riesgo | Alto — requiere estimación estable de Fisher inversa; sobrecarga compute no despreciable |
| Señal éxito | η crece sostenidamente post-expansión vs estancamiento sin expansión |

**Variantes**:
- D.1: **SENN parcial** — solo para decidir cuándo (no dónde), trigger basado en η
- D.2: **SENN completo** — decide tipo de expansión (width vs depth vs SSM)

### 6.5 Opción F: SSM dentro de la Resonancia (SSM + Resonancia)

**Descripción**: SSM se añade **dentro** del loop de resonancia para proveer memoria comprimida, mientras la resonancia se mantiene intacta como refinamiento iterativo.

> ⚠️ **Ambigüedad a resolver antes de implementar** — el pseudocódigo de abajo recurre sobre *steps* de resonancia (estado `(batch, seq, state_dim)`, actualizado una vez por step): eso es memoria **inter-step**, coherente con el mapeo del §5.4 pero NO da el beneficio O(L) vs O(L²) en contexto. Para memoria **inter-token** (contexto largo, argumento O(L)), el selective scan debe recorrer la **dimensión de secuencia** dentro de cada step. Son dos arquitecturas distintas con el mismo nombre:
> - **F-step**: estado recurrente entre iteraciones de resonancia → enriquece el refinamiento iterativo, no toca el coste de atención
> - **F-token**: scan sobre la secuencia → memoria de largo alcance real, habilita reducir atención (Jamba-style) y contextos más largos

**Arquitectura**: la resonancia actual itera transformer blocks stateless (sin memoria entre tokens). El SSM se coloca en paralelo o serie con la atención dentro de cada paso, dando al loop un estado persistente.

```
# Antes (stateless por paso):
for step in range(n_steps):
    h += clock[:, step]          # señal temporal
    h = attention(h)             # recalcula todo — O(L²)
    h = mlp(h)

# Después (SSM + Resonancia):
ssm_state = torch.zeros(batch, seq, state_dim)   # ← estado persistente

for step in range(n_steps):
    h += clock[:, step]                          # resonancia: señal temporal
    
    # SSM: memoria comprimida entre tokens — O(L)
    Δ_t = softplus(W_Δ · h)                     # step size dependiente del input
    Ā_t = exp(-Δ_t · A)                          # discretización
    B̄_t = (1 - Ā_t) · (W_B · h) / A
    ssm_state = Ā_t · ssm_state + B̄_t · h       # actualización de estado
    ssm_out = W_C · ssm_state                    # lectura
    
    h = h + ssm_out                              # inyecta memoria de largo alcance
    
    # Atención (solo 1 de cada N pasos, estilo Jamba)
    if step % attn_interval == 0:
        h = attention(h)                         # recall preciso
    
    h = mlp(h)                                   # procesamiento local
```

| Aspecto | Detalle |
|---------|---------|
| Cambio | No se quita nada, se **añade** SSM dentro de cada paso de resonancia |
| Atención | Opcional por paso — 1 de cada N pasos usa atención (Jamba-style), el resto solo SSM+MLP |
| state_dim | N=16 (caben 16 heads paralelas con hidden_dim=1024) |
| Inicialización | Ā ≈ 0.5 (mismo factor que el h_prev * 0.5 actual) |
| Preserva resonancia | ✅ **Totalmente** — el loop y la señal clock se mantienen |
| Compute | **Menos** que hoy si atención es parcial (SSM O(L) vs attn O(L²)) |
| Parámetros extra | ~3 × state_dim × hidden_dim (Δ, B, C) + out_proj ≈ 50K params adicionales |

**Riesgo**: Bajo-Medio — no se reemplaza nada existente, se añade un mecanismo de memoria. Si el SSM no funciona, se puede desactivar sin afectar la resonancia actual.

**Variantes**:
- F.1: **SSM en paralelo con atención** en cada paso (ambos se suman)
- F.2: **SSM reemplaza atención en pasos alternos** — atención solo cada 3-5 pasos
- F.3: **SSM como pre-procesamiento** antes de la resonancia (una pasada SSM, luego loop de resonancia normal)
- F.4: **SSM bidireccional** (para tareas que no son causales) — escanea secuencia forward+backward

### 6.6 Opción E: Combinaciones

No son excluyentes. Posible roadmap multi-fase:

| Fase | Opciones | Orden sugerido |
|------|----------|---------------|
| 1 | **A.1** (Net2DeeperNet 6→8) | Primero, bajo riesgo, resultados rápidos |
| 2 | **A.2** (Net2DeeperNet 8→10) | Si A.1 funciona |
| 3 | **F.1** (SSM + Resonancia) | Añade SSM al loop sin romper nada |
| 4 | **F.2 → C.1** (SSM gradual) | Eliminar atención progresivamente si SSM funciona |
| 5 | **B.3** (SSM puro) | Si F puro funciona, probar sin atención |
| 6 | **D.2** (SENN completo) | Como meta-capa sobre el experimento ganador |

### 6.7 Matriz de Decisión

| Criterio | A (Depth) | B (SSM puro) | C (Hybrid Jamba) | D (SENN) | **F (SSM+Resonancia)** |
|----------|-----------|--------------|------------------|----------|------------------------|
| Riesgo técnico | Bajo | Medio-Alto | Medio | Alto | **Bajo-Medio** |
| Impacto potencial | Medio | Alto | Muy Alto | Máximo | **Alto** |
| Tiempo implementación | ~1 día | ~2 semanas | ~3 semanas | ~1 mes+ | **~1 semana** |
| Compute extra (200 épocas) | ~18 GPU-h | ~5 GPU-h (O(L)) | ~10 GPU-h | ~20 GPU-h | **~8 GPU-h** |
| Preserva función actual | ✅ Sí | ❌ No | ❌ No | ⚠️ Depende | **✅ Sí (añade, no reemplaza)** |
| Escalabilidad contexto | ❌ O(L²) | ✅ O(L) | ✅ Casi O(L) | ⚠️ Depende | **✅ Casi O(L)** (attn parcial) |
| Literatura ternario | BitNet b1.58 | Ternary Mamba, Bi-Mamba | Sin precedentes directos | SENN original FP16 | **Sin precedentes directos** |

### 6.8 Recomendación Inicial

Basado en la literatura y el estado actual del modelo:

1. **Opción A** (depth) es el movimiento más natural: bajo riesgo, preserva todo, nos mueve a una ratio W/D=128 que ya ha sido validada en BitNet de 30B parámetros.
2. **Opción F** (SSM + Resonancia) es la que mejor captura la visión del modelo: la resonancia se queda como refinamiento iterativo, y el SSM le da la memoria secuencial que necesita. No hay que elegir entre una u otra.
3. **Opción C** (hybrid SSM Jamba-style) es complementaria a F: si el SSM funciona dentro de la resonancia, el siguiente paso natural es reemplazar atención por SSM en algunas capas.
4. **Opción B** (SSM puro) debería esperar a que F y C demuestren que el SSM ternario funciona en este contexto.
5. **Opción D** (SENN) como meta-capa si las decisiones manuales no son concluyentes.

> **Decisión final**: se tomará al terminar el entrenamiento actual (época ~1305/1408 o cuando plateau patience se agote). Este documento será la base para esa decisión.

---

## Referencias

1. Petty et al., "The Impact of Depth on Compositional Generalization in Transformer LMs", NAACL 2024
2. Saunshi et al., "Reasoning with Latent Thoughts: On the Power of Looped Transformers", NeurIPS 2025
3. "Disentangling Recall and Reasoning in Transformer Models", OpenReview 2025
4. "Layer Specialization Underlying Compositional Reasoning in Transformers", arXiv 2510.17469
5. Fahim & Karim, "The Depth Delusion: Why Transformers Should Be Wider, Not Deeper", arXiv 2601.20994, 2026
6. Hayou et al., "Width and Depth Limits Commute in Residual Networks", ICML 2023
7. Bu et al., "Scaling Depth Capacity via Zero/One-Layer Model Expansion", Meta FAIR 2025
8. Du et al., "Depth up-scaling vs width up-scaling", 2024
9. Mitchell, Mundt, Kersting, "Self-Expanding Neural Networks", NeurIPS 2023
10. Ma et al., "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits", 2024
11. Nielsen et al., "BitNet b1.58 Reloaded: State-of-the-art Performance Also on Smaller Networks", 2024
