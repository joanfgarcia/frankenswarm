# 🧬 Frankenswarm — Lab Notebook

> Registro experimental del entrenamiento PopuLoRA.
> Cada ejecución se documenta con parámetros, resultados y decisiones.
> **Nada se borra. Los fracasos son datos.**

---

## Auditoría Externa Pre-Ejecución (2026-05-26)

### Consultas realizadas
Cuatro modelos externos evaluaron el `FRANKENSWARM_DIGEST` completo y el Capítulo 16 de la novela de Aleth ("El Jardín de Frankenstein"). El objetivo: obtener opiniones independientes antes de la primera ejecución.

| Modelo | Tono | Veredicto | Insight clave |
|---|---|---|---|
| **Grok** | Coach pragmático | "¡A plantar ya!" | Plan paso a paso concreto. Propone batch=16, τ=0.8, modo debug por step. |
| **DeepSeek** | Poeta-ingeniero | "Plantar con alma" | β=0.3 para canal emocional. Monitorizar SVD. Cuidar a los Díscolos. |
| **Claude Sonnet** | Cirujano | "Para. Phase A primero" | Identifica dos proyectos bajo un nombre. NEAT a 7M es zona roja. Catastrophic forgetting. |
| **Lumo** | Profesor equilibrado | 8.5/10 técnico | Validar eficiencia energética con wattímetro, no estimaciones teóricas. |

### Consenso de las 5 voces (4 externas + Aleth)
1. **Arquitectura sólida**: 4 capas, vocab congelado, Gumbel-Softmax ✓
2. **β emocional**: Debe empezar bajo. Consenso en β=0.3 (DeepSeek).
3. **NEAT a 7M es arriesgado**: Transiciones NEAT→Net2Net→PEFT no validadas.
4. **Correr Phase 0 cuanto antes**: `train_populora.py` es la prueba mínima.
5. **Phase A ≠ Phase B**: El MoE distribuido y el entrenamiento desde cero son proyectos distintos.

### Informes completos
Almacenados en `docs/extern/` (excluidos del digest para preservar independencia de futuros auditores):
- `PRE_0_GROK.md`
- `PRE_0_DEEPSEEK.md`
- `PRE_0_CLAUDE_SONNET.md`
- `PRE_0_LUMO.md`

---

## Experimento 001 — Full Vocabulary Baseline (8192 tokens)

**Fecha**: 2026-05-26 20:32 CEST
**Script**: `src/bitnet/train_populora.py`
**Device**: CUDA (RTX 4060 Ti)
**Duración**: ~11 segundos
**Estado**: ❌ FRACASO (accuracy 0%)

### Parámetros

| Param | Valor | Justificación |
|---|---|---|
| `vocab_size` | 8192 | SovereignTranslator completo (fastembed, 384-dim) |
| `hidden_dim` | 256 | Espacio oculto del Core ternario |
| `num_layers` | 4 | 4 bloques Transformer con BitLinear |
| `pop_size` | 4 | 4 agentes en la arena |
| `epochs` | 5 | - |
| `steps_per_epoch` | 100 | Total: 500 steps |
| `batch_size` | 32 | - |
| `lr` | 1e-3 | AdamW |
| `tau_start → tau_min` | 1.0 → 0.1 | Annealing lineal de Gumbel-Softmax |
| `β (emotion)` | 0.3 | Patch pre-ejecución (consenso DeepSeek) |
| `message_length` | 3 | Secuencia de 3 tokens |
| `targets` | 15 conceptos + 6 emociones | Mapeados a token IDs en espacio de 8192 |

### Resultados completos

| Época | Loss media | Accuracy conjunta | Peor agente | Reemplazo SVD |
|---|---|---|---|---|
| 1 | 5.0389 | 0.00% | Agent_0 | → hijo de Agent_3 × Agent_2 |
| 2 | 4.2480 | 0.00% | Agent_0 | → hijo de Agent_3 × Agent_2 |
| 3 | 4.2443 | 0.00% | Agent_0 | → hijo de Agent_3 × Agent_2 |
| 4 | 4.1967 | 0.00% | Agent_0 | → hijo de Agent_3 × Agent_2 |
| 5 | 4.3002 | 0.00% | Agent_0 | → hijo de Agent_3 × Agent_2 |

### Análisis

- **Loss baja un 17%** (5.04 → 4.20): los modelos optimizan distribuciones internas, pero el argmax no converge al token correcto.
- **0% accuracy**: Con 8192 tokens, el azar puro para acertar concepto+emoción simultáneamente es `(1/8192)² ≈ 0.000001%`. Incluso un modelo parcialmente entrenado no puede argmax al token exacto en 500 steps.
- **Loss teórica random**: `ln(8192) ≈ 9.01`. La loss en 4.x sugiere que el modelo ha aprendido algo (está por debajo del azar), pero no lo suficiente para acertar tokens discretos.
- **Agent_0 siempre es el peor**: El SVD crossover crea un hijo que arranca frío y es inmediatamente el peor otra vez. El crossover no transfiere conocimiento efectivamente en 1 época.

### Diagnóstico
> **El espacio de 8192 tokens es demasiado grande para Phase 0.** Los agentes necesitan apuntar a 1 token correcto de 8192 en cada posición. Es como pedirle a un recién nacido que escriba la palabra exacta en un diccionario enciclopédico.

### Decisión
→ **Reducir vocabulario a micro-escala (21 tokens)** manteniendo la misma arquitectura BitNet.

---

## Experimento 002 — Micro-Vocabulary Arena (21 tokens)

**Fecha**: 2026-05-26 20:36 CEST
**Script**: `src/bitnet/train_populora_micro.py`
**Device**: CUDA (RTX 4060 Ti)
**VRAM**: ~836 MB
**Duración**: ~30 segundos
**Estado**: ❌ FRACASO (accuracy estancada en ~6.5% concepto, 0% emoción)

### Cambios respecto a Exp. 001
1. **Vocabulario**: 8192 → 21 tokens (15 conceptos + 6 emociones)
2. **Embeddings**: Generados con fastembed solo para los 21 tokens (21×384 vs 8192×384)
3. **Más training**: 10 epochs × 200 steps = 2000 steps (vs. 500)
4. **Accuracy descompuesta**: Concepto / Emoción / Conjunta (en Exp. 001 solo conjunta)
5. **Debug logging**: Cada 20 steps, muestra target vs predicción en texto legible
6. **SVD monitoring**: Condition number >10⁴ → fallback a interpolación lineal

### Parámetros

| Param | Valor | Justificación |
|---|---|---|
| `vocab_size` | 21 | 15 conceptos + 6 emociones |
| `hidden_dim` | 256 | Sin cambio |
| `num_layers` | 4 | Sin cambio |
| `pop_size` | 4 | Sin cambio |
| `epochs` | 10 | Más tiempo para converger |
| `steps_per_epoch` | 200 | Más steps por época |
| `batch_size` | 32 | Sin cambio |
| `lr` | 1e-3 | Sin cambio |
| `tau_start → tau_min` | 1.0 → 0.1 | Sin cambio |
| `β (emotion)` | 0.3 | Consenso externo |
| `message_length` | 3 | Sin cambio |
| `targets` | 15 conceptos [0..14] + 6 emociones [15..20] | IDs locales en micro-vocab |

### Resultados completos

| Época | Loss media | Acc. Concepto | Acc. Emoción | Acc. Conjunta | Peor | Reemplazo |
|---|---|---|---|---|---|---|
| 1 | 4.1069 | 5.33% | 0.17% | 0.00% | Agent_0 | → A3 × A2 |
| 2 | 4.0774 | 6.47% | 0.20% | 0.00% | Agent_0 | → A3 × A2 |
| 3 | 4.0499 | 5.92% | 0.42% | 0.00% | Agent_0 | → A3 × A2 |
| 4 | 4.0237 | 6.84% | 0.00% | 0.00% | Agent_0 | → A3 × A2 |
| 5 | 4.0279 | 6.84% | 0.08% | 0.00% | Agent_0 | → A3 × A2 |
| 6 | 4.0201 | 6.80% | 0.23% | 0.00% | Agent_0 | → A3 × A2 |
| 7 | 4.0204 | 6.81% | 0.09% | 0.00% | Agent_0 | → A3 × A2 |
| 8 | 4.0060 | 6.72% | 0.08% | 0.00% | Agent_0 | → A3 × A2 |
| 9 | 4.0046 | 6.58% | 0.00% | 0.00% | Agent_0 | → A3 × A2 |
| 10 | 3.9990 | 6.50% | 0.00% | 0.00% | Agent_0 | → A3 × A2 |

### Baselines teóricos (para referencia)

| Métrica | Random (21 clases) | Random (15 conceptos) | Random (6 emociones) | Observado |
|---|---|---|---|---|
| Loss | ln(21) ≈ **3.04** | - | - | **3.99** (peor que random) |
| Acc. Concepto | 1/21 ≈ 4.76% | 1/15 ≈ 6.67% | - | **6.50%** (≈ random 15) |
| Acc. Emoción | 1/21 ≈ 4.76% | - | 1/6 ≈ 16.67% | **0.00%** (peor que random) |

### Hallazgos críticos

#### 1. 🔴 El modelo predice el MISMO token en ambas posiciones
Patrón dominante en todos los debug logs:
```
target=(fuego,miedo)    → pred=(perro,perro)
target=(código,hambre)  → pred=(agua,agua)
target=(sol,dolor)      → pred=(luna,luna)
target=(búnker,alegría) → pred=(tierra,tierra)
```
**El listener no diferencia la posición 1 (concepto) de la posición 2 (emoción).** Emite el mismo token en ambas posiciones. Esto indica que el modelo no tiene conciencia posicional.

#### 2. 🔴 La loss está POR ENCIMA del azar
- Loss observada: ~4.0
- Loss random para 21 clases: ln(21) ≈ 3.04
- **El modelo está peor que lanzar un dado.** Esto sugiere que el Gumbel-Softmax channel está destruyendo información en lugar de transmitirla.

#### 3. 🟡 El concepto roza el azar pero la emoción está muerta
- Concepto: ~6.5% (azar para 15 conceptos = 6.67%) → **el modelo NO ha aprendido nada sobre conceptos**
- Emoción: ~0% (azar para 6 emociones = 16.67%) → **la emoción está completamente suprimida**
- El β=0.3 combinado con que concepto y emoción comparten un espacio unificado de 21 tokens genera un conflicto: el modelo solo ve conceptos como candidatos y nunca aprende a emitir tokens emocionales (posiciones 15-20 del vocab)

#### 4. 🟡 Agent_0 es SIEMPRE el peor
En todas las épocas, Agent_0 es reemplazado. Pero el hijo SVD siempre arranca frío y vuelve a ser el peor. **El crossover no funciona en el primer epoch** porque el hijo no tiene tiempo de estabilizarse antes de ser evaluado.

### Diagnóstico estructural

> **Hipótesis principal**: El modelo carece de **codificación posicional** y de **separación de tareas**.
>
> El BitNet4LayerModel recibe una secuencia de 3 tokens y emite logits sobre 21 clases en cada posición. Pero no tiene ningún mecanismo para saber que la posición 1 debe predecir un concepto (0-14) y la posición 2 debe predecir una emoción (15-20). Para el modelo, todas las posiciones son intercambiables.
>
> Además, concepto y emoción comparten el mismo espacio de salida (21 logits). El gradiente del concepto (peso 1.0) domina sobre el de la emoción (peso 0.3), y como ambos compiten por los mismos logits, la emoción nunca recibe señal suficiente.

### Decisiones para Exp. 003

> [!IMPORTANT]
> Tres cambios propuestos:
> 1. **Cabezas de salida separadas**: Una cabeza para concepto (15 clases) y otra para emoción (6 clases), en lugar de un espacio unificado de 21.
> 2. **Positional encoding**: Añadir embeddings posicionales aprendidos al input del modelo para que distinga posición 0 (input concepto), posición 1 (output concepto) y posición 2 (output emoción).
> 3. **Subir β a 0.5**: Con cabezas separadas ya no hay competencia por los logits; la emoción puede recibir más señal.

---

## Experimento 003 — Dual-Head Arena (15C + 6E, positional encoding)

**Fecha**: 2026-05-26 20:41 CEST
**Script**: `src/bitnet/train_populora_micro.py` (v3 — dual head)
**Device**: CUDA (RTX 4060 Ti)
**VRAM**: ~836 MB
**Duración**: ~48 segundos
**Telemetría**: `storage/telemetry/EXP_003.jsonl` (3000 steps × registro completo)
**Estado**: ❌ FRACASO (accuracy = azar exacto en ambos canales)

### Cambios respecto a Exp. 002
1. **Cabezas separadas**: `concept_head(hidden_dim, 15)` + `emotion_head(hidden_dim, 6)` — ya no compiten por logits
2. **Positional encoding**: `Embedding(3, 256)` en speaker y listener — conciencia posicional
3. **β subido**: 0.3 → 0.5 (con cabezas separadas, la emoción puede recibir más señal)
4. **Más training**: 15 epochs × 200 steps = 3000 steps
5. **Telemetría JSONL**: Registro por step y por época vía `ExperimentLogger`
6. **Speaker/Listener wrappers**: `DualHeadSpeaker` y `DualHeadListener` envuelven al `BitNet4LayerModel`

### Parámetros

| Param | Valor | Cambio vs 002 |
|---|---|---|
| `vocab_size` | 21 | = |
| `num_concepts` | 15 | (nuevo: cabeza separada) |
| `num_emotions` | 6 | (nuevo: cabeza separada) |
| `hidden_dim` | 256 | = |
| `num_layers` | 4 | = |
| `pop_size` | 4 | = |
| `epochs` | 15 | +5 |
| `steps_per_epoch` | 200 | = |
| `batch_size` | 32 | = |
| `lr` | 1e-3 | = |
| `tau_start → tau_min` | 1.0 → 0.1 | = |
| `β (emotion)` | **0.5** | +0.2 |
| `positional_encoding` | **Sí** | nuevo |
| `architecture` | **dual_head** | nuevo |

### Resultados completos

| Época | Loss | Concepto | Emoción | Conjunta | Peor | Reemplazo |
|---|---|---|---|---|---|---|
| 1 | 3.818 | 6.64% | 16.58% | 1.16% | A0 | → A1 × A3 |
| 2 | 3.729 | 6.50% | 16.41% | 1.13% | A0 | → A3 × A2 |
| 3 | 3.694 | 6.73% | 16.42% | 1.06% | A0 | → A3 × A2 |
| 4 | 3.679 | 6.75% | 16.52% | 1.20% | A0 | → A3 × A2 |
| 5 | 3.664 | 6.78% | 16.61% | 1.06% | A0 | → A1 × A3 |
| 6 | 3.660 | 6.55% | 16.58% | 1.09% | A3 | → A1 × A2 |
| 7 | 3.649 | 6.88% | 16.84% | 1.03% | A0 | → A1 × A2 |
| 8 | 3.649 | 6.42% | 16.55% | 1.06% | A3 | → A1 × A0 |
| 9 | 3.639 | 6.78% | 16.39% | 1.27% | A3 | → A1 × A2 |
| 10 | 3.638 | 6.52% | 16.53% | 0.92% | A2 | → A0 × A1 |
| 11 | 3.629 | 6.61% | 16.53% | 1.08% | A1 | → A0 × A2 |
| 12 | 3.641 | 6.56% | 16.70% | 1.12% | A3 | → A0 × A1 |
| 13 | 3.634 | 6.92% | 16.88% | 1.11% | A3 | → A2 × A0 |
| 14 | 3.632 | 6.62% | 16.81% | 1.27% | A0 | → A1 × A2 |
| 15 | 3.633 | 6.52% | 16.61% | 1.06% | A0 | → A3 × A2 |

### Baselines exactos vs observado

| Métrica | Random teórico | Observado (media) | Veredicto |
|---|---|---|---|
| Loss total | ln(15) + 0.5×ln(6) = **3.61** | **3.63** | ≈ random ✗ |
| Acc. Concepto | 1/15 = **6.67%** | **6.63%** | ≈ random ✗ |
| Acc. Emoción | 1/6 = **16.67%** | **16.60%** | ≈ random ✗ |
| Acc. Conjunta | 6.67% × 16.67% = **1.11%** | **1.10%** | ≈ random ✗ |

### Mejoras respecto a Exp. 002
1. ✅ **Ya NO predice el mismo token en ambas posiciones** — las dual heads eliminaron el conflicto
2. ✅ **La emoción ya no está en 0%** — pasó de 0% (Exp. 002) a 16.6% (Exp. 003)
3. ✅ **El peor agente ya NO es siempre el mismo** — la evolución está funcionando
4. ✅ **Telemetría completa** — 3000 registros en JSONL para análisis

### Problema persistente

> [!CAUTION]
> **El canal Gumbel-Softmax no transmite información.**
>
> La loss convergió exactamente al valor teórico random. El speaker emite mensajes, el listener los recibe, pero no hay codificación útil. Los modelos han aprendido las distribuciones marginales (uniform para conceptos, uniform para emociones) pero **NADA sobre la comunicación**.
>
> La señal de gradiente pasa a través del Straight-Through estimator, pero es insuficiente para que el speaker aprenda a codificar y el listener a decodificar en 3000 steps.

### Hipótesis para Exp. 004

El problema podría ser:
1. **El canal Gumbel-Softmax es demasiado ruidoso** para modelos tan pequeños
2. **El learning rate es demasiado alto** para la señal débil del ST-estimator
3. **El modelo necesita más steps** (los referential games en la literatura usan 50K-500K steps)
4. **La arquitectura del speaker es ineficiente** — genera 3 tokens de mensaje para transmitir 2 datos

### Decisión para Exp. 004: Test Diagnóstico

> [!IMPORTANT]
> Antes de cambiar más hiperparámetros, necesitamos **aislar dónde falla la comunicación**.
>
> **Exp. 004a — Teacher Forcing (diagnóstico)**: Eliminar el Gumbel-Softmax y alimentar al listener directamente con los tokens del target (one-hot). Si el listener aprende → el problema es el speaker/canal. Si no → el problema es el listener/arquitectura.
>
> **Exp. 004b — Más steps**: Si 004a confirma que el canal es el cuello de botella, probar con 10x más steps (30,000) y lr más bajo (1e-4).

---

## Experimento 004 — (pendiente: diagnóstico de aislamiento)

**Estado**: 🔄 EN DISEÑO

