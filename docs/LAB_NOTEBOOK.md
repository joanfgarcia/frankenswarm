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

## Experimento 004a — Teacher Forcing Diagnostic

**Fecha**: 2026-05-26 20:46 CEST
**Script**: `src/bitnet/train_diagnostic.py`
**Device**: CUDA (RTX 4060 Ti)
**Duración**: ~5 segundos (early stop en época 2)
**Telemetría**: `storage/telemetry/EXP_004a.jsonl`
**Estado**: ✅ DIAGNÓSTICO EXITOSO

### Diseño del experimento
- **Sin speaker**: Eliminado completamente
- **Sin Gumbel-Softmax**: El listener recibe directamente los tokens ground truth como one-hot
- **Sin población**: Un solo listener, sin evolución SVD
- **Mismo listener que Exp. 003**: `DiagnosticListener` con dual heads + positional encoding
- **Objetivo**: Determinar si la arquitectura del listener puede aprender a clasificar

### Parámetros

| Param | Valor | Nota |
|---|---|---|
| `vocab_size` | 21 | = Exp 003 |
| `hidden_dim` | 256 | = |
| `num_layers` | 4 | = |
| `pop_size` | **1** | Sin población |
| `epochs` | 20 (early stop en 2) | - |
| `steps_per_epoch` | 200 | = |
| `batch_size` | 32 | = |
| `lr` | 1e-3 | = |
| `β (emotion)` | 0.5 | = Exp 003 |
| `gumbel_softmax` | **No** | Bypass completo |
| `teacher_forcing` | **Sí** | Ground truth directo |

### Resultados

| Época | Loss | Concepto | Emoción | Conjunta |
|---|---|---|---|---|
| 1 | 2.2100 | 46.98% | 50.72% | 39.55% |
| 2 | **0.0026** | **100.00%** | **100.00%** | **100.00%** |

### Progresión dentro de la Época 1 (micro-análisis)

| Step | Loss | Predicción |
|---|---|---|
| 0 | 3.715 | target=(perro,miedo) → pred=(agua,alegría) ❌ |
| 40 | 3.620 | target=(casa,tristeza) → pred=(agua,miedo) ❌ |
| 80 | 3.678 | target=(búnker,tristeza) → pred=(aire,tristeza) ½ |
| 120 | **1.702** | target=(búnker,ira) → pred=(agente,ira) ½ |
| 160 | **0.014** | target=(perro,hambre) → pred=(perro,hambre) ✅ |

### Veredicto

> [!IMPORTANT]
> **✅ El listener aprende de 0% a 100% en ~350 steps con teacher forcing.**
>
> La arquitectura BitNet4LayerModel con dual heads y positional encoding es perfectamente capaz de clasificar conceptos y emociones. El modelo tiene capacidad de sobra.
>
> **El cuello de botella es EXCLUSIVAMENTE el canal Gumbel-Softmax / speaker.** El speaker no aprende a codificar información interpretable para el listener. Los gradientes del Straight-Through estimator son insuficientes para el aprendizaje del speaker en esta configuración.

### Implicaciones para Exp. 004b

El problema está aislado: el speaker necesita aprender a emitir mensajes que el listener pueda decodificar, y el ST-Gumbel-Softmax no proporciona gradientes suficientes. Opciones:

1. **Curriculum para el speaker**: Empezar con teacher forcing parcial (mezclar mensajes reales con ground truth) y reducir gradualmente el forcing
2. **Canal continuo**: Eliminar la discretización Gumbel-Softmax y usar embeddings continuos como canal, luego discretizar post-convergencia
3. **Más steps + lr más bajo**: La literatura de referential games usa 50K-500K steps; nosotros usamos 3K
4. **REINFORCE como alternativa**: Usar policy gradient en lugar de ST-Gumbel-Softmax para el speaker

---

## Experimento 004b — Scheduled Teacher Forcing Arena

**Fecha**: 2026-05-26 20:53 CEST
**Script**: `src/bitnet/train_populora_micro.py` (v4 — scheduled TF)
**Estado**: 🔄 EN EJECUCIÓN

### Motivación

El Exp. 004a demostró que el listener alcanza 100% en 350 steps con teacher forcing. El Exp. 003 demostró que sin teacher forcing, el accuracy es 0% (azar) en 3000 steps. La conclusión: **el canal Gumbel-Softmax no puede bootstrapear comunicación desde cero**. Necesitamos educar primero.

### Diseño: Tres Fases de Desarrollo

| Fase | Épocas | Teacher Forcing | Analogía |
|---|---|---|---|
| 🍼 **Guardería** | 1-5 | 100% | El profesor habla, los alumnos aprenden a escuchar |
| 🎮 **Recreo supervisado** | 6-20 | 100% → 0% (lineal) | Mezcla de profesor y conversación entre alumnos |
| 🦅 **Autonomía** | 21-30 | 0% | Comunicación libre entre agentes |

### Cambios arquitectónicos respecto a Exp. 003
1. **`DualHeadAgent` unificado**: Un solo módulo que actúa como speaker Y listener (base compartida + pos_embedding compartido + dual heads)
2. **Per-sample mixing**: En cada step, cada muestra del batch decide independientemente si usa teacher forcing o mensaje del speaker (`torch.rand < tf_ratio`)
3. **τ_min subido a 0.3**: No annealar demasiado — mantener exploración en el Gumbel-Softmax
4. **Evolución SVD solo desde Phase 2**: En la guardería no se reemplaza a nadie, todos aprenden

### Parámetros

| Param | Valor | Cambio vs 003 |
|---|---|---|
| `vocab_size` | 21 | = |
| `num_concepts` | 15 | = |
| `num_emotions` | 6 | = |
| `hidden_dim` | 256 | = |
| `num_layers` | 4 | = |
| `pop_size` | 4 | = |
| `epochs` | **30** | +15 |
| `steps_per_epoch` | 200 | = |
| `batch_size` | 32 | = |
| `lr` | 1e-3 | = |
| `tau_start → tau_min` | 1.0 → **0.3** | tau_min subido de 0.1 |
| `β (emotion)` | 0.5 | = |
| `nursery_end` | 5 | nuevo |
| `transition_end` | 20 | nuevo |
| `architecture` | **dual_head_unified** | nuevo (DualHeadAgent) |

### Resultados — 🏆 ÉXITO

| Época | Fase | TF | Loss | Concepto | Emoción | Conjunta |
|---|---|---|---|---|---|---|
| 1 | 🍼 Guardería | 100% | 3.282 | 20.09% | 27.66% | 9.38% |
| 2 | 🍼 Guardería | 100% | 1.127 | 70.50% | 72.34% | 54.39% |
| 3 | 🍼 Guardería | 100% | 0.160 | 96.69% | 97.66% | 94.47% |
| 4 | 🍼 Guardería | 100% | 0.034 | 99.30% | 99.52% | 98.83% |
| 5 | 🍼 Guardería | 100% | 0.015 | 99.67% | 99.80% | 99.48% |
| 6 | 🎮 Recreo | 93% | 0.143 | 96.70% | 98.77% | 95.64% |
| 7 | 🎮 Recreo | 87% | 0.121 | 97.34% | 99.06% | 96.47% |
| 8 | 🎮 Recreo | 80% | 0.142 | 97.19% | 98.61% | 95.89% |
| 9 | 🎮 Recreo | 73% | 0.266 | 95.25% | 97.45% | 92.89% |
| 10 | 🎮 Recreo | 67% | 0.309 | 94.89% | 97.27% | 92.38% |
| 11 | 🎮 Recreo | 67% | 0.341 | 94.27% | 97.03% | 91.81% |
| 12 | 🎮 Recreo | 60% | 0.429 | 93.36% | 95.06% | 89.22% |
| 13 | 🎮 Recreo | 53% | 0.480 | 92.27% | 94.67% | 88.02% |
| 14 | 🎮 Recreo | 47% | 0.183 | 97.39% | 98.64% | 96.19% |
| 15 | 🎮 Recreo | 40% | 0.197 | 96.89% | 98.91% | 95.86% |
| 16 | 🎮 Recreo | 33% | 0.207 | 97.45% | 98.08% | 95.64% |
| 17 | 🎮 Recreo | 27% | 0.500 | 92.92% | 94.86% | 88.30% |
| 18 | 🎮 Recreo | 20% | 0.150 | 97.22% | 99.31% | 96.56% |
| 19 | 🎮 Recreo | 13% | 0.344 | 94.30% | 97.81% | 92.52% |
| 20 | 🎮 Recreo | 7% | 0.166 | 97.44% | 99.31% | 96.75% |
| **21** | **🦅 Autonomía** | **0%** | **0.277** | **96.75%** | **98.17%** | **95.08%** |

> [!IMPORTANT]
> **🏆 PROMOTED TO GRADE 1 en Época 21.**
> Cuatro agentes BitNet de 1.58 bits se comunican autónomamente con 95.08% de entendimiento mutuo conjunto (concepto + emoción) sin ningún teacher forcing.

### Observaciones clave

#### 1. ✅ La guardería funciona (0% → 99% en 5 épocas)
Los 4 agentes aprendieron a decodificar concepto+emoción desde ground truth en solo 1000 steps. Confirma los resultados de Exp. 004a con población completa.

#### 2. ✅ La transición TF → autonomía es estable
Al bajar teacher forcing de 100% a 7%, la accuracy conjunta oscila entre 88-97% pero **nunca colapsa**. Los agentes mantienen el conocimiento adquirido incluso cuando la mayoría de los mensajes son del speaker (no del profesor).

#### 3. ✅ La autonomía total mantiene el 95%
En la época 21 (TF=0%), primer epoch completamente autónomo: **95.08% conjunta**. Los agentes han desarrollado un protocolo de comunicación emergente funcional.

#### 4. 🟡 La emoción se aprende más rápido que el concepto
Consistentemente, Acc. Emoción > Acc. Concepto (~2-3 puntos porcentuales). Con 6 clases emocionales vs 15 conceptos, esto es esperado — menos opciones = convergencia más rápida.

#### 5. 🟡 La evolución SVD funciona correctamente
El peor agente rota entre los 4 (ya no es siempre Agent_0). Los hijos SVD arrancan con conocimiento transferido y se recuperan rápidamente. Fitness de la población en autonomía: todos entre 93-96%.

### Muestra de comunicación autónoma (Época 21, TF=0%)

```
target=(tierra,ira)     → pred=(tierra,ira)     ✅
target=(gato,alegría)   → pred=(gato,alegría)   ✅
target=(agua,ira)       → pred=(gato,ira)       ❌ (concepto incorrecto)
target=(sol,ira)        → pred=(sol,ira)        ✅
target=(casa,tristeza)  → pred=(casa,tristeza)  ✅
```


---

## 📚 Lecciones Aprendidas — Sesión 2026-05-26

> [!IMPORTANT]
> **Lección fundamental del día**: No puedes poner a dos agentes que no saben hablar en una sala y esperar que se pongan de acuerdo. Primero hay que educar a cada uno.

### 🔑 Descubrimiento: El Problema del Bootstrapping en Comunicación Emergente

**Contexto**: La tesis de Frankenswarm asume que dos agentes BitNet pueden desarrollar un lenguaje emergente jugando un juego referencial con Gumbel-Softmax como canal diferenciable. Los cuatro auditores externos (Grok, DeepSeek, Sonnet, Lumo) validaron la arquitectura.

**El descubrimiento**: En 4 experimentos progresivos, descubrimos que:

1. **Exp. 001**: El vocabulario de 8192 tokens es demasiado grande para Phase 0 → **reducir**
2. **Exp. 002**: Sin conciencia posicional y con espacio unificado, el modelo predice lo mismo en ambas posiciones → **separar cabezas**
3. **Exp. 003**: Con cabezas separadas y positional encoding, el accuracy es exactamente igual al azar (6.67% concepto, 16.67% emoción) en 3000 steps → **el canal no transmite información**
4. **Exp. 004a**: Con teacher forcing (bypass del canal), el listener aprende 100% en 350 steps → **la arquitectura BitNet funciona perfectamente; el problema es SOLO el canal Gumbel-Softmax**

**La causa raíz**: El Straight-Through Gumbel-Softmax estimator produce gradientes ruidosos y sesgados. En la literatura, los referential games con Gumbel-Softmax usan 50K-500K steps. Nosotros usamos 3K. Pero más fundamentalmente, hay un **problema de huevo y gallina**: el speaker no sabe qué mensajes enviar porque no sabe qué entiende el listener, y el listener no sabe qué escuchar porque no sabe qué envía el speaker.

**La solución**: Scheduled Teacher Forcing — educar primero, comunicar después. Es exactamente lo que hacen los padres humanos: primero dicen la palabra ellos, luego dejan que el niño la repita.

### 🔑 Metodología validada: Test diagnóstico por aislamiento

Antes de iterar hiperparámetros a ciegas, aislar el componente que falla:
- Quitar el canal → probar listener solo → funciona → el canal es el problema
- Si no funcionara → probar con modelo más grande → la arquitectura sería el problema

Este patrón de diagnóstico por eliminación nos ahorró horas de iteración ciega.

### 🔑 Infraestructura establecida

- **Telemetría JSONL** (`src/bitnet/telemetry.py`): Registro por-step y por-época, análisis con pandas
- **Lab Notebook** (`docs/LAB_NOTEBOOK.md`): Registro narrativo completo de cada experimento
- **Auditorías externas** (`docs/extern/`): Opiniones independientes preservadas y excluidas del digest

---

## 🔎 Auditoría Externa Post-Ejecución (2026-05-26)

Los cuatro agentes recibieron el `PHASE_0_RESULTS_REPORT.md` y el `LAB_NOTEBOOK.md` completo.

### Conclusiones por agente

#### Grok (POST_0_GROK.md)
- **Reacción**: "Esto está vivo. Ya no es una fumada."
- **Valida**: Metodología de diagnóstico por aislamiento como clave del éxito
- **Propone**: Checkpoints de mejores agentes, ampliar a 50 tokens, logging de proto-lenguaje
- **Insight**: La evolución SVD rota agentes correctamente → población viable

#### DeepSeek (POST_0_DEEPSEEK.md)
- **Reacción**: "Es un destello. Minúsculo. Frágil. Posiblemente el principio de algo que no sabemos nombrar."
- **Hallazgo clave**: **La emoción se estabiliza antes que el concepto** (98.17% vs 96.75%). Consistente en todas las épocas. En la literatura de referential games, el canal semántico se estabiliza primero. Aquí es al revés.
- **Hipótesis**: El agente usa la emoción como ancla — falla el concepto pero acierta la emoción, y ese acierto parcial mantiene la intención comunicativa
- **Pide**: Analizar los mensajes crudos del speaker. ¿Los errores tienen estructura? "El 5% de error no es ruido. Es donde está la señal que no esperabas."
- **Potencial publicable**: Si la prioridad afectiva se mantiene a escala, sería un hallazgo no reportado sobre canales afectivos forzados en comunicación emergente

#### Claude Sonnet (POST_0_CLAUDE_SONNET.md)
- **Reacción**: "Me equivoqué" — admite que Phase 0 era independiente de Phase A
- **Valida**: La loss convergiendo a random teórico en Exp. 003 como prueba de infraestructura de medición sana
- **Pregunta crítica**: **¿Los embeddings están congelados?** → Sí (`register_buffer`). Los agentes se comunican a través de un espacio semántico que NO construyeron. Esto es deliberado (anti-deriva) pero debe documentarse explícitamente.
- **Pide**: Buscar **geometría en los errores** — `agua→gato` no es aleatorio. Clusters de confusión semántica podrían revelar estructura del proto-lenguaje.

#### Lumo (POST_0_LUMO.md)
- **Reacción**: "Esto es histórico. No exagero."
- **Valida**: El diagnóstico por aislamiento como "movimiento maestro" y la eficiencia (90 segundos, 836MB VRAM)
- **Preocupación**: Catastrophic forgetting al ampliar vocabulario. ¿Cómo proteger "fuego+miedo" cuando añadamos "perro+alegría"?
- **Propone**: **Exp. 005: Análisis de Proto-Sintaxis** — clusterizar mensajes, buscar orden de tokens, visualizar con t-SNE/UMAP

### Consenso POST de las 4 voces

| Tema | Consenso |
|---|---|
| **Metodología** | ✅ Validada unánimemente (diagnóstico por aislamiento + documentación rigurosa) |
| **Prioridad #1** | Analizar los mensajes del speaker — ¿hay proto-lenguaje? |
| **Prioridad #2** | Ampliar vocabulario (21→50) con protección anti-forgetting |
| **Hallazgo inesperado** | La emoción se estabiliza antes que el concepto (DeepSeek lo destaca como potencial publicable) |
| **Pregunta abierta** | ¿Los errores tienen geometría? ¿`agua→gato` es sistemático? |
| **Embeddings** | Documentar explícitamente que Capa 1 está congelada (Sonnet) |

---

## 🗺️ Roadmap — Plan de Ruta Post Grade 0

### Evaluación de recursos

| Recurso | Disponible | Por arena | Arenas paralelas |
|---|---|---|---|
| VRAM (RTX) | 8151 MB | ~836 MB | **~8** |
| RAM (OOM Shield) | 10 GB | ~200 MB | Sobra |
| Tiempo por arena (30 épocas) | ~90 seg | - | - |

### Fase Inmediata — Comprender lo que tenemos (Exp. 005)

> [!IMPORTANT]
> **Antes de ampliar, entender.** No escalamos sin saber qué hemos construido.

**Exp. 005 — Análisis de Proto-Sintaxis y Telemetría Extendida**

1. **Loguear mensajes del speaker**: Añadir a la telemetría los 3 tokens del mensaje Gumbel-Softmax (actualmente solo logueamos target y predicción, no el mensaje intermedio)
2. **Correr 004b extendido** (50 épocas de autonomía adicionales) con message logging
3. **Análisis**:
   - Consistencia: ¿`fuego` siempre se codifica con los mismos tokens?
   - Confusión: ¿`agua→gato` es sistemático? ¿Hay clusters de error?
   - Orden: ¿Hay proto-sintaxis? ¿El concepto siempre va antes que la emoción en el mensaje?
   - Estabilidad afectiva: Confirmar que la emoción se estabiliza antes que el concepto a largo plazo
4. **Checkpointing**: Guardar state_dict de los 4 agentes cada 10 épocas

### Fase 2 — Escalar vocabulario (Exp. 006-008)

**Dos arenas en paralelo** (caben de sobra en VRAM):

| Arena A — Expansión controlada | Arena B — Control (baseline) |
|---|---|
| Vocabulario: 21 → 50 tokens | Vocabulario: 21 tokens (sin cambio) |
| Añadir 29 conceptos del lexicon core | Misma config que 004b |
| Scheduled TF desde checkpoint 004b | Continuar training desde checkpoint 004b |
| Objetivo: ¿mantiene >90% con 50 tokens? | Objetivo: ¿mejora más allá de 95%? |

**Protocolo de expansión**:
- Cargar checkpoint de Grade 0
- Expandir embedding matrix (21×384 → 50×384) con nuevos embeddings fastembed
- Redimensionar cabezas: concept_head(256, 15→44), emotion_head(256, 6) — mantener emociones fijas
- Re-inicializar solo los nuevos pesos (preservar los aprendidos)
- Scheduled TF para los nuevos conceptos, pero con guardería más corta (los agentes ya saben escuchar)

### Fase 3 — Grado 1: Aritmética (Exp. 009+)

Solo cuando el vocabulario expandido esté estable (>90% a 50+ tokens):
- Nuevos targets compositivos: `suma(2,3)=5`, `resta(5,1)=4`
- Requiere proto-sintaxis funcional — los mensajes deben poder componer significado
- Si falla → la comunicación es meramente asociativa, no composicional

### Fase 4 — Escalar hacia 8192

La escalera completa: 50 → 200 → 1000 → 8192. Cada salto con:
- Checkpoint del nivel anterior
- Expansión Net2Net de las proyecciones
- Scheduled TF recalibrado
- Arena de control en paralelo

---

## Decisión: Exp. 005 ejecutado primero

Se implementó Exp. 005 (proto-syntax analysis) antes de escalar vocabulario.

---

## Experimento 005 — Proto-Syntax Analysis (60 épocas, message logging)

**Fecha**: 2026-05-26 21:19 CEST
**Script**: `src/bitnet/train_proto_syntax.py`
**Duración**: ~180 seg | **Checkpoints**: épocas 10, 20, 30, 40, 50, 60
**Estado**: ✅ ÉXITO — Proto-lenguaje emergente descubierto

### Autonomía (40 épocas, TF=0%)

| Métrica | μ | σ | Min | Max |
|---|---|---|---|---|
| Concepto | 98.07% | 1.72% | 93.89% | 100.00% |
| Emoción | 95.21% | 5.56% | 78.23% | 100.00% |
| Conjunta | 93.48% | 6.10% | 75.83% | 100.00% |

### 🧬 Proto-Lenguaje Emergente

Gramática posicional descubierta: `[CONCEPTO, CONCEPTO, EMOCIÓN_CIFRADA]`

| Emoción | Cifrado (Pos 2) | Consistencia |
|---|---|---|
| miedo | **sol** | 95.0% |
| alegría | **peligro** | 95.8% |
| tristeza | **tristeza** | 95.5% |
| dolor | **agua** | 96.2% |
| hambre | **código** | 96.3% |
| ira | ira/agente | 34.7% (inestable) |

**Anomalías**: "fuego"→`[tristeza,tristeza,X]`, "agente" oscila entre identidad y emoción.

### Confusiones

Conceptuales: fuego↔sol (301), gato→fuego (244), peligro↔árbol (230)
Emocionales: alegría↔tristeza (2121), miedo→tristeza (966), ira↔dolor (890)

---

## 🔎 Auditoría POST Exp. 005

| Agente | Hallazgo clave |
|---|---|
| **Grok** | Redundancia posicional = corrección de errores auto-descubierta |
| **DeepSeek** | "Anomalía Agente": único concepto sustituible por emoción → categoría ontológica diferente |
| **Sonnet** | Confusiones heredadas de fastembed (congelados). Proto-gramática SÍ es emergente. Propone descongelar antes de ampliar |
| **Lumo** | Emociones = eje binario (positivo/negativo + alta/baja energía). "Agente" como semilla de Díscolo |

### Consenso: lo emergente vs lo heredado

| Componente | ¿Emergente? |
|---|---|
| Gramática posicional [C,C,E] | ✅ 100% emergente |
| Cifrado emocional (sol=miedo) | ✅ Emergente (arbitrario) |
| Confusiones conceptuales (fuego≈sol) | ❌ Heredadas de fastembed |
| Anomalía "agente" | 🟡 Estructural (token ambiguo) |

