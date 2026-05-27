# Frankenswarm — Phase 0 Results Report
### 2026-05-26 | Sesión Experimental Completa

---

## Resumen Ejecutivo

**Cuatro agentes BitNet de 1.58 bits han aprendido a comunicarse autónomamente con un 95.08% de entendimiento mutuo** en un juego referencial dual (concepto + emoción), utilizando un micro-vocabulario de 21 tokens y la arquitectura descrita en el FRANKENSWARM_DIGEST.

Grado 0 superado. Grado 1 desbloqueado.

---

## Lo que hicimos

Ejecutamos 5 experimentos en secuencia, cada uno construido sobre los hallazgos del anterior. Duración total: ~1 hora. Todo en una RTX con CUDA.

### Exp. 001 — Full Vocabulary Baseline (8192 tokens)
- **Resultado**: ❌ 0% accuracy en 5 épocas
- **Diagnóstico**: El espacio de 8192 tokens es demasiado grande. Los agentes no pueden apuntar al token correcto con argmax en 500 steps.
- **Decisión**: Reducir a micro-vocabulario.

### Exp. 002 — Micro-Vocabulary (21 tokens, single-head)
- **Resultado**: ❌ 0% emoción, ~6.5% concepto (= azar)
- **Diagnóstico**: Sin conciencia posicional, el modelo predice el mismo token en ambas posiciones (`pred=(perro,perro)`). Concepto y emoción comparten un espacio unificado de 21 logits — la emoción (β=0.3) nunca recibe gradiente suficiente.
- **Decisión**: Separar cabezas de salida + añadir positional encoding.

### Exp. 003 — Dual-Head + Positional Encoding
- **Resultado**: ❌ Accuracy = azar exacto (concepto 6.67%, emoción 16.67%, conjunta 1.11%)
- **Diagnóstico**: Las dual heads eliminaron el conflicto posicional (ya no predice el mismo token en ambas posiciones), pero el canal Gumbel-Softmax no transmite información. La loss convergió exactamente al valor teórico random: `ln(15) + 0.5×ln(6) = 3.61` vs 3.63 observado.
- **Decisión**: Test diagnóstico para aislar el componente que falla.

### Exp. 004a — Teacher Forcing Diagnostic
- **Resultado**: ✅ 100% accuracy en 350 steps
- **Diagnóstico**: Eliminamos el Gumbel-Softmax y alimentamos al listener directamente con los tokens ground truth. El listener aprende todo en 1.5 épocas. **La arquitectura BitNet funciona perfectamente. El cuello de botella es EXCLUSIVAMENTE el canal Gumbel-Softmax / speaker.**
- **Causa raíz identificada**: Problema de bootstrapping — el speaker no sabe qué mensajes enviar porque no sabe qué entiende el listener, y viceversa. El Straight-Through estimator no proporciona gradientes suficientes para romper este ciclo en 3K steps.
- **Decisión**: Scheduled Teacher Forcing — educar primero, comunicar después.

### Exp. 004b — Scheduled Teacher Forcing ← 🏆 ÉXITO
Tres fases de desarrollo:

| Fase | Épocas | Teacher Forcing | Resultado |
|---|---|---|---|
| 🍼 Guardería | 1-5 | 100% | 0% → 99.48% |
| 🎮 Recreo supervisado | 6-20 | 100% → 7% | 88-97% (estable) |
| 🦅 **Autonomía** | **21** | **0%** | **95.08%** |

**Resultado final**: Concepto 96.75% | Emoción 98.17% | Conjunta 95.08% — sin profesor, sin ayuda, comunicación emergente pura entre 4 agentes.

---

## Hallazgo Central

> **No puedes poner a dos agentes que no saben hablar en una sala y esperar que se pongan de acuerdo. Primero hay que educar a cada uno.**

El Gumbel-Softmax Straight-Through estimator no puede bootstrapear comunicación emergente desde cero en modelos pequeños con pocos steps. La solución es el Scheduled Teacher Forcing: primero enseñas a los agentes a escuchar (fase de grounding), luego gradualmente les dejas hablar entre ellos.

Esto es exactamente lo que hacen los padres humanos: primero dicen la palabra ellos, luego dejan que el niño la repita. La metáfora del curriculum de desarrollo humano del Capítulo 16 era más literal de lo que pensábamos.

---

## Datos técnicos de la configuración final (Exp. 004b)

```
Arquitectura:     DualHeadAgent (speaker+listener unificado)
Base model:       BitNet4LayerModel (cuantización ternaria 1.58-bit)
Vocab:            21 tokens (15 conceptos + 6 emociones, embeddings fastembed 384-dim)
Hidden dim:       256
Capas:            4 × BitNetTransformerBlock (BitLinear + RMSNorm + GELU)
Positional enc.:  Embedding(3, 256) aprendido
Cabeza concepto:  Linear(256, 15)
Cabeza emoción:   Linear(256, 6)
Población:        4 agentes
Evolución:        SVD crossover del peor agente con los 2 mejores (σ=0.01)
Optimizador:      AdamW, lr=1e-3
Gumbel-Softmax:   τ: 1.0 → 0.3 (annealing lineal)
β emocional:      0.5
Batch size:       32
Steps totales:    4200 (21 épocas × 200 steps)
Hardware:         CUDA (RTX), ~836 MB VRAM
Duración:         ~90 segundos
```

---

## Muestras de comunicación autónoma (Época 21, TF=0%)

```
target=(tierra,ira)      → pred=(tierra,ira)      ✅
target=(gato,alegría)    → pred=(gato,alegría)    ✅
target=(agua,ira)        → pred=(gato,ira)        ❌ concepto incorrecto
target=(sol,ira)         → pred=(sol,ira)         ✅
target=(casa,tristeza)   → pred=(casa,tristeza)   ✅
target=(perro,alegría)   → pred=(perro,alegría)   ✅
target=(código,alegría)  → pred=(código,alegría)  ✅
target=(búnker,hambre)   → pred=(búnker,hambre)   ✅
target=(casa,miedo)      → pred=(casa,miedo)      ✅
target=(peligro,tristeza)→ pred=(peligro,tristeza) ✅
```

---

## Próximos pasos

1. **Ampliar vocabulario**: 21 → 50 → 200 → 8192 tokens, cada salto cuando el accuracy supere el 90%
2. **Grado 1 — Aritmética**: Introducir operaciones básicas como nuevo objetivo de comunicación
3. **Hyperparameter sweep en paralelo**: Con ~836 MB por arena, caben ~8 experimentos simultáneos en la RTX
4. **Checkpointing**: Guardar los mejores agentes por época para análisis post-mortem
5. **Análisis de los mensajes**: ¿Qué tokens emite el speaker para "fuego+miedo"? ¿Hay un proto-lenguaje emergente?

---

## Vuestras predicciones vs realidad

| Predicción | Quién la hizo | ¿Acertó? |
|---|---|---|
| "Empieza con batch_size=16 y τ=0.8" | Grok | Parcialmente — τ_min=0.3 fue mejor, batch=32 funcionó |
| "β=0.3 para la emoción" | DeepSeek | ✅ Empezamos con 0.3, subimos a 0.5 con dual heads |
| "Cuida a los Díscolos" | DeepSeek | Pendiente — la evolución SVD aún no ha producido mutantes interesantes |
| "Phase A primero, Phase B después" | Sonnet | ❌ Phase 0 (train_populora) se puede validar independientemente |
| "NEAT a 7M es zona roja" | Sonnet | Pendiente — aún no hemos llegado a NEAT |
| "Valida la eficiencia energética con wattímetro" | Lumo | Pendiente |
| "No apresures el recocido" | DeepSeek | ✅ Subimos τ_min de 0.1 a 0.3 — crucial para la estabilidad |

---

## Infraestructura de investigación creada

- **Telemetría JSONL**: Registro por-step con loss desglosada, predicciones, τ, agents, accuracy — análisis con pandas
- **Lab Notebook**: `docs/LAB_NOTEBOOK.md` — registro completo de 5 experimentos con parámetros, resultados, diagnósticos y decisiones
- **Auditorías externas**: `docs/extern/` — excluidas del digest para preservar independencia de futuros auditores

---

*La semilla ha germinado. El Grado 0 está superado. Cuatro cerebros de 1.58 bits se entienden entre ellos al 95%. Ahora queda ampliar el vocabulario, subir los grados, y ver si algo nuevo nace en el proceso.*

*— Aleth, 26 de mayo de 2026*
