# Frankenswarm: Emergent Reasoning in Ternary Neural Networks
## Technical Brief — Mayo 2026

**Objetivo**: Documentar resultados experimentales de razonamiento emergente en modelos BitNet (pesos ternarios {-1, 0, 1}) entrenados con un curriculum progresivo, presión evolutiva y modulación de pérdida por "miedo".

**Solicitud**: Opinión experta sobre la validez de los hallazgos, limitaciones no detectadas y direcciones de investigación.

---

## 1. Arquitectura

```
BitNet4LayerModel (≈2M params)
├── Vocab Embeddings:  (8192, 384) — buffer fijo, pre-entrenado con SentenceTransformers
├── Inbound Proj:      Linear(384 → 256)
├── Positional Embed:  Learnable (1, 4, 256)
├── Core:              3× BitNetTransformerBlock(256, 4 heads, MLP ratio=4)
│   ├── BitLinear:     Pesos cuantizados a {-1, 0, 1} via STE (Straight-Through Estimator)
│   ├── Attention:     Multi-head con Q,K,V BitLinear
│   └── MLP:           BitLinear up(256→1024) + GELU + BitLinear down(1024→256)
├── RMSNorm
└── Outbound Proj:     Linear(256 → 384) → matmul con vocab_embeddings.T → logits (8192)
```

**Comunicación inter-agente**: Gumbel-Softmax (`hard=False` para gradientes, `hard=True` para evaluación). Los mensajes son vectores one-hot relajados de dimensión 8192, transmitidos entre agentes de forma diferenciable.

**Vocabulario operativo**: 24 tokens de un espacio de 8192. Logit mask restringe la generación a estos 24 tokens:
- Conceptos (12): gato, perro, casa, árbol, agua, fuego, tierra, aire, sol, luna, peligro, seguridad
- Operadores (4): implica, contradice, cadena, niega
- Resultados (2): verdad, falsedad
- Emociones (6): miedo, alegría, ira, tristeza, dolor, hambre

**Formato de secuencia**: 4 slots fijos — `[concepto_A, operador, concepto_B, resultado]`

---

## 2. Grafo Causal (Mundo del Agente)

13 reglas de implicación causal definen el "mundo" del agente:

```
fuego → peligro    (fear=0.9)     luna → agua       (fear=0.2)
fuego → tierra     (fear=0.7)     tierra → agua     (fear=0.3)
agua → seguridad   (fear=0.1)     seguridad → casa  (fear=0.0)
casa → seguridad   (fear=0.1)     sol → aire        (fear=0.3)
gato → seguridad   (fear=0.1)     árbol → aire      (fear=0.2)
perro → seguridad  (fear=0.1)     aire → sol        (fear=0.1)
peligro → fuego    (fear=0.95)
```

De estas 13 reglas directas emergen **12 cadenas transitivas** de 2-3 saltos. Ejemplos:
- `luna → agua → seguridad → casa` (3 saltos, fear_max=0.20)
- `fuego → tierra → agua → seguridad` (3 saltos, fear_max=0.70)
- `peligro → fuego → tierra → agua` (3 saltos, fear_max=0.95)

---

## 3. Mecanismo de Miedo (Asymmetric Loss)

El "miedo" no es un canal separado ni una emoción simulada. Es un **multiplicador escalar de la función de pérdida**:

```python
fear_weight = 1.0 + concept_fear_score × fear_amplifier
loss_per_sample = cross_entropy(prediction, target) × fear_weight
```

Efecto: equivocarse en `fuego implica peligro` (fear=0.9, amplifier=3.0) genera **3.7×** más gradiente que equivocarse en `gato implica seguridad` (fear=0.1). Esto crea presión selectiva asimétrica: los agentes que internalizan primero las reglas de supervivencia tienen mayor fitness y sobreviven al crossover SVD.

---

## 4. Evolución (SVD Crossover)

Población de 4 agentes. Cada N épocas:
1. El agente con peor fitness es eliminado
2. Se genera un hijo cruzando los 2 mejores agentes vía SVD:
   - Promedio ponderado de pesos: `W_avg = α·W_a + (1-α)·W_b`
   - Descomposición SVD: `U, S, Vh = svd(W_avg)`
   - Perturbación de valores singulares: `S' = S + N(0, σ)`
   - Reconstrucción: `W_child = U · diag(S') · Vh`
3. El hijo reemplaza al peor agente con un optimizador nuevo

---

## 5. Curriculum y Resultados

Cada experimento hace hotstart desde el checkpoint del anterior. Los pesos acumulan capacidades:

### Nivel 3: Implicación Causal (EXP_025)
- **Tarea**: `fuego implica peligro = verdad/falsedad` (154 ecuaciones)
- **Config**: 4 agentes, 60 épocas, 200 steps, batch=32, lr=0.001, fear_amp=2.0
- **Resultado**: Train=99%, Test=89%, Survival=84%

### Nivel 3.5: Transitividad (EXP_026) — Hotstart desde EXP_025
- **Tarea**: `luna cadena seguridad = verdad` — la cadena `luna→agua→seguridad` nunca fue enseñada directamente (79 ecuaciones)
- **Config**: 80 épocas, 250 steps, fear_amp=2.5, cadena loss_weight=1.5
- **Resultado**: Train=99.88%, **Test=94.75%**, **Survival=100%**
- **Significado**: El modelo infiere relaciones transitivas no vistas. Test contiene pares (inicio, fin) que solo existen como cadenas multi-salto.

### Nivel 4: Modus Tollens (EXP_027) — Hotstart desde EXP_026
- **Tarea**: `seguridad niega luna = verdad` — si `luna→agua→seguridad`, entonces `¬seguridad → ¬luna` (96 ecuaciones)
- **Config**: 80 épocas, 300 steps, fear_amp=3.0, niega loss_weight=2.0, TF_min=8%
- **Resultado**: Train=99.86%, **Test=92.54%**, Survival=86%
- **Significado**: Razonamiento contrafactual. El modelo invierte la dirección de inferencia incluyendo cadenas transitivas inversas.

### Nivel 4.5: Emoción como Contexto (EXP_028A vs 028B)

**A/B test** — ambos hotstart desde EXP_027:

| | EXP_028A (Input) | EXP_028B (Target) |
|---|---|---|
| **Mecanismo** | Emoción inyectada en slot 3 del speaker | Loss auxiliar CE para predecir emoción en posición 0 |
| **Train** | 99.83% | 87.91% |
| **Test (pico)** | **96.49%** | 79.82% |
| **Survival** | **92%** | 77% |
| **Train 100%** | Sí (época 56) | Nunca |

**Observación**: En modo target, el agente predice correctamente la emoción "miedo" para conceptos peligrosos (fuego, peligro) pero falla en conceptos neutros. La emoción emerge donde hay stakes, no donde no las hay.

### Nivel 5: Razonamiento Multi-Paso en Loop (EXP_029, 030A, 030B)

Tres configuraciones, todas hotstart desde EXP_028A:

| Experimento | Configuración | Paso 1 | Cadena Completa |
|---|---|---|---|
| EXP_029 | Un modelo, 2 pasos encadenados (A→A) | **100%** | **70%** |
| EXP_030A | Dos modelos, pipeline (🧠0→🧠1) | ~100% | 67% |
| EXP_030B | Dos modelos, loop 3 iter (🧠0→🧠1→🧠0) | ~100% | 66% |

**Mecanismo del loop**: La salida Gumbel-Softmax (`hard=False`) del paso N se inyecta como input del paso N+1. Los gradientes fluyen a través de todo el loop vía backpropagation.

**Resultado**: El techo de ~67% es **independiente** de la configuración (1 modelo vs 2 modelos vs loop). El paso 1 siempre acierta. La degradación ocurre en la transferencia de información entre pasos.

---

## 6. Hallazgos Principales

### H1: El curriculum progresivo con hotstart permite acumulación de capacidades
Sin hotstart, EXP_027 (modus tollens) no converge. Con hotstart desde EXP_026, converge al 92% en test. La transferencia de conocimiento entre niveles de abstracción es la clave.

### H2: La presión selectiva asimétrica (fear) funciona como esperado
Los agentes priorizan reglas de supervivencia. Survival (precisión solo en ejemplos con fear>1.5) alcanza 100% repetidamente, incluso cuando la precisión general cae.

### H3: La emoción es más efectiva como input que como target
Conflicto de objetivos: predecir emoción Y resolver lógica con el mismo cerebro de 4 slots degrada ambas tareas. Como input contextual, la emoción modula sin interferir.

### H4: El canal Gumbel-Softmax pierde ~30% de resolución por paso
El hallazgo más importante de la sesión. Los vectores one-hot relajados pierden información al pasar entre pasos/agentes. Este es un límite **del canal**, no de la inteligencia de los agentes. Evidencia: los tres experimentos de loop (029, 030A, 030B) convergen al mismo techo (~67%) independientemente de la topología.

---

## 7. Preguntas para los Expertos

1. **Validez del test de transitividad**: ¿95% en cadenas no vistas (EXP_026) constituye evidencia de composición relacional, o podría explicarse por memorización de patrones estadísticos en un vocabulario de 12 conceptos?

2. **El techo Gumbel-Softmax**: ¿Existen técnicas conocidas para reducir la degradación de señal en chains diferenciables multi-paso? ¿Vector Quantization (VQ-VAE style)? ¿Straight-Through con más bits? ¿Alguna alternativa a Gumbel que preserve más información?

3. **Escalabilidad del curriculum**: ¿Hay evidencia en la literatura de que este tipo de curriculum progresivo (hotstart capa sobre capa) escala más allá de ~2M parámetros? ¿O colapsa en modelos más grandes?

4. **Fear vs Reward Shaping**: ¿El multiplicador de loss asimétrico es funcionalmente equivalente a reward shaping en RL? ¿Hay razones para preferir uno sobre otro en este contexto?

5. **La pregunta de fondo**: A este nivel (24 tokens, 4 slots, 2M params), ¿estamos observando razonamiento emergente o pattern matching sofisticado? ¿Qué test adicional propondrían para distinguir uno del otro?

---

## 8. Reproducibilidad

- **Hardware**: RTX 5070 Ti (16GB VRAM), portátil
- **Framework**: PyTorch, todo en un solo repositorio
- **Tiempo por experimento**: 3-5 minutos (60-80 épocas)
- **Seeds**: Todos los experimentos usan seed=42
- **Código fuente**: Disponible en el repositorio Frankenswarm
- **Checkpoints**: Preservados en `storage/experiments/EXP_0XX/`
