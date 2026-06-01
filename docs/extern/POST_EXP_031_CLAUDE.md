# Expert Review #4 — Frankenswarm Technical Brief
## Fecha: 29 Mayo 2026

### Reviewer: Modelo Externo #4 (Cloud)
### Veredicto: "Más que memorización pura, pero no podéis descartar lookup implícito sobre el grafo."
### Estilo: Metodológicamente el más riguroso de los 4 reviewers.

---

## Respuestas a las 5 Preguntas

### P1. Transitividad
- Con 12 conceptos y 13 reglas, no se puede descartar memorización estadística.
- **Test propuesto (SCAN-style)**: Entrenar sin un subconjunto de REGLAS directas (no solo pares inicio/fin), testar si infiere cadenas que DEPENDEN de esas reglas ausentes.
- "El diseño actual — excluir pares inicio/fin pero incluir reglas intermedias — deja abierta la posibilidad de combinación superficial."

### P2. Techo Gumbel-Softmax
- "El hallazgo más sólido del documento porque es reproducible en tres configuraciones."
- **Opciones, por orden de recomendación**:
  1. **VQ-VAE** — codebook discreto, STE para gradientes. "La más prometedora para vuestro caso."
  2. **Discrete bottleneck τ fijo (~0.1)** — discreto desde el inicio, más difícil de bootstrapear.
  3. **argmax + STE puro** — canal completamente discreto, gradiente menos informativo pero canal sin degradación.

### P3. Escalabilidad del curriculum
- Funciona en modelos pequeños donde capacidad es bottleneck. En grandes, memorizan en vez de generalizar.
- Literatura de emergent communication (Lazaridou, Mordatch): todos los éxitos son modelos pequeños.
- "Evidencia razonable de que escala a ~7M con restricciones adecuadas."
- **Riesgo**: hotstart propaga sesgos. **Solución**: layer freezing selectivo al escalar.

### P4. Fear vs Reward Shaping
- "Funcionalmente sí, con diferencia importante." Fear opera per-sample, RL opera per-episodio.
- **Clasificación técnica correcta**: Lo que tenéis es **importance weighting en loss** — técnica establecida en class imbalance.
- "Llamarle fear es correcto semánticamente y técnicamente honesto."

### P5. ¿Razonamiento real?
- "Los agentes generalizan a combinaciones no vistas. Eso es más que memorización pura."
- "Lo que no podéis descartar: función de lookup implícita sobre el grafo."

---

## EL EXPERIMENTO DEFINITIVO (propuesto por R4)

### Test de Transferencia Estructural

> **Construid un grafo causal ISOMORFO con vocabulario completamente distinto.**

1. Grafo original: `fuego→peligro`, `luna→agua→seguridad`, etc.
2. Grafo nuevo (misma topología, mismos fear scores): `tormenta→inundación`, `estrella→río→refugio`, etc.
3. Cargar pesos de EXP_028A (SIN hotstart del nuevo dominio).
4. Entrenar el grafo nuevo.
5. **Medir: ¿cuántas épocas necesita para converger vs desde cero?**

**Interpretación:**
- Si aprende SIGNIFICATIVAMENTE más rápido → **hay transferencia de estructura relacional**. Los pesos codifican la TOPOLOGÍA del grafo, no solo los nombres.
- Si aprende a la MISMA velocidad → los pesos codifican CONTENIDO, no estructura. Era pattern matching.

> "Ese experimento sería **el más publicable** de todos los que habéis hecho."

---

## Insights Nuevos (no en reviews anteriores)

1. **SCAN-style test**: Entrenar sin REGLAS (no solo pares), testar inferencia dependiente de reglas ausentes. Más limpio que excluir solo pares inicio/fin.
2. **argmax + STE puro**: Canal completamente discreto. Más simple que VQ-VAE. Bootstrapeado por Teacher Forcing existente.
3. **Layer freezing selectivo**: Solución concreta para escalar curriculum sin propagación de sesgos.
4. **Fear = importance weighting**: Clasificación técnica precisa que ancla la narrativa del "miedo" a literatura establecida.
5. **Transferencia estructural**: EL test definitivo para distinguir composición de memorización.

---

## Síntesis Final — 4 Reviewers

### Tabla comparativa

| Aspecto | R1 | R2 | R3 | R4 |
|---------|-----|-----|-----|-----|
| **Veredicto** | "Frontera" | "Pattern matching" | "Composición con salvedad" | "Más que memorización, pero..." |
| **Transitividad** | Moderada-fuerte | Bias inductivo | Composición real | Lookup implícito posible |
| **Fear** | Elegante | Atención selectiva | Más estable que RL | Importance weighting |
| **Bottleneck** | ✅ | ✅ | ✅ | ✅ "Hallazgo más sólido" |
| **Solución canal** | VQ-VAE | STE hard τ=0.5 | VQ-VAE + residuals | VQ-VAE o argmax+STE |
| **Escalabilidad** | 7M-70M | ≤10M | Desconocido | ~7M con restricciones |
| **Test propuesto** | Reglas aleatorias | Concepto nuevo + inversión | Perturbación + abducción | **Transferencia estructural** |

### UNANIMIDAD (4/4):
1. ✅ Bottleneck Gumbel-Softmax → hallazgo real y reproducible
2. ✅ Curriculum con hotstart → sólido
3. ✅ 24 tokens → insuficiente para afirmar razonamiento definitivo
4. ✅ Necesita tests fuera de distribución
5. ✅ VQ-VAE como dirección más prometedora para el canal

### Roadmap Priorizado (consolidado 4 reviews)

| Prioridad | Experimento | Propuesto por | Qué resuelve |
|-----------|-------------|---------------|---------------|
| 🔴 #1 | **Transferencia estructural** (grafo isomorfo) | R4 | ¿Estructura o contenido? PUBLICABLE |
| 🔴 #2 | **100 conceptos + reglas procedurales** | R1 | ¿Escala el razonamiento? |
| 🔴 #3 | **STE hard / argmax** | R2 + R4 | ¿El techo 67% sube? |
| 🟡 #4 | Concepto nuevo (ratón/tormenta) | R2 + R3 | Composición vs memorización |
| 🟡 #5 | Perturbación semántica | R3 | Estructura vs nombres |
| 🟡 #6 | SCAN-style (excluir reglas) | R4 | Composición más limpia |
| 🟢 #7 | VQ-VAE channel | R1 + R3 + R4 | Alternativa Gumbel (siguiente fase) |
| 🟢 #8 | Test de abducción | R3 | Razonamiento backward |
