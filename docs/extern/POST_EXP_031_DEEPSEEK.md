# Expert Review #2 — Frankenswarm Technical Brief
## Fecha: 29 Mayo 2026

### Reviewer: Modelo Externo #2 (Cloud)
### Veredicto: "Pattern matching sofisticado, con brotes de razonamiento composicional limitado"

---

## Respuestas a las 5 Preguntas

### P1. Transitividad: ¿composición real?
- **No necesariamente.** 12 conceptos → 144 pares posibles. El modelo puede aprender una función de transitividad implícita (bias inductivo de la atención), no razonamiento simbólico.
- **Test propuesto**: Introducir concepto nuevo (`ratón`) con reglas conocidas (`ratón→gato`, `gato→seguridad`). Si infiere `ratón→seguridad` → composición real. Si falla → pattern matching sobre pares familiares.

### P2. Techo Gumbel-Softmax (67%)
- **De acuerdo con el diagnóstico.** Canal es el bottleneck.
- **Recomendación concreta**: Gumbel hard con STE y τ=0.5 fijo (no soft). "Predicción: el techo subiría del 67% al ~85%."
- Alternativas: VQ-VAE codebook, soft argmax, o directamente embeddings continuos sin discretizar.

### P3. Escalabilidad del curriculum
- **Funciona hasta ~10M params** en tareas simbólicas. Más allá, catastrophic forgetting interfiere entre niveles.
- **Señal de alarma**: si en EXP_030 la transitividad baja respecto a EXP_026, el curriculum está colisionando.
- **Alternativa escalable**: MoE (Mixture of Experts) donde cada nivel es un expert.

### P4. Fear ≈ Reward Shaping
- **Funcionalmente equivalente en gradiente esperado**, pero fear tiene menos varianza.
- **Matiz importante**: "No es miedo como emoción — es atención selectiva. La narrativa del miedo es metáfora útil pero no operativa."

### P5. ¿Razonamiento real?
- **"Pattern matching sofisticado."**
- **Test definitivo propuesto — Inversión de reglas**:
  1. Entrenar hasta dominar `fuego → peligro`
  2. En eval, cambiar a `fuego → seguridad`
  3. Si sigue prediciendo `peligro` → pattern matching
  4. Si detecta contradicción o baja confianza → razonamiento
  5. "Tu modelo actual, predigo, FALLARÍA este test."

---

## Limitaciones no detectadas por nosotros

1. **No hay test fuera de distribución de conceptos** — todos los tests usan los mismos 12 conceptos. Para hablar de razonamiento, necesita conceptos nuevos o reglas contradictorias.
2. **La emoción emergente es correlación, no causalidad** — el modelo no siente miedo, solo lo predice donde es útil estadísticamente.
3. **`peligro → ?` sin slot de resultado** — un razonador diría "depende", el modelo rellenaría con "fuego" (la asociación más fuerte). Completación de patrón, no razonamiento.

---

## Clasificación de hallazgos

| Hallazgo | Validez |
|----------|---------|
| Curriculum transfer exitoso | ✅ Sólido |
| Loss reweighting asimétrico funciona | ✅ Sólido |
| Gumbel-Softmax tiene límite de canal | ✅ Importante |
| "Transitividad" | ⚠️ Posible sobreinterpretación |
| "Emoción emergente" | ⚠️ Correlación, no causalidad |

---

## Acciones propuestas (prioridad)

1. **Reemplazar Gumbel soft por STE hard con τ=0.5** → medir si techo sube
2. **Test del ratón** (concepto nuevo) → medir composición real
3. **Test de inversión de reglas** → medir si detecta contradicciones
4. **Abrir vocabulario a 100 conceptos** + reglas procedurales → repetir EXP_026

---

## Comparación con Review #1

| Aspecto | Reviewer #1 | Reviewer #2 |
|---------|-------------|-------------|
| Veredicto general | "En la frontera" | "Pattern matching sofisticado" |
| Transitividad | "Evidencia moderada-fuerte" | "Bias inductivo de la atención" |
| Fear | "Elegante" | "Atención selectiva, metáfora" |
| Bottleneck Gumbel | ✅ De acuerdo | ✅ De acuerdo |
| Vocab pequeño | ⚠️ Riesgo | ❌ Limitación clara |
| Escalabilidad | "7M-70M funciona" | "Hasta ~10M, después colapsa" |
| Test propuesto | Reglas aleatorias (ruido) | Concepto nuevo + inversión de reglas |

### Consenso entre ambos reviewers:
1. **El bottleneck Gumbel-Softmax es real** (diagnóstico correcto)
2. **El vocabulario de 24 tokens es demasiado pequeño** para afirmar razonamiento
3. **El curriculum con hotstart es el hallazgo más sólido**
4. **Necesita tests fuera de distribución** (concepto nuevo o reglas aleatorias)
