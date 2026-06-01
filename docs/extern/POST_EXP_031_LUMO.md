# Expert Review #3 — Frankenswarm Technical Brief
## Fecha: 29 Mayo 2026

### Reviewer: Modelo Externo #3 (Cloud)
### Veredicto: "Investigación de frontera real. Composición relacional, con salvedad del vocabulario pequeño."
### Calificación: 9/10 validez, 8.5/10 innovación, 9/10 impacto potencial

---

## Respuestas a las 5 Preguntas

### P1. Transitividad: ¿composición real?
- **"Es composición relacional, pero con salvedad crítica."**
- Evidencia a favor: cadenas nunca enseñadas, hotstart necesario, test out-of-distribution.
- Salvedad: vocabulario de 12 conceptos → grafo latente memorizable.
- **Test propuesto (Perturbación Semántica)**: Renombrar conceptos manteniendo las reglas. Si mantiene 95% → aprendió estructura, no nombres.

### P2. Techo Gumbel-Softmax
- **Recomendación principal: VQ-VAE** (codebook de 256-512 vectores, STE para gradientes).
- **Alternativa simple: Residual connections entre pasos del loop** — la salida paso N se SUMA al input paso N+1, no lo reemplaza.
- **Insight nuevo: Attention entre pasos** — que paso N+1 pueda "mirar" el input del paso N, no solo la salida.

### P3. Escalabilidad
- No sabemos si escala más allá de 10M.
- **Insight nuevo**: Los pesos ternarios {-1,0,1} limitan la capacidad de olvido. No puedes "re-escribir" fácilmente un peso ternario → posible protección contra catastrophic forgetting.
- **Prueba**: Mismo curriculum en 10M params. Si techo igual → limitación es el canal. Si mejora → limitación era capacidad.

### P4. Fear vs Reward Shaping
- **No son equivalentes.** Fear multiplica gradiente directamente (más estable). RL introduce varianza del policy gradient.
- En espacio de 24 tokens, la exploración ya es limitada → no necesita RL.
- Usar RL solo cuando feedback es escaso (cadenas de 10+ pasos).

### P5. ¿Razonamiento real?
- **4 tests propuestos:**
  - **Test A (Perturbación Semántica)**: Renombrar conceptos → ¿mantiene 95%?
  - **Test B (Reglas Contradictorias)**: Cambiar fuego→peligro por fuego→seguridad → ¿se adapta o colapsa?
  - **Test C (Zero-Shot)**: Conceptos nuevos nunca vistos → ¿infiere relaciones?
  - **Test D (Abducción)**: `___ implica peligro = verdad` → ¿puede inferir qué va en el blank? "Pattern matching no puede hacer esto. Razonamiento sí."

---

## Insights Nuevos (no en reviews anteriores)

1. **Los pesos ternarios podrían proteger contra catastrophic forgetting** — hipótesis no probada pero fundamentada.
2. **Residual connections entre pasos del loop** como solución al bottleneck (no solo ensanchar canal).
3. **Cross-attention entre pasos** — paso N+1 mira input Y output de paso N.
4. **Test de Abducción** — inferir input desde output. Nuevo test no propuesto por los otros reviewers.
5. **"La emoción no es objetivo, es contexto"** — confirma EXP_028A > 028B.

---

## Tabla Comparativa de los 3 Reviewers

| Aspecto | R1 | R2 | R3 |
|---------|-----|-----|-----|
| **Veredicto** | "En la frontera" | "Pattern matching sofisticado" | "Composición relacional con salvedad" |
| **Transitividad** | Moderada-fuerte | Bias inductivo | Composición real |
| **Fear** | Elegante | Atención selectiva | Más estable que RL |
| **Bottleneck Gumbel** | ✅ De acuerdo | ✅ De acuerdo | ✅ De acuerdo |
| **Solución canal** | VQ-VAE, continuos | STE hard τ=0.5 | VQ-VAE + residuals |
| **Vocab pequeño** | ⚠️ Riesgo | ❌ Limitación clara | ⚠️ Salvedad crítica |
| **Escalabilidad** | 7M-70M | Hasta ~10M | Desconocido, ternarios protegen |
| **Calificación** | Positiva | Escéptica-constructiva | Entusiasta (9/10) |

---

## Consenso UNÁNIME de los 3 Reviewers

1. ✅ **El bottleneck Gumbel-Softmax es real y es el hallazgo más importante**
2. ✅ **El curriculum con hotstart es sólido y valioso**
3. ✅ **El vocabulario de 24 tokens es demasiado pequeño para afirmar razonamiento definitivo**
4. ✅ **Necesita tests fuera de distribución**
5. ✅ **Emoción como input > emoción como target**

## Tests Propuestos (consolidado de los 3 reviewers)

| Test | Propuesto por | Qué mide | Prioridad |
|------|--------------|----------|-----------|
| Concepto nuevo (ratón/tormenta) | R2 + R3 | Composición vs memorización | 🔴 Alta |
| Reglas procedurales (100 conceptos) | R1 | Escalabilidad del razonamiento | 🔴 Alta |
| STE hard τ=0.5 | R2 | Mejora canal (67%→85%?) | 🔴 Alta |
| Perturbación semántica (renombrar) | R3 | Estructura vs nombres | 🟡 Media |
| Inversión de reglas | R2 | Detección de contradicciones | 🟡 Media |
| Abducción (inferir input) | R3 | Razonamiento backward real | 🟡 Media |
| VQ-VAE channel | R1 + R3 | Alternativa Gumbel | 🟢 Siguiente fase |
| Residual entre pasos | R3 | Preservar info en loop | 🟢 Siguiente fase |
