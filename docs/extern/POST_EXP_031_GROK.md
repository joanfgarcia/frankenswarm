# Expert Review — Frankenswarm Technical Brief
## Fecha: 29 Mayo 2026

### Reviewer: Modelo Externo (Cloud)
### Documento revisado: `FRANKENSWARM_TECHNICAL_BRIEF.md`

---

## Valoración: ✅ Trabajo serio y valioso

### Fortalezas validadas:
1. **Curriculum + Hotstart** — "uno de los hallazgos más sólidos"
2. **Fear weighting** — "forma limpia de reward shaping diferenciada"
3. **H4 (bottleneck Gumbel-Softmax)** — "diagnóstico preciso, hallazgo más importante"
4. **Control experimental** — "metodología reproducible"

### Riesgos identificados:
1. **Vocabulario demasiado pequeño (24 tokens)** — 24⁴ = ~331k combinaciones manejables. Posible pattern matching sofisticado en vez de composición real.
2. **Posible leakage en cadenas** — Embeddings congelados podrían capturar correlaciones indirectas. Necesita test de control con reglas aleatorias.
3. **Escalabilidad incierta** — Net2Net en pesos ternarios podría ser inestable.

### Respuestas a las 5 preguntas:
1. **Transitividad**: Evidencia moderada-fuerte, no concluyente. Necesita control con reglas aleatorias.
2. **Gumbel-Softmax**: VQ-VAE, embeddings continuos durante training, discretizar solo en inferencia.
3. **Escalabilidad curriculum**: Funciona mejor en 7M-70M que en ultra-pequeños.
4. **Fear ≈ Reward Shaping**: Funcionalmente equivalentes. Fear es más interpretable.
5. **¿Razonamiento real?**: "Estás en la frontera." Con 24 tokens → más cerca de pattern matching. Con 80+ tokens + reglas procedurales → razonamiento emergente real.

### Recomendación principal:
> **EXP_031**: Aumentar vocabulario a 60-80 tokens + reglas causales generadas proceduralmente (no hardcodeadas). Si mantiene >75% en cadenas de 4-5 saltos → evidencia fuerte de razonamiento emergente.

### Cita clave:
> "El camino más interesante ahora mismo no es hacer el modelo más grande, sino más inteligente en su nicho: mejor canal, mejor curriculum, más presión pragmática."
