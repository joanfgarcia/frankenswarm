# Revisión Pendiente: Escalado Curricular

**Fecha:** 2026-07-22
**Estado:** Pendiente de revisión
**Prioridad:** Media (no urgente, pero fundamental para tesis)

## Problema

El escalado de epochs por stage es **exponencial** (`epochs = base × 2^(stage × 0.5)`), lo que significa que cada stage tarda el **doble** que el anterior:

```
Stage 0 (0-1 años):   64 epochs  (~21h)
Stage 1 (1-2 años):   90 epochs  (~30h)
Stage 2 (2-3 años):  128 epochs  (~43h)
Stage 3 (3-4 años):  181 epochs  (~60h)
Stage 4 (4-5 años):  256 epochs  (~85h)
Stage 5 (5-6 años):  362 epochs (~121h)
Stage 6 (6-7 años):  512 epochs (~171h)
Stage 7 (7-8 años):  724 epochs (~241h)
TOTAL: 2,317 epochs (~772h)
```

## Por qué es un problema

1. **No es biológicamente preciso:** Un niño de 7 años no necesita 128x más tiempo de escuela que un bebé de 1 año
2. **Penaliza el aprendizaje avanzado:** Los stages finales toman una proporción desproporcionada del tiempo total
3. **Ignora transfer learning:** El conocimiento previo debería acelerar el aprendizaje, no ralentizarlo
4. **Duplica innecesariamente:** La neurogénesis ya crece el modelo; no necesitamos crecer el tiempo también

## Comparación con tiempo humano real

- **Humano:** ~8,760h por año (24h × 365 días)
- **Nuestro stage 7:** ~241h
- **Ratio:** Stage 7 es 11.5x más corto que un año humano, pero 128x más largo que stage 0

## Alternativas propuestas

| Opción | Escalado | Total epochs | Total horas | Precisión biológica |
|--------|----------|--------------|-------------|---------------------|
| **A) Fijo** | 100×8 stages | 800 | ~267h | Alta |
| **B) Logarítmico** | 32→96 | 512 | ~171h | Media-Alta |
| **C) Adaptativo** | Para cuando val_loss < umbral | Variable | Variable | Alta |
| **D) Exponencial (actual)** | 32→4096 | 2,317 | ~772h | Baja |

### Opción A: Tiempo fijo por stage
- **Ventaja:** Más predecible, más biológicamente preciso
- **Desventaje:** Puede ser insuficiente para stages complejos
- **Implementación:** Cambiar `stage_scale` a 0 o usar formula lineal

### Opción B: Escalado logarítmico
- **Ventaja:** Crecimiento suave, mantiene algo de escalado
- **Desventaje:** Aún puede ser excesivo en stages finales
- **Implementación:** Usar `log2(stage + 1)` en lugar de `2^(stage × 0.5)`

### Opción C: Escalado adaptativo
- **Ventaja:** El modelo aprende cuando está listo, no cuando el schedule dice
- **Desventaje:** Menos predecible, necesita umbral de validación
- **Implementación:** Añadir `--adaptive_stopping` con umbral de val_loss

## Código afectado

- `src/bitnet/training/train_sovereign_school.py`: Línea ~850 (cálculo de epochs)
- Parámetros: `--base_epochs`, `--stage_scale`

## Decisión pendiente

- [ ] Revisar si la tesis requiere escalado exponencial explícito
- [ ] Evaluar si el cambio afecta los resultados actuales
- [ ] Decidir entre opciones A, B o C
- [ ] Implementar cambio y re-entrenar desde checkpoint más reciente

## Nota

**NO TOCAR AHORA.** El entrenamiento actual lleva ~31h invertidas y la tesis se basa en esta formulación. Cambiar ahora podría invalidar los resultados. Revisar después de completar el milestone 7_years.
