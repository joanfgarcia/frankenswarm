# RFC — Instrumento v3 y Protocolo de Cobertura (DL-011)

> **Estado**: PROTOCOLO IMPLEMENTADO · batería 80/50 pendiente de generación
> **Autor**: Aleth + Joan García · **Fecha**: 30-ago-2026
> **Contexto**: la puerta de avance (examen de hito) y la medición de cognición
> estaban fundidas; el ×300 y el muestreo aleatorio gateaban por memorización.
> **Decisión registrada en**: `docs/DECISION_LOG.md` (DL-011)

---

## 1. Resumen

Tres cambios de protocolo que separan **avance** de **cognición**:

1. **Factor de examen ×300 → ×10** (`--exam_repeat_factor`): ~9.000
   exposiciones por etapa producían lookup de pares exactos. ×10 fija el hecho
   sin contaminar la medición. Efecto colateral deseado: el gate se endurece y
   la escalera de neurogénesis puede ejercitarse por fin.
2. **Muestreo cíclico permutado** (`CyclicPoolSampler`): cada etapa recorre su
   pool completo en permutaciones sucesivas — garantiza que TODAS las
   secuencias del nivel se exponen al menos una vez. El muestreo aleatorio
   anterior cubría solo ~25-30% del pool por etapa.
3. **Examen gateado por cobertura**: la etapa solo puede cerrarse
   (plateau o tope) cuando el pool está cubierto → la afirmación "Bit graduó N
   años habiendo visto todo su corpus gateado" pasa a ser verdadera por
   construcción. Mínimos por etapa: E0 ~5 épocas … E7 ~106 épocas (a 400k/época).

## 2. Batería 80/50 (instrumento de evaluación — retroactiva)

Por edad: **80 preguntas vistas** (gate de avance, entrenadas ×10) + **50
preguntas NO vistas** (métrica de cognición, jamás entrenadas). Total 130/edad,
910 en total.

- **Decisiones**: 50 y no 30 por IC95 — a 50% de acierto, ±13.9% con n=50 vs
  ±17.9% (n=30) / ±21.9% (n=20); la comparativa entre brazos necesita resolver
  5-10 puntos.
- **"No vista" = composición nueva de hechos conocidos**: paráfrasis y
  recombinaciones del material curricular (inversión sujeto-predicado,
  formas interrogativas, pares aritméticos con otra superficie). In-gateo,
  in-vocab, y **verificadas ausentes del entrenamiento por script** contra el
  store CSR (hash de secuencia).
- **Retroactiva**: es evaluación pura — aplicable a TODOS los checkpoints,
  incluidos los del run ×300 (ablación v2→v3: misma arquitectura, mismo
  currículo; el ×300 debería hundir el score de las 50 no vistas y apenas
  mover el de las 80 vistas → cifra directa del coste de memorización).
- **La métrica primaria de la tesis vive aquí**: la ventaja composicional de
  los glifos, si existe, se manifiesta en las no vistas (las vistas las
  memorizan ambos brazos).

Estado de generación: pendiente (script generador + verificación de ausencia).

## 3. Ablación v2 → v3

El run v2 (`bit003_glyph_x300_v2`, 4 hitos: 2-5 años, época 229) se ARCHIVA
como documento experimental, no se borra. El run v3 parte de cero con el mismo
corpus/gateo/instrumento de entrenamiento salvo los tres cambios de este RFC.

## 4. Limitaciones

1. El cursor del sampler no persiste entre steps del job: cada proceso reinicia
   el barajado (la cobertura se garantiza dentro de cada proceso; entre
   procesos hay re-exposición parcial — aceptado, documentado).
2. La cobertura se define sobre el split de ENTRENAMIENTO (el val 10% nunca se
   entrena por diseño; se mide, no se ve).
3. La batería 80/50 necesita Samantha (o exact-match) como corrector; las 50
   no vistas deben re-verificarse tras cada regeneración de vocab (los scripts
   de validación existentes cubren in-vocab, la ausencia es del generador).
