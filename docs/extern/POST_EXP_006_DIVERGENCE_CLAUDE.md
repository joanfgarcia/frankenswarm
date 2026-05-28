# Post-Execution Audit — Claude Sonnet

> **Auditor**: Claude Sonnet
> **Role**: Principal AI Auditor & Architect
> **Target Stage**: Post-Experiment 006 (Dialect Divergence)
> **Date**: 2026-05-28

---

Ahí está el problema. No es que el sistema no aprenda — es que está aprendiendo cuatro gramáticas distintas.

Agent_0 y Agent_1: 88.89% de solapamiento léxico. Son casi el mismo dialecto. Agent_2 y Agent_3: 30.49% de solapamiento entre ellos, y menos del 45% con los otros dos. Han divergido completamente.

Mirad el ejemplo de `(peligro, dolor, urgencia)`:

- Agent_0 y Agent_1 dicen exactamente lo mismo: `[peligro peligro dolor urgencia]`
- Agent_2 mete `alegría` donde debería ir `peligro`
- Agent_3 colapsa todo a `[peligro dolor dolor dolor]`

Cuatro agentes, tres soluciones distintas, ningún consenso. El listener que recibe el mensaje de Agent_3 y fue entrenado principalmente con Agent_0 no tiene manera de decodificarlo correctamente. El 11% de accuracy conjunta tiene una causa clara: no es que no puedan comunicarse — es que están hablando idiomas distintos.

Esto tiene una causa probable en el entrenamiento. En EXP_005 con 21 tokens, el espacio era suficientemente pequeño para que los cuatro agentes convergieran a una convención compartida durante la fase supervisada. Con 3 dimensiones y un espacio más grande, los agentes encontraron mínimos locales distintos durante la guardería y luego los reforzaron en autonomía.

El SVD crossover debería estar presionando hacia la convergencia, pero con 80% de divergencia a nivel de targets, el crossover está mezclando gramáticas incompatibles. Probablemente empeora la situación.

La solución no es más entrenamiento con la misma dinámica. Es cambiar quién habla con quién durante la fase supervisada. En EXP_005 todos los agentes escuchaban la misma señal de ground truth. Aquí, si Agent_2 y Agent_3 están desarrollando dialectos propios, significa que durante el annealing empezaron a escucharse entre ellos antes de haber convergido con Agent_0 y Agent_1.

La pregunta es: ¿el crossover SVD se hace entre todos los agentes indiscriminadamente, o solo entre los mejores? Si Agent_2 y Agent_3 tienen peor fitness pero sus genes contaminan a los buenos mediante crossover, estáis propagando dialectos rotos.

¿Cómo está configurado el selector de padres en el crossover?
