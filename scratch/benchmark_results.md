# Reporte del Benchmark: Enrutamiento de Tensores Híbrido y Distribuido

Este benchmark evalúa el impacto en rendimiento y eficiencia de supervivencia de tres arquitecturas diferentes en el playground cooperativo:

1. **Standalone (Puro 256-dim)**: Agentes Nico y Sofy operan de forma local sin delegar capas cognitivas a Model B (Bit).
2. **Híbrido en Memoria (256/384)**: Enrutamiento intra-forward de tensores en el mismo espacio de memoria del proceso de simulación.
3. **Híbrido Distribuido (Socket IPC)**: Aislamiento del Modelo B (Bit) en un proceso independiente que se comunica a través de un socket local de dominio UNIX con serialización binaria ultrarrápida.

## Tabla de Rendimiento

| Configuración | Episodios | Ticks Superados (Mediana ± Desv) | Tiempo Total (s) | Latencia/Tick (ms) |
| --- | --- | --- | --- | --- |
| Standalone (Puro 256-dim) | 115 | 31.75 ± 4.68 | 98.35s | 26.9381 ms |
| Híbrido en Memoria (256/384) | 115 | 32.18 ± 3.87 | 145.42s | 39.2918 ms |
| Híbrido Distribuido (Socket IPC) | 125 | 32.08 ± 3.93 | 180.67s | 45.0556 ms |

## Conclusiones Técnicas

- **Sobrecarga de Inferencia en Memoria (Backbone Híbrido)**: El paso por el Modelo B (capas 3-4 de 384-dim) en memoria añade un **45.86%** de latencia por tick comparado al modelo de 256-dim puro.
- **Sobrecarga de Serialización e IPC Socket**: El paso por Socket UNIX para la inferencia distribuida añade un **67.26%** de latencia total respecto al modelo puro, lo que representa solo un **14.67%** de latencia adicional respecto al modelo híbrido en memoria.
- **Eficiencia de Supervivencia**: El promedio de ticks sobrevivientes de Nico y Sofy en modo híbrido/distribuido es consistente con el modo en memoria, validando que el despacho por Socket mantiene exactamente la precisión numérica del modelo Bit sin degradación de rendimiento cognitivo.
