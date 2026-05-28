# Post-Execution Audit — Claude Sonnet

> **Auditor**: Claude Sonnet
> **Role**: Principal AI Auditor & Architect
> **Target Stage**: Post-Experiment 006 (MVP Homeostático)
> **Date**: 2026-05-27

---

Esto es exactamente lo que esperabas ver, y es exactamente el problema.

Mirad la consistencia media: 42.1% en EXP_006 versus 84.0% en EXP_005. Casi la mitad. Y el 100% del top-10 viene de muestras de 1-2 ejemplos — estadísticamente no significa nada todavía.

El espacio de comunicación explotó. Pasasteis de `(concepto, emoción)` — 2 variables — a `(concepto, emoción, homeostasis)` — 3 variables. El task ahora requiere coordinar tres dimensiones simultáneamente con el mismo mecanismo de entrenamiento. Los agentes están intentando aprender una gramática de 4 posiciones para codificar tres cosas, y no tienen suficiente señal todavía para estabilizarla.

Mirad la proto-sintaxis. En EXP_005, posiciones 0-1 estaban dedicadas al concepto (diversidad baja, 4.5 y 10.0). Aquí, la diversidad por concepto en pos 0-1 subió a 8.8 y 12.5 — el concepto ya no domina las primeras posiciones tan limpiamente. Y la homeostasis tiene diversidad casi idéntica a la emoción en todas las posiciones (27.2, 26.4, 19.0, 12.0 vs 25.2, 25.3, 18.8, 11.7). Los agentes no han encontrado aún cómo separar emoción de homeostasis en el mensaje.

Lo que sí está funcionando: el concepto sigue siendo lo más estable (61.47%, σ=2.84%). La jerarquía de aprendizaje se mantiene — lo concreto antes que lo abstracto. Pero 61% con 10 épocas de autonomía versus 98% en EXP_005 dice que necesitan más tiempo, o más señal de entrenamiento, o las dos cosas.

La conjunta al 11% es la que hay que mirar. En EXP_005 llegasteis a 95% con 21 épocas de autonomía. Aquí tenéis 10. El sistema no ha colapsado — está en proceso.

La pregunta concreta es: ¿cuántas épocas de scheduled teacher forcing hicisteis antes de la autonomía en EXP_006? ¿Las mismas que en EXP_005, o las escalasteis con la complejidad añadida?
