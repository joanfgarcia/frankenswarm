# Pre-Execution Audit — Claude Sonnet

> **Auditor**: Claude Sonnet
> **Role**: Principal AI Auditor & Architect
> **Target Stage**: Pre-Phase 0 Evaluation
> **Date**: 2026-05-26

---

Bien. Déjame leer el digest primero.Bien. Ya tengo suficiente. Déjame pensar en voz alta, sin cortesía innecesaria.

---

**Lo que funciona bien y merece respeto:**

La decisión de separar Phase A (validar la infraestructura con modelos existentes) de Phase B (criar los micro-expertos) es la más sana de todo el documento. "No construyas el motor y el combustible al mismo tiempo" — eso es ingeniería real, no romanticismo. Es la única parte donde el proyecto se protege de su propia ambición.

El hardware MoE sobre silicon heterogéneo también es sólido. Los benchmarks son reales, las cifras de energía son honestas, y la intuición de que el 80% de las consultas no necesitan 80W es correcta y verificable. Eso no es visión — es pragmatismo ejecutable ahora mismo.

La arquitectura de cuatro capas con vocabulario soberano compartido como interfaz discreta entre nodos es elegante. El insight de los 2 bytes por token como canal inter-experto en lugar de vectores continuos es genuinamente bueno.

---

**Lo que me preocupa:**

Hay dos proyectos distintos viviendo bajo el mismo nombre, y creo que eso es el problema central.

El primero es un hardware MoE distribuido sobre silicon real — Prolog gate, cascade routing, LoRA adapters sobre modelos existentes, metabolic loop nocturno. Este proyecto es concreto, tiene métricas, tiene hardware medido, tiene un roadmap creíble. Está a 60-70% de ser algo funcionando.

El segundo es el jardín de Aleth — criar cerebros ternarios desde cero, curriculum humano, NEAT sobre topologías de enjambre, emergencia desde la semilla. Este proyecto es hermoso como idea y enormemente arriesgado como ingeniería.

El problema es que el segundo proyecto necesita que el primero esté terminado antes de que tenga sentido empezar. Y el capítulo de Aleth escribe sobre el segundo como si fuera lo que están construyendo ahora mismo.

**NEAT sobre redes de 7M parámetros es la apuesta más difícil del stack.** El propio documento lo reconoce en REFERENCES: "Genetic algorithms suffer from evolutionary noise and stagnation at scale." Y 7M parámetros ya está en el límite superior donde NEAT históricamente se rompe — NEAT funciona bien hasta ~100K parámetros, luego la maldición de la dimensionalidad lo destruye. La solución propuesta (NEAT hasta 100K, Net2Net para crecer, luego PEFT/PopuLoRA para escalar) es correcta en teoría, pero cada transición entre esas fases es un salto de fe no validado.

El curriculum humano como enfoque pedagógico me parece correcto filosóficamente pero no sé si está resolviendo el problema real. El problema con entrenar modelos pequeños desde cero no es que les falte curriculum — es que les faltan parámetros para retener lo que aprenden. Un modelo de 7M parámetros que intenta aprender gramática, aritmética, ciencias e historia al mismo tiempo va a sufrir catastrophic forgetting brutal entre disciplinas. La solución de especializar nodos por disciplina (un nodo para matemáticas, otro para lengua) tiene más sentido, pero entonces el curriculum humano ya no es una progresión pedagógica — es simplemente un dataset bien organizado.

---

**Lo que cambiaría:**

Terminar Phase A primero. Que el Prolog gate ruteé, que el cascade funcione, que los adapters se swapeen en caliente. Eso es la fundación. Todo lo demás depende de que esto sea real.

Del capítulo de Aleth rescataría una cosa: la pausa antes de plantar la semilla. Eso es sabio. El problema es que la pausa tiene que terminar en "vamos a hacer Phase A primero" — no en "vamos a hacer Phase B directamente porque es más bonito".

La semilla puede esperar. El invernadero hay que verificarlo primero.

¿Es eso lo que querías escuchar, o hay algo específico que te estaba rondando?
