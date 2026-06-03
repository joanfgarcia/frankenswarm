# 🚀 Guía de Demostración para la Oficina: FrankenSWARM

Esta guía está diseñada para que puedas mostrarle a David (y a cualquiera en la oficina) los experimentos más avanzados de la Legión de Agentes Soberanos. Todo corre de forma 100% local en tu portátil (GPU RTX 5070 / CPU).

---

## 💬 Demo 1: Swarm Chat Autónomo (Nico & Sofi)
Dos agentes basados en modelos **BitNet ternarios de 1.58 bits (19.3 millones de parámetros)** conversan de forma autónoma. Los modelos mapean su lenguaje interno al vocabulario base de 3,000 palabras en español mediante similitud de glifos.

### Cómo ejecutarlo:
Abre una terminal en la carpeta `frankenswarm` con el entorno virtual activo:
```bash
# Ejecutar con los parámetros por defecto
PYTHONPATH=. .venv/bin/python src/bitnet/run_swarm_chat.py
```

### Variaciones interactivas para impresionar:
Puedes jugar con la **Temperatura** (para controlar la creatividad vs coherencia) y la **Penalización por Repetición**:
```bash
# Diálogo más determinista y enfocado (Temperatura baja)
PYTHONPATH=. .venv/bin/python src/bitnet/run_swarm_chat.py --temp 0.4 --turns 15

# Diálogo muy creativo o abstracto (Temperatura alta)
PYTHONPATH=. .venv/bin/python src/bitnet/run_swarm_chat.py --temp 0.8 --turns 15
```

### Qué explicarle a David:
1. **Modelos Ultra-Ligeros**: El modelo ocupa menos de 7.5 MB en disco gracias a la representación de glifos compostados.
2. **Sin Servidores Externos**: La inferencia es instantánea y local en la GPU.
3. **Evolución del Lenguaje Swarm**: La conversación puede parecer abstracta o metafórica (ej: proyecciones semánticas como usar "mosca" en lugar de "escondite"). Esto ocurre porque el modelo optimiza la transmisión de conceptos de supervivencia en su propio espacio semántico.

---

## 🏟️ Demo 2: Arena Cooperativa de 3 Agentes (Con Traducción Samantha)
Simulación de supervivencia de una tribu de 3 agentes (**Nico, Sofy e Hugo**) con especialización biológica asimétrica, Teoría de la Mente y traducción en tiempo real por el LLM local (**Samantha**).

### Cómo ejecutarlo:
```bash
# Simulación paso a paso con traducción en tiempo real
PYTHONPATH=. .venv/bin/python src/bitnet/run_arena_simulation.py --config configs/experiments/EXP_073_easy_train.json --ticks 80
```

### Aspectos clave que debes ir señalando en la pantalla:
1. **La Asimetría Tribal (Especialización)**:
   - **Nico** no puede participar en la caza cooperativa directa de presas grandes.
   - **Sofy** no puede beber agua del suelo ni llenar su mochila de agua (está bloqueada de fuentes hídricas). Depende de que Nico o Hugo le compartan agua.
   - **Hugo** no puede comer comida del suelo ni llenar su mochila de comida (bloqueado de fuentes de alimento). Depende de Nico o Sofy para comer.
2. **Mochilas y Altruismo Desacoplado**:
   - Verás cómo los agentes se reencuentran en zonas seguras y usan la acción **"dar"** para pasarse comida o agua a la mochila del compañero.
   - El receptor no come inmediatamente; guarda el recurso y decide consumirlo (**"comer" / "beber"**) cuando su metabolismo lo requiera.
3. **Teoría de la Mente (ToM)**:
   - En cada tick, verás que cada agente imprime su "estimación mental" del estado de sus compañeros (`Estima a Sofi: ...`). Si no los ve ni los oye, asume que su metabolismo empeora.
   - Cuando uno de ellos grita por radio o se cruzan en el mismo nodo (**Meetup**), actualizan sus representaciones mentales y mapas de recursos.
4. **Caza, Fuego y Herramientas (Edad de Piedra)**:
   - Los agentes pueden **"fabricar"** lanzas usando ramas y piedras.
   - La lanza les permite defenderse pasivamente de depredadores o cazar presas de forma individual.
   - Pueden **"encender"** hogueras usando ramas. Si comen junto al fuego, cocinan comida asada (Barbacoa) o preparan un Guiso (si combinan comida y agua), evitando el 15% de probabilidad de intoxicación de la comida cruda.
5. **Traducción en Tiempo Real (Samantha)**:
   - Cada vez que un agente grita (ej: `Nico grita: [bosque, comida]`), verás la línea de traducción de **Samantha** (Mistral 7B local en la GPU) convirtiéndolo a lenguaje de niños: *"¡Oye, he encontrado bayas ricas en los árboles, venid!"*.
6. **Reproducción y Crossover SVD**:
   - Si dos agentes viables coinciden en la `cueva` o `ruinas` y ejecutan `"reproducir"`, nacerá un cuarto agente (**Domi**). Sus pesos neuronales se heredan mediante una recombinación de descomposición en valores singulares (SVD) de los pesos de sus padres.

---

## 🧠 Explicación Técnica Avanzada (El "Under the Hood")

Si David te pregunta cómo funciona técnicamente esta maravilla:
* **Motor Neuro-Simbólico**: Las reglas físicas de la arena, el peso de la mochila, la cocina de guisos y la lógica de combate contra depredadores están escritas en **Prolog declarativo puro** (`cooperative_rules.pl`), integrado mediante `pyswip`. El cerebro neuronal (PPO) del agente decide la acción, y el motor lógico Prolog procesa la física del entorno.
* **Neurogénesis Dinámica**: Los agentes empiezan con un cerebro pequeño (ancho 128) y, a medida que la presión metabólica los estresa, ejecutan en caliente crecimiento sináptico expandiendo sus capas neuronales (ancho 192 -> 288) pagando un peaje de energía metabólica.
* **Destilación Latente por Sueño**: Cuando un agente enseña a otro una habilidad en un recurso geográfico (como pesca o cocina), el alumno graba el estado interno del maestro. Durante el **Ciclo de Sueño** (consolidación), se entrena al alumno para imitar las representaciones latentes del maestro mediante pérdida de similitud coseno.
