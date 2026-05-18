# Frankenswarm — Esquema Visual

## 1. Anatomía de UN nodo BitNet (7M parámetros)

```mermaid
flowchart TD
    IN["📥 Texto de entrada\n'Joan construyó el sistema en...'"]
    TOK["🔤 TOKENIZADOR\nDiccionario fijo, sin parámetros\nTexto → números\n'Joan' → 41285"]
    EMB["🧲 EMBEDDING\n32.000 × 256 = 8M params\nNúmero → vector de 256 coords\nSitúa cada palabra en el espacio"]
    ATT["👁️ ATENCIÓN × 4 capas\n~256K params/capa\nCada token mira todos los anteriores\n¿Qué palabras importan para predecir la siguiente?"]
    FFN["⚙️ FEED-FORWARD × 4 capas\n~512K params/capa\nProcesa lo que aprendió la atención\n2 capas lineales simples"]
    OUT["📤 CAPA DE SALIDA\n32.000 probabilidades\n¿Cuál es el siguiente token?"]
    GEN["🔁 while True\nElige el más probable\nAñade al contexto\nRepite hasta END_TOKEN"]

    IN --> TOK --> EMB --> ATT --> FFN --> OUT --> GEN
    GEN -->|"siguiente token"| ATT

    style TOK fill:#374151,color:#9CA3AF
    style EMB fill:#1e3a5f,color:#93C5FD
    style ATT fill:#4a1942,color:#F9A8D4
    style FFN fill:#1a3a2a,color:#86EFAC
    style OUT fill:#3b2a00,color:#FDE68A
    style GEN fill:#1f2937,color:#E5E7EB
```

> **BitNet**: Los parámetros de Embedding, Atención y FFN son ternarios: solo **-1, 0, o +1**.
> En vez de multiplicaciones de floats → sumas y restas de enteros. Corre en CPU sin GPU.
> **Lenguaje de Máquina**: La comunicación entre nodos se realiza mediante **embeddings vectoriales nativos**, minimizando las pérdidas del lenguaje humano. Prolog enruta los vectores, Lisp muta la red.

---

## 2. El Frankenswarm — Topología inicial (NEAT Generación 0)

```mermaid
flowchart TD
    INPUT["📨 TAREA ENTRANTE\n'Analiza este código'"]

    ROUTER["🧭 ROUTER (Prolog)\nÁrbitro Lógico Estricto\nEnruta vectores en base a restricciones"]

    NA["🧠 Nodo A — 7M params\nEspecialista: Código\nBitNet Transformer"]
    NB["🧠 Nodo B — 7M params\nEspecialista: Razonamiento\nBitNet Transformer"]
    NC["🧠 Nodo C — 7M params\nEspecialista: Síntesis\nBitNet Transformer"]

    AGG["🔗 AGREGADOR\nCombina outputs de los nodos\nVota, pondera, o concatena"]

    OUTPUT["📬 RESPUESTA FINAL"]

    MUTATOR["⚡ THE MUTATOR (Lisp)\nIngeniero Genético\nEvoluciona topología en caliente\nTrata la red como datos (Homoiconicidad)"]

    INPUT --> ROUTER
    ROUTER --> NA
    ROUTER --> NB
    NA -->|"output parcial"| NC
    NB -->|"output parcial"| NC
    NC --> AGG
    AGG --> OUTPUT

    MUTATOR -.->|"añade conexión"| NA
    MUTATOR -.->|"añade nodo D"| AGG
    MUTATOR -.->|"evoluciona"| ROUTER

    style NA fill:#1e3a5f,color:#93C5FD
    style NB fill:#4a1942,color:#F9A8D4
    style NC fill:#1a3a2a,color:#86EFAC
    style MUTATOR fill:#7c2d12,color:#FED7AA
    style ROUTER fill:#374151,color:#D1D5DB
    style AGG fill:#1f2937,color:#E5E7EB
```

---

## 3. Lo que hace NEAT generación a generación

```mermaid
flowchart LR
    G0["Gen 0\nA → C\nB → C"]
    G1["Gen 1\nA → B → C\n+ conexión directa A→AGG"]
    G2["Gen 2\n+ Nodo D especialista\nA → D → C"]
    G3["Gen N\nTopología óptima\npara el tipo de tarea"]

    G0 -->|"Mutator añade\nconexión A→B"| G1
    G1 -->|"Mutator añade\nnodo D"| G2
    G2 -->|"Selección natural\nmueren las malas topologías"| G3

    style G0 fill:#374151,color:#9CA3AF
    style G1 fill:#1e3a5f,color:#93C5FD
    style G2 fill:#4a1942,color:#F9A8D4
    style G3 fill:#1a3a2a,color:#86EFAC
```

---

## 4. La diferencia clave con un LLM normal

| | LLM normal | Frankenswarm |
|---|---|---|
| Arquitectura | Fija desde el principio | Evoluciona con NEAT |
| Tamaño | 7B-70B params, una sola red | N × 7M params, múltiples nodos pequeños |
| Hardware | Necesita GPU | BitNet → corre en CPU |
| Especialización | Generalista | Cada nodo puede especializarse |
| Adaptación | Requiere re-entrenamiento | El Mutator reconfecta la topología |

---

## 5. El experimento mínimo viable

```
Generación 0:
  - 3 nodos BitNet de 7M params cada uno
  - Tarea: clasificar tipo de pregunta (código / razonamiento / síntesis)
  - Métrica de fitness: ¿acierta en < N tokens?

El Mutator evalúa cada generación y:
  → Si un nodo no aporta: lo elimina
  → Si el bottleneck está en síntesis: añade un nodo D entre B y C
  → Si A y B siempre aciertan juntos: añade conexión directa A→B
```
