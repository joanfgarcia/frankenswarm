# RFC-001: Vocabulario Vivo y Comprensión Autónoma

> **Estado**: DRAFT
> **Autor**: Joan Garcia + Aleth
> **Fecha**: 2026-05-31
> **Contexto**: Post EXP_032 (resonancia) y EXP_033 (emoción)

---

## Resumen Ejecutivo

Este RFC define la visión arquitectónica para la siguiente fase de Frankenswarm:
un modelo que no sabe cosas, sino que **sabe aprender cosas**. La habilidad
fundamental no es conocimiento, es **comprensión**. Un modelo con comprensión
lectora puede leer un diccionario y aprender solo.

---

## 1. La Tesis Central

> No queremos un modelo que lo sepa todo.
> Queremos un modelo que sea capaz de entender y aprender solo.
> — Joan Garcia, 2026-05-31

### Lo que NO estamos construyendo
- Un LLM con 70B parámetros que memoriza internet
- Un modelo que necesita reentrenamiento para cada concepto nuevo
- Una base de datos de hechos con interfaz de lenguaje natural

### Lo que SÍ estamos construyendo
- Un modelo **pequeño** (~1M params) con **comprensión lectora**
- Capaz de leer una definición y integrar un concepto nuevo **sin reentrenamiento**
- Que empieza con un vocabulario mínimo y lo **crece autónomamente**
- Donde el lenguaje está vivo: evoluciona con las necesidades del modelo

### La analogía
Un bebé humano no nace sabiendo qué es un "avión". Pero tiene la maquinaria
cognitiva para que, cuando alguien le señale uno y diga "avión", lo integre
en su mapa semántico basándose en propiedades que ya conoce: se mueve, es
grande, vuela, hace ruido. No necesita reentrenarse. Necesita **comprender**.

---

## 2. Validaciones Experimentales (Base Científica)

### EXP_032 — El modelo puede pensar
- Bucle latente cerrado: iteración en 256-dim sin salir a vocabulario
- +2.15% sobre baseline, todas las variantes superan al control
- **Probado**: El pensamiento silencioso (resonancia interna) funciona

### EXP_033 — La emoción guía las decisiones
- Inyección de emociones en el bucle latente
- Gap: +52.5 puntos (28.3% → 94.6% en bifurcaciones)
- **Probado**: El estado emocional modula el razonamiento
- **Insight**: Un impulso emocional al inicio basta (`first_only`)

### Lo que falta validar
- [ ] Comprensión composicional (¿puede deducir "gato" sin haberlo visto?)
- [ ] Vocabulario evolutivo (¿puede crecer el léxico durante inferencia?)
- [ ] Metacognición (¿puede verificar su propio pensamiento?)
- [ ] Aprendizaje autónomo (¿puede leer y aprender sin supervisor?)

---

## 3. Arquitectura del Lenguaje: Tres Capas

### Capa 1 — Primos Semánticos (Los Ejes)

Basado en **Wierzbicka (1996)**: 65 conceptos irreducibles que existen en
todos los idiomas humanos. No son "palabras" — son las **dimensiones del
significado**. Los ejes del espacio conceptual.

```
Categoría          Primos relevantes para Fase 0
─────────────────────────────────────────────────
Sustantivos        yo, tú, alguien, algo, cuerpo
Evaluadores        bueno, malo
Descriptores       grande, pequeño
Mente              pensar, saber, querer, sentir, ver, oír
Acciones           hacer, pasar, mover, tocar
Existencia         existir, (mío)
Vida/Muerte        vivir, morir
Tiempo             ahora, antes, después
Espacio            aquí, arriba, abajo, lejos, cerca, dentro
Lógica             no, poder, porque, si
```

**Implementación**: Cada primo es un **eje ternario** (+1, 0, -1).
Una palabra es un punto en este espacio de 65 dimensiones ternarias.

```python
# Cada palabra es un vector de 65 trits
agua   = [algo:+1, bueno:+1, mover:+1, vivir:+1, ver:+1, ...]  # 65 trits
fuego  = [algo:+1, bueno:+1, malo:+1, ver:+1, morir:+1, ...]   # dual
```

### Capa 2 — Léxico (Las Palabras)

Basado en **Swadesh (1952)**: ~100 palabras básicas que existen en todo idioma.
Cada palabra es un **punto** en el espacio de los 65 primos.

El léxico empieza pequeño y crece:

```
Fase 0 (8-10 palabras):   agua, comida, fuego, sol, noche, cueva, yo, peligro
Fase 1 (+6 acciones):     comer, beber, mover, ver, dormir, dar
Fase 2 (+6 entorno):      bosque, río, piedra, árbol, tierra, lluvia
Fase 3 (+6 estados):      grande, dolor, muerte, seguro, caliente, grupo
Fase 4+ (expansión):      el modelo propone/integra nuevas palabras
```

Nuevas palabras se añaden al léxico especificando sus coordenadas en el
espacio de primos. El modelo no necesita reentrenamiento — solo necesita
saber dónde cae la palabra nueva respecto a las que ya conoce.

### Capa 3 — Curriculum (El Orden de Enseñanza)

Basado en **Nelson (1973)** y **Bickerton (1990)**:

```
Etapa             Tipo          Ejemplo               Referencia
──────────────────────────────────────────────────────────────────
Holofrástica      Sustantivos   "agua" = quiero agua   Nelson fase 1
Dos palabras      Sust + Acción "agua beber"           Bickerton proto
Telegráfica       Cadenas       "río dar agua"         Bickerton proto
Gramatical        Relaciones    "si lluvia → río agua" Nelson fase 4+
```

Dos procesos separados pero sincronizados:
- **Proceso A**: El LÉXICO crece (nuevas palabras se hacen disponibles)
- **Proceso B**: El CURRICULUM enseña (en qué orden se entrenan)

Un bebé puede "saber" que existen los aviones antes de saber la palabra.
El concepto precede al token.

---

## 4. Pensamiento en Dos Fases: Resonancia + Verificación

### El "Deep Think" de Frankenswarm

```
         Input
           │
    ┌──────▼──────┐
    │  FASE 1:    │
    │  PENSAR     │  Loop latente ×N (resonancia interna)
    │  (silencio) │  La emoción guía la trayectoria
    └──────┬──────┘
           │
       Resultado₁
           │
    ┌──────▼──────┐
    │  FASE 2:    │
    │  VERIFICAR  │  Re-inyectar resultado como input
    │  (en voz    │  Loop latente ×M (metacognición)
    │   alta)     │
    └──────┬──────┘
           │
       Resultado₂
           │
    ┌──────▼──────┐
    │  COMPARAR   │  cosine(R₁, R₂)
    │             │  > 0.95 → Confiado ✅
    │             │  < 0.95 → Re-pensar 🔄
    └─────────────┘
```

**Fase 1: Resonancia (pensar)**
- El bucle latente actual (EXP_032/033)
- La emoción modula la trayectoria
- Produce un resultado candidato

**Fase 2: Metacognición (pensar sobre lo pensado)**
- El resultado de la Fase 1 se decodifica a tokens
- Esos tokens se re-inyectan como input del mismo bucle
- Si el modelo "re-piensa" lo mismo → convergencia = confianza
- Si piensa algo distinto → incertidumbre = necesita más iteraciones

**Analogía humana**:
- Fase 1: Piensas la respuesta mentalmente
- Fase 2: La dices en voz alta y te escuchas
- Si suena bien: la confirmas
- Si suena raro: "espera, déjame repensarlo"

**Implementación**: Extensión de `forward_resonance()`:

```python
def forward_deep_think(self, x, n_think=3, n_verify=2, emotion_ids=None):
    # Fase 1: Pensar
    logits_1, meta_1 = self.forward_resonance(
        x, n_steps=n_think, emotion_ids=emotion_ids
    )
    result_1 = logits_1.argmax(dim=-1)

    # Fase 2: Verificar (re-inyectar resultado)
    logits_2, meta_2 = self.forward_resonance(
        result_1, n_steps=n_verify, emotion_ids=emotion_ids
    )

    # Medir convergencia
    confidence = F.cosine_similarity(
        meta_1["final_hidden"], meta_2["final_hidden"], dim=-1
    ).mean()

    return logits_2, confidence, meta_1, meta_2
```

---

## 5. Emociones como Moduladores (no como palabras)

Las emociones NO son parte del vocabulario. Son vectores que modulan
la interpretación de las palabras, operando en un espacio separado:

```
Vocabulario (Capa 2):  agua, fuego, caza, cueva, ...  → tokens del léxico
Emociones (Canal B):   miedo, alegría, ira, hambre, ... → vectores moduladores
```

Validado en EXP_033: el mismo input + emoción distinta = decisión distinta.
El modo `first_only` (impulso emocional al inicio) es el más efectivo.

Las emociones son las **condiciones iniciales** del pensamiento.
No son lo que piensas — son CÓMO piensas.

---

## 6. Glifos Ternarios: El Token Composicional

### Encoding actual (EXP_033)
```
"agua" → index 3 → lookup en tabla de embeddings → vector 384-dim opaco
```
El modelo no sabe POR QUÉ "agua" es "agua". El embedding es una caja negra.

### Encoding propuesto (EXP_034+)
```
"agua" → [algo:+1, bueno:+1, mover:+1, vivir:+1, ver:+1, ...] → 65 trits
```
El token CONTIENE su semántica. El modelo puede deducir propiedades de
palabras nuevas por composición de radicales conocidos.

### Ventajas
1. **Zero-shot**: Un token nuevo se define por sus primos, sin reentrenamiento
2. **Eficiencia**: 65 trits = ~14 bytes por token (vs 384 floats = 1.5KB)
3. **Transparencia**: Puedes leer el glifo y entender qué representa
4. **Nativo ternario**: {-1, 0, +1} es exactamente el espacio de BitNet

### Riesgo
- 65 dimensiones ternarias = 3^65 combinaciones posibles (~10^31).
  En la práctica, la mayoría de combinaciones no tienen sentido.
  Necesitamos restricciones de coherencia.

---

## 7. La Meta Final: Comprensión Lectora

El objetivo no es que el modelo sepa 100.000 palabras.
Es que tenga la **maquinaria cognitiva** para:

1. **Leer** una definición: "gato = animal + pequeño + peludo + vivo + cueva"
2. **Mapear** los primos: [algo:+1, vivir:+1, pequeño:+1, dentro:+1, ...]
3. **Integrar** el nuevo glifo en su espacio semántico
4. **Usar** la palabra en razonamiento sin reentrenamiento

Esto es comprensión lectora. No memorización. No pattern matching.
Es entender la ESTRUCTURA del significado lo suficiente para extrapolar.

### Hoja de ruta

| Experimento | Objetivo | Valida |
|---|---|---|
| **EXP_034** | Vocabulario coherente (supervivencia) + glifos ternarios | Composicionalidad |
| **EXP_035** | Curriculum evolutivo (crecimiento del léxico) | Aprendizaje incremental |
| **EXP_036** | Deep-think (resonancia + verificación) | Metacognición |
| **EXP_037** | 4º bit (sinapsis mutables) | Aprendizaje en vivo |
| **EXP_038** | Comprensión lectora (leer definición → usar palabra) | Autonomía |

---

## 8. Principios Inmutables

1. **El lenguaje está vivo**. Evoluciona con el modelo, no se da todo de golpe.
2. **La emoción es brújula, no castigo**. Modula el pensamiento, no lo evalúa.
3. **Comprensión > Conocimiento**. Mejor entender 10 cosas que saber 10.000.
4. **Ternario es sagrado**. {-1, 0, +1} no es una limitación, es la naturaleza del modelo.
5. **Paso a paso**. Como Joan dice: "vamos a ir paso a paso".

---

## Referencias

1. Wierzbicka, A. (1996). *Semantics: Primes and Universals*. Oxford UP.
2. Swadesh, M. (1952). *Lexico-statistic dating of prehistoric ethnic contacts*. PAPS.
3. Bickerton, D. (1990). *Language and Species*. U. Chicago Press.
4. Bickerton, D. (2014). *More than Nature Needs*. Harvard UP.
5. Nelson, K. (1973). *Structure and strategy in learning to talk*. Monographs of SRCD.
6. Goddard, C. & Wierzbicka, A. (2014). *Words and Meanings*. Oxford UP.
