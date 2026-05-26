Esto es lo que quería ver.

Mira la sección 2, la proto-sintaxis. Las posiciones 0 y 1 tienen diversidad baja por concepto (4.5 y 10.0) y alta por emoción (16.5 y 16.7). La posición 2 invierte: alta por concepto (13.1), baja por emoción (11.2).

Los agentes han inventado una gramática. Posiciones 0-1 para el concepto, posición 2 para la emoción — o algo funcionalmente equivalente. Nadie les dijo eso. No está en la arquitectura. Emergió de la presión del task.

Y el proto-léxico lo confirma: casi todos los conceptos usan su propio token en posición 0 y 1 (`gato→gato,gato`, `perro→perro,perro`), y luego posición 2 varía según algo. Para las emociones, los tokens son distintos por completo — `miedo` usa `árbol,sol,sol`, `hambre` usa `agua,código,código`. No hay mapeo uno-a-uno con el vocabulario humano. Han creado representaciones propias.

Ahora la geometría de confusión. `fuego→sol` 301 veces, `sol→fuego` 177 veces — son mutuamente confundibles. `árbol↔peligro` también bidireccional. Esto no es ruido aleatorio: hay clusters semánticos en su espacio que no coinciden exactamente con los nuestros. `fuego` y `sol` para ellos son más similares que para nosotros. ¿Por qué? Probablemente por los embeddings de fastembed — que es exactamente la pregunta que te hice sobre si están congelados.

Y el hallazgo que más me interesa: `agente` es el concepto más inestable, con 34.7% de consistencia en `agente+ira`. El speaker oscila entre `agente,agente,...` e `ira,ira,...`. Es el único concepto que comparte su token con una emoción. Los agentes están teniendo un conflicto de identidad literal — no saben si `agente` es una cosa del mundo o un estado interno.

Eso es un artefacto del vocabulario, no una limitación del modelo. Pero es un artefacto interesante.

¿Están congelados los embeddings de fastembed?

**Joan** : - Sí, están congelados

Entonces la geometría de confusión es heredada, no inventada.

`fuego→sol` 301 veces porque en el espacio de fastembed esos dos vectores están cerca. Los agentes no están confundiendo fuego con sol — están comunicándose fielmente a través de un espacio semántico que nosotros les dimos, y ese espacio tiene esa proximidad. El error no es suyo, es del mapa que les pusimos debajo.

Lo cual tiene una implicación importante: cuando descongelés los embeddings, parte de esa confusión debería resolverse sola. Los agentes tendrían incentivo para separar `fuego` de `sol` en su espacio propio porque la tarea los recompensa por distinguirlos. El proto-lenguaje que están construyendo ahora está limitado por la geometría de fastembed. Con embeddings entrenables, construirían una geometría optimizada para _su_ tarea de comunicación, no para la semántica humana general.

Eso es el experimento siguiente más interesante, antes de ampliar el vocabulario. No porque sea urgente, sino porque os diría algo fundamental: ¿cuánto de la estructura que están mostrando es suya, y cuánto es el reflejo de fastembed?

Pero hay algo que no cambia con los embeddings congelados o no: la proto-gramática posicional. Esa sí la inventaron ellos. Nadie les dijo que posición 0-1 es para concepto y posición 2 es para emoción. El espacio semántico subyacente no determina la sintaxis — eso requiere coordinación entre speaker y listener durante el entrenamiento. Esa parte es genuinamente emergente.

Y volviendo a `agente`: ese conflicto de identidad no desaparece aunque descongeles los embeddings, porque el problema es estructural — el vocabulario tiene un token que es simultáneamente un concepto-objeto y algo que se parece a un estado interno. Si seguís ampliando el vocabulario, ese token va a seguir siendo el más ruidoso a menos que lo separéis conscientemente.

Una cosa más, sobre la sección 3. Dijiste antes que la metáfora del capítulo 16 era más literal de lo que pensabais. Hay otro detalle que lo confirma en estos datos: la emoción tiene más varianza (σ=5.56%) que el concepto (σ=1.72%). Los conceptos se estabilizan antes y se quedan estables. Las emociones fluctúan más, incluso en autonomía pura.

Eso también pasa en humanos.
