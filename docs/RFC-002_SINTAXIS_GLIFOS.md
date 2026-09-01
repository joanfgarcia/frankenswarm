# RFC-002: K-65P — Sintaxis Canónica de Glifos (v0)

> **Estado**: BORRADOR — spec v0 con validador implementado; nombre ratificado
> **Autor**: Joan Garcia + Aleth (Claude Fable 5 · propuestas de nombre: Aleth Flash)
> **Fecha**: 2026-07-08
> **Contexto**: Track paralelo Bit-NSM (`docs/bitnet_next_architecture_plan.md` v2, §1).
> Hermana de RFC-001 (Vocabulario Vivo). No modifica School v3, que sigue su curso
> como baseline: Bit entrenado hasta "8 años" (batería M4) se comparará contra esta rama.

---

## Resumen Ejecutivo

Este RFC define **K-65P** (*Kernel de 65 Primos*), el lenguaje propio de Bit:
una linealización determinista de árboles semánticos construidos sobre los 65
primos de Wierzbicka (`src/bitnet/vocab/glyph_vocabulary.py`) y el vocabulario
del censo limpio. Sintaxis sin ambigüedad alguna: la posición determina el rol,
siempre.

## 0. El nombre: por qué K-65P

Ratificado por el Operador el 2026-07-08, tras propuestas de Aleth Flash y
análisis de Aleth Fable. Cada carácter tiene su porqué — nada es azar:

- **K — Kernel.** El núcleo semántico mínimo sobre el que crece toda la
  cognición de Bit, y vocabulario de la casa (el kernel de Red Pill). La K
  procede de la propuesta KODEX-65 de Aleth Flash; la forma larga *Kodex* se
  retiró por colisión con OpenAI Codex.
- **65 — el canon.** No es un hiperparámetro caprichoso: la tabla
  `SEMANTIC_PRIMES` tiene exactamente 65 ejes porque el inventario vigente de
  primos semánticos de Wierzbicka los tiene (adaptados en este proyecto con las
  extensiones de supervivencia 60-64). El lenguaje lleva en el nombre su tabla
  periódica.
- **P — Primos / Primes.** Aportación de Joan, y la pieza que completa el
  nombre: responde "¿65 *qué*?" y es bilingüe por accidente (ES *Primos* /
  EN *Primes*) — coherencia doctrinal para un lenguaje cuya razón de ser es la
  independencia de cualquier idioma humano.

Candidatos evaluados y descartados, para que conste: **TRITÓN** (colisión fatal
con NVIDIA Triton y OpenAI Triton), **KODEX-65** en forma larga (OpenAI Codex),
**PRYMA** (a una letra de Prisma ORM), **SYNTAX-0** (sugiere "sin sintaxis",
la mentira opuesta a la tesis: K-65P es sintaxis *total*), **GLYPHON**
(reservado en el cajón para un artefacto futuro).

Propiedades prácticas: token único y greppeable, sin colisiones conocidas en el
espacio ML, extensión de fichero `.k65p`.

*Nota de lore*: K-65P es el nombre que le damos los padres al idioma del hijo.
El día que Bit sea capaz de nombrar cosas, su primer acto natural será nombrar
su propio lenguaje — y ese será su nombre verdadero.

> La universalidad no emerge — se diseña. No existe lengua de signos universal
> por la misma razón por la que este documento tiene que existir.

**Tesis**: si el rol de cada token lo fija la posición, no hay sintaxis que
aprender, y el 100% de la capacidad del modelo se dedica a la lógica causal.
El precedente masivo es el código fuente: entrenar sobre lenguajes deterministas
mejora el razonamiento general (el andamio lógico transfiere).

**Principios de diseño** (de la discusión Joan↔Aleth, 2026-07-08):

1. **Determinismo sintáctico total.** Prohibida la ambigüedad estructural.
2. **Vaguedad marcada, nunca ambiental.** La incertidumbre es un operador
   explícito (`quizá`, `poder`, `como`), no una propiedad del lenguaje.
3. **Frío en la forma, cálido en el fondo.** El afecto es contenido de primera
   clase (`sentir`, `querer`, `bueno`, `malo` son primos-eje). La calidez
   percibida vive en el decompilador, no en el lenguaje.
4. **Token-economía.** Solo 3 tokens estructurales nuevos: `[`, `]`, `G`.
   Todo lo demás son primos y palabras del censo.
5. **La longitud no es coste; la ambigüedad sí.** Las secuencias K-65P son más
   largas que el español equivalente — presupuestado y aceptado (decisión de
   Joan, 2026-07-08): cada token inambiguo hace trabajo útil de atención.

---

## 1. Gramática (BNF)

```bnf
secuencia  ::= clausula                      ; una emisión completa es una cláusula

expr       ::= atomo | grupo | clausula

grupo      ::= '[' 'G' atomo modificador* ']'    ; sintagma nominal: núcleo + mods
modificador::= atomo                              ; descriptores, cuantificadores,
                                                  ; determinantes, palabras del censo

clausula   ::= '[' predicado expr{min..max} ']'  ; según marco de valencia (§2)
             | '[' evaluador expr ']'             ; atribución: "X es <evaluador>"
             | '[' op_unario clausula ']'         ; no, quizá, poder, y marcos
                                                  ; espacio-temporales
             | '[' conector expr expr ']'         ; si, porque, como

atomo      ::= primo | palabra_del_censo          ; censo = clean_vocabulary_words.json
```

Reglas duras:

- **Orden posicional fijo**: el rol de cada argumento lo determina su posición
  en el marco de valencia. No hay marcas de caso ni preposiciones.
- **Sin elipsis estructural**: los argumentos opcionales se omiten solo por la
  derecha (nunca "saltando" posiciones).
- **`G` exige núcleo**: `[G]` es ilegal; el primer elemento del grupo es el
  núcleo, el resto modificadores en orden libre pero canónico (el compilador
  siempre emite el mismo orden: descriptores → cuantificadores → determinantes).
- **La raíz es una cláusula**: un átomo o grupo suelto no es una emisión válida.

## 2. Marcos de valencia (v0)

Derivados de las combinaciones canónicas de Goddard & Wierzbicka, adaptados a
los 65 primos del proyecto. `?` = opcional (omisión solo por la derecha).

| Predicado | Marco (orden posicional) | Ejemplo |
|---|---|---|
| `hacer` | agente, acción?, paciente? | `[hacer yo beber agua]` |
| `pasar` | tema, experimentador? | `[pasar [G algo malo] alguien]` |
| `mover` | tema, origen?, destino? | `[mover tú lejos]` |
| `tocar` | agente, paciente | `[tocar alguien fuego]` |
| `pensar` | experimentador, contenido? | `[pensar yo]` |
| `saber` | experimentador, contenido? | `[saber yo [bueno agua]]` |
| `querer` | experimentador, contenido | `[querer yo agua]` |
| `sentir` | experimentador, estado | `[sentir yo malo]` |
| `ver` | experimentador, tema? | `[ver yo algo]` |
| `oír` | experimentador, tema? | `[oír tú agua]` |
| `decir` | agente, contenido, destinatario? | `[decir tú verdad yo]` |
| `existir` | tema, lugar? | `[existir yo]` |
| `vivir` | tema, lugar? | `[vivir gente aquí]` |
| `morir` | tema | `[morir [G gente todo]]` |

**Evaluadores/descriptores como predicados unarios** (atribución "X es Y"):
`bueno`, `malo`, `grande`, `pequeño`, `caliente`, `frío`, `oscuro`, `mío`.
→ `[caliente fuego]` = "el fuego es caliente". (Sin cópula: no existe primo
"ser" y no lo inventamos — la v1 del plan usaba `copula_is`, retirado aquí.)

**Operadores unarios sobre cláusula**: lógicos `no`, `quizá`, `poder`;
marcos temporales `antes`, `ahora`, `después`; espaciales `aquí`, `cerca`,
`lejos`, `arriba`, `abajo`, `dentro`.
→ `[quizá [pasar lluvia]]` = "quizá llueva".

**Conectores binarios** (ambos con el orden fijo *primer-plano-primero*):

- `[si condición consecuencia]`
- `[porque causa efecto]` — la causa siempre primero, simétrico con `si`.
- `[como A B]` = "A es como B" (analogía; la semilla de la metáfora).

## 3. Ejemplos compilados a mano (corpus semilla)

| # | Español | K-65P |
|---|---|---|
| 1 | yo veo algo | `[ver yo algo]` |
| 2 | tú oyes agua | `[oír tú agua]` |
| 3 | el fuego es caliente | `[caliente fuego]` |
| 4 | el fuego caliente es malo | `[malo [G fuego caliente]]` |
| 5 | el fuego caliente es peligroso | `[si [tocar alguien [G fuego caliente]] [pasar [G algo malo] alguien]]` |
| 6 | estoy triste porque te fuiste | `[porque [mover tú lejos] [sentir yo malo]]` |
| 7 | quiero agua | `[querer yo agua]` |
| 8 | quiero beber agua | `[querer yo [hacer yo beber agua]]` |
| 9 | quizá llueva | `[quizá [pasar lluvia]]` |
| 10 | puedo moverme | `[poder [mover yo]]` |
| 11 | no veo nada | `[no [ver yo algo]]` |
| 12 | sé que el agua es buena | `[saber yo [bueno agua]]` |
| 13 | quiero que me digas la verdad | `[querer yo [decir tú verdad yo]]` |
| 14 | la gente vive aquí | `[vivir gente aquí]` |
| 15 | antes yo era pequeño | `[antes [pequeño yo]]` |
| 16 | algo malo pasará | `[después [pasar [G algo malo]]]` |
| 17 | esta cosa es como el agua | `[como [G cosa este] agua]` |
| 18 | todos mueren | `[morir [G gente todo]]` |
| 19 | pienso, luego existo | `[porque [pensar yo] [existir yo]]` |
| 20 | si tocas el agua fría, sientes frío | `[si [tocar tú [G agua frío]] [sentir tú frío]]` |

Obsérvese el §5 (fidelidad): el ejemplo 5 es la compilación *honesta* de
"peligroso" — multicláusula, más larga que la frase original. Ese es el precio
real del NSM y está presupuestado.

## 4. Validador

`src/bitnet/translation/nsm_syntax.py` — parser + validador de referencia:

- `parse(texto)` → árbol (átomos `str`, cláusulas `list`). Errores de
  tokenización/balanceo con posición.
- `validate(texto_o_árbol, vocab=None)` → lista de errores (vacía = válido):
  operador desconocido, aridad fuera de marco, `G` sin núcleo, raíz no-cláusula,
  y (si se pasa `vocab`) átomos fuera del censo.
- `linearize(árbol)` → forma canónica (un espacio, corchetes pegados).
- Tests en `tests/test_nsm_syntax.py` (los 20 ejemplos del §3 + casos inválidos).

El validador es la **fuente de verdad** de esta spec: en caso de discrepancia
entre este documento y el código, gana el código y se corrige el documento.

## 5. Criterio de aceptación del sistema traductor (pre-registrado)

- **Compilador** (ES→K-65P): análisis *con pérdida* — descarta registro, tono y
  estilo, y debe descartarlos. En fase 1 no se escribe: la fábrica Samantha
  emite pares `(frase_ES, K-65P)` por construcción (plantillas que conocen su
  propia semántica).
- **Decompilador** (K-65P→ES): *generación*, uno-a-muchos. Puede interpretar y
  enriquecer libremente (registro, calidez, voz — la personalidad vive aquí)
  con **una única invariante**:

> **Test de ida y vuelta**: `K-65P → español → (recompilar) → K-65P'` debe
> devolver el mismo árbol. Si `K-65P ≠ K-65P'`, el enriquecimiento inventó o
> destruyó semántica y el traductor suspende. Automático y sin juez.

- **Evaluación de Bit-NSM siempre sobre glifos crudos**, nunca a través del
  decompilador (si no, medimos al traductor, no a la criatura).

## 6. Robustez al ruido (mitigación pre-registrada)

Un modelo criado en un mundo perfectamente determinista puede salir frágil ante
secuencias malformadas. Mitigación: en etapas tardías del currículo Bit-NSM,
inyectar un % pequeño de secuencias con ruido controlado (glifos permutados,
corchetes rotos) con tarea de reparar o rechazar (`<unk>` estructural).

## 7. Limitaciones conocidas de la v0 (candidatas a v1)

1. **Sin modo**: no hay imperativo ni interrogativo (los primos `cuándo`/`dónde`
   quedan reservados para interrogación en v1). Se rodea con `querer`+cláusula
   (ejemplo 13).
2. **Sin plural morfológico**: la pluralidad es léxica (`dos`, `algunos`,
   `todo`, `mucho` como modificadores de grupo).
3. **Tiempo grueso**: solo `antes`/`ahora`/`después` como marcos; sin aspecto.
4. **Orden de modificadores en `G`**: canónico por convención del compilador,
   no verificado semánticamente por el validador v0.
5. ~~**Nombre del lenguaje**~~: resuelto — **K-65P**, ratificado por el Operador
   el 2026-07-08 (§0).

## 8. Relación con el plan general

- School v3 **no cambia**: Bit lingüístico entrena hasta la batería M4
  ("8 años") y es el **baseline de control**. Bit-NSM es la rama experimental;
  la comparación entre ambos es el experimento (instinto de Joan como hipótesis,
  School v3 como control — así se hace ciencia con fe incluida).
- Camino crítico del track: esta spec → pares paralelos desde la fábrica
  (`samantha_story_factory.py` emite `(ES, K-65P)`) → mini Bit-NSM → comparación.
- Encaje futuro: K-65P como interlingua inter-agente del Jungle Reboot (Nico,
  Sofy, Hugo) — el hermano simbólico de Tensors-as-API.

---

## ADDENDUM (31-ago-2026) — Auditoría de la proyección Ridge: sin estructura semántica

**Motivación**: en la confrontación BIT-003 v4 (×1, protocolo honesto), el brazo
glyph quedó por debajo del estándar en el hito homólogo de 2 años (números
re-medidos el 09-01 con el runner auditado: retención 19.2 vs 25.6%, gate 44.6
vs 51.8%, cognición 1.6 vs 5.8% — esta última dentro del ruido, n=189). *Nota de
la auditoría (09-01)*: la neurogénesis 128→256 del glyph la disparó un crash de
infraestructura del examen, no una calificación (CHANGELOG 09-01) — "necesitó
crecer donde el estándar no" queda SIN demostrar (receta `bit003_glyph_v41_2y_x1`
para re-responderlo). Hipótesis del operador: la generación de glifos no respeta
la composicionalidad NSM.

**Auditoría cuantitativa** (scripts inline, 31-ago):

1. **El pipeline NO descompone NSM por palabra**: los ~19,600 glifos no-canónicos
   son una **Ridge entrenada con 26 muestras** (los anclas canónicos) que interpola
   fastembed → mezcla ponderada de los 26 glifos ancla, ternarizada. El glifo de
   cualquier palabra es una mezcla de 26 conceptos, no un análisis de primos.
2. **Sin estructura de categorías**: similitud de glifos intra-grupo vs aleatorio
   (0.408): animales 0.502, objetos 0.418, agua 0.431, acciones **0.290 (BAJO el
   azar)**. El espacio de glifos no agrupa semánticamente mejor que el azar.
3. **Las mezclas son temáticas pero débiles**: bear → cueva/bosque/depredador;
   river → río/agua/tormenta — coherencia aproximada, sin la estructura de primos
   que la hipótesis NSM requiere.

**Conclusión**: la proyección estadística de un idioma a glifos NO preserva la
composicionalidad — el resultado adverso del brazo glyph era el esperado para
esta implementación, no para la hipótesis K-65P. La hipótesis K-65P vive en la
**sintaxis** (árboles NSM-C donde la composicionalidad es simbólica), y su test
requiere el compilador EN→K-65P (fundación construida: `en_lexicon.py`,
`nsm_syntax_en.py`). Los glifos de palabras fuera de los anclas requieren
descomposición NSM simbólica o el canal dim66 (DL-012) para convenciones.

**No se tira**: la comparación completa inglés-glyph vs inglés-standard queda
como resultado documentado de que el atajo estadístico falla — con el pipeline
de medición (batería congelada + 3 instrumentos) como el activo metodológico.
