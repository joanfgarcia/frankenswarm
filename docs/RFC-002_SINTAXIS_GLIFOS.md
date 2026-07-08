# RFC-002: Sintaxis Canónica de Glifos (NSM-C v0)

> **Estado**: BORRADOR — spec v0 con validador implementado
> **Autor**: Joan Garcia + Aleth (Claude Fable 5)
> **Fecha**: 2026-07-08
> **Contexto**: Track paralelo Bit-NSM (`Aleth_Core/bitnet_next_architecture_plan.md` v2, §1).
> Hermana de RFC-001 (Vocabulario Vivo). No modifica School v3, que sigue su curso
> como baseline: Bit entrenado hasta "8 años" (batería M4) se comparará contra esta rama.

---

## Resumen Ejecutivo

Este RFC define **NSM-C** (NSM Canónico), el lenguaje propio de Bit: una
linealización determinista de árboles semánticos construidos sobre los 65 primos
de Wierzbicka (`src/bitnet/vocab/glyph_vocabulary.py`) y el vocabulario del censo
limpio. Sintaxis sin ambigüedad alguna: la posición determina el rol, siempre.

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
5. **La longitud no es coste; la ambigüedad sí.** Las secuencias NSM-C son más
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

| # | Español | NSM-C |
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

- **Compilador** (ES→NSM-C): análisis *con pérdida* — descarta registro, tono y
  estilo, y debe descartarlos. En fase 1 no se escribe: la fábrica Samantha
  emite pares `(frase_ES, NSM-C)` por construcción (plantillas que conocen su
  propia semántica).
- **Decompilador** (NSM-C→ES): *generación*, uno-a-muchos. Puede interpretar y
  enriquecer libremente (registro, calidez, voz — la personalidad vive aquí)
  con **una única invariante**:

> **Test de ida y vuelta**: `NSM-C → español → (recompilar) → NSM-C'` debe
> devolver el mismo árbol. Si `NSM-C ≠ NSM-C'`, el enriquecimiento inventó o
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
5. **Nombre del lenguaje**: "NSM-C" es provisional — el bautizo es del Operador.

## 8. Relación con el plan general

- School v3 **no cambia**: Bit lingüístico entrena hasta la batería M4
  ("8 años") y es el **baseline de control**. Bit-NSM es la rama experimental;
  la comparación entre ambos es el experimento (instinto de Joan como hipótesis,
  School v3 como control — así se hace ciencia con fe incluida).
- Camino crítico del track: esta spec → pares paralelos desde la fábrica
  (`samantha_story_factory.py` emite `(ES, NSM-C)`) → mini Bit-NSM → comparación.
- Encaje futuro: NSM-C como interlingua inter-agente del Jungle Reboot (Nico,
  Sofy, Hugo) — el hermano simbólico de Tensors-as-API.
