# Cata gate-0: clasificación y traducción justa (para curación)

> DL-016→023. Actualizado 3-sep. Ideas fuente en inglés (convención);
> descomposición NSM y cláusulas K-65P por molécula. Vocabulario v2 vivo:
> 16 moléculas sembradas (`scripts/seed_vocab_v2.py`).

## 1. Los 65 primos (Capa 0 — indefinibles por definición)

| id | en | es | categoría | valencia |
|---|---|---|---|---|
| 0 | i | yo | atom |  |
| 1 | you | tú | atom |  |
| 2 | someone | alguien | atom |  |
| 3 | people | gente | atom |  |
| 4 | something | algo | atom |  |
| 5 | thing | cosa | atom |  |
| 6 | body | cuerpo | atom |  |
| 7 | part | parte | atom |  |
| 8 | good | bueno | evaluator |  |
| 9 | bad | malo | evaluator |  |
| 10 | big | grande | evaluator |  |
| 11 | small | pequeño | evaluator |  |
| 12 | think | pensar | predicate | 1–2: experiencer, content |
| 13 | know | saber | predicate | 1–2: experiencer, content |
| 14 | want | querer | predicate | 2–2: experiencer, content |
| 15 | feel | sentir | predicate | 2–2: experiencer, state |
| 16 | see | ver | predicate | 1–2: experiencer, theme |
| 17 | hear | oír | predicate | 1–2: experiencer, theme |
| 18 | say | decir | predicate | 2–3: agent, content, addressee |
| 19 | word | palabra | atom |  |
| 20 | true | verdad | atom |  |
| 21 | do | hacer | predicate | 1–3: agent, action, patient |
| 22 | happen | pasar | predicate | 1–2: theme, experiencer |
| 23 | move | mover | predicate | 1–3: theme, origin, destination |
| 24 | touch | tocar | predicate | 2–2: agent, patient |
| 25 | exist | existir | predicate | 1–2: theme, place |
| 26 | mine | mío | evaluator |  |
| 27 | live | vivir | predicate | 1–2: theme, place |
| 28 | die | morir | predicate | 1–1: theme |
| 29 | when | cuándo | atom |  |
| 30 | now | ahora | unary_operator |  |
| 31 | before | antes | unary_operator |  |
| 32 | after | después | unary_operator |  |
| 33 | long_time | mucho_tiempo | atom |  |
| 34 | short_time | poco_tiempo | atom |  |
| 35 | moment | momento | atom |  |
| 36 | where | dónde | atom |  |
| 37 | here | aquí | unary_operator |  |
| 38 | above | arriba | unary_operator |  |
| 39 | below | abajo | unary_operator |  |
| 40 | far | lejos | unary_operator |  |
| 41 | near | cerca | unary_operator |  |
| 42 | side | lado | atom |  |
| 43 | inside | dentro | unary_operator |  |
| 44 | not | no | unary_operator |  |
| 45 | maybe | quizá | unary_operator |  |
| 46 | can | poder | unary_operator |  |
| 47 | because | porque | binary_connector |  |
| 48 | if | si | binary_connector |  |
| 49 | very | muy | atom |  |
| 50 | more | más | atom |  |
| 51 | like | como | binary_connector |  |
| 52 | this | este | atom |  |
| 53 | same | mismo | atom |  |
| 54 | other | otro | atom |  |
| 55 | one | uno | atom |  |
| 56 | two | dos | atom |  |
| 57 | some | algunos | atom |  |
| 58 | all | todo | atom |  |
| 59 | much | mucho | atom |  |
| 60 | hot | caliente | evaluator |  |
| 61 | cold | frío | evaluator |  |
| 62 | water | agua_prima | atom |  |
| 63 | light | luz | atom |  |
| 64 | dark | oscuro | evaluator |  |

## 2. Moléculas confirmadas (sembradas — `configs/k65p_v2/moleculas.json`)

| molécula | idea fuente (EN) | descomposición NSM | K-65P |
|---|---|---|---|
| **fire** | hot light that can burn | es de la familia de la luz; es caliente; no es frío; puede hacer algo malo (quemar) | `[G light fire] [hot fire] [not [cold fire]] [can [do fire [G something bad]]]` |
| **sun** | the very big bright thing above, far away, of the fire family | es de la familia del fuego; es muy grande; está muy lejos; está arriba | `[G fire sun] [very [big sun]] [very [far [exist sun]]] [above [exist sun]]` |
| **earth** | the very big ground everything lives on, near under our feet, that does not move | es cosa; es muy grande; la gente vive en él abajo; es lo cercano que se toca; no se mueve | `[G thing earth] [very [big earth]] [below [live people earth]] [very [near [touch people earth]]] [not [move earth]]` |
| **blind** | if someone is blind, then they cannot see | si alguien es ciego, entonces no puede ver | `[if [G blind someone] [not [can [see blind]]]]` |
| **black** | without light; like dying and like cold; like blindness | es oscuro; no es de la familia de la luz; como morir; como el frío; como la ceguera | `[dark black] [not [G light black]] [like black die] [like black cold] [like black blind]` |
| **white** | all the light | toda la luz (saturación) | `[very [G light white]]` |
| **red** | light like fire | es de la luz; como el fuego | `[G light red] [like red fire]` |
| **yellow** | light like the sun | es de la luz; como el sol | `[G light yellow] [like yellow sun]` |
| **blue** | light like water, cold | es de la luz; es fría | `[G light blue] [cold blue]` |
| **brown** | the dark color of the earth | como la tierra; no es luz plena | `[like brown earth] [not [very [G light brown]]]` |
| **purple** | the light mix of red and blue | es de la luz; como el rojo; como el azul (frío: ambivalente = 0) | `[G light purple] [like purple red] [like purple blue]` |
| **green** | the light mix of blue and yellow | es de la luz; como el azul; como el amarillo | `[G light green] [like green blue] [like green yellow]` |
| **joy** | the state people feel when something very good happens; canon: yellow | la gente siente alegría; es muy buena; su color es el amarillo | `[feel people joy] [very [good joy]] [yellow joy]` |
| **sadness** | the state people feel when something bad happens; canon: blue, studies: blue-sadness 53% | la gente siente tristeza; es mala; su color es el azul | `[feel people sadness] [bad sadness] [blue sadness]` |
| **anger** | the hot bad state; canon: red, studies: red-anger 73% | la gente siente ira; es mala; es caliente; su color es el rojo | `[feel people anger] [bad anger] [hot anger] [red anger]` |
| **fear** | the bad state of expecting bad; canon: purple, studies: fear→purple/grey/black | la gente siente miedo; es malo; su color es el morado | `[feel people fear] [bad fear] [purple fear]` |
| **disgust** | the bad state of rejecting contamination; canon: green (broccoli), studies: brown (excrement) — many-to-many | la gente siente asco; es malo; sus colores son el verde (canon) y el marrón (estudios) | `[feel people disgust] [bad disgust] [green disgust] [brown disgust]` |

## 3. Borradores pendientes de tu validación (por capas — cada capa solo referencia lo definido)

| capa | molécula | idea fuente (EN) | descomposición NSM | K-65P (borrador) |
|---|---|---|---|---|
| **A — la base animal** | | | | |
| animal | a living thing that moves by itself and can act | vive; se mueve; puede hacer | `[live animal] [move animal] [can [do animal]]` |
| **B — gente y familia** | | | | |
| child | a young person | es alguien; es pequeño | `[small child]` |
| baby | a very young child | es una clase de child; muy pequeño | `[very [small baby]]` |
| parent | a person who feels something good toward their baby (care = feel-good-toward, NSM) | es alguien; siente algo bueno hacia el bebé | `[feel parent [G baby good]]` |
| friend | a person one feels good with | es alguien; se siente algo bueno juntos | `[feel people [G friend good]]` |
| **C — animales y sonidos** | | | | |
| bark | the sound of the dog — a big sound people hear | la gente lo oye; es un sonido grande | `[hear people bark] [big bark]` |
| meow | the sound of the cat — a small sound people hear | la gente lo oye; es un sonido pequeño | `[hear people meow] [small meow]` |
| dog | an animal that says bark and lives with people | es animal; dice bark; vive con la gente | `[say dog bark] [live dog people]` |
| cat | an animal that says meow and lives with people | es animal; dice meow; vive con la gente | `[say cat meow] [live cat people]` |
| **D — naturaleza** | | | | |
| tree | a tall living thing with green parts | vive; es grande; tiene partes verdes | `[live tree] [big tree] [G green tree]` |
| forest | many trees together (collection = group + quantity, DL-017) | árboles juntos; muchos juntos | `[G tree forest] [G much forest]` |
| river | water that moves | es de la familia del agua; se mueve | `[G water river] [move river]` |
| rain | water that falls from above | es agua; se mueve hacia abajo | `[G water rain] [below [move rain]]` |
| storm | bad much-water weather | es agua; es malo; es mucho | `[G water storm] [bad storm] [G much storm]` |
| night | the time when it is dark | es oscuro; es un momento | `[dark night] [G moment night]` |
| moon | the light in the dark sky | es de la luz; es del oscuro — la luz de la noche | `[G light moon] [G dark moon]` |
| stone | a hard thing that does not move by itself | es cosa; no se mueve | `[G thing stone] [not [move stone]]` |
| cave | a dark thing | es oscuro; es cosa | `[dark cave] [G thing cave]` |
| **E — comida y acciones** | | | | |
| food | the things living beings eat — what keeps alive | es cosa; es buena | `[G thing food] [good food]` |
| play | do things for feel-good | se hace; es bueno | `[do people play] [good play]` |
| eat | put food in the body | se hace; es de la familia de la comida; es del cuerpo | `[do people eat] [G food eat] [G body eat]` |
| drink | put water in the body | se hace; es de la familia del agua; es del cuerpo | `[do people drink] [G water drink] [G body drink]` |
| sleep | the body rests and does not move | el cuerpo no se mueve; es un momento | `[not [move sleep]] [G moment sleep] [G body sleep]` |
| give | do, touching, something good for someone | se hace; se toca; es bueno | `[do people give] [touch people give] [G good give]` |
| help | do something so someone feels good and can do it | se hace; se siente algo bueno | `[do people help] [feel people [G help good]]` |
| kiss | touch with the mouth to feel-good (mouth → capa de partes del cuerpo) | se toca; se siente algo bueno | `[touch people kiss] [feel people [G kiss good]]` |
| make | do so a new thing exists | se hace; hace existir | `[do people make] [G exist make]` |
| need | want something very much | se quiere; es muy bueno tenerlo | `[want people need] [very [good need]]` |
| put | move a thing to a place and touch it there | se hace; se mueve; se toca; en un lado | `[do people put] [move people put] [touch people put] [G side put]` |
| read | see words and know them | se ven las palabras; se saben | `[see people read] [know people read] [G word read]` |
| sit | the body moves down and then does not move | el cuerpo se mueve hacia abajo; luego no se mueve | `[below [move sit]] [not [move sit]] [G body sit]` |
| take | touch a thing and move it to oneself | se toca; se mueve | `[touch people take] [move people take]` |
| try | do wanting it to work, not knowing if it can — and if it works, it turns out good | se hace; se quiere; no se sabe si se puede; y si sale, queda bueno | `[do people try] [want people try] [not [know people [can [do try]]]] [if [do people try] [good try]]` |
| turn | the body or a thing changes side | el cuerpo se mueve; cambia de lado | `[move turn] [G side turn] [G body turn]` |
| way | the thing one moves along from here | es cosa; desde aquí; se mueve por él | `[G thing way] [G here way] [move way]` |
| **F — cosas** | | | | |
| ball | a thing children play with | es cosa; se juega con ella; de niños | `[G play ball] [G child ball]` |
| apple | a sweet food eaten with the mouth (sweet = very-good; mouth → capa de partes del cuerpo) | es comida; es MUY buena; se come | `[G food apple] [very [good apple]] [do people [G eat apple]]` |
| bread | the food from the earth | es comida; viene de la tierra | `[G food bread] [G earth bread]` |
| bed | the thing one sleeps on | es cosa; de la familia de dormir | `[G thing bed] [G sleep bed]` |
| book | things with words inside to read | es cosa; tiene palabras; se lee; tiene dentro | `[G word book] [G read book] [G inside book]` |
| box | the thing for putting things in | es cosa; de la familia de put | `[G thing box] [G put box]` |
| car | a thing that moves people | es cosa; se mueve; lleva gente | `[G thing car] [move car] [G people car]` |
| train | the big thing that moves people | es cosa; se mueve; lleva gente; es grande | `[G thing train] [move train] [G people train] [big train]` |
| house | the place where people live | es cosa; la gente vive en él | `[G thing house] [live people house]` |
| home | the place where the family lives — a good house | es cosa; la gente vive; es bueno | `[G thing home] [live people home] [good home]` |
| milk | the white food one drinks | es comida; es blanca; de la familia de beber | `[G food milk] [G white milk] [G drink milk]` |
| toys | the things children play with | es cosa; se juega; de niños | `[G thing toys] [G play toys] [G child toys]` |
| **G — cualidades y cuerpo** | | | | |
| color | what things look like in the light | es cosa; se ve; es de la luz | `[G thing color] [see people color] [G light color]` |
| hurt | the body feels bad | el cuerpo siente algo malo | `[feel body hurt] [bad hurt]` |
| wound | the place in the body that hurts | es del cuerpo; es malo | `[G body wound] [bad wound]` |

## 4. Superposiciones léxicas (sinónimos → surfaces, no moléculas nuevas)

- **daddy** → dad (mismo concepto: parent — pendiente decisión de género, ⚠ P8)
- **mommy** → mom (mismo concepto: parent — pendiente ⚠ P8)
- **mummy** → mom (idem)
- **kid** → child (mismo concepto; kid queda como surface)

## Preguntas abiertas para el operador

1. ~~Thomas~~ → RESUELTA (DL-021): `[N thomas]`, marcador de nombre.
2. ~~Hueco interrogativo~~ → RESUELTA (DL-023): `[Q cláusula]` con glifo querer+saber; la fuerza ilocutiva es del intérprete.
3. ~~Colores~~ → RESUELTA (DL-022): paleta del pintor + arrastre emoción→color.
4. ~~'get'~~ → RESUELTA (DL-023): solo-en-contexto — la unidad de traducción es la frase-idea.
5. **'oh no' / 'yeah'**: ¿caen o son contextos de SILENCIO/alarm?
6. **child/baby** (DL-020): ¿re-siembra tal cual o esperan a la capa B del lote?
7. **Sonidos** (bark/meow): borrados como MOLÉCULAS con distinción por tamaño (bark=grande, meow=pequeño — los animales los arrastran por la malla). ¿Apruebas el convenio?
8. **⚠ NUEVA — El eje de género no existe en los 65 primos**: mom/dad y boy/man colisionan sin él. Opciones: (a) un solo concepto *parent* con mom/dad como surfaces hasta que exista la capa, (b) diseñar género como moléculas (NSM: "una clase de gente que puede hacer bebés" vs "no"), (c) esperar. Mi recomendación: (a) para la cata — *parent* cubre mom y dad sin doctrina forzada.
9. **⚠ NUEVA — El eje del gusto no existe** (dulce/amargo): apple se distingue de bread por prototipos (apple=buena, bread=de la tierra) — suficiente para la cata, pero un día el corpus de comida lo pedirá.
10. **⚠ NUEVA — El arrastre arrastra también los contrastes**: las referencias a moléculas traen sus trits −1 (earth=mover:−1 contaminó river en un borrador; resuelto usando la familia `[G water river]` en vez de citar earth). Convención: **preferir familia-G a referencia directa cuando el rol es locativo/categorial**, reservar la referencia directa para cuando el contenido ES definicional.

## Convención de referencia K-65P (cómo "seguir hablando de lo dicho" sin marcador nuevo)

**No hace falta un marcador tipo G o N para la anáfora** — tres mecanismos existentes
la cubren (auditoría del ejemplo fire, 3-sep):
1. **El ancla ES la anáfora**: la palabra definida repetida en sus cláusulas (`[hot fire]`
   + `[G light fire]` — los dos `fire` son "el mismo fuego", como el esto/ello NSM).
2. **El grupo-G ES la pertenencia a familia**: `[G light fire]` = "el fuego es de la
   familia de la luz" — el "es luz" que la idea pedía y las cláusulas no tenían.
3. **El anidamiento ES la referencia a lo dicho**: `[think people [can [happen bad]]]`
   — la cláusula interior es el objeto de la exterior (el "eso" de los predicados
   mentales). Los primos THIS(52)/SAME(53) existen para los casos explícitos.
4. **Regla modal** (refinamiento DL-014): la negación de un predicado MENTAL niega
   el estado, no el complemento — "no sé si puedo hacerlo" niega el SABER; el hacer
   hipotético no cancela el hacer afirmado (`try` = hacer queriendo sin saber, no
   "hacer cancelado"). La negación de contenido FÍSICO sí atribuye el contraste
   (`[not [move sleep]]` → mover:−1, el patrón hielo).
5. **Regla de completitud**: todo contenido material de la idea fuente debe aparecer
   en las cláusulas (lo que no aparece, el glifo no lo sabrá) — y cada trit debe
   trazarse a una palabra de la idea. La auditoría idea→huella forma parte de la
   generación de este documento.
6. **Inglés para humanos, números para la máquina**: las cláusulas curadas usan
   nombres EN (`[below ...]`, `[can ...]`) — nunca ids numéricos, que colisionan
   visualmente (`[43 ...]` es dentro, NO poder — confusión real detectada en fire).
   La forma canónica numérica la genera `linearize()`.

## El pipeline tras tu curación

Capas aprobadas → `seed_vocab_v2.py` extendido → compilador de patrones (los marcos × rellenos del pool) → corpus mínimo validado → **cata: Bit v2 a 32d** con la escalera DL-019 y el test de permutación como primera lectura.
