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
| **fire** | hot light that can burn | es de la familia de la luz; es caliente; no es frío; puede hacer algo malo (quemar) | `[G fire light] [hot fire] [not [cold fire]] [can [do fire [G something bad]]]` |
| **sky** | the big place above where sun, moon and stars live |  | `[G thing sky] [big sky] [above [exist sky]]` |
| **sun** | the very big bright thing above, far away, of the fire family | es de la familia del fuego; es muy grande; está muy lejos; está arriba | `[G sun fire] [very [big sun]] [very [far [exist sun]]] [above [exist sun]]` |
| **earth** | the very big ground everything lives on, near under our feet, that does not move | es cosa; es muy grande; la gente vive en él abajo; es lo cercano que se toca; no se mueve | `[G earth thing] [very [big earth]] [below [live people earth]] [very [near [touch people earth]]] [not [move earth]]` |
| **new** | like the short-time thing | como la cosa de poco-tiempo | `[like new [G thing short_time]]` |
| **young** | like the short-time someone | como el alguien de poco-tiempo | `[like young [G someone short_time]]` |
| **old** | like the long-time thing and the long-time someone | como lo de mucho-tiempo (cosa y alguien) | `[like old [G thing long_time]] [like old [G someone long_time]]` |
| **blind** | if someone is blind, then they cannot see (condition as predication, not group) | si alguien es ciego, entonces no puede ver (el antecedente es predicación, no grupo) | `[if [blind someone] [not [can [see blind]]]]` |
| **void** | what does not exist | lo que no existe | `[not [exist void]]` |
| **black** | without light; like dying and like cold; like blindness; like the void | es oscuro; no es de la familia de la luz; como morir; como el frío; como la ceguera; como el vacío | `[dark black] [not [G black light]] [like black die] [like black cold] [like black blind] [like black void]` |
| **red** | light like fire | es de la luz; como el fuego | `[G red light] [like red fire]` |
| **yellow** | light like the sun | es de la luz; como el sol | `[G yellow light] [like yellow sun]` |
| **blue** | light like water, cold | es de la luz; es fría | `[G blue light] [cold blue]` |
| **brown** | the dark color of the earth | como la tierra; no es luz plena | `[like brown earth] [not [very [G brown light]]]` |
| **dirty** | like what is not good; like brown (its badness carried by [bad dirty]) | como lo no-bueno; como el marrón; es malo | `[like [not [good something]] dirty] [like dirty brown] [bad dirty]` |
| **clean** | not dirty — defined by opposition (dirty must come first) | no es sucio — definido por oposición (la oposición entera de dirty, trits invertidos) | `[not [dirty clean]]` |
| **white** | all the light; like living (life/death axis with black↔die); like clean — AFTER clean (definitional layering) | toda la luz; como vivir (eje vida/muerte con black↔morir); como lo limpio | `[very [G white light]] [like white live] [like white clean]` |
| **purple** | the light mix of red and blue | es de la luz; como el rojo; como el azul (frío: ambivalente = 0) | `[G purple light] [like purple red] [like purple blue]` |
| **green** | the light mix of blue and yellow | es de la luz; como el azul; como el amarillo | `[G green light] [like green blue] [like green yellow]` |
| **joy** | the state people feel when something very good happens; canon: yellow | la gente siente alegría; es muy buena; su color es el amarillo | `[feel people joy] [very [good joy]]` + *arrastre:* [yellow joy] |
| **sadness** | the state people feel when something bad happens; canon: blue, studies: blue-sadness 53% | la gente siente tristeza; es mala; su color es el azul | `[feel people sadness] [bad sadness]` + *arrastre:* [blue sadness] |
| **anger** | the hot bad state; canon: red, studies: red-anger 73% | la gente siente ira; es mala; es caliente; su color es el rojo | `[feel people anger] [bad anger] [hot anger]` + *arrastre:* [red anger] |
| **fear** | the bad state of expecting bad; canon: purple, studies: fear→purple/grey/black | la gente siente miedo; es malo; su color es el morado | `[feel people fear] [bad fear]` + *arrastre:* [purple fear] |
| **disgust** | the bad state of rejecting contamination; canon: green (broccoli), studies: brown (excrement) — many-to-many | la gente siente asco; es malo; sus colores son el verde (canon) y el marrón (estudios) | `[feel people disgust] [bad disgust]` + *arrastre:* [green disgust] [brown disgust] |
| **child** | a young person | es joven; es pequeño; es de la gente | `[young child] [small child] [G people child]` |
| **baby** | a very young child | muy joven; muy pequeño; es una clase de child | `[very [young baby]] [very [small baby]] [G child baby]` |
| **animal** | a living thing that moves by itself and can act — a little like someone, but not of the people | como alguien (poco); es alguien; no es de la gente; vive; se mueve; puede hacer | `[like animal someone] [G animal someone] [not [G people animal]] [live animal] [move animal] [can [do animal]]` |
| **domestic** | living with people, in a good way | vive con la gente; es bueno | `[live domestic people] [good domestic]` |
| **friend** | a known someone one feels good with | alguien conoce a friend; ese alguien se siente bien con friend; es alguien | `[know someone friend] [feel people [G friend good]] [G someone friend]` |
| **dog** | a domestic animal that says bark — and is a friend (in cultures where dogs are companions) | es de la familia animal; dice bark; es doméstico; es friend | `[G dog animal] [say dog bark] [G dog domestic] [friend dog]` |
| **cat** | a domestic animal that says meow | es de la familia animal; dice meow; es doméstico | `[G cat animal] [say cat meow] [G cat domestic]` |
| **bark** | the sound of the dog — a big (loud, scandalous) sound even if the dog is small | la gente lo oye; es un sonido grande (fuerte aunque el perro sea pequeño) | `[hear people bark] [big bark] [G dog bark]` |
| **meow** | the sound of the cat — a small sound people hear | la gente lo oye; es un sonido pequeño | `[hear people meow] [small meow] [G cat meow]` |

## 3. Borradores pendientes de tu validación (por capas — cada capa solo referencia lo definido)

*(el parentesco diferido vive en §3b — no sembrar hasta la doctrina de reciprocidad)*

| capa | molécula | idea fuente (EN) | descomposición NSM | K-65P (borrador) |
|---|---|---|---|---|
| **A — la base animal** | | | | |
| vegetal | a living thing not at all like someone | vive; no es como alguien; no es alguien | `[live vegetal] [not [like vegetal someone]]` |
| **B — gente y familia** | | | | |
| parent | a person who feels something good toward their baby and acts for it (care = feel-good + do-for, NSM) | es alguien; siente algo bueno hacia el bebé; hace por el bebé | `[feel parent [G baby good]] [do parent [G baby good]]` |
| offspring | someone who belongs to a parent; who comes from a body (hijo-descendiente, NO baby-niño) | es alguien; es de parent; viene de un cuerpo | `[G offspring someone] [G offspring parent] [G offspring body]` |
| male | someone in whom babies do not live (NSM: the male cannot bear) | es alguien; los bebés no viven en él | `[G male someone] [not [live baby male]]` |
| female | someone in whom babies live (NSM: can bear) | es alguien; los bebés viven en ella | `[G female someone] [live baby female]` |
| boy | a young male person | es child; es macho | `[G boy child] [G boy male]` |
| man | a big male someone (an adult) | es alguien; es macho; es grande | `[G man someone] [G man male] [G man big]` |
| **D — naturaleza** | | | | |
| tree | a tall living vegetal with green parts that needs light to live | vive; es grande; tiene partes verdes; es vegetal; es de la luz | `[live tree] [big tree] [G tree green] [G tree vegetal] [G tree light]` |
| forest | many trees together (collection = group + quantity, DL-017) | árboles juntos; muchos juntos | `[G forest tree] [G forest much] [G forest dark]` |
| river | water that moves | es de la familia del agua; se mueve | `[G river water] [move river]` |
| sea | very big water that does not move, of the earth (the opposite of river) | es agua; no se mueve; es muy grande; es de la tierra | `[G sea water] [not [move sea]] [very [big sea]] [G sea earth]` |
| rain | water that falls from above | es agua; se mueve hacia abajo | `[G rain water] [below [move rain]]` |
| storm | bad much-rain weather from the sky, like dark | es de la familia de rain; es malo; es mucho; como lo oscuro; es del cielo | `[G storm rain] [bad storm] [G storm much] [like storm dark] [G storm sky]` |
| night | the time when it is dark | es oscuro; es un momento | `[dark night] [G night moment]` |
| moon | the light in the dark sky | es de la luz; es del oscuro — la luz de la noche | `[G moon light] [G moon dark]` |
| stone | a hard thing that does not move by itself | es cosa; no se mueve | `[G stone thing] [not [move stone]]` |
| cave | a dark thing | es oscuro; es cosa | `[dark cave] [G cave thing]` |
| **E — comida y acciones** | | | | |
| food | the things living beings eat — what keeps alive | es cosa; es buena | `[G food thing] [good food]` |
| play | do things for feel-good | se hace; es bueno | `[do people play] [good play]` |
| eat | put food in the body | se hace; es de la familia de la comida; es del cuerpo | `[do people eat] [G eat food] [G eat body]` |
| drink | put water in the body | se hace; es de la familia del agua; es del cuerpo | `[do people drink] [G drink water] [G drink body]` |
| taste | what you feel when you eat, by touch (the contact sense of eating) | se siente; es de la familia de comer; es de la familia de tocar | `[feel people taste] [G taste eat] [G taste touch]` |
| sweet | very good taste | es de la familia del gusto; es muy bueno | `[G sweet taste] [very [good sweet]]` |
| salty | sea flavor | es de la familia del gusto; como el mar | `[G salty taste] [like salty sea]` |
| bitter | very bad taste (the poison signal) | es de la familia del gusto; es muy malo | `[G bitter taste] [very [bad bitter]]` |
| sour | bad green taste (unripe) | es de la familia del gusto; es malo; es verde (no maduro) | `[G sour taste] [bad sour] [G sour green]` |
| spicy | hot taste | es de la familia del gusto; como lo caliente | `[G spicy taste] [like spicy hot]` |
| smell | what you feel at a distance, like hearing (the distant-perception family) | se siente; es de la familia de oír | `[feel people smell] [G smell hear]` |
| stink | smell very bad, brown-associated (rot/excrement — the Rozin biology) | es de la familia de oler; es muy malo; es marrón | `[G stink smell] [very [bad stink]] [G stink brown]` |
| sleep | the body rests and does not move | el cuerpo no se mueve; es un momento | `[not [move sleep]] [G sleep moment] [G sleep body]` |
| give | do, touching, something good for someone | se hace; se toca; es bueno | `[do people give] [touch people give] [G give good]` |
| help | do something so someone feels good and can do it | se hace; se siente algo bueno | `[do people help] [feel people [G help good]]` |
| kiss | touch with the mouth to feel-good (mouth → capa de partes del cuerpo) | se toca; se siente algo bueno | `[touch people kiss] [feel people [G kiss good]]` |
| make | do so a new thing exists | se hace; hace existir | `[do people make] [G make exist]` |
| need | want something very much | se quiere; es muy bueno tenerlo | `[want people need] [very [good need]]` |
| put | move a thing to a place and touch it there | se hace; se mueve; se toca; en un lado | `[do people put] [move people put] [touch people put] [G put side]` |
| read | see words and know them | se ven las palabras; se saben | `[see people read] [know people read] [G read word]` |
| sit | the body moves down and then does not move | el cuerpo se mueve hacia abajo; luego no se mueve | `[below [move sit]] [not [move sit]] [G sit body]` |
| take | touch a thing and move it to oneself | se toca; se mueve | `[touch people take] [move people take]` |
| try | do wanting it to work, not knowing if it can — and if it works, it turns out good | se hace; se quiere; no se sabe si se puede; y si sale, queda bueno | `[do people try] [want people try] [not [know people [can [do try]]]] [if [do people try] [good try]]` |
| turn | the body or a thing changes side | el cuerpo se mueve; cambia de lado | `[move turn] [G turn side] [G turn body]` |
| way | the thing one moves along from here | es cosa; desde aquí; se mueve por él | `[G way thing] [G way here] [move way]` |
| **T — taxonomía animal** | | | | |
| mammal | an animal whose babies drink its milk | es animal; es de la familia de la leche | `[G animal mammal] [G milk mammal] [G live mammal] [G hot mammal]` |
| fish | a cold animal that lives in water | es animal; vive en el agua; es frío (sangre fría) | `[G animal fish] [live fish water] [G cold fish]` |
| bird | an animal that moves above | es animal; se mueve arriba | `[G animal bird] [above [move bird]]` |
| reptile | a crawling cold animal (cold-blooded) | es animal; se mueve abajo; no es caliente | `[G animal reptile] [below [move reptile]] [not [hot reptile]]` |
| amphibian | an animal that lives in water and on earth | es animal; vive en el agua y en la tierra | `[G animal amphibian] [live amphibian water] [live amphibian earth]` |
| terrestrial | an animal that lives on earth | es animal; vive en la tierra | `[G animal terrestrial] [live terrestrial earth] [G move terrestrial]` |
| aquatic | an animal that lives in water | es animal; vive en el agua | `[G animal aquatic] [live aquatic water]` |
| flier | an animal that moves above and far (range) | es animal; se mueve arriba; va lejos | `[G animal flier] [above [move flier]] [far [move flier]]` |
| prey | an animal that is food (for someone) | es animal; es comida | `[G animal prey] [G food prey] [G move prey]` |
| predator | an animal that kills prey and eats it | es animal; hace morir a la presa; come presa | `[G animal predator] [do predator [G die prey]] [do predator [G eat prey]]` |
| **F — cosas** | | | | |
| ball | a thing children play with | es cosa; se juega con ella; de niños | `[G ball thing] [G ball move] [G ball play] [G ball child]` |
| apple | a sweet food eaten with the mouth, also sour when small and green (covers sweet and sour kinds) | es comida; es MUY buena; se come; como lo agrio-pequeño; como lo dulce-pequeño | `[G apple food] [very [good apple]] [do people [G apple eat]] [like apple [G sour small]] [like apple [G sweet small]]` |
| bread | the hot good brown food from the earth | es comida; viene de la tierra; es marrón; es caliente; es buena | `[G bread food] [G bread earth] [G bread brown] [G bread hot] [G bread good]` |
| bed | the thing one sleeps on, for the night hours | es cosa; de la familia de dormir; es de la noche | `[G bed thing] [G bed sleep] [G bed night]` |
| book | things with words inside to read | es cosa; tiene palabras; se lee; tiene dentro | `[G book word] [G book read] [G book inside]` |
| box | the thing for putting things in | es cosa; de la familia de put | `[G box thing] [G box inside] [G box put]` |
| car | a thing that moves people | es cosa; se mueve; lleva gente | `[G car thing] [move car] [G car people]` |
| train | the big thing that moves people | es cosa; se mueve; lleva gente; es grande | `[G train thing] [move train] [G train people] [big train]` |
| house | the place where people live | es cosa; la gente vive en él | `[G house thing] [live people house]` |
| home | the place where the family lives — a good house | es cosa; la gente vive; es bueno | `[G home thing] [live people home] [good home]` |
| milk | the white food one drinks | es comida; es blanca; de la familia de beber | `[G milk food] [G milk white] [G milk drink] [G milk water] [G milk cold]` |
| toys | the things children play with | es cosa; se juega; de niños | `[G toys thing] [G toys small] [G toys play] [G toys child]` |
| **G — cualidades y cuerpo** | | | | |
| color | what things look like in the light | es cosa; se ve; es de la luz | `[G color thing] [see people color] [G color light]` |
| hurt | the body feels bad | el cuerpo siente algo malo | `[feel body hurt] [bad hurt]` |
| wound | the place in the body that hurts; big ones may kill; it hurts | es del cuerpo; es malo; es grande si puede matar; duele | `[G wound body] [bad wound] [if [G wound big Xsomeone] [maybe [die Xsomeone]]] [hurt wound] [G wound mine]` |

## 3b. Diferidos a la doctrina de reciprocidad/relacional (NO sembrar)

Sibling necesita sibling/2 + axioma de simetría; el parentesco migra con él
para no re-sembrar bajo L2 (la inyectividad impediría cambiar huellas). Gramática verde,
pero fuera de la siembra hasta que las relaciones simétricas tengan diseño.

| molécula | idea fuente (EN) | descomposición NSM | K-65P (diseño en espera) |
|---|---|---|---|
| mom | a female parent | es hembra; es de la clase parent | `[female mom] [G mom parent]` |
| dad | a male parent | es macho; es de la clase parent | `[male dad] [G dad parent]` |
| grandpa | an old male parent (the parent of a parent) | es macho; es parent; es viejo | `[male grandpa] [G grandpa parent] [G grandpa old]` |
| grandma | an old female parent (the parent of a parent) | es hembra; es parent; es vieja | `[female grandma] [G grandma parent] [G grandma old]` |
| son | a male offspring who belongs to a parent | es offspring; es macho; es DE parent; es de alguien | `[male son] [G son offspring] [G son parent] [mine son]` |
| daughter | a female offspring who belongs to a parent | es offspring; es hembra; es DE parent; es de alguien | `[female daughter] [G daughter offspring] [G daughter parent] [mine daughter]` |
| uncle | a male son of a grandparent (the parent's brother) | es macho; es son; es de la familia grandpa | `[male uncle] [G uncle son] [G uncle grandpa]` |
| aunt | a female daughter of a grandparent (the parent's sister) | es hembra; es daughter; es de la familia grandpa | `[female aunt] [G aunt daughter] [G aunt grandpa]` |
| sibling | those who share parents — the same ones | es alguien; es de parent; son los mismos | `[G sibling someone] [G sibling parent] [G sibling same]` |
| brother | a male sibling whose mom and dad are the same (ones) | es macho; es sibling | `[male brother] [G brother sibling]` |
| sister | a female sibling whose mom and dad are the same (ones) | es hembra; es sibling | `[female sister] [G sister sibling]` |

## 4. Superposiciones léxicas (sinónimos → surfaces, no moléculas nuevas)

- **null** → void (el valor nulo ES el vacío: definirlo aparte colisionaría — misma definición, misma huella)
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
8. ~~**El eje de género**~~ → **RESUELTO (opción b)**: male = `[G someone male] [not [live baby male]]`
   ("alguien en quien no viven bebés") → {alguien}−{vivir}; female = `[G someone female]
   [live baby female]` ("alguien en quien viven bebés") → {alguien, vivir}+baby.
   son/daughter = offspring ∓ vivir (difieren EXACTAMENTE en el signo de vivir).
   mom/dad quedan como surfaces de parent hasta que la capa de género entre en la siembra.
11. ~~Reciprocidad~~ + **sibling**: hermano NO cabe en forma unaria (colapsaría con
    parent: le faltaría la mismidad — necesitaría SAME u operador) → hermano es
    **molécula-relación sibling/2** (`[sibling [N x] [N y]]`) con axioma de simetría
    en la KB (la doctrina de reciprocidad pendiente). Las relaciones simétricas viven
    en la base de hechos, no en el léxico unario.
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

11. **Parentesco relacional (diseñado, implementación diferida con la reciprocidad)**:
    sibling/2 y cousin-of/2 sobre `[N ...]` (`[sibling [N x] [N y]]`,
    `[cousin-of [N x] [N y]]`) con axiomas de simetría en la KB.
    **Materno ≠ paterno por construcción del sistema de símbolos**: `[N grandpaX]`
    y `[N grandpaY]` son átomos distintos por grafía (Prolog-style) — no hace
    falta ningún hecho de distinción: `[son-of [N dad] [N grandpaX]]` y
    `[son-of [N mom] [N grandpaY]]` ya dicen que los abuelos son distintos.
    Los conceptos (grandpa) son unarios; las instancias ([N ...]) son símbolos.
12. **Reciprocidad (doctrina pendiente, NO para ahora)**: friend(A,B) ⟺ friend(B,A) —
    las relaciones simétricas necesitan diseño propio (¿axioma de simetría en la KB?
    ¿cláusula espejo en la definición?). Se diseña cuando toque, no como parche.

## El pipeline tras tu curación

Capas aprobadas → `seed_vocab_v2.py` extendido → compilador de patrones (los marcos × rellenos del pool) → corpus mínimo validado → **cata: Bit v2 a 32d** con la escalera DL-019 y el test de permutación como primera lectura.
