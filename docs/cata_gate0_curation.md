# Cata gate-0: clasificación y traducción justa (borrador para curación)

> DL-016→022 aplicados. Fuente: pool de etapa 0-1 (1.66M frases, 247 palabras reales).
> **Actualizado 3-sep**: nombres propios (DL-021), paleta y emociones sembradas (DL-022),
> definiciones en cadena (DL-020). Vocabulario v2 vivo: 16 moléculas sembradas
> (`scripts/seed_vocab_v2.py`, reproducible).

## Estado de la siembra v2 (lo que ya NO hace falta curar)

Registradas y sembradas (`configs/k65p_v2/moleculas.json`):
- **Del gate (5/59)**: fire, sun, earth, blue, red
- **Paleta extra (DL-022)**: black, white, yellow, brown, purple, green — el pintor completo
- **Emociones film 1 (DL-022)**: joy, sadness, anger, fear, disgust — arrastre emoción→color
  (asco many-to-many: verde canon Pixar + marrón estudios/biología)
- **Pre-aprobadas por el operador (DL-020)**: child, baby — la cadena definicional
  person(primo) → child → baby; a re-sembrar con el lote


## Resumen

| destino | n | nota |
|---|---|---|
| primo directo/flexionado | 111 | mapeo literal, incl. water/hot/move/want/see/say |
| **molécula v2 a curar** | 59 → **54 pendientes** | 5 ya sembradas (fire/sun/earth/blue/red); child/baby pre-aprobadas (DL-020) |
| marco composicional | 42 | la traducción justa NSM (love, have, preguntas, again...) |
| cae justamente | 38 | artículos, copulativos, interjecciones |
| nombre propio | 1 | Thomas → `[N thomas]` (DL-021, resuelto) |
| sin clasificar | 2 | get (verbo ambiguo), group (→ marcador G) |

## Los 59 candidatos a molécula (tú curas: aprueba / corrige la idea / rechaza)

| palabra | idea fuente (borrador) |
|---|---|
| apple | a round sweet fruit |
| baby | a very young person |
| ball | a round thing children play with |
| bark | the sound of the dog |
| bed | the thing one sleeps on |
| ~~blue~~ | ~~the color of the sky~~ → **REGISTRADA (DL-022)** |
| book | things with words inside to read |
| box | a thing with empty inside, for putting things |
| boy | a young male person |
| bread | the food made of flour |
| car | a thing with wheels that moves people |
| cat | a small animal that says meow, lives with people |
| cave | a dark place inside the earth |
| child | a young person |
| color | what makes things look different: red, blue... |
| dad | the male parent |
| daddy | the male parent |
| dog | an animal that says bark, lives with people |
| drink | put water in the body |
| ~~earth~~ | ~~the ground; the big thing we stand on~~ → **REGISTRADA (DL-022)** |
| eat | put food in the body |
| ~~fire~~ | ~~hot light that can burn and kill~~ → **REGISTRADA (DL-022)** |
| food | the things living beings eat |
| forest | many trees together |
| friend | a person one feels good with |
| give | make someone have something |
| help | do something so someone can do it |
| home | the place where the family lives |
| house | the place where people live |
| hurt | the body feels bad |
| kid | a young person |
| kiss | touch with the mouth to feel-good |
| make | do something so a new thing exists |
| man | an adult male person |
| meow | the sound of the cat |
| milk | the white drink from the mom animal |
| mom | the female parent; the one who cares for the child |
| moon | the bright thing in the dark sky |
| need | want something very much: without it, something bad happens |
| night | the time when it is dark |
| play | do things for feel-good, not for need |
| put | move a thing to a place and touch it there |
| rain | water that falls from above |
| read | see the words of a book and know them |
| ~~red~~ | ~~the color of blood/fire~~ → **REGISTRADA (DL-022)** |
| river | water that moves on the earth |
| sit | the body goes down and rests on something |
| sleep | the body rests: eyes closed, not moving |
| stone | a hard thing that does not move by itself |
| storm | bad weather: much water and wind from the sky |
| ~~sun~~ | ~~the bright thing above, far, that makes the day~~ → **REGISTRADA (DL-022)** |
| take | touch a thing and move it to oneself |
| toys | the things children play with |
| train | the big thing on rails that moves people |
| tree | a tall living thing with parts that are green |
| try | do something wanting it to work, not knowing if it can |
| turn | the body or a thing changes direction |
| way | the place where one moves from here to there |
| wound | the place in the body where it hurts |

## Los marcos (la parte interpretativa — revisa las fórmulas)

- **again** → `[more [one moment]]` — AGAIN = un momento más
- **and** → `[G {a} {b}]` — conjunción = grupo G de átomos
- **back** → `[move {x}] [before [move {x}]]` — BACK = mover hacia lo anterior
- **called** → `[say {x} [G name {y}]]` — 'se llama X'
- **come** → `[move {x}] [here {x}]` — COME = moverse hacia aquí
- **drinks** → `→ clasificar como 'drink'` — flexión/contracción: se normaliza antes del mapa
- **find** → `[know {x} [where {y}]] [not [before [know {x} [where {y}]]]]` — FIND = sé dónde está y no lo sabía
- **first** → `[before all]` — FIRST
- **full** → `[much inside {x}]` — FULL = mucho dentro
- **had** → `→ clasificar como 'have'` — flexión/contracción: se normaliza antes del mapa
- **has** → `→ clasificar como 'have'` — flexión/contracción: se normaliza antes del mapa
- **have** → `[exist {y}] [mine {y} {x}]` — HAVE no es primo (NSM): 'y existe y es mío/de X'
- **havent** → `→ clasificar como 'have'` — flexión/contracción: se normaliza antes del mapa
- **hes** → `→ clasificar como 'he'` — flexión/contracción: se normaliza antes del mapa
- **how** → `[want {x} [know {x} [like {y} [do {z}]]]]` — HOW = de qué manera
- **learn** → `[know {x} {y}] [not [before [know {x} {y}]]]` — LEARN = sé ahora lo que no sabía antes
- **let** → `[want {x} [can {y} {z}]]` — LET = quiero que puedas
- **lets** → `→ clasificar como 'let'` — flexión/contracción: se normaliza antes del mapa
- **love** → `[feel {x} [G {y} good]]` — X feels something good toward Y (explicación NSM de LOVE)
- **mommy** → `→ clasificar como 'mom'` — flexión/contracción: se normaliza antes del mapa
- **mummy** → `→ clasificar como 'mom'` — flexión/contracción: se normaliza antes del mapa
- **out** → `[not inside {x}]` — OUTSIDE = no dentro
- **plays** → `→ clasificar como 'play'` — flexión/contracción: se normaliza antes del mapa
- **remember** → `[know {x} [before [know {x} {y}]]]` — REMEMBER = sabía antes y sé ahora
- **runs** → `→ clasificar como 'run'` — flexión/contracción: se normaliza antes del mapa
- **safe** → `[not danger {x}]` — SAFE = no peligro
- **shall** → `→ clasificar como 'will'` — flexión/contracción: se normaliza antes del mapa
- **shes** → `→ clasificar como 'she'` — flexión/contracción: se normaliza antes del mapa
- **show** → `[do {x}] [see {y} {z}]` — SHOW = hacer ver
- **sleeps** → `→ clasificar como 'sleep'` — flexión/contracción: se normaliza antes del mapa
- **teach** → `[say {x} {y} {z}] [know {z} {y}]` — TEACH = decir para que sepa
- **theres** → `[exist {y}]` — 'there is' = EXIST
- **theyre** → `→ clasificar como 'they'` — flexión/contracción: se normaliza antes del mapa
- **weve** → `→ clasificar como 'we'` — flexión/contracción: se normaliza antes del mapa
- **what** → `[want {x} [know {x} [do {y} [what]]]]` — PREGUNTA = quiero saber (la palabra interrogativa es el hueco cuestionado)
- **whats** → `→ clasificar como 'what'` — flexión/contracción: se normaliza antes del mapa
- **who** → `[want {x} [know {x} [someone]]]` — WHO
- **whos** → `→ clasificar como 'who'` — flexión/contracción: se normaliza antes del mapa
- **why** → `[want {x} [know {x} [because [happen {y}]]]]` — WHY
- **will** → `[after [do {x}]]` — FUTURE via AFTER (NSM no tiene futuro como primo)
- **with** → `[G {x} {y}]` — 'jugar conmigo' = grupo compartido
- **without** → `[not [G {x} {y}]]` — WITHOUT

## Ejemplos de traducción justa (patrones top del pool)

| frase real del corpus | K-65P v2 | nota |
|---|---|---|
| *she said* | `[say someone]` | SAY con solo el agente |
| *i love you* | `[feel I [G YOU VERY GOOD]]` | LOVE = explicar NSM: siento algo muy bueno hacia ti |
| *what are you doing* | `[want I [know I [do YOU WHAT]]]` | PREGUNTA = quiero saber; WHAT = hueco cuestionado |
| *go away* | `[move YOU] [not [near I]]` | solo con primos: muévete, no cerca de mí |
| *mom said* | `[say mom]` | mom = molécula cabeza-agente |
| *you are a good friend* | `[friend YOU] [good YOU]` | molécula como cabeza unaria + atribución; sin copulativo |
| *do you want to play with me* | `[want I [know I [want YOU [G play I]]]]` | pregunta + compuesta G 'play with me' |
| *i dont know* | `[not [know I]]  → ␀ SILENCIO` | el ancla natural de la ley L1 (abstención) |
| *are you okay* | `[want I [know I [good YOU]]]` | pregunta sobre el atributo good |
| *yes please* | `[TRUE]` | please cae; yes = TRUE |
| *there is a big tree* | `[exist tree] [big tree]` | 'there is' = EXIST; artículo cae |
| *i want to eat* | `[want I [eat I]]` | molécula eat como contenido de want |
| *the dog drinks water* | `[drink dog] [drink water dog]` | WATER es primo: 'el agua' va de primo |
| *he is in the house* | `[inside SOMEONE house]` | locativo = primo INSIDE |
| *it is hot* | `[hot something]` | atribución directa del primo HOT |

## Orden de curación recomendado (capas definicionales — DL-020)

Cada molécula solo referencia las ya definidas, así que el lote se cura por capas:
1. **Gente y familia** (8): mom, dad, baby✝, child✝, boy, kid, man, friend
2. **Animales y sonidos** (5): dog, cat, bark, meow (+animal ya en seed demos)
3. **Naturaleza** (6): tree, forest, river, rain, storm, night, moon, stone, cave
4. **Cosas** (11): ball, apple, bed, book, box, bread, car, food, house, home, milk, toys, train
5. **Acciones** (12): eat, drink, play, sleep, give, help, kiss, make, need, put, read, sit, take, try, turn
6. **Cualidades/estados** (3): color, hurt, wound, way

✝ child/baby ya tienen borrador aprobado (cadena person→child→baby, DL-020).

## Preguntas abiertas para el operador

1. ~~**Thomas** (nombres propios)~~ → **RESUELTA (DL-021)**: marcador de nombre — `[N thomas]`, símbolo sin huella, fuera de la malla. Los nombres jamás encabezan; las relaciones encabezan.
2. **El hueco interrogativo** (WHAT en preguntas): ¿lo dejamos como posición cuestionada implícita o pedimos un marcador estructural?
3. ~~**Colores**~~ → **RESUELTA (DL-022)**: paleta del pintor (blanco/rojo/amarillo/azul/negro
   + marrón y mezclas) sobre los primos luz/oscuro/caliente/frío; arrastre emoción→color
   (la asimetría de contaminación: la pelota roja queda limpia, la ira lleva su rojo).
4. **'get'** (el verbo más ambiguo del inglés): ¿molécula o frame take/come según contexto?
5. **'oh no' / 'yeah'**: ¿caen o son contextos de SILENCIO/alarm?
6. **child/baby** (DL-020): ¿las re-sembramos tal cual (cadena person→child→baby) o
   esperan a la curación del lote gente?
7. **Los sonidos** (bark/meow): ¿moléculas de sonido o marcos `[say dog [G bark]]`
   ("el perro dice bark")? El marco es más NSM-justo: el sonido es lo que se dice.