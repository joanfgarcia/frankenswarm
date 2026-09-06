"""Regenera docs/cata_gate0_curation.md: primos + confirmadas + borradores.

Las borradoras llevan descomposición NSM y cláusulas K-65P por capa definicional.
Los huecos doctrinales que aparecen al redactar se marcan ⚠ y suben como preguntas.
"""

from pathlib import Path
import json

BASE = Path(__file__).resolve().parents[1]
primos = json.loads((BASE / "configs" / "k65p_v2" / "primos.json").read_text(encoding="utf-8"))
seed = json.loads((BASE / "configs" / "k65p_v2" / "moleculas.json").read_text(encoding="utf-8"))

# (palabra, idea fuente EN, descomposición NSM, cláusulas K-65P, capa)
# ⚠ = hueco doctrinal detectado al redactar (ver Preguntas nuevas)
DRAFTS = [
	# ── capa A: la base animal ──
	("animal", "a living thing that moves by itself and can act — a little like someone, but not of the people",
	 "como alguien (poco); es alguien; no es de la gente; vive; se mueve; puede hacer",
	 ["[like animal someone]", "[G animal someone]", "[not [G animal people]]", "[live animal]", "[move animal]", "[can [do animal]]"], "A"),
	("vegetal", "a living thing not at all like someone",
	 "vive; no es como alguien; no es alguien", ["[live vegetal]", "[not [like vegetal someone]]"], "A"),
	# ── capa B: gente y familia (compuestos de SOMEONE) ──
	("child", "a young person", "es joven; es pequeño; es de la gente (el alguien llega por young: ya no hace falta [G child someone])", ["[young child]", "[small child]", "[G child people]"], "B"),
	("baby", "a very young child", "muy joven; muy pequeño; es una clase de child", ["[very [young baby]]", "[very [small baby]]", "[G baby child]"], "B"),
	("parent", "a person who feels something good toward their baby and acts for it (care = feel-good + do-for, NSM)",
	 "es alguien; siente algo bueno hacia el bebé; hace por el bebé", ["[feel parent [G baby good]]", "[do parent [G baby good]]"], "B"),
	("offspring", "someone who belongs to a parent; who comes from a body (hijo-descendiente, NO baby-niño)",
	 "es alguien; es de parent; viene de un cuerpo", ["[G offspring someone]", "[G offspring parent]", "[G offspring body]"], "B"),
	("male", "someone in whom babies do not live (NSM: the male cannot bear)",
	 "es alguien; los bebés no viven en él", ["[G male someone]", "[not [live baby male]]"], "B"),
	("female", "someone in whom babies live (NSM: can bear)",
	 "es alguien; los bebés viven en ella", ["[G female someone]", "[live baby female]"], "B"),
	# cousin: NO cabe en forma unaria (colisiona con uncle: child∪uncle ≡ son como conjuntos) —
	# es molécula-relación cousin-of/2 sobre nombres, con la simetría en la KB (igual que sibling/2).
	("friend", "a known person one feels good with",
	 "alguien conoce a friend; ese alguien se siente bien con friend; es alguien",
	 ["[know someone friend]", "[feel people [G friend good]]", "[G friend someone]"], "B"),
	# ⚠ boy/man: sin primos de género no se distinguen (ver Preguntas 8)
	# ── capa C: animales y sonidos ──
	("domestic", "living with people, in a good way",
	 "vive con la gente; es bueno", ["[live domestic people]", "[good domestic]"], "C"),
	("bark", "a big sound people hear",
	 "la gente lo oye; es un sonido grande", ["[hear people bark]", "[big bark]"], "C"),
	("meow", "a small sound people hear",
	 "la gente lo oye; es un sonido pequeño", ["[hear people meow]", "[small meow]"], "C"),
	("dog", "a domestic animal that says bark",
	 "es de la familia animal; dice bark; es doméstico", ["[G dog animal]", "[say dog bark]", "[G dog domestic]"], "C"),
	("cat", "a domestic animal that says meow",
	 "es de la familia animal; dice meow; es doméstico", ["[G cat animal]", "[say cat meow]", "[G cat domestic]"], "C"),
	# ── capa D: naturaleza ──
	("tree", "a tall living vegetal with green parts that needs light to live",
	 "vive; es grande; tiene partes verdes; es vegetal; es de la luz",
	 ["[live tree]", "[big tree]", "[G tree green]", "[G tree vegetal]", "[G tree light]"], "D"),
	("forest", "many trees together (collection = group + quantity, DL-017)",
	 "árboles juntos; muchos juntos", ["[G forest tree]", "[G forest much]", "[G forest dark]"], "D"),
	("river", "water that moves", "es de la familia del agua; se mueve", ["[G river water]", "[move river]"], "D"),
	("rain", "water that falls from above", "es agua; se mueve hacia abajo", ["[G rain water]", "[below [move rain]]"], "D"),
	("storm", "bad much-water weather", "es agua; es malo; es mucho", ["[G storm water]", "[bad storm]", "[G storm much]"], "D"),
	("night", "the time when it is dark", "es oscuro; es un momento", ["[dark night]", "[G night moment]"], "D"),
	("moon", "the light in the dark sky", "es de la luz; es del oscuro — la luz de la noche",
	 ["[G moon light]", "[G moon dark]"], "D"),
	("stone", "a hard thing that does not move by itself",
	 "es cosa; no se mueve", ["[G stone thing]", "[not [move stone]]"], "D"),
	("cave", "a dark thing", "es oscuro; es cosa", ["[dark cave]", "[G cave thing]"], "D"),
	# ── capa E: acciones ──
	("food", "the things living beings eat — what keeps alive", "es cosa; es buena", ["[G food thing]", "[good food]"], "E"),
	("play", "do things for feel-good", "se hace; es bueno", ["[do people play]", "[good play]"], "E"),
	("eat", "put food in the body", "se hace; es de la familia de la comida; es del cuerpo", ["[do people eat]", "[G eat food]", "[G eat body]"], "E"),
	("drink", "put water in the body", "se hace; es de la familia del agua; es del cuerpo", ["[do people drink]", "[G drink water]", "[G drink body]"], "E"),
	("sleep", "the body rests and does not move", "el cuerpo no se mueve; es un momento", ["[not [move sleep]]", "[G sleep moment]", "[G sleep body]"], "E"),
	("give", "do, touching, something good for someone", "se hace; se toca; es bueno", ["[do people give]", "[touch people give]", "[G give good]"], "E"),
	("help", "do something so someone feels good and can do it", "se hace; se siente algo bueno", ["[do people help]", "[feel people [G help good]]"], "E"),
	("kiss", "touch with the mouth to feel-good (mouth → capa de partes del cuerpo)", "se toca; se siente algo bueno", ["[touch people kiss]", "[feel people [G kiss good]]"], "E"),
	("make", "do so a new thing exists", "se hace; hace existir", ["[do people make]", "[G make exist]"], "E"),
	("need", "want something very much", "se quiere; es muy bueno tenerlo", ["[want people need]", "[very [good need]]"], "E"),
	("put", "move a thing to a place and touch it there", "se hace; se mueve; se toca; en un lado", ["[do people put]", "[move people put]", "[touch people put]", "[G put side]"], "E"),
	("read", "see words and know them", "se ven las palabras; se saben", ["[see people read]", "[know people read]", "[G read word]"], "E"),
	("sit", "the body moves down and then does not move", "el cuerpo se mueve hacia abajo; luego no se mueve",
	 ["[below [move sit]]", "[not [move sit]]", "[G sit body]"], "E"),
	("take", "touch a thing and move it to oneself", "se toca; se mueve", ["[touch people take]", "[move people take]"], "E"),
	("try", "do wanting it to work, not knowing if it can — and if it works, it turns out good", "se hace; se quiere; no se sabe si se puede; y si sale, queda bueno",
	 ["[do people try]", "[want people try]", "[not [know people [can [do try]]]]", "[if [do people try] [good try]]"], "E"),
	("turn", "the body or a thing changes side", "el cuerpo se mueve; cambia de lado", ["[move turn]", "[G turn side]", "[G turn body]"], "E"),
	("way", "the thing one moves along from here", "es cosa; desde aquí; se mueve por él", ["[G way thing]", "[G way here]", "[move way]"], "E"),
	# ── capa F: cosas (referencian acciones ya definidas) ──
	("ball", "a thing children play with", "es cosa; se juega con ella; de niños", ["[G ball thing]", "[G ball move]", "[G ball play]", "[G ball child]"], "F"),
	("apple", "a sweet food eaten with the mouth (sweet = very-good; mouth → capa de partes del cuerpo)", "es comida; es MUY buena; se come", ["[G apple food]", "[very [good apple]]", "[do people [G apple eat]]"], "F"),
	("bread", "the food from the earth", "es comida; viene de la tierra", ["[G bread food]", "[G bread earth]", "[G bread brown]"], "F"),
	("bed", "the thing one sleeps on", "es cosa; de la familia de dormir", ["[G bed thing]", "[G bed sleep]"], "F"),
	("book", "things with words inside to read", "es cosa; tiene palabras; se lee; tiene dentro", ["[G book word]", "[G book read]", "[G book inside]"], "F"),
	("box", "the thing for putting things in", "es cosa; de la familia de put", ["[G box thing]", "[G box inside]", "[G box put]"], "F"),
	("car", "a thing that moves people", "es cosa; se mueve; lleva gente", ["[G car thing]", "[move car]", "[G car people]"], "F"),
	("train", "the big thing that moves people", "es cosa; se mueve; lleva gente; es grande", ["[G train thing]", "[move train]", "[G train people]", "[big train]"], "F"),
	("house", "the place where people live", "es cosa; la gente vive en él", ["[G house thing]", "[live people house]"], "F"),
	("home", "the place where the family lives — a good house", "es cosa; la gente vive; es bueno",
	 ["[G home thing]", "[live people home]", "[good home]"], "F"),
	("milk", "the white food one drinks", "es comida; es blanca; de la familia de beber", ["[G milk food]", "[G milk white]", "[G milk drink]", "[G milk agua]", "[G milk cold]"], "F"),
	("toys", "the things children play with", "es cosa; se juega; de niños", ["[G toys thing]", "[G toys small]", "[G toys play]", "[G toys child]"], "F"),
	# ── capa G: cualidades y cuerpo ──
	("color", "what things look like in the light", "es cosa; se ve; es de la luz", ["[G color thing]", "[see people color]", "[G color light]"], "G"),
	("hurt", "the body feels bad", "el cuerpo siente algo malo", ["[feel body hurt]", "[bad hurt]"], "G"),
	("wound", "the place in the body that hurts", "es del cuerpo; es malo", ["[G wound body]", "[bad wound]"], "G"),
]

# superposiciones léxicas (synónimos → surfaces, no moléculas nuevas)

# ── DIFERIDAS a la doctrina de reciprocidad/relacional (auditoría 3-sep) ──
# Sibling necesita sibling/2 + simetría; el resto del parentesco migra con él
# para no re-sembrar bajo L2. Se validan igual (gramática verde) pero NO son sembrables.
# ORDEN definicional dentro del bloque (mom→…→sister): la batería registra en orden.
DEFERRED_DRAFTS = [
	("mom", "a female parent",
	 "es hembra; es de la clase parent", ["[female mom]", "[G mom parent]"], "B"),
	("dad", "a male parent",
	 "es macho; es de la clase parent", ["[male dad]", "[G dad parent]"], "B"),
	("grandpa", "an old male parent (the parent of a parent)",
	 "es macho; es parent; es viejo", ["[male grandpa]", "[G grandpa parent]", "[G grandpa old]"], "B"),
	("grandma", "an old female parent (the parent of a parent)",
	 "es hembra; es parent; es vieja", ["[female grandma]", "[G grandma parent]", "[G grandma old]"], "B"),
	("son", "a male offspring who belongs to a parent", "es offspring; es macho; es DE parent (pertenencia)",
	 ["[male son]", "[G son offspring]", "[G son parent]"], "B"),
	("daughter", "a female offspring who belongs to a parent", "es offspring; es hembra; es DE parent (pertenencia)",
	 ["[female daughter]", "[G daughter offspring]", "[G daughter parent]"], "B"),
	("uncle", "a male son of a grandparent (the parent's brother)",
	 "es macho; es son; es de la familia grandpa", ["[male uncle]", "[G uncle son]", "[G uncle grandpa]"], "B"),
	("aunt", "a female daughter of a grandparent (the parent's sister)",
	 "es hembra; es daughter; es de la familia grandpa", ["[female aunt]", "[G aunt daughter]", "[G aunt grandpa]"], "B"),
	("sibling", "those who share parents — the same ones",
	 "es alguien; es de parent; son los mismos", ["[G sibling someone]", "[G sibling parent]", "[G sibling same]"], "B"),
	("brother", "a male sibling whose mom and dad are the same (ones)",
	 "es macho; es sibling", ["[male brother]", "[G brother sibling]"], "B"),
	("sister", "a female sibling whose mom and dad are the same (ones)",
	 "es hembra; es sibling", ["[female sister]", "[G sister sibling]"], "B"),


]

SYNONYMS = {
	"null": "void (el valor nulo ES el vacío: definirlo aparte colisionaría — misma definición, misma huella)",
	"daddy": "dad (mismo concepto: parent — pendiente decisión de género, ⚠ P8)",
	"mommy": "mom (mismo concepto: parent — pendiente ⚠ P8)",
	"mummy": "mom (idem)",
	"kid": "child (mismo concepto; kid queda como surface)",
}


def main() -> None:
	L = []
	L.append("# Cata gate-0: clasificación y traducción justa (para curación)\n")
	L.append("> DL-016→023. Actualizado 3-sep. Ideas fuente en inglés (convención);")
	L.append("> descomposición NSM y cláusulas K-65P por molécula. Vocabulario v2 vivo:")
	L.append("> 16 moléculas sembradas (`scripts/seed_vocab_v2.py`).\n")

	# ── 1. los 65 primos ──
	L.append("## 1. Los 65 primos (Capa 0 — indefinibles por definición)\n")
	L.append("| id | en | es | categoría | valencia |")
	L.append("|---|---|---|---|---|")
	for p in primos["primes"]:
		v = p.get("valency", {})
		val = f"{v['args'][0]}–{v['args'][1]}: {', '.join(v['roles'])}" if v else ""
		L.append(f"| {p['id']} | {p['names']['en']} | {p['names']['es']} | {p['kind']} | {val} |")
	L.append("")

	# ── 2. confirmadas ──
	L.append("## 2. Moléculas confirmadas (sembradas — `configs/k65p_v2/moleculas.json`)\n")
	L.append("| molécula | idea fuente (EN) | descomposición NSM | K-65P |")
	L.append("|---|---|---|---|")
	NSM_SEED = {
		"fire": "es de la familia de la luz; es caliente; no es frío; puede hacer algo malo (quemar)",
		"sun": "es de la familia del fuego; es muy grande; está muy lejos; está arriba",
		"earth": "es cosa; es muy grande; la gente vive en él abajo; es lo cercano que se toca; no se mueve",
		"black": "es oscuro; no es de la familia de la luz; como morir; como el frío; como la ceguera; como el vacío",
		"void": "lo que no existe",
		"new": "como la cosa de poco-tiempo",
		"young": "como el alguien de poco-tiempo",
		"old": "como lo de mucho-tiempo (cosa y alguien)",
		"blind": "si alguien es ciego, entonces no puede ver (el antecedente es predicación, no grupo)",
		"white": "toda la luz; como vivir (eje vida/muerte con black↔morir); como lo limpio",
		"dirty": "como lo no-bueno; como el marrón; es malo",
		"clean": "no es sucio — definido por oposición (la oposición entera de dirty, trits invertidos)",
		"red": "es de la luz; como el fuego",
		"yellow": "es de la luz; como el sol",
		"blue": "es de la luz; es fría",
		"brown": "como la tierra; no es luz plena",
		"purple": "es de la luz; como el rojo; como el azul (frío: ambivalente = 0)",
		"green": "es de la luz; como el azul; como el amarillo",
		"joy": "la gente siente alegría; es muy buena; su color es el amarillo",
		"sadness": "la gente siente tristeza; es mala; su color es el azul",
		"anger": "la gente siente ira; es mala; es caliente; su color es el rojo",
		"fear": "la gente siente miedo; es malo; su color es el morado",
		"disgust": "la gente siente asco; es malo; sus colores son el verde (canon) y el marrón (estudios)",
	}
	for name, m in seed.items():
		cl = " ".join(m["clauses"])
		dr = (" + *arrastre:* " + " ".join(m.get("drags", []))) if m.get("drags") else ""
		L.append(f"| **{name}** | {m['idea']} | {NSM_SEED.get(name, '')} | `{cl}`{dr} |")
	L.append("")

	# ── 3. borradores por capas ──
	L.append("## 3. Borradores pendientes de tu validación (por capas — cada capa solo referencia lo definido)\n")
	L.append("*(el parentesco diferido vive en §3b — no sembrar hasta la doctrina de reciprocidad)*\n")
	L.append("| capa | molécula | idea fuente (EN) | descomposición NSM | K-65P (borrador) |")
	L.append("|---|---|---|---|---|")
	last = None
	for w, idea, nsm, clauses, layer in DRAFTS:
		if layer != last:
			titles = {"A": "A — la base animal", "B": "B — gente y familia", "C": "C — animales y sonidos",
				"D": "D — naturaleza", "E": "E — comida y acciones", "F": "F — cosas", "G": "G — cualidades y cuerpo"}
			L.append(f"| **{titles[layer]}** | | | | |")
			last = layer
		cl = " ".join(clauses)
		L.append(f"| {w} | {idea} | {nsm} | `{cl}` |")
	L.append("")

	# ── 3b. diferidos a la doctrina relacional ──
	L.append("## 3b. Diferidos a la doctrina de reciprocidad/relacional (NO sembrar)\n")
	L.append("Sibling necesita sibling/2 + axioma de simetría; el parentesco migra con él")
	L.append("para no re-sembrar bajo L2 (la inyectividad impediría cambiar huellas). Gramática verde,")
	L.append("pero fuera de la siembra hasta que las relaciones simétricas tengan diseño.\n")
	L.append("| molécula | idea fuente (EN) | descomposición NSM | K-65P (diseño en espera) |")
	L.append("|---|---|---|---|")
	for w, idea, nsm, clauses, layer in DEFERRED_DRAFTS:
		cl = " ".join(clauses)
		L.append(f"| {w} | {idea} | {nsm} | `{cl}` |")
	L.append("")

	# ── 4. superficies (sinónimos) ──
	L.append("## 4. Superposiciones léxicas (sinónimos → surfaces, no moléculas nuevas)\n")
	for s, note in SYNONYMS.items():
		L.append(f"- **{s}** → {note}")
	L.append("")

	# ── 5. preguntas ──
	L.append("""## Preguntas abiertas para el operador

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
""")
	(BASE / "docs" / "cata_gate0_curation.md").write_text("\n".join(L), encoding="utf-8")
	print(f"doc regenerado: {len(primos['primes'])} primos, {len(seed)} confirmadas, {len(DRAFTS)} borradores")


if __name__ == "__main__":
	main()
