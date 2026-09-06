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
	("animal", "a living thing that moves by itself and can act",
	 "vive; se mueve; puede hacer", ["[live animal]", "[move animal]", "[can [do animal]]"], "A"),
	# ── capa B: gente y familia (compuestos de SOMEONE) ──
	("child", "a young person", "es alguien; es pequeño", ["[small child]"], "B"),
	("baby", "a very young child", "es una clase de child; muy pequeño", ["[very [small baby]]"], "B"),
	("parent", "a person who feels something good toward their baby (care = feel-good-toward, NSM)",
	 "es alguien; siente algo bueno hacia el bebé", ["[feel parent [G baby good]]"], "B"),
	("friend", "a person one feels good with",
	 "es alguien; se siente algo bueno juntos", ["[feel people [G friend good]]"], "B"),
	# ⚠ boy/man: sin primos de género no se distinguen (ver Preguntas 8)
	# ── capa C: animales y sonidos ──
	("bark", "the sound of the dog — a big sound people hear",
	 "la gente lo oye; es un sonido grande", ["[hear people bark]", "[big bark]"], "C"),
	("meow", "the sound of the cat — a small sound people hear",
	 "la gente lo oye; es un sonido pequeño", ["[hear people meow]", "[small meow]"], "C"),
	("dog", "an animal that says bark and lives with people",
	 "es animal; dice bark; vive con la gente", ["[say dog bark]", "[live dog people]"], "C"),
	("cat", "an animal that says meow and lives with people",
	 "es animal; dice meow; vive con la gente", ["[say cat meow]", "[live cat people]"], "C"),
	# ── capa D: naturaleza ──
	("tree", "a tall living thing with green parts",
	 "vive; es grande; tiene partes verdes", ["[live tree]", "[big tree]", "[G green tree]"], "D"),
	("forest", "many trees together (collection = group + quantity, DL-017)",
	 "árboles juntos; muchos juntos", ["[G tree forest]", "[G much forest]"], "D"),
	("river", "water that moves", "es de la familia del agua; se mueve", ["[G water river]", "[move river]"], "D"),
	("rain", "water that falls from above", "es agua; se mueve hacia abajo", ["[G water rain]", "[below [move rain]]"], "D"),
	("storm", "bad much-water weather", "es agua; es malo; es mucho", ["[G water storm]", "[bad storm]", "[G much storm]"], "D"),
	("night", "the time when it is dark", "es oscuro; es un momento", ["[dark night]", "[G moment night]"], "D"),
	("moon", "the light in the dark sky", "es de la luz; es del oscuro — la luz de la noche",
	 ["[G light moon]", "[G dark moon]"], "D"),
	("stone", "a hard thing that does not move by itself",
	 "es cosa; no se mueve", ["[G thing stone]", "[not [move stone]]"], "D"),
	("cave", "a dark thing", "es oscuro; es cosa", ["[dark cave]", "[G thing cave]"], "D"),
	# ── capa E: acciones ──
	("food", "the things living beings eat — what keeps alive", "es cosa; es buena", ["[G thing food]", "[good food]"], "E"),
	("play", "do things for feel-good", "se hace; es bueno", ["[do people play]", "[good play]"], "E"),
	("eat", "put food in the body", "se hace; es de la familia de la comida", ["[do people eat]", "[G food eat]"], "E"),
	("drink", "put water in the body", "se hace; es de la familia del agua", ["[do people drink]", "[G water drink]"], "E"),
	("sleep", "the body rests and does not move", "no se mueve; es un momento", ["[not [move sleep]]", "[G moment sleep]"], "E"),
	("give", "do, touching, something good for someone", "se hace; se toca; es bueno", ["[do people give]", "[touch people give]", "[G good give]"], "E"),
	("help", "do something so someone feels good and can do it", "se hace; se siente algo bueno", ["[do people help]", "[feel people [G help good]]"], "E"),
	("kiss", "touch with the mouth to feel-good", "se toca; se siente algo bueno", ["[touch people kiss]", "[feel people [G kiss good]]"], "E"),
	("make", "do so a new thing exists", "se hace; hace existir", ["[do people make]", "[G exist make]"], "E"),
	("need", "want something very much", "se quiere; es muy bueno tenerlo", ["[want people need]", "[very [good need]]"], "E"),
	("put", "move a thing to a place and touch it there", "se hace; se mueve; se toca", ["[do people put]", "[move people put]", "[touch people put]"], "E"),
	("read", "see words and know them", "se ve; se sabe", ["[see people read]", "[know people read]"], "E"),
	("sit", "the body moves down and then does not move", "se mueve hacia abajo; luego no se mueve",
	 ["[below [move sit]]", "[not [move sit]]"], "E"),
	("take", "touch a thing and move it to oneself", "se toca; se mueve", ["[touch people take]", "[move people take]"], "E"),
	("try", "do wanting it to work, not knowing if it can", "se hace; se quiere; no se sabe si se puede",
	 ["[do people try]", "[want people try]", "[not [know people [can [do try]]]]"], "E"),
	("turn", "the body or a thing changes side", "se mueve; cambia de lado", ["[move turn]", "[G side turn]"], "E"),
	("way", "the thing one moves along from here", "es cosa; desde aquí; se mueve por él", ["[G thing way]", "[G here way]", "[move way]"], "E"),
	# ── capa F: cosas (referencian acciones ya definidas) ──
	("ball", "a thing children play with", "es cosa; se juega con ella; de niños", ["[G play ball]", "[G child ball]"], "F"),
	("apple", "a sweet food — very good in the mouth (sweet = very-good)", "es comida; es MUY buena (dulce = very-good)", ["[G food apple]", "[very [good apple]]"], "F"),
	("bread", "the food from the earth", "es comida; viene de la tierra", ["[G food bread]", "[G earth bread]"], "F"),
	("bed", "the thing one sleeps on", "es cosa; de la familia de dormir", ["[G thing bed]", "[G sleep bed]"], "F"),
	("book", "things with words inside to read", "es cosa; tiene palabras; se lee", ["[G word book]", "[G read book]"], "F"),
	("box", "the thing for putting things in", "es cosa; de la familia de put", ["[G thing box]", "[G put box]"], "F"),
	("car", "a thing that moves people", "es cosa; se mueve; lleva gente", ["[G thing car]", "[move car]", "[G people car]"], "F"),
	("train", "the big thing that moves people", "es cosa; se mueve; lleva gente; es grande", ["[G thing train]", "[move train]", "[G people train]", "[big train]"], "F"),
	("house", "the place where people live", "es cosa; la gente vive en él", ["[G thing house]", "[live people house]"], "F"),
	("home", "the place where the family lives — a good house", "es cosa; la gente vive; es bueno",
	 ["[G thing home]", "[live people home]", "[good home]"], "F"),
	("milk", "the white food one drinks", "es comida; es blanca; de la familia de beber", ["[G food milk]", "[G white milk]", "[G drink milk]"], "F"),
	("toys", "the things children play with", "es cosa; se juega; de niños", ["[G thing toys]", "[G play toys]", "[G child toys]"], "F"),
	# ── capa G: cualidades y cuerpo ──
	("color", "what things look like in the light", "es cosa; se ve; es de la luz", ["[G thing color]", "[see people color]", "[G light color]"], "G"),
	("hurt", "the body feels bad", "el cuerpo siente algo malo", ["[feel body hurt]", "[bad hurt]"], "G"),
	("wound", "the place in the body that hurts", "es del cuerpo; es malo", ["[G body wound]", "[bad wound]"], "G"),
]

# superposiciones léxicas (synónimos → surfaces, no moléculas nuevas)
SYNONYMS = {
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
		"fire": "es caliente; no es frío",
		"sun": "muy caliente; no es frío",
		"earth": "es cosa; no se mueve",
		"black": "es oscuro; no es de la familia de la luz",
		"white": "toda la luz (saturación)",
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
		L.append(f"| **{name}** | {m['idea']} | {NSM_SEED.get(name, '')} | `{cl}` |")
	L.append("")

	# ── 3. borradores por capas ──
	L.append("## 3. Borradores pendientes de tu validación (por capas — cada capa solo referencia lo definido)\n")
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
8. **⚠ NUEVA — El eje de género no existe en los 65 primos**: mom/dad y boy/man colisionan sin él. Opciones: (a) un solo concepto *parent* con mom/dad como surfaces hasta que exista la capa, (b) diseñar género como moléculas (NSM: "una clase de gente que puede hacer bebés" vs "no"), (c) esperar. Mi recomendación: (a) para la cata — *parent* cubre mom y dad sin doctrina forzada.
9. **⚠ NUEVA — El eje del gusto no existe** (dulce/amargo): apple se distingue de bread por prototipos (apple=buena, bread=de la tierra) — suficiente para la cata, pero un día el corpus de comida lo pedirá.
10. **⚠ NUEVA — El arrastre arrastra también los contrastes**: las referencias a moléculas traen sus trits −1 (earth=mover:−1 contaminó river en un borrador; resuelto usando la familia `[G water river]` en vez de citar earth). Convención: **preferir familia-G a referencia directa cuando el rol es locativo/categorial**, reservar la referencia directa para cuando el contenido ES definicional.

## El pipeline tras tu curación

Capas aprobadas → `seed_vocab_v2.py` extendido → compilador de patrones (los marcos × rellenos del pool) → corpus mínimo validado → **cata: Bit v2 a 32d** con la escalera DL-019 y el test de permutación como primera lectura.
""")
	(BASE / "docs" / "cata_gate0_curation.md").write_text("\n".join(L), encoding="utf-8")
	print(f"doc regenerado: {len(primos['primes'])} primos, {len(seed)} confirmadas, {len(DRAFTS)} borradores")


if __name__ == "__main__":
	main()
