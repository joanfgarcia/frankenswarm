"""Clasificación del gate etapa-0 (253 palabras) → destinos K-65P v2.

Categorías:
  prime    → mapea a un primo (directo o por forma flexionada)
  molecule → molécula v2 a curar (idea fuente en inglés incluida)
  frame    → construcción composicional con primos (la traducción justa NSM)
  drop     → cae justamente (artículos/copulativos absorbidos, interjecciones)
  name     → nombre propio (Thomas) — decisión de curación pendiente

Uso:
	.venv/bin/python scripts/classify_gate0.py  → emite map JSON + doc de curación
"""

import json
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]

# ── PRIMOS: palabra del gate → primo k65p (renderización EN del PRIMES) ──
PRIMES_MAP = {
	"i": "I", "me": "I", "im": "I", "ive": "I", "ill": "I",
	"you": "YOU", "youre": "YOU", "youve": "YOU", "your": "YOU",
	"he": "SOMEONE", "she": "SOMEONE", "him": "SOMEONE", "her": "SOMEONE",
	"they": "SOMEONE", "them": "SOMEONE", "us": "SOMEONE", "we": "SOMEONE",
	"his": "SOMEONE", "their": "SOMEONE", "our": "SOMEONE",
	"my": "MINE", "something": "SOMETHING", "something-else": "SOMETHING",
	"thing": "THING", "things": "THING",
	"good": "GOOD", "better": "GOOD", "fine": "GOOD", "nice": "GOOD", "okay": "GOOD",
	"big": "BIG", "little": "SMALL", "bit": "SMALL",
	"think": "THINK", "know": "KNOW",
	"want": "WANT", "wants": "WANT", "wanna": "WANT",
	"say": "SAY", "said": "SAY", "says": "SAY", "tell": "SAY",
	"see": "SEE", "look": "SEE", "looks": "SEE",
	"do": "DO", "does": "DO", "did": "DO", "doing": "DO",
	"can": "CAN", "could": "CAN", "might": "CAN", "would": "CAN", "should": "CAN",
	"not": "NOT", "no": "NOT", "dont": "NOT", "doesnt": "NOT", "didnt": "NOT",
	"cant": "NOT", "wont": "NOT", "isnt": "NOT", "arent": "NOT",
	"because": "BECAUSE", "if": "IF",
	"all": "ALL", "some": "SOME", "any": "SOME", "something2": "SOME",
	"one": "ONE", "ones": "ONE", "two": "TWO",
	"this": "THIS", "these": "THIS", "those": "THIS", "thats": "THIS",
	"other": "OTHER", "another": "OTHER",
	"more": "MORE", "too": "MORE", "much": "MUCH",
	"very": "VERY", "really": "VERY",
	"now": "NOW", "today": "NOW", "then": "AFTER", "when": "WHEN",
	"where": "WHERE", "wheres": "WHERE", "here": "HERE", "heres": "HERE",
	"up": "ABOVE", "over": "ABOVE", "down": "BELOW",
	"in": "INSIDE", "into": "INSIDE",
	"near": "NEAR", "far": "FAR", "away": "FAR",
	"water": "WATER", "hot": "HOT",
	"move": "MOVE", "go": "MOVE", "goes": "MOVE", "going": "MOVE", "went": "MOVE",
	"like": "LIKE", "maybe": "MAYBE",
	"yes": "TRUE", "right": "TRUE",
	"time": "MOMENT", "moment": "MOMENT",
	"it": "SOMETHING", "its": "SOMETHING",
}

# ── FRAMES: construcciones con primos (la traducción justa NSM) ──
FRAMES = {
	"love": {"frame": "feel-good-toward", "k65p": "[feel {x} [G {y} good]]",
		"nsm": "X feels something good toward Y (explicación NSM de LOVE)"},
	"love2": {"frame": "feel-very-good-toward", "k65p": "[feel {x} [G {y} very-good]]", "nsm": "LOVE intenso"},
	"have": {"frame": "possession", "k65p": "[exist {y}] [mine {y} {x}]",
		"nsm": "HAVE no es primo (NSM): 'y existe y es mío/de X'"},
	"better": {"frame": "comparative", "k65p": "[more good]", "nsm": "comparativo = more + quality"},
	"full": {"frame": "quantity-locative", "k65p": "[much inside {x}]", "nsm": "FULL = mucho dentro"},
	"safe": {"frame": "negation-of-danger", "k65p": "[not danger {x}]", "nsm": "SAFE = no peligro"},
	"remember": {"frame": "know-before", "k65p": "[know {x} [before [know {x} {y}]]]",
		"nsm": "REMEMBER = sabía antes y sé ahora"},
	"learn": {"frame": "come-to-know", "k65p": "[know {x} {y}] [not [before [know {x} {y}]]]",
		"nsm": "LEARN = sé ahora lo que no sabía antes"},
	"teach": {"frame": "cause-to-know", "k65p": "[say {x} {y} {z}] [know {z} {y}]",
		"nsm": "TEACH = decir para que sepa"},
	"let": {"frame": "want-can", "k65p": "[want {x} [can {y} {z}]]", "nsm": "LET = quiero que puedas"},
	"will": {"frame": "future", "k65p": "[after [do {x}]]", "nsm": "FUTURE via AFTER (NSM no tiene futuro como primo)"},
	"what": {"frame": "want-know", "k65p": "[want {x} [know {x} [do {y} [what]]]]",
		"nsm": "PREGUNTA = quiero saber (la palabra interrogativa es el hueco cuestionado)"},
	"why": {"frame": "want-know-because", "k65p": "[want {x} [know {x} [because [happen {y}]]]]", "nsm": "WHY"},
	"who": {"frame": "want-know-who", "k65p": "[want {x} [know {x} [someone]]]", "nsm": "WHO"},
	"how": {"frame": "want-know-how", "k65p": "[want {x} [know {x} [like {y} [do {z}]]]]", "nsm": "HOW = de qué manera"},
	"theres": {"frame": "existential", "k65p": "[exist {y}]", "nsm": "'there is' = EXIST"},
	"out": {"frame": "outside", "k65p": "[not inside {x}]", "nsm": "OUTSIDE = no dentro"},
	"and": {"frame": "group", "k65p": "[G {a} {b}]", "nsm": "conjunción = grupo G de átomos"},
	"with": {"frame": "companion", "k65p": "[G {x} {y}]", "nsm": "'jugar conmigo' = grupo compartido"},
	"without": {"frame": "not-companion", "k65p": "[not [G {x} {y}]]", "nsm": "WITHOUT"},
	"again": {"frame": "one-more-time", "k65p": "[more [one moment]]", "nsm": "AGAIN = un momento más"},
	"back": {"frame": "move-back", "k65p": "[move {x}] [before [move {x}]]", "nsm": "BACK = mover hacia lo anterior"},
	"called": {"frame": "say-name", "k65p": "[say {x} [G name {y}]]", "nsm": "'se llama X'"},
	"come": {"frame": "move-toward", "k65p": "[move {x}] [here {x}]", "nsm": "COME = moverse hacia aquí"},
	"find": {"frame": "know-where", "k65p": "[know {x} [where {y}]] [not [before [know {x} [where {y}]]]]",
		"nsm": "FIND = sé dónde está y no lo sabía"},
	"first": {"frame": "before-all", "k65p": "[before all]", "nsm": "FIRST"},
	"show": {"frame": "cause-to-see", "k65p": "[do {x}] [see {y} {z}]", "nsm": "SHOW = hacer ver"},
}

# ── DROP: caen justamente ──
DROPS = {
	"a", "an", "the",  # sin artículos en K-65P
	"be", "been", "was", "were", "is", "am", "are",  # copulativo absorbido por atribución/predicación
	"of", "to", "for", "at", "as", "from", "on", "off", "about", "else", "just",
	"so", "well", "oh", "ah", "uh", "um", "huh", "yeah", "hello", "please",
	"or", "but", "which", "that", "there", "way2", "gonna", "hafta", "let-s",
	"got", "get2",  # auxiliares/partículas del inglés hablado
}

# ── MOLÉCULAS v2 a curar (idea fuente en inglés) ──
MOLECULES = {
	# gente
	"mom": "the female parent; the one who cares for the child",
	"dad": "the male parent", "daddy": "the male parent", "baby": "a very young person",
	"boy": "a young male person", "child": "a young person", "kid": "a young person",
	"man": "an adult male person", "friend": "a person one feels good with",
	# animales
	"cat": "a small animal that says meow, lives with people", "dog": "an animal that says bark, lives with people",
	# naturaleza
	"sun": "the bright thing above, far, that makes the day", "moon": "the bright thing in the dark sky",
	"rain": "water that falls from above", "river": "water that moves on the earth",
	"earth": "the ground; the big thing we stand on", "forest": "many trees together",
	"tree": "a tall living thing with parts that are green", "stone": "a hard thing that does not move by itself",
	"storm": "bad weather: much water and wind from the sky", "fire": "hot light that can burn and kill",
	"cave": "a dark place inside the earth",
	# cosas
	"ball": "a round thing children play with", "apple": "a round sweet fruit",
	"bed": "the thing one sleeps on", "book": "things with words inside to read",
	"box": "a thing with empty inside, for putting things", "bread": "the food made of flour",
	"car": "a thing with wheels that moves people", "food": "the things living beings eat",
	"house": "the place where people live", "milk": "the white drink from the mom animal",
	"toys": "the things children play with", "train": "the big thing on rails that moves people",
	"color": "what makes things look different: red, blue...",
	"home": "the place where the family lives",
	"make": "do something so a new thing exists",
	"need": "want something very much: without it, something bad happens",
	"put": "move a thing to a place and touch it there",
	"read": "see the words of a book and know them",
	"sit": "the body goes down and rests on something",
	"take": "touch a thing and move it to oneself",
	"try": "do something wanting it to work, not knowing if it can",
	"turn": "the body or a thing changes direction",
	"way": "the place where one moves from here to there",
	"run": "move the body very fast",
	# acciones (verbos sin primo)
	"eat": "put food in the body", "drink": "put water in the body",
	"play": "do things for feel-good, not for need", "sleep": "the body rests: eyes closed, not moving",
	"give": "make someone have something", "kiss": "touch with the mouth to feel-good",
	"help": "do something so someone can do it",
	# sonidos
	"bark": "the sound of the dog", "meow": "the sound of the cat",
	# cualidades/estados
	"blue": "the color of the sky", "red": "the color of blood/fire",
	"night": "the time when it is dark", "danger": None,  # ya existe en v2
	"hurt": "the body feels bad", "wound": "the place in the body where it hurts",
	"name": None,
}

# nombre propio — decisión de curación
NAMES = {"thomas": "TinyStories character — ¿molécula por entidad o marcador de nombre?"}

# flexiones que se normalizan antes del mapa (irregulares frecuentes)
NORMALIZE = {
	"drinks": "drink", "plays": "play", "sleeps": "sleep", "runs": "run",
	"love2": "love", "mommy": "mom", "mummy": "mom", "daddy": "dad",
	"hes": "he", "shes": "she", "theyre": "they", "weve": "we", "ive": "i",
	"whats": "what", "whos": "who", "had": "have", "has": "have",
	"havent": "have", "lets": "let", "shall": "will", "ill": "will",
	"im": "i'm", "dont": "don't",
}


def classify() -> dict:
	words = json.load(open(BASE / "configs" / "expanded_glyphs.json"))["words"]
	import numpy as np
	masks = json.load(open(BASE / "storage" / "checkpoints" / "bit003_glyph" / "stage_gate_masks.json"))
	gate = sorted(words[i] for i in masks["stage_allowed"][0] if words[i] != "<pad>")

	result = {"prime": {}, "molecule": {}, "frame": {}, "drop": [], "name": {}}
	unmapped = []
	for w in gate:
		wl = w.casefold()
		if wl in PRIMES_MAP:
			result["prime"][w] = PRIMES_MAP[wl]
		elif wl in FRAMES:
			result["frame"][w] = FRAMES[wl]
		elif wl in DROPS:
			result["drop"].append(w)
		elif wl in MOLECULES and MOLECULES[wl]:
			result["molecule"][w] = MOLECULES[wl]
		elif wl == "predator":  # ya existe en v2
			result["prime"][w] = "(v2 ya registrada)"
		elif wl in MOLECULES:  # ya existe en v2
			result["prime"][w] = "(v2 ya registrada)"
		elif wl in NAMES:
			result["name"][w] = NAMES[wl]
		elif wl in NORMALIZE and NORMALIZE[wl] != wl:
			result["frame"][w] = {"frame": f"morph→{NORMALIZE[wl]}", "k65p": f"→ clasificar como '{NORMALIZE[wl]}'",
				"nsm": "flexión/contracción: se normaliza antes del mapa"}
		else:
			unmapped.append(w)
	result["unmapped"] = unmapped
	result["gate_size"] = len(gate)
	return result


def main() -> None:
	r = classify()
	print(f"gate: {r['gate_size']} palabras")
	for cat in ("prime", "molecule", "frame", "name"):
		print(f"  {cat:9s}: {len(r[cat]):3d}")
	print(f"  drop     : {len(r['drop']):3d}")
	print(f"  unmapped : {len(r['unmapped'])} {r['unmapped']}")

	out = BASE / "configs" / "k65p_v2"
	out.mkdir(exist_ok=True)
	(out / "gate0_map.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
	print(f"→ {out / 'gate0_map.json'}")


if __name__ == "__main__":
	main()
