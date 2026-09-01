"""Generador del BANK de examen por etapa (DL-013) — el corpus de preguntas.

Un bank por segmento de etapa (0-1 … 7-8): Q→A sobre los hechos que ESA etapa
introduce, expandidos con cruces/marcos semánticamente válidos y verificados:
in-vocab, in-gate de la etapa, única-duplicación entre banks. Los banks son
material de entrenamiento (×exam_repeat_factor) y de donde se muestrean los
exámenes de hito (muestreo aleatorio ponderado por recencia, 10 formas).

Salida: configs/exam_banks/stage_{i}.json + resumen.
"""
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.generate_battery_v3 import seq_hash, trained_exam_forms  # noqa: E402
from src.bitnet.training.modules.tokenization import tokenize  # noqa: E402
from src.bitnet.vocab.dictionary_tool import SovereignDictionary  # noqa: E402

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DIALOGUE = {"hello", "how are you", "who are you", "what is your name", "where are you from", "what is your home", "do you like books"}

# ── Hechos base por etapa (los que la etapa INTRODUCE; AGE[age] + examen) ────
STAGE_FACTS = {
	0: [("hello", "hello"), ("water", "water"), ("fire", "bad"), ("mom", "dad"),
		("dog", "bark"), ("the ball is", "big"), ("the baby wants", "milk"),
		("night time to", "sleep"), ("the cat wants to", "eat"), ("i want", "milk"),
		("dad is", "here"), ("mom is", "here"), ("water is", "good"), ("time to", "sleep")],
	1: [("the dog runs", "much"), ("the bear eats", "honey"), ("the bird flies", "high"),
		("the lion has hair", "big"), ("the flowers drink", "water"), ("i count one two", "three"),
		("one plus one is", "two"), ("the cat sleeps", "happy"), ("i see the", "moon"),
		("what is your name", "baby"), ("i want", "bread"), ("if i touch the fire", "burns"),
		("where is dad", "here"), ("the sun is", "hot"), ("at night we", "sleep"),
		("the cat drinks", "milk"), ("one and one are", "two"), ("the dog is my", "friend"),
		("i play with", "you"), ("the sun is", "hot"), ("i eat the", "apple")],
	2: [("the fish lives in the", "water"), ("at night i see the", "moon"), ("the bird has", "wings"),
		("the snow is", "cold"), ("we read a", "book"), ("the sun shines", "much"),
		("what is your home", "cave"), ("do you like books", "yes"), ("where are you from", "cave"),
		("the heart pumps blood to the", "body"), ("the water of the river runs towards the", "sea")],
	3: [("what is five plus three? it is", "eight"), ("what is ten minus two? it is", "eight"),
		("what is three plus three? it is", "six"), ("what is six minus four? it is", "two"),
		("two plus two is", "four"), ("two plus three is", "five"), ("three plus one is", "four"),
		("what is six minus one? it is", "five"), ("what is eight minus two? it is", "six"),
		("i count two three", "four"), ("i count three four", "five"),
		("the bear drinks", "water"), ("the dog drinks", "water"), ("the fire is", "hot"),
		("the moon shines at", "night"), ("i sleep at", "night"), ("we sleep at", "night")],
	4: [("i count one two", "three"), ("one plus one is", "two"), ("the bear eats", "honey"),
		("the bird flies", "high"), ("the flowers drink", "water"), ("two plus two is", "four"),
		("the week has seven", "days"), ("the fish swims and the bird", "flies"),
		("four plus one is", "five"), ("we sleep in a", "bed"), ("the earth moves around the", "sun"),
		("maps show rivers and", "countries"), ("rain falls from the", "sky"),
		("what is your name", "bit"), ("where are you from", "cave")],
	5: [("what is three plus three? it is", "six"), ("what is six minus four? it is", "two"),
		("the water of the river runs towards the", "sea"), ("the trees give oxygen and", "shade"),
		("the heart pumps blood to the", "body"), ("what is four plus five? it is", "nine"),
		("what is ten minus five? it is", "five"), ("the moon shines at", "night"),
		("the roots of the tree are under the", "ground"), ("we write words on a", "page")],
	6: [("what is your name", "bit"), ("where are you from", "cave"), ("what is seven plus three? it is", "ten"),
		("the heart moves the", "blood"), ("a story lives inside a", "book"), ("the sun gives light and", "heat"),
		("winter is the season of", "snow"), ("the earth moves around the", "sun"),
		("maps show rivers and", "countries"), ("rain falls from the", "sky")],
	7: [("if x plus two is five then x is", "three"), ("every cause produces an", "effect"),
		("the long halls of the library are full of", "mirrors"), ("a perfect memory keeps the shape of each", "cloud"),
		("one point can contain the whole", "universe"), ("what is your home", "cave"),
		("do you like books", "yes"), ("written words overcome the passage of", "time"),
		("what is nine minus six? it is", "three"), ("the poet sings to the moon in the cold", "night")],
}

# ── Expansiones programáticas (cruces semánticamente válidos por nivel) ─────
ANIMALS_WATER = ["bear", "dog", "lion", "horse", "cow", "goat", "duck", "mouse", "pig", "bird", "wolf"]
ANIMALS_APPLE = ["horse", "cow", "goat", "pig", "mouse", "monkey", "bear", "dog"]
SAY_SOUND = [("dog", "bark"), ("cat", "meow"), ("duck", "quack"), ("cow", "moo"), ("owl", "hoot"),
	("pig", "oink"), ("wolf", "howl"), ("lion", "roar"), ("snake", "hiss"), ("monkey", "chatter")]
PLACES = [("house", "dog"), ("house", "cat"), ("house", "baby"), ("house", "mom"), ("house", "dad"),
	("forest", "bear"), ("forest", "wolf"), ("forest", "monkey"), ("river", "fish"), ("river", "duck"),
	("road", "dog"), ("bed", "baby"), ("bed", "cat"), ("table", "cup"), ("table", "food")]


# ── Tiers semánticos (mundo curricular ampliado — 99 moléculas del k65p) ──
TIER_ANIMALS = ["dog", "cat", "bird", "horse", "fish", "wolf", "bear", "mouse",
	"rabbit", "fox", "snake", "cow", "pig", "elephant", "lion", "tiger", "monkey",
	"owl", "duck", "eagle", "whale", "shark", "goat", "deer"]
TIER_OBJECTS = ["chair", "table", "bed", "cup", "knife", "door", "house",
	"bridge", "road", "roof", "hammer", "basket", "ball", "book", "apple", "milk"]
TIER_PLACES = ["house", "forest", "river", "road", "water", "tree", "cave", "bed"]
TIER_ACTIONS = ["run", "jump", "swim", "fly", "climb", "hide", "play", "sleep", "eat", "drink"]


# ── Partición del mundo por etapa + frames acumulativos ─────────────────────
# Cada etapa INTRODUCE animales/objetos/lugares nuevos (partición) y frames
# nuevos; los frames nuevos cruzan con TODOS los sujetos acumulados y los
# sujetos nuevos con TODOS los frames acumulados — crecimiento combinatorio
# sin duplicados (cada hecho vive en el bank de su etapa).

ANIMALS_PER_STAGE = 3
NEW_ANIMALS = {k: TIER_ANIMALS[k*3:(k+1)*3] for k in range(8)}
NEW_OBJECTS = {k: TIER_OBJECTS[k*2:(k+1)*2] for k in range(8)}
NEW_PLACES = {k: TIER_PLACES[k*1:(k+1)*1] for k in range(8)}

SAYS = {a: s for a, s in SAY_SOUND}


def frames_available(stage: int) -> list:
	"""Frames (sujeto → (q, a) o None) introducidos en o antes de `stage`."""
	F = []
	if stage >= 0:
		F.append(lambda s: (f"the {s} drinks", "water"))
	if stage >= 1:
		F.append(lambda s: (f"the {s} eats", "food"))
		F.append(lambda s: (f"the {s} sleeps", "now"))
	if stage >= 2:
		F.append(lambda s: (f"the {s} is here", "now"))
		F.append(lambda s: (f"what does the {s} say", SAYS.get(s)) if s in SAYS else None)
	if stage >= 3:
		F.append(lambda s: (f"the {s} eats the", "apple"))
		F.append(lambda s: (f"the {s} sleeps at", "night"))
	if stage >= 4:
		F.append(lambda s: (f"the {s} can", "jump"))
		F.append(lambda s: (f"the {s} is in the", "house"))
	if stage >= 5:
		F.append(lambda s: (f"the {s} plays", "ball"))
		F.append(lambda s: (f"the {s} drinks from the", "river"))
	if stage >= 6:
		F.append(lambda s: (f"the {s} climbs the", "tree"))
		F.append(lambda s: (f"the {s} hides in the", "forest"))
	if stage >= 7:
		F.append(lambda s: (f"the {s} hunts the", "mouse") if s in ("wolf", "fox", "snake", "owl", "lion", "tiger", "bear") else None)
	return [f for f in F if f is not None]


def bank_for_stage(stage: int, gate_idx: set, gate_words: set, used_hashes: set, w2i: dict, dictionary) -> tuple:
	"""Hechos base de la etapa + cruces combinatorios (nuevos sujetos × frames
	acumulados, sujetos acumulados × frames nuevos). Todo verificado: in-vocab,
	in-gate de etapa, no duplicado. Cap: 200.

	`gate_idx` son los índices permitidos por la máscara de la etapa: una
	respuesta fuera del gate es inaprendible (la pérdida excluye targets
	vetados) e inalcanzable en examen (el argmax enmascarado no puede
	producirla) — se rechaza. Auditoría 1-sep: este check nació muerto
	(gate_idx local a None) y dejó pasar burns/howl/hiss."""
	items, rej = [], []

	def try_add(q: str, a: str):
		mapped_a = dictionary.map_to_base_word(a)
		a_idx = w2i.get(mapped_a, 1)
		if a_idx not in gate_idx:
			rej.append((q, a, "fuera de gate")); return False
		toks_q = tokenize(q, w2i)
		toks_a = [a_idx]
		if len(toks_q) < 2:
			rej.append((q, a, "corta")); return False
		hs = {seq_hash(toks_q + toks_a)}
		for f in trained_exam_forms(q, a, w2i, dictionary):
			hs.add(seq_hash(f))
		if hs & used_hashes:
			rej.append((q, a, "duplicado")); return False
		used_hashes.update(hs)
		items.append({"q": q, "a": a})
		return True

	# sujetos: partición nueva + acumulados
	new_animals = [a for a in NEW_ANIMALS.get(stage, []) if a in gate_words]
	acc_animals = [a for k in range(stage + 1) for a in NEW_ANIMALS.get(k, []) if a in gate_words]
	people = [w for w in ("i", "you", "baby", "mom", "dad") if w in gate_words]
	F = frames_available(stage)
	prev_F = frames_available(max(0, stage - 1)) if stage > 0 else []
	new_frames = F[len(prev_F):]

	# 1) hechos base de la etapa
	for q, a in STAGE_FACTS.get(stage, []):
		try_add(q, a)
	# 2) sujetos NUEVOS × frames acumulados
	for a in new_animals:
		for f in F:
			r = f(a)
			if r: try_add(*r)
	for s in people:
		for f in F:
			r = f(s)
			if r: try_add(*r)
	# 3) sujetos acumulados × frames NUEVOS
	for f in new_frames:
		for a in acc_animals:
			r = f(a)
			if r: try_add(*r)
		for s in people:
			r = f(s)
			if r: try_add(*r)

	return items[:200], rej


def main() -> None:
	words = json.load(open(os.path.join(BASE, "configs", "expanded_glyphs.json")))["words"]
	w2i = {w: i for i, w in enumerate(words)}
	dictionary = SovereignDictionary(os.path.join(BASE, "configs", "expanded_glyphs.json"))
	masks = json.load(open(os.path.join(BASE, "storage", "checkpoints", "bit003_glyph_v4_x1", "stage_gate_masks.json")))

	used_hashes: set = set()
	banks: dict = {}
	for stage in range(8):
		gate_idx = set(masks["stage_allowed"][stage])
		gate_words = {words[i] for i in gate_idx}
		items, rej = bank_for_stage(stage, gate_idx, gate_words, used_hashes, w2i, dictionary)
		reasons = Counter(r[-1] for r in rej)
		banks[stage] = items
		print(f"  bank etapa {stage}: {len(items)} preguntas | rechazos: {dict(reasons)}")

	os.makedirs(os.path.join(BASE, "configs", "exam_banks"), exist_ok=True)
	total = sum(len(v) for v in banks.values())
	manifest = {"protocol": "DL-013", "total": total, "per_stage": {k: len(v) for k, v in banks.items()}}
	for stage, items in banks.items():
		path = os.path.join(BASE, "configs", "exam_banks", f"stage_{stage}.json")
		json.dump({"stage_idx": stage, "questions": items}, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
	json.dump(manifest, open(os.path.join(BASE, "configs", "exam_banks", "manifest.json"), "w", encoding="utf-8"), indent=1)
	print(f"\n── BANK total: {total} preguntas en 8 etapas → configs/exam_banks/")


if __name__ == "__main__":
	main()

