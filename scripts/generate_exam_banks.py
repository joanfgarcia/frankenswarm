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


def expansions_for_stage(stage: int) -> list:
	out = []
	if stage >= 1:
		for a in ANIMALS_WATER[:6 + stage * 2]:
			out.append((f"the {a} drinks", "water"))
	for a, s in SAY_SOUND[:4 + stage * 2]:
		out.append((f"what does the {a} say", s))
	for place, who in PLACES:
		if stage >= 2 or who in ("dog", "baby", "cat"):
			out.append((f"the {who} is in the", place))
	if stage >= 3:
		for a in ANIMALS_APPLE[:6 + (stage - 3) * 2]:
			out.append((f"the {a} eats the", "apple"))
		out.append(("the fire is", "hot")); out.append(("the water is", "cold"))
		out.append(("the moon shines at", "night")); out.append(("i sleep at", "night"))
	if stage >= 4:
		out.append(("the dog reads a", "book")); out.append(("the baby reads a", "book"))
		out.append(("the cow gives", "milk")); out.append(("the bird gives", "light"))
	return out


def main() -> None:
	words = json.load(open(os.path.join(BASE, "configs", "expanded_glyphs.json")))["words"]
	w2i = {w: i for i, w in enumerate(words)}
	dictionary = SovereignDictionary(os.path.join(BASE, "configs", "expanded_glyphs.json"))
	masks = json.load(open(os.path.join(BASE, "storage", "checkpoints", "bit003_glyph_v4_x1", "stage_gate_masks.json")))

	used_hashes: set = set()
	banks: dict = {}
	stats = {}
	for stage in range(8):
		gate_idx = set(masks["stage_allowed"][stage])
		candidates = STAGE_FACTS.get(stage, []) + expansions_for_stage(stage)
		items, rej = [], []
		for q, a in candidates:
			mapped_a = dictionary.map_to_base_word(a)
			toks_a = [w2i.get(mapped_a, 1)]
			if w2i.get(mapped_a, 1) not in gate_idx:
				rej.append((q, a, "fuera de gate")); continue
			forms = trained_exam_forms(q, mapped_a, w2i, dictionary)
			hs = {seq_hash(f) for f in forms}
			hs.add(seq_hash(tokenize(q, w2i) + toks_a))
			if hs & used_hashes:
				rej.append((q, a, "duplicado entre banks")); continue
			used_hashes |= hs
			items.append({"q": q, "a": a})
		banks[stage] = items
		stats[stage] = (len(items), len(rej))
		print(f"  bank etapa {stage} ({['0-1','1-2','2-3','3-4','5','6','7','8'][stage]}): {len(items)} preguntas ({len(rej)} rechazadas)")

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
