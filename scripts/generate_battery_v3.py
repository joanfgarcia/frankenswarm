"""Generador de la batería 80/50 v3 (DL-011) — edad 4 (hitos 2-4 años entrenados).

VISTAS: pares Q→A con superficie ENTRENADA (examen ×10 + AGE_QUESTIONS v2 +
completados del currículo preescolar) — verificados PRESENTES en el store.
NO VISTAS: recomposiciones nuevas de los mismos hechos (superficies que el
entrenamiento nunca vio) — verificadas AUSENTES del store completo (42.3M
secuencias), in-vocab y dentro del gateo de la etapa del hito.

Salida: configs/battery_v3/age4.json (+ cache de hashes en storage/datasets/).
"""
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.bitnet.training.modules.tokenization import generate_question_variations, tokenize, words_of  # noqa: E402

from src.bitnet.vocab.dictionary_tool import SovereignDictionary  # noqa: E402

DIALOGUE_TRIGGERS = {"hello", "how are you", "who are you", "what is your name", "where are you from", "what is your home", "do you like books"}

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AGE = 4
STAGE_IDX = 3  # etapa 3-4 (examen de 4 años)
DIALOGUE = {"hello", "how are you", "who are you", "what is your name", "where are you from", "what is your home", "do you like books"}


def seq_hash(tokens) -> int:
	return int.from_bytes(hashlib.blake2b(np.asarray(tokens, dtype=np.int32).tobytes(), digest_size=8).digest(), "big")


def build_store_hash_set(store_path: str, cache_path: str) -> set:
	"""Set de hashes (blake2b-8B) de TODAS las secuencias del store. Cacheable."""
	if os.path.exists(cache_path):
		return set(np.load(cache_path).tolist())
	d = np.load(store_path)
	flat, offsets = d["flat"], d["offsets"]
	print(f"── hasheando {len(offsets)-1:,} secuencias del store (una vez, ~4 min)…")
	h = hashlib.blake2b(digest_size=8)
	out = np.empty(len(offsets) - 1, dtype=np.int64)
	for i in range(len(offsets) - 1):
		a, b = int(offsets[i]), int(offsets[i + 1])
		h = hashlib.blake2b(flat[a:b].tobytes(), digest_size=8)
		out[i] = int.from_bytes(h.digest(), "big", signed=True)
	np.save(cache_path, out)
	return set(out.tolist())


# ── Hechos entrenados (universo edades 2-4) ──────────────────────────────────
# Pares Q→A entrenados: examen preschool (13) + AGE_QUESTIONS v2 edades 2-4 (30)
EXAM_QA = [
	("the cat sleeps", "happy"), ("the dog runs", "much"), ("i see the", "moon"),
	("the bear eats", "honey"), ("i have two apples and one cat plus one are", "two"),
	("i count one two", "three"), ("one plus one is", "two"), ("the bird flies", "high"),
	("the lion has hair", "big"), ("the flowers drink", "water"), ("hello", "hello"),
	("how are you", "fine"), ("who are you", "boy"),
]
AGE_QA = [
	("hello", "hello"), ("cat", "meow"), ("water", "water"), ("fire", "bad"), ("mom", "dad"),
	("dog", "bark"), ("the baby wants", "milk"), ("night time to", "sleep"),
	("the cat wants to", "eat"), ("the ball is", "big"),
	("what is your name", "baby"), ("the dog runs", "much"), ("i want", "bread"),
	("if i touch the fire", "burns"), ("where is dad", "here"), ("the sun is", "hot"),
	("at night we", "sleep"), ("the cat drinks", "milk"), ("one and one are", "two"),
	("the dog is my", "friend"),
	("hello", "hello"), ("how are you", "fine"), ("who are you", "boy"),
	("the sun shines", "much"), ("if i touch the fire", "hurt"),
	("the fish lives in the", "water"), ("at night i see the", "moon"), ("the bird has", "wings"),
	("the snow is", "cold"), ("we read a", "book"),
]
# Completados del currículo preescolar (statement → contexto, última palabra)
CURRICULUM_COMPLETIONS = [
	("dad is", "here"), ("mom is", "here"), ("i want", "milk"), ("water is", "good"),
	("time to", "sleep"), ("the red", "ball"), ("the dog", "runs"), ("the cat drinks", "milk"),
	("i play with", "you"), ("the sun is", "hot"), ("i eat the", "apple"),
	("the baby wants", "mom"), ("we go to", "bed"), ("i love you", "mom"),
	("i want to", "play"), ("we eat bread and drink", "water"), ("the baby sleeps in the", "night"),
	("come here and", "play"), ("the cat and the dog", "play"),
	("i give you food and you give me a", "kiss"),
]

SEEN_QA = EXAM_QA + [(q, a) for q, a in AGE_QA if (q, a) not in EXAM_QA]
SEEN_COMPLETIONS = CURRICULUM_COMPLETIONS

# ── NO VISTAS: recomposiciones nuevas de hechos conocidos ────────────────────
# Superficies que generate_question_variations NO genera y el entrenamiento
# nunca vio (verificación de ausencia contra el store completo abajo).
UNSEEN_CANDIDATES = [
	# aritmética: pares nuevos con superficies entrenadas
	("two plus two is", "four"), ("three plus one is", "four"), ("two plus three is", "five"),
	("five plus one is", "six"), ("what is two plus two? it is", "four"),
	("what is three plus one? it is", "four"), ("what is six minus one? it is", "five"),
	("what is eight minus two? it is", "six"), ("what is five minus one? it is", "four"),
	("i count two three", "four"), ("i count three four", "five"), ("i count four five", "six"),
	# cruces sujeto×verbo/predicado nuevos (semánticamente válidos)
	("the bear drinks", "water"), ("the dog drinks", "water"), ("the lion drinks", "water"),
	("the baby drinks", "milk"), ("the bird drinks", "water"), ("the mom drinks", "water"),
	("the fire is", "hot"), ("the water is", "cold"), ("the baby is", "big"),
	("the dog has", "hair"), ("the lion eats", "honey"), ("the baby sees the", "moon"),
	("the dog wants", "water"), ("the baby eats the", "apple"), ("the dad drinks", "water"),
	("the cat sees the", "dog"), ("the bird sees the", "moon"), ("the mom sees the", "baby"),
	# marcos interrogativos/afirmativos nuevos para hechos entrenados
	("the moon shines at", "night"), ("i sleep at", "night"), ("the bird swims in the", "water"),
	("the fish is in the", "water"), ("the baby sleeps at", "night"), ("we sleep at", "night"),
	("the bird is in the", "moon"), ("i see the", "dog"), ("i see the", "baby"),
	("the dog sees the", "cat"), ("the flowers are in the", "water"), ("the baby reads a", "book"),
	("the dog reads a", "book"), ("i read the", "book"), ("the baby plays with the", "ball"),
	("the dad plays with the", "ball"), ("the mom reads a", "book"), ("the baby loves the", "dog"),
	("the cat plays with the", "ball"), ("i love the", "dog"), ("the bird loves the", "moon"),
	("the dog loves the", "ball"), ("the baby loves the", "mom"), ("i see the", "fire"),
	("the dad eats the", "apple"), ("the mom eats the", "apple"), ("the baby drinks", "water"),
	("the dog is in the", "water"), ("the bird eats the", "apple"), ("the lion is in the", "water"),
]


def trained_exam_forms(q: str, a: str, w2i: dict, dictionary) -> list:
	"""Réplica exacta de compile_exam_sequences_for_age: variaciones oficiales +
	mapeo a palabra base + wrapper de diálogo. Devuelve las secuencias entrenadas."""
	import re
	forms = []
	for var in generate_question_variations(q):
		q_words = re.findall(r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_]+", var.lower())
		mapped_q = [dictionary.map_to_base_word(w) for w in q_words]
		mapped_a = dictionary.map_to_base_word(a)
		q_tokens = [w2i.get(w, 1) for w in mapped_q]
		a_token = w2i.get(mapped_a, 1)
		if q.strip().lower() in DIALOGUE_TRIGGERS:
			tokens = [w2i.get("you", 1)] + q_tokens + [w2i.get("me", 1)] + [a_token]
		else:
			tokens = q_tokens + [a_token]
		forms.append(tokens)
	return forms


def main() -> None:
	import argparse
	ap = argparse.ArgumentParser()
	ap.add_argument("--set_version", type=int, default=2, help="Versión del set congelado (v1 = pre-banks, intocable; v2+ = con banks en el universo)")
	ap.add_argument("--age", type=int, default=4, choices=[2, 3, 4], help="Edad del hito (2-4: bucket preschool)")
	ap.add_argument("--stage_idx", type=int, default=None, help="Etapa del gateo (default: age-1)")
	args = ap.parse_args()
	global AGE, STAGE_IDX
	AGE = args.age
	STAGE_IDX = args.stage_idx if args.stage_idx is not None else AGE - 1

	words = json.load(open(os.path.join(BASE, "configs", "expanded_glyphs.json")))["words"]
	w2i = {w: i for i, w in enumerate(words)}
	dictionary = SovereignDictionary(os.path.join(BASE, "configs", "expanded_glyphs.json"))
	masks = json.load(open(os.path.join(BASE, "storage", "checkpoints", "bit003_glyph", "stage_gate_masks.json")))
	gate_idx = set(masks["stage_allowed"][STAGE_IDX])

	store_path = os.path.join(BASE, "storage", "datasets", "tokenized_store.npz")
	cache_path = os.path.join(BASE, "storage", "datasets", "store_hash_set.npy")
	store_hashes = build_store_hash_set(store_path, cache_path)
	# El universo entrenado = store + exámenes (con variaciones/mapeo) + currículo
	# preescolar. Los exámenes/currículo NO viven en el store: se inyectan en
	# compile_data_for_stage (×exam_repeat_factor).
	age_slice = {2: 10, 3: 20, 4: 30}[AGE]  # AGE_QUESTIONS: 10 por edad, 2→3→4
	seen_source = EXAM_QA + AGE_QA[:age_slice]
	for q, a in seen_source:
		for f in trained_exam_forms(q, a, w2i, dictionary):
			store_hashes.add(seq_hash(f))
	curr_json = json.load(open(os.path.join(BASE, "configs", "school_curriculum_structured_en.json")))["curriculum"]
	for it in curr_json.get("preschool", []):
		store_hashes.add(seq_hash(tokenize(it["text"], w2i)))
	# Los BANKS de examen (DL-013) son material de entrenamiento en cuanto se
	# cableen: sus hechos NO pueden ser "no vistos" (auditoría 1-sep: 28/195
	# del set v1 colisionaban con los banks). Todas sus formas entran al
	# universo entrenado.
	n_bank = 0
	bank_dir = os.path.join(BASE, "configs", "exam_banks")
	if os.path.isdir(bank_dir):
		for fn in sorted(os.listdir(bank_dir)):
			if not fn.startswith("stage_"):
				continue
			for it in json.load(open(os.path.join(bank_dir, fn)))["questions"]:
				store_hashes.add(seq_hash(tokenize(it["q"], w2i) + [w2i.get(dictionary.map_to_base_word(it["a"]), 1)]))
				for f in trained_exam_forms(it["q"], it["a"], w2i, dictionary):
					store_hashes.add(seq_hash(f))
				n_bank += 1
	print(f"── universo entrenado: {len(store_hashes):,} hashes únicos (store + exámenes + currículo + {n_bank} preguntas de bank)")

	seen, rejected = [], []
	seen_qs: set = set()
	seen_pairs: set = set()
	for q, a in seen_source:
		mapped_a = dictionary.map_to_base_word(a)
		if w2i.get(mapped_a, 1) not in gate_idx:
			rejected.append((q, a, "respuesta fuera de gate/vocab")); continue
		if (q, a) in seen_pairs:
			rejected.append((q, a, "duplicado exacto")); continue
		if q in seen_qs:
			# misma pregunta con OTRA respuesta: bajo argmax solo una puede
			# acertar — la segunda garantiza un fallo del instrumento, no del
			# modelo (auditoría 1-sep: 'if i touch the fire' → burns Y hurt).
			rejected.append((q, a, "pregunta duplicada con otra respuesta")); continue
		forms = trained_exam_forms(q, a, w2i, dictionary)
		present = any(seq_hash(f) in store_hashes for f in forms)
		if not present:
			rejected.append((q, a, "no verificada presente en el universo entrenado")); continue
		seen_pairs.add((q, a)); seen_qs.add(q)
		seen.append({"q": q, "a": a, "verified_present": present, "n_forms": len(forms)})
	for ctx, last in SEEN_COMPLETIONS:
		toks_q = tokenize(ctx, w2i)
		toks_a = tokenize(last, w2i)
		if len(toks_a) != 1 or w2i.get(last, 1) not in gate_idx:
			rejected.append((ctx, last, "respuesta fuera de gate/vocab")); continue
		if (ctx, last) in seen_pairs:
			rejected.append((ctx, last, "duplicado exacto")); continue
		if ctx in seen_qs:
			rejected.append((ctx, last, "pregunta duplicada con otra respuesta")); continue
		present = seq_hash(toks_q + toks_a) in store_hashes
		if not present:
			rejected.append((ctx, last, "no verificada presente en el universo entrenado")); continue
		seen_pairs.add((ctx, last)); seen_qs.add(ctx)
		seen.append({"q": ctx, "a": last, "verified_present": present})
	seen = seen[:80]

	STRATA = [
		("S1_cruce", UNSEEN_CANDIDATES + [
			("the horse drinks", "water"), ("the cow drinks", "water"), ("the goat drinks", "water"),
			("the duck drinks", "water"), ("the mouse drinks", "water"), ("the pig drinks", "water"),
			("the horse eats", "apple"), ("the cow eats", "apple"), ("the goat eats", "apple"),
			("the mouse eats", "apple"), ("the duck eats", "apple"),
			("the horse sees the", "moon"), ("the cow sees the", "moon"), ("the pig sees the", "moon"),
			("the horse plays", "ball"), ("the goat plays", "ball"), ("the mouse plays", "ball"),
			("the horse sleeps at", "night"), ("the cow sleeps at", "night"),
			("the baby sees the", "dog"), ("the dad sees the", "cat"), ("the mom sees the", "bird"),
			("the dad drinks", "milk"), ("the baby loves the", "ball"), ("the dad loves the", "book"),
		]),
		("S2_interrogativo", [
			("what does the baby drink", "milk"), ("what does the bear drink", "water"),
			("what does the dog say", "bark"), ("what does the cat say", "meow"),
			("what does the duck say", "quack"), ("what does the cow say", "moo"),
			("what does the owl say", "hoot"), ("where is the fish", "water"),
			("where does the bird live", "tree"), ("what does the baby eat", "apple"),
			("what does the bear eat", "honey"), ("what does the dog eat", "apple"),
			("what does the mom drink", "water"), ("what does the cat drink", "milk"),
			("what does the bird do", "fly"), ("what does the fish do", "swim"),
			("what does the baby want", "milk"), ("what does the dog want", "water"),
			("what does the baby play", "ball"), ("what does the cat play", "ball"),
			("what is the snow", "cold"), ("what is the fire", "hot"),
			("what is the ball", "big"), ("who is my friend", "dog"),
			("who drinks water", "baby"), ("who wants milk", "baby"),
			("who reads a book", "baby"), ("who plays the ball", "baby"),
			("what is in the night", "moon"), ("what is hot", "fire"),
			("what is cold", "snow"), ("what flies", "bird"),
			("what swims", "fish"), ("what barks", "dog"),
			("what drinks milk", "cat"), ("what gives a kiss", "mom"),
			("what gives food", "dad"), ("what sleeps at night", "baby"),
			("what runs", "dog"), ("what eats honey", "bear"),
		]),
		("S3_inversion_riddle", [
			("it says bark", "dog"), ("it says meow", "cat"), ("it says quack", "duck"),
			("it says moo", "cow"), ("it says hoot", "owl"),
			("it has wings", "bird"), ("it has hair", "lion"),
			("it drinks milk and says meow", "cat"), ("it barks and plays", "dog"),
			("it swims and says quack", "duck"), ("it says moo and drinks", "cow"),
			("it flies at night", "owl"), ("it flies and sings", "bird"),
			("it sleeps in the house", "dog"), ("it eats honey", "bear"),
			("it runs and barks", "dog"), ("it runs and meows", "cat"),
			("it swims in the river", "fish"), ("it lives in the water", "duck"),
			("the baby of the mom", "baby"), ("the friend of the baby", "dog"),
			("it gives milk", "cow"), ("it gives a kiss", "mom"),
			("it gives food", "dad"), ("it says oink", "pig"),
			# ("it says caw", "owl") y ("it says croak", "duck") RETIRADAS
			# (auditoría 1-sep): caw es del cuervo y croak de la rana — el
			# gold era factualmente incorrecto.
			("it climbs", "monkey"), ("it howls", "wolf"),
			("it hides and runs", "mouse"), ("it is big and grey", "elephant"),
			("it says roar", "lion"), ("it says hiss", "snake"),
			("it hunts at night", "owl"), ("it eats apples", "horse"),
			("it plays with the baby", "dog"), ("it sleeps all day", "cat"),
			("it flies and says hoot", "owl"), ("it says chatter", "monkey"),
		]),
		("S4_aritmetica", [
			("two plus two is", "four"), ("three plus one is", "four"), ("two plus three is", "five"),
			("five plus one is", "six"), ("three plus two is", "five"), ("four plus one is", "five"),
			("one plus two is", "three"), ("two plus one is", "three"), ("three plus three is", "six"),
			("four plus two is", "six"), ("what is two plus two? it is", "four"),
			("what is three plus one? it is", "four"), ("what is two plus three? it is", "five"),
			("what is four plus one? it is", "five"), ("what is six minus one? it is", "five"),
			("what is eight minus two? it is", "six"), ("what is five minus one? it is", "four"),
			("what is four minus one? it is", "three"), ("what is seven minus two? it is", "five"),
			("what is six minus two? it is", "four"), ("what is nine minus four? it is", "five"),
			("i count two three", "four"), ("i count three four", "five"), ("i count four five", "six"),
			("i count five six", "seven"), ("i count one two three", "four"),
			("i have two apples and i eat one i have", "one"),
			("two and two are", "four"), ("three and one are", "four"),
			("two and one are", "three"), ("one and two are", "three"),
			("what is one plus three? it is", "four"), ("what is two plus one? it is", "three"),
			("what is three plus two? it is", "five"), ("what is five plus one? it is", "six"),
			("what is nine minus five? it is", "four"), ("what is seven minus three? it is", "four"),
			("what is eight minus three? it is", "five"), ("what is six minus three? it is", "three"),
			("i count three and then four", "five"),
		]),
		("S5_predicado_nuevo", [
			("the baby sleeps in the", "house"), ("the dog sleeps in the", "house"),
			("the cat is in the", "house"), ("the bird is in the", "forest"),
			("the bear is in the", "forest"), ("the fish is in the", "river"),
			("the water is in the", "river"), ("the dog drinks from the", "river"),
			("the baby drinks from the", "cup"), ("the water is in the", "cup"),
			("the cat is on the", "table"), ("the food is on the", "table"),
			("the ball is in the", "house"), ("the book is on the", "table"),
			("the dog is at the", "door"), ("the baby is in the", "bed"),
			("the cat is in the", "bed"), ("the bird is in the", "forest"),
			("the mom is in the", "house"), ("the dad is in the", "house"),
			("the bear drinks from the", "river"), ("the horse drinks from the", "river"),
			("the food is in the", "cup"), ("the milk is in the", "cup"),
			("the dog eats in the", "house"), ("the baby plays in the", "house"),
			("the bird sings in the", "forest"), ("the moon is in the", "night"),
			("the sun is in the", "day"), ("the rain is in the", "night"),
			("the dog is in the", "road"), ("the cat walks on the", "road"),
			("the baby sleeps with the", "mom"), ("the baby plays with the", "dad"),
			("the dog plays with the", "cat"), ("the bird plays with the", "bird"),
			("the apple is for the", "baby"), ("the milk is for the", "baby"),
			("the book is for the", "baby"), ("the ball is for the", "dog"),
		]),
	]
	unseen, u_rej = [], []
	unseen_qs: set = set()
	quota = 200 // len(STRATA)
	rng = np.random.default_rng(42)  # semilla fija: el set congelado es reproducible
	for stratum, candidates in STRATA:
		cands = list(candidates)
		rng.shuffle(cands)
		count = 0
		for q, a in cands:
			if count >= quota:
				break
			toks_q = tokenize(q, w2i)
			toks_a = tokenize(a, w2i)
			if len(toks_q) < 2 or len(toks_a) != 1:
				u_rej.append((stratum, q, a, "tokenización")); continue
			if w2i.get(a, 1) not in gate_idx:
				u_rej.append((stratum, q, a, "respuesta fuera del gate de la etapa")); continue
			if q in unseen_qs:
				# misma pregunta con otra respuesta: el argmax solo puede dar
				# una — la 2ª entra como fallo estructural, no cognitivo.
				u_rej.append((stratum, q, a, "pregunta duplicada")); continue
			hashes = {seq_hash(toks_q + toks_a)}
			for f in trained_exam_forms(q, a, w2i, dictionary):
				hashes.add(seq_hash(f))
			if hashes & store_hashes:
				u_rej.append((stratum, q, a, "¡existe en el entrenamiento!")); continue
			unseen_qs.add(q)
			unseen.append({"q": q, "a": a, "stratum": stratum, "hash": seq_hash(toks_q + toks_a)})
			count += 1

	seen_present = sum(1 for x in seen if x["verified_present"])
	from collections import Counter
	strata_counts = Counter(x["stratum"] for x in unseen)
	os.makedirs(os.path.join(BASE, "configs", "battery_v3"), exist_ok=True)
	# Set congelado: id = hash de la lista ordenada de hashes → el mismo fichero
	# evalúa a TODOS los modelos (comparativa con preguntas idénticas).
	set_id = hashlib.sha256(
		",".join(str(x["hash"]) for x in sorted(unseen, key=lambda x: x["hash"])).encode()
	).hexdigest()[:12]
	out = {
		"age": AGE, "stage_idx": STAGE_IDX, "protocol": "v3 DL-011",
		"set_version": args.set_version,
		"set_id": f"age{AGE}_v{args.set_version}_{set_id}", "sampling_seed": 42,
		"banks_in_universe": n_bank,
		"seen_gate": seen, "unseen_cognition": unseen,
		"rejected_seen": rejected, "rejected_unseen": u_rej,
		"meta": {
			"seen_total": len(seen), "seen_verified_present": seen_present,
			"unseen_total": len(unseen),
			"store_seqs_hashed": len(store_hashes),
		},
	}
	# v1 (pre-banks) queda congelado en age{AGE}.json — la confrontación v4 en
	# curso se mide con él; v2+ (banks en el universo) usa fichero versionado.
	suffix = "" if args.set_version <= 1 else f"_v{args.set_version}"
	path = os.path.join(BASE, "configs", "battery_v3", f"age{AGE}{suffix}.json")
	json.dump(out, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
	print(f"\n── BATERÍA edad {AGE} ──")
	print(f"  vistas (gate):   {len(seen)} | verificadas presentes: {seen_present}")
	print(f"  no vistas (cognición): {len(unseen)} | por estrato: {dict(strata_counts)}")
	if rejected: print("  vistas rechazadas:", rejected[:5])
	if u_rej: print("  no-vistas rechazadas:", u_rej[:8])
	print(f"  → {path}")


if __name__ == "__main__":
	main()
