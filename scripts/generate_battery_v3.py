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
	for q, a in SEEN_QA:
		for f in trained_exam_forms(q, a, w2i, dictionary):
			store_hashes.add(seq_hash(f))
	curr_json = json.load(open(os.path.join(BASE, "configs", "school_curriculum_structured_en.json")))["curriculum"]
	for it in curr_json.get("preschool", []):
		store_hashes.add(seq_hash(tokenize(it["text"], w2i)))
	print(f"── universo entrenado: {len(store_hashes):,} hashes únicos (store + exámenes + currículo)")

	seen, rejected = [], []
	for q, a in SEEN_QA:
		mapped_a = dictionary.map_to_base_word(a)
		if w2i.get(mapped_a, 1) not in gate_idx:
			rejected.append((q, a, "respuesta fuera de gate/vocab")); continue
		forms = trained_exam_forms(q, a, w2i, dictionary)
		present = any(seq_hash(f) in store_hashes for f in forms)
		seen.append({"q": q, "a": a, "verified_present": present})
	for ctx, last in SEEN_COMPLETIONS:
		toks_q = tokenize(ctx, w2i)
		toks_a = tokenize(last, w2i)
		if len(toks_a) != 1 or w2i.get(last, 1) not in gate_idx:
			rejected.append((ctx, last, "respuesta fuera de gate/vocab")); continue
		present = seq_hash(toks_q + toks_a) in store_hashes
		seen.append({"q": ctx, "a": last, "verified_present": present})

	unseen, u_rej = [], []
	for q, a in UNSEEN_CANDIDATES:
		if len(unseen) >= 50:
			break
		toks_q = tokenize(q, w2i)
		toks_a = tokenize(a, w2i)
		if len(toks_q) < 2 or len(toks_a) != 1:
			u_rej.append((q, a, "tokenización")); continue
		if w2i.get(a, 1) not in gate_idx:
			u_rej.append((q, a, "respuesta fuera del gate de la etapa")); continue
		full_plain = toks_q + toks_a
		hashes = {seq_hash(full_plain)}
		for f in trained_exam_forms(q, a, w2i, dictionary):
			hashes.add(seq_hash(f))
		if hashes & store_hashes:
			u_rej.append((q, a, "¡la secuencia SÍ existe en el entrenamiento!")); continue
		unseen.append({"q": q, "a": a, "hash": seq_hash(full_plain)})

	seen_present = sum(1 for x in seen if x["verified_present"])
	os.makedirs(os.path.join(BASE, "configs", "battery_v3"), exist_ok=True)
	out = {
		"age": AGE, "stage_idx": STAGE_IDX, "protocol": "v3 DL-011",
		"seen_gate": seen, "unseen_cognition": unseen,
		"rejected_seen": rejected, "rejected_unseen": u_rej,
		"meta": {
			"seen_total": len(seen), "seen_verified_present": seen_present,
			"unseen_total": len(unseen),
			"store_seqs_hashed": len(store_hashes),
		},
	}
	path = os.path.join(BASE, "configs", "battery_v3", f"age{AGE}.json")
	json.dump(out, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
	print(f"\n── BATERÍA edad {AGE} ──")
	print(f"  vistas (gate):   {len(seen)} | verificadas presentes en entrenamiento: {seen_present}")
	print(f"  no vistas (cognición): {len(unseen)} | verificadas ausentes: {len(unseen)}")
	if rejected: print("  vistas rechazadas:", rejected[:5])
	if u_rej: print("  no-vistas rechazadas:", u_rej[:8])
	print(f"  → {path}")


if __name__ == "__main__":
	main()
