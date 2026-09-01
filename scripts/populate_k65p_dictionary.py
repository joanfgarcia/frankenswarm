"""Población idempotente del diccionario K-65P↔humano (DL-014, F4 del plan DL-015).

Puebla: 65 primos + 28 canónicos (glifos del operador) + 99 moléculas k65p
(status=molecule, glifo provisional Ridge hasta la descomposición NSM) +
el vocabulario restante como pending + las phrases de compresión N→1.

Aceptación (DL-015 F4): stats() cuadra con la doctrina (65/28/99) y el test
de ida-vuelta RFC-002 §5 pasa DESDE LA BD (glifo → features → glifo).

Uso: .venv/bin/python scripts/populate_k65p_dictionary.py [--db ruta]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.bitnet.vocab.k65p_dictionary import K65PDictionary  # noqa: E402
from src.bitnet.vocab.glyph_vocabulary import SEMANTIC_PRIMES, VOCABULARY  # noqa: E402
from src.bitnet.vocab.nsm_explication import explication_to_glyph, glyph_to_features  # noqa: E402

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Las moléculas k65p (los tiers del mundo semántico — train_sovereign_school_k65p)
TIER_BASE = ["fire", "water", "sun", "night", "tree", "rock", "earth", "river",
	"cave", "forest", "food", "predator", "myself", "group", "wound", "storm",
	"rain", "danger", "safe", "full", "chair", "table", "bed", "cup", "knife",
	"door", "house", "bridge", "road", "roof", "hammer", "basket",
	"eat", "drink", "sleep", "move_action", "see_action", "cut", "open", "build", "sit"]
TIER_ANIMALS = ["dog", "cat", "bird", "horse", "fish", "wolf", "bear", "mouse",
	"rabbit", "fox", "snake", "cow", "pig", "elephant", "lion", "tiger", "monkey",
	"owl", "duck", "eagle", "whale", "shark", "ant", "goat", "deer",
	"learn", "teach", "give"]
TIER_BEHAVIOR = ["bark", "meow", "sing", "roar", "howl", "chirp", "growl", "hiss",
	"moo", "oink", "hoot", "quack", "caw", "croak", "buzz", "trumpet", "chatter",
	"run", "jump", "swim", "fly", "climb", "hunt", "hide", "play"]
MOLECULES = TIER_BASE + TIER_ANIMALS + TIER_BEHAVIOR

PHRASES = [
	("give up", None, "phrasal verb: renunciar/dejar de intentar — concepto único en 2 palabras"),
	("it says bark", "dog", "riddle 3→1: el sonido colapsa en el animal"),
	("it says meow", "cat", "riddle 3→1"),
	("it says quack", "duck", "riddle 3→1"),
	("it says moo", "cow", "riddle 3→1"),
	("baby dog", None, "puppy — pendiente de su propia entrada"),
]


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--db", default="storage/k65p_dictionary.db")
	args = ap.parse_args()
	d = K65PDictionary(os.path.join(BASE, args.db))

	# 1) los 65 primos (átomos: glifo = solo su propio primo activo)
	for i, name in enumerate(SEMANTIC_PRIMES):
		d.upsert_prime(i, name)
		d.upsert_word(name, glyph=explication_to_glyph({name: 1}), status="prime",
			source="canonical_primes")

	# 2) los 28 canónicos (glifos del operador, la fuente de la convención)
	for name, glyph in VOCABULARY.items():
		feats = {SEMANTIC_PRIMES[i]: int(glyph[i]) for i in np.nonzero(np.array(glyph))[0]}
		d.upsert_word(name, glyph=glyph, explication=feats, status="canonical",
			source="operator_glyphs")

	# 3) las 99 moléculas k65p (glifo provisional Ridge — la descomposición NSM
	#    las reemplazará por el programa DL-014)
	g = json.load(open(os.path.join(BASE, "configs", "expanded_glyphs.json")))
	w2i = {w: i for i, w in enumerate(g["words"])}
	n_mol = 0
	for m in MOLECULES:
		if m in w2i:
			# insert_if_new: NO resetea drafted/explicated (bug 1-sep: el
			# re-populate machacaba los drafted del programa NSM a molecule)
			d.insert_if_new(m, glyph=np.array(g["glyphs"][w2i[m]], dtype=np.int8),
				status="molecule", source="k65p_tiers")
			n_mol += 1

	# 4) el vocabulario restante como pending — INSERT OR IGNORE: JAMÁS resetea
	#    el estado de una palabra ya clasificada (bug 1-sep: el segundo populate
	#    machacaba los drafted del generador NSM)
	for w in g["words"]:
		d.ensure_pending(w)

	# 5) las phrases de compresión N→1
	for span, word, note in PHRASES:
		d.upsert_phrase(span, word_surface=word, context_note=note, source="curated")

	# ── aceptación DL-015 F4 ──
	st = d.stats()
	print(f"stats: {st}")
	assert st.get("prime", 0) == 65, f"primos: {st.get('prime')}"
	assert st.get("canonical", 0) == 28, f"canónicos: {st.get('canonical')}"
	assert st.get("molecule", 0) == n_mol, f"moléculas: {st.get('molecule')} != {n_mol}"
	# round-trip RFC-002 §5 DESDE LA BD: glifo → features → glifo
	for name, canon in VOCABULARY.items():
		row = d.en_to_k65p(name)
		g = np.frombuffer(row["glyph"], dtype=np.int8)
		feats = {SEMANTIC_PRIMES[i]: int(g[i]) for i in np.nonzero(g)[0]}
		assert (explication_to_glyph(feats) == canon).all(), f"round-trip falla en {name}"
	print(f"✅ aceptación: 65 primos / 28 canónicos / {n_mol} moléculas | round-trip 28/28 desde la BD")


if __name__ == "__main__":
	main()
