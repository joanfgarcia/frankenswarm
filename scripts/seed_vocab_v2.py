"""Vocabulario semilla K-65P v2 (DL-016→022): la siembra reproducible.

Capas definicionales (cada molécula solo referencia las ya definidas):
  1. prototipos (de primos)          → fire, sun
  2. paleta del pintor (luz + protos) → black, white, red, yellow, blue, brown
  3. secundarios = mezclas duales     → purple, green  (NOTA: orange≡yellow
     colisionan en la huella — límite DL-018, pendiente de plano estructural)
  4. emociones (núcleo NSM + arrastre emoción→color, doctrina Inside Out +
     estudios Jonauskaitė&Möhr 2025: amarillo-alegría 90%, rojo-ira 73%,
     azul-tristeza 53%; asco many-to-many: verde=canon Pixar, marrón=ciencia)

Uso:
	PYTHONPATH=.:../k65p/src .venv/bin/python scripts/seed_vocab_v2.py
"""

import sys
from pathlib import Path

import json

base = Path(__file__).resolve().parents[1]
sys.path.append(str(base))

from src.bitnet.vocab.registry_v2 import VocabularyV2  # noqa: E402

SEED = [
	# ── capa 1: prototipos ──
	("fire", "hot light that can burn", ["[hot fire]", "[not [cold fire]]"]),
	("sun", "the very hot bright thing above", ["[very [hot sun]]", "[not [cold sun]]"]),
	("earth", "the ground; the big thing that does not move", ["[G thing earth]", "[not [move earth]]", "[big earth]"]),
	# ── capa 2: paleta del pintor ──
	("black", "without light", ["[dark black]", "[not [G light black]]"]),
	("white", "all the light", ["[very [G light white]]"]),
	("red", "light like fire", ["[G light red]", "[like red fire]"]),
	("yellow", "light like the sun", ["[G light yellow]", "[like yellow sun]"]),
	("blue", "light like water, cold", ["[G light blue]", "[cold blue]"]),
	("brown", "the dark color of the earth", ["[like brown earth]", "[not [very [G light brown]]]"]),
	# ── capa 3: mezclas duales ──
	("purple", "the light mix of red and blue", ["[G light purple]", "[like purple red]", "[like purple blue]"]),
	("green", "the light mix of blue and yellow", ["[G light green]", "[like green blue]", "[like green yellow]"]),
	# ── capa 4: emociones (núcleo NSM como estado sentido + arrastre de color) ──
	("joy", "the state people feel when something very good happens; canon: yellow",
		["[feel people joy]", "[very [good joy]]", "[yellow joy]"]),
	("sadness", "the state people feel when something bad happens; canon: blue, studies: blue-sadness 53%",
		["[feel people sadness]", "[bad sadness]", "[blue sadness]"]),
	("anger", "the hot bad state; canon: red, studies: red-anger 73%",
		["[feel people anger]", "[bad anger]", "[hot anger]", "[red anger]"]),
	("fear", "the bad state of expecting bad; canon: purple, studies: fear→purple/grey/black",
		["[feel people fear]", "[bad fear]", "[purple fear]"]),
	("disgust", "the bad state of rejecting contamination; canon: green (broccoli), studies: brown (excrement) — many-to-many",
		["[feel people disgust]", "[bad disgust]", "[green disgust]", "[brown disgust]"]),
]


def main() -> None:
	v = VocabularyV2()
	for surface, idea, clauses in SEED:
		try:
			v.register(surface, idea, clauses)
			print(f"  ✓ {surface}")
		except Exception as e:
			print(f"  ✗ {surface}: {e}")
			raise
	n = len(v.molecules)
	print(f"\n✓ {n} moléculas sembradas → configs/k65p_v2/moleculas.json")
	gl = {m["surface"]: sum(1 for t in m["glyph"] if t != 0) for m in v.molecules.values()}
	print("  trits activos por molécula:", gl)


if __name__ == "__main__":
	main()
