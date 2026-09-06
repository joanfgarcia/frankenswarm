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
	("fire", "hot light that can burn",
		["[G light fire]", "[hot fire]", "[not [cold fire]]",
		 "[can [do fire [G something bad]]]"]),  # "es de la luz; es caliente; no es frío; puede hacer algo malo"
	("sun", "the very big bright thing above, far away, of the fire family",
		["[G fire sun]", "[very [big sun]]", "[very [far [exist sun]]]", "[above [exist sun]]"]),
	("earth", "the very big ground everything lives on, near under our feet, that does not move",
		["[G thing earth]", "[very [big earth]]", "[below [live people earth]]",
		 "[very [near [touch people earth]]]", "[not [move earth]]"]),
	# ── capa 2: paleta del pintor ──
	("new", "like the short-time thing",
		["[like new [G short_time thing]]"]),
	("young", "like the short-time someone",
		["[like young [G short_time someone]]"]),
	("old", "like the long-time thing and the long-time someone",
		["[like old [G long_time thing]]", "[like old [G long_time someone]]"]),
	("blind", "if someone is blind, then they cannot see",
		["[if [G blind someone] [not [can [see blind]]]]"]),
	("void", "what does not exist",
		["[not [exist void]]"]),
	("black", "without light; like dying and like cold; like blindness; like the void",
		["[dark black]", "[not [G light black]]", "[like black die]", "[like black cold]",
		 "[like black blind]", "[like black void]"]),
	("red", "light like fire", ["[G light red]", "[like red fire]"]),
	("yellow", "light like the sun", ["[G light yellow]", "[like yellow sun]"]),
	("blue", "light like water, cold", ["[G light blue]", "[cold blue]"]),
	("brown", "the dark color of the earth", ["[like brown earth]", "[not [very [G light brown]]]"]),
	("dirty", "like what is not good; like brown (its badness carried by [bad dirty])",
		["[like [not [good something]] dirty]", "[like dirty brown]", "[bad dirty]"]),
	("clean", "not dirty — defined by opposition (dirty must come first)",
		["[not [dirty clean]]"]),
	("white", "all the light; like living (life/death axis with black↔die); like clean — AFTER clean (definitional layering)",
		["[very [G light white]]", "[like white live]", "[like white clean]"]),
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
