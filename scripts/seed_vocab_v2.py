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
		["[G fire light]", "[hot fire]", "[not [cold fire]]",
		 "[can [do fire [G something bad]]]"]),  # "es de la luz; es caliente; no es frío; puede hacer algo malo"
	("sky", "the big place above where sun, moon and stars live",
		["[G thing sky]", "[big sky]", "[above [exist sky]]"]),
	("sun", "the very big bright thing above, far away, of the fire family",
		["[G sun fire]", "[very [big sun]]", "[very [far [exist sun]]]", "[above [exist sun]]"]),
	("earth", "the very big ground everything lives on, near under our feet, that does not move",
		["[G earth thing]", "[very [big earth]]", "[below [live people earth]]",
		 "[very [near [touch people earth]]]", "[not [move earth]]"]),
	# ── capa 2: paleta del pintor ──
	("new", "like the short-time thing",
		["[like new [G thing short_time]]"]),
	("young", "like the short-time someone",
		["[like young [G someone short_time]]"]),
	("old", "like the long-time thing and the long-time someone",
		["[like old [G thing long_time]]", "[like old [G someone long_time]]"]),
	("blind", "if someone is blind, then they cannot see (condition as predication, not group)",
		["[if [blind someone] [not [can [see blind]]]]"]),
	("void", "what does not exist",
		["[not [exist void]]"]),
	("black", "without light; like dying and like cold; like blindness; like the void",
		["[dark black]", "[not [G black light]]", "[like black die]", "[like black cold]",
		 "[like black blind]", "[like black void]"]),
	("red", "light like fire", ["[G red light]", "[like red fire]"]),
	("yellow", "light like the sun", ["[G yellow light]", "[like yellow sun]"]),
	("blue", "light like water, cold", ["[G blue light]", "[cold blue]"]),
	("brown", "the dark color of the earth", ["[like brown earth]", "[not [very [G brown light]]]"]),
	("dirty", "like what is not good; like brown (its badness carried by [bad dirty])",
		["[like [not [good something]] dirty]", "[like dirty brown]", "[bad dirty]"]),
	("clean", "not dirty — defined by opposition (dirty must come first)",
		["[not [dirty clean]]"]),
	("white", "all the light; like living (life/death axis with black↔die); like clean — AFTER clean (definitional layering)",
		["[very [G white light]]", "[like white live]", "[like white clean]"]),
	# ── capa 3: mezclas duales ──
	("purple", "the light mix of red and blue", ["[G purple light]", "[like purple red]", "[like purple blue]"]),
	("green", "the light mix of blue and yellow", ["[G green light]", "[like green blue]", "[like green yellow]"]),
	# ── capa 4: emociones (núcleo NSM como estado sentido + arrastre de color) ──
	("joy", "the state people feel when something very good happens; canon: yellow",
		["[feel people joy]", "[very [good joy]]"], ["[yellow joy]"]),
	("sadness", "the state people feel when something bad happens; canon: blue, studies: blue-sadness 53%",
		["[feel people sadness]", "[bad sadness]"], ["[blue sadness]"]),
	("anger", "the hot bad state; canon: red, studies: red-anger 73%",
		["[feel people anger]", "[bad anger]", "[hot anger]"], ["[red anger]"]),
	("fear", "the bad state of expecting bad; canon: purple, studies: fear→purple/grey/black",
		["[feel people fear]", "[bad fear]"], ["[purple fear]"]),
	("disgust", "the bad state of rejecting contamination; canon: green (broccoli), studies: brown (excrement) — many-to-many",
		["[feel people disgust]", "[bad disgust]"], ["[green disgust]", "[brown disgust]"]),
	# ── capa 5: gente y sonidos (lote aprobado por el operador 3-sep) ──
	("child", "a young person",
		["[young child]", "[small child]", "[G people child]"]),
	("baby", "a very young child",
		["[very [young baby]]", "[very [small baby]]", "[G child baby]"]),
	("animal", "a living thing that moves by itself and can act — a little like someone, but not of the people",
		["[like animal someone]", "[G animal someone]", "[not [G people animal]]", "[live animal]", "[move animal]", "[can [do animal]]"]),
	("domestic", "living with people, in a good way",
		["[live domestic people]", "[good domestic]"]),
	("friend", "a known someone one feels good with",
		["[know someone friend]", "[feel people [G friend good]]", "[G someone friend]"]),
	("dog", "a domestic animal that says bark — and is a friend (in cultures where dogs are companions)",
		["[G dog animal]", "[say dog bark]", "[G dog domestic]", "[friend dog]"]),
	("cat", "a domestic animal that says meow",
		["[G cat animal]", "[say cat meow]", "[G cat domestic]"]),
	("bark", "the sound of the dog — a big (loud, scandalous) sound even if the dog is small",
		["[hear people bark]", "[big bark]", "[G dog bark]"]),
	("meow", "the sound of the cat — a small sound people hear",
		["[hear people meow]", "[small meow]", "[G cat meow]"]),
]


def main() -> None:
	v = VocabularyV2()
	for entry in SEED:
		surface, idea, clauses = entry[0], entry[1], entry[2]
		drags = entry[3] if len(entry) > 3 else []
		try:
			v.register(surface, idea, clauses, drag_clauses=drags)
			print(f"  ✓ {surface}" + (f" (+{len(drags)} arrastre)" if drags else ""))
		except Exception as e:
			print(f"  ✗ {surface}: {e}")
			raise
	# fail-fast: toda molécula referenciada debe existir (nada cae en silencio)
	import re as _re
	known = set(v.molecules) | {r[0].lower() for r in __import__('k65p.primes', fromlist=['PRIMES']).PRIMES} | {r[1].lower() for r in __import__('k65p.primes', fromlist=['PRIMES']).PRIMES}
	missing = set()
	for _m in v.molecules.values():
		for _c in _m["clauses"] + _m.get("drags", []):
			for _t in _re.findall(r'[a-zA-Z_]+', _c):
				_tl = _t.casefold()
				if _tl in ("g", "G") or _tl.lstrip("-").isdigit() or _tl in known:
					continue
				try:
					from src.bitnet.vocab.structured_explication import SYMBOL_TO_ID as _S
					if _tl in _S:
						continue
				except ImportError:
					pass
				missing.add((_m["surface"], _t))
	assert not missing, f"referencias colgadas: {sorted(missing)}"
	v.reproject_all()
	from src.bitnet.vocab.registry_v2 import InjectionError as _IE
	seen = {}
	for _n, _m in v.molecules.items():
		_g = tuple(_m["glyph"])
		if _g in seen:
			raise _IE(f"colisión tras punto fijo: '{_n}' ≡ '{seen[_g]}'")
		seen[_g] = _n
	n = len(v.molecules)
	print(f"\n✓ {n} moléculas sembradas (punto fijo) → configs/k65p_v2/moleculas.json")
	gl = {m["surface"]: sum(1 for t in m["glyph"] if t != 0) for m in v.molecules.values()}
	print("  trits activos por molécula:", gl)


if __name__ == "__main__":
	main()
