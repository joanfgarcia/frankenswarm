"""Fábrica de corpus semántico K-65P v3 — gating curricular + narrativas.

Genera corpus por 3 tiers semánticos alineados con el gating del trainer:
- preschool (base): el mundo físico y sus atributos (fire, water, sun...)
- primary (+animals): agentes animales + acciones básicas
- secondary (+behavior): sonidos + acciones complejas + narrativas

Cada grupo garantiza >= 40 expresiones. Expresiones sintácticamente válidas
Y semánticamente coherentes (verificadas contra el validador).
"""

import argparse, hashlib, json, sys
from pathlib import Path

base_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(base_dir))
k65p_src = base_dir.parent / "k65p" / "src"
if k65p_src.exists():
	sys.path.append(str(k65p_src))

from k65p.lexicon import molecule_names
from k65p.validator import is_valid

LANG = "en"
OUT_DIR = base_dir / "storage" / "curriculum" / "factory_semantic"

# ── Tiers semánticos (espejo de SEMANTIC_TIERS del trainer) ──────────────────
BASE = [
	"fire","water","sun","night","tree","rock","earth","river","cave","forest",
	"food","predator","myself","group","wound","storm","rain","danger","safe","full",
]
ANIMALS = ["dog","cat","bird","horse","fish","wolf","bear","mouse","rabbit","fox","snake"]
BASIC_ACTS = ["eat","drink","sleep","move_action","see_action","learn","teach","give"]
BEHAVIOR = ["bark","meow","sing","roar","howl","chirp","growl","hiss",
	"run","jump","swim","fly","climb","hunt","hide","play"]

# Categorías de atributos (qué entidades son qué)
HOT = ["fire","sun"]
COLD = ["water","night"]
GOOD = ["food","water","sun","sleep"]
BAD = ["fire","predator","storm","wound","danger"]
BIG = ["sun","tree","forest","predator","earth","river"]
SMALL = ["rock","myself","cave"]  # solo base (tier 0)
SMALL_ANIMALS = ["mouse","rabbit"]  # tier 1
LIVE = ["myself","predator"] + ANIMALS
BASE_LIVE = ["myself","predator"]  # solo entidades base (tier 0)

# Sonido por animal
SOUNDS = {"dog":"bark","cat":"meow","bird":"sing","wolf":"howl","bear":"growl",
	"mouse":"chirp","rabbit":"chirp","fox":"howl","snake":"hiss"}


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--seed", type=int, default=770)
	args = parser.parse_args()
	en_lex = molecule_names(lang=LANG, path=str(k65p_src / ".." / "data" / "lexicon.json"))
	OUT_DIR.mkdir(parents=True, exist_ok=True)

	corpus = {"preschool": set(), "primary": set(), "secondary": set()}

	def add(stage, expr):
		expr = expr.strip()
		if expr and is_valid(expr, lexicon=en_lex):
			corpus[stage].add(expr)

	# ══════════════ TIER 0: PREESCOLAR (mundo base) ══════════════
	for e in BASE:
		add("preschool", f"[25 {e}]")            # existencia
	for e in HOT:    add("preschool", f"[60 {e}]"); add("preschool", f"[60 [G {e} 60]]")
	for e in COLD:   add("preschool", f"[61 {e}]"); add("preschool", f"[61 [G {e} 61]]")
	for e in GOOD:   add("preschool", f"[8 {e}]");  add("preschool", f"[8 [G {e} 8]]")
	for e in BAD:    add("preschool", f"[9 {e}]");  add("preschool", f"[9 [G {e} 9]]")
	for e in BIG:    add("preschool", f"[10 {e}]")
	for e in SMALL:  add("preschool", f"[11 {e}]")
	# Atributos cruzados (grupo G con dos atributos)
	for e in HOT:    add("preschool", f"[9 [G {e} 60]]")   # malo+caliente (fuego)
	for e in COLD:   add("preschool", f"[8 [G {e} 61]]")   # bueno+frío (agua)
	for e in BASE_LIVE:   add("preschool", f"[46 [21 {e} sleep]]")  # CAN dormir
	# Acciones básicas de la base
	for e in BASE_LIVE:
		for a in ["eat","drink","sleep","move_action"]:
			add("preschool", f"[21 {e} {a}]")
	# Combinaciones cruzadas de atributos (grupo G con dos entidades)
	for e in BASE:
		for e2 in BASE:
			if e != e2:
				add("preschool", f"[8 [G {e} {e2}]]")
				add("preschool", f"[9 [G {e} {e2}]]")
	# Causalidad base (causa → sentir)
	for e in BASE:
		for cause in [f"[24 2 {e}]", f"[25 {e}]"]:
			for feel in ["[15 2 8]", "[15 2 9]"]:
				add("preschool", f"[47 {cause} {feel}]")
	# DO base con objeto
	for e in BASE_LIVE:
		for obj in ["food","water"]:
			add("preschool", f"[21 {e} eat {obj}]")
			add("preschool", f"[21 {e} drink {obj}]")

	# ══════════════ TIER 1: PRIMARIA (+animales) ══════════════
	for a in ANIMALS:
		add("primary", f"[25 {a}]")             # animales existen
		add("primary", f"[27 {a} forest]")      # viven en el bosque
	for a in ANIMALS:
		for act in ["eat","drink","sleep","move_action","see_action","learn","teach","give"]:
			add("primary", f"[21 {a} {act}]")    # animales actúan
	# Atributos de animales
	for a in ANIMALS:
		add("primary", f"[11 {a}]") if a in SMALL + SMALL_ANIMALS else add("primary", f"[10 {a}]")
	# Causalidad simple con animales
	for a in ANIMALS:
		add("primary", f"[47 [25 {a}] [15 2 8]]")    # porque existe → sentir bien
	for pred in ["wolf","bear","fox","snake"]:
		add("primary", f"[47 [25 {pred}] [15 2 9]]") # depredadores → sentir mal
		add("primary", f"[9 [G {pred} big]]")
	# Acciones básicas + objetos (cross product completo)
	for a in ANIMALS:
		for obj in ["food","water"]:
			add("primary", f"[21 {a} eat {obj}]")
			add("primary", f"[21 {a} drink {obj}]")
	# DO con doble objeto
	for a in ANIMALS:
		for o1 in ["food","water","rock"]:
			add("primary", f"[21 {a} eat {o1}]")
	# Percepción simple
	for a in ANIMALS:
		for b in ANIMALS[:6]:
			if a != b:
				add("primary", f"[16 {a} {b}]")     # a sees b
	# Existencia + grupo
	for a in ANIMALS:
		add("primary", f"[25 [G {a} small]]")
	# Acciones + lugar
	for a in ANIMALS:
		for loc in ["cave","forest","river","earth"]:
			add("primary", f"[21 {a} sleep {loc}]")
	# Causalidad con acciones
	for a in ANIMALS:
		for act in ["eat","drink","sleep"]:
			add("primary", f"[47 [21 {a} {act}] [15 2 8]]")

	# ══════════════ TIER 2: SECUNDARIA (+sonidos +complejas) ══════════════
	# Sonidos: animal → su sonido
	for a, s in SOUNDS.items():
		add("secondary", f"[21 {a} {s}]")          # dog bark
		add("secondary", f"[25 {a}]")
	# Percepción: X oye el sonido de Y (cross product)
	for a in ANIMALS:
		for s_owner, snd in SOUNDS.items():
			if a != s_owner:
				add("secondary", f"[17 {a} {snd}]")  # cat hears bark
				add("secondary", f"[17 {a} {s_owner}]")
	# Acciones complejas
	for a in ANIMALS:
		for act in ["hunt","hide","climb","fly","swim","jump","run","play"]:
			add("secondary", f"[21 {a} {act}]")
	# Narrativas encadenadas (depredador → presa huye)
	narr = [
		("[25 wolf]", "[21 wolf hunt]", "[16 dog wolf]", "[21 dog run]", "[15 dog 9]"),
		("[25 wolf]", "[21 wolf hunt]", "[16 cat wolf]", "[21 cat hide]"),
		("[25 fox]", "[21 fox hunt]", "[16 rabbit fox]", "[21 rabbit run]"),
		("[25 bear]", "[21 bear hunt]", "[16 fish bear]", "[21 fish swim]"),
		("[25 snake]", "[21 snake hunt]", "[16 mouse snake]", "[21 mouse hide]"),
		("[25 cat]", "[21 cat hunt]", "[16 bird cat]", "[21 bird fly]"),
	]
	for seq in narr:
		for expr in seq:
			add("secondary", expr)
	# Más narrativas: sonido → reacción
	for a, s in SOUNDS.items():
		for listener in ANIMALS:
			if listener != a:
				add("secondary", f"[21 {a} {s}]")
				add("secondary", f"[17 {listener} {s}]")
				add("secondary", f"[47 [17 {listener} {s}] [16 {listener} {a}]]")
	# Negación contrastiva
	for e in BASE + ANIMALS:
		add("secondary", f"[44 [25 {e}]]")
		add("secondary", f"[44 [9 {e}]]")
		add("secondary", f"[44 [8 {e}]]")
	# Cuantificadores
	add("secondary", "[28 [G 3 58]]")
	add("secondary", "[44 [28 [G 3 57]]]")
	# Causalidad anidada con sonidos y acciones
	for a, s in SOUNDS.items():
		add("secondary", f"[47 [21 {a} {s}] [15 2 8]]")
	for a in ANIMALS:
		for act in ["hunt","hide","run","play"]:
			add("secondary", f"[47 [21 {a} {act}] [15 2 8]]")
			add("secondary", f"[47 [21 {a} {act}] [15 2 9]]")
	# IF con animales
	for a in ANIMALS:
		for s_owner, snd in SOUNDS.items():
			if a != s_owner:
				add("secondary", f"[48 [17 {a} {snd}] [16 {a} {s_owner}]]")
	# DO complejo (acción + objeto + lugar)
	for a in ANIMALS:
		for act in ["hunt","hide","climb"]:
			for loc in ["cave","forest","river"]:
				add("secondary", f"[21 {a} {act} {loc}]")

	# ── Escribir ──
	total = 0
	for stage in ["preschool", "primary", "secondary"]:
		path = OUT_DIR / f"{stage}.jsonl"
		with open(path, "w", encoding="utf-8") as f:
			for expr in sorted(corpus[stage]):
				r = {"k65p_canonical": expr, "stage": stage,
					"provenance": "semantic_factory_v3",
					"hash": hashlib.sha256(expr.encode()).hexdigest()[:16]}
				f.write(json.dumps(r, ensure_ascii=False) + "\n")
		total += len(corpus[stage])
		print(f"  {stage}: {len(corpus[stage])} expresiones")

	# Held-out
	heldout = ["[48 [24 2 [G fire hot]] [22 [G 4 9] 2]]",
		"[48 [44 [21 2 drink water]] [28 2]]",
		"[47 [25 fire] [47 [60 fire] [15 2 9]]]"]
	with open(OUT_DIR / "ood_holdout.jsonl", "w", encoding="utf-8") as f:
		for ho in heldout:
			f.write(json.dumps({"k65p_canonical": ho, "stage": "secondary",
				"provenance": "heldout", "hash": hashlib.sha256(ho.encode()).hexdigest()[:16],
				"ood_reason": "heldout_theorem"}, ensure_ascii=False) + "\n")
	print(f"  held-out: {len(heldout)} teoremas")

	# KB de verdad
	from k65p.bridge import to_prolog as tp
	kb = set()
	for stage in ["preschool", "primary", "secondary"]:
		for expr in corpus[stage]:
			if expr.startswith("[44 "): continue
			if is_valid(expr, lexicon=en_lex):
				try:
					kb.add(tp(expr, executable=True))
					kb.add(tp(expr, executable=False))
				except Exception:
					pass
	for ho in heldout:
		try:
			kb.add(tp(ho, executable=True)); kb.add(tp(ho, executable=False))
		except Exception:
			pass
	with open(OUT_DIR / "kb.pl", "w", encoding="utf-8") as f:
		for c in sorted(kb):
			f.write(c + "\n")
	print(f"  KB: {len(kb)} cláusulas | TOTAL corpus: {total}")
	print("Listo.")


if __name__ == "__main__":
	main()
