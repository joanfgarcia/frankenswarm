"""Generador composicional de corpus K-65P para la Escuela Soberana de Bit v2.

Genera expresiones K-65P válidas POR CONSTRUCCIÓN (cada árbol se construye
respetando las valencias del validador de referencia y se re-verifica con
`k65p.validator.validate` antes de escribirse). Sustituye al pipeline de
traducción semántica ES→K-65P, deprecado en DL-004 por fidelidad nula.

Estratificación (alineada con las máscaras de etapa del trainer):
- preschool  → moléculas[:10], profundidad ≤ 2, sin conectores binarios
- primary    → moléculas[:20], profundidad ≤ 2, operadores unarios + grupos G
- secondary  → moléculas completas, profundidad ≤ 3, conectores binarios

Además reserva un conjunto OOD real: todas las expresiones cuyo (cabeza,
primer argumento) cae en HOLDOUT_PAIRS se excluyen del corpus escolar y se
escriben en ood_holdout.jsonl — el examen adversarial evalúa sobre ellas.

Uso:
	PYTHONPATH=.:../k65p/src .venv/bin/python scripts/generate_k65p_corpus.py [--seed 770]
"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

base_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(base_dir))

k65p_src = base_dir.parent / "k65p" / "src"
if k65p_src.exists():
	sys.path.append(str(k65p_src))

from k65p.lexicon import load_lexicon, molecule_names
from k65p.validator import is_valid, linearize

OUT_DIR = base_dir / "storage" / "curriculum" / "factory_k65p"

# Valencias espejo del validador (fuente de verdad: k65p/validator.py).
PREDICATES = {
	21: (1, 3), 22: (1, 2), 23: (1, 3), 24: (2, 2), 12: (1, 2), 13: (1, 2),
	14: (2, 2), 15: (2, 2), 16: (1, 2), 17: (1, 2), 18: (2, 3), 25: (1, 2),
	27: (1, 2), 28: (1, 1),
}
EVALUATORS = [8, 9, 10, 11, 26, 60, 61, 64]
UNARY = [44, 45, 46, 31, 30, 32, 37, 41, 40, 38, 39, 43]
BINARY = [48, 47, 51]

# Primos utilizables como átomos-entidad (sustantivos del kernel).
ENTITY_PRIMES = [0, 1, 2, 3, 4, 5, 6, 7, 19, 62, 63]

# Pares (cabeza, primer argumento) reservados como OOD: nunca en train/val.
HOLDOUT_PAIRS = {(21, "fuego"), (25, 63), (14, 1), (24, "agua"), (47, 48)}

# Tamaños ×2.5 tras el probe DL-005: con 1.200 muestras el bloque preescolar
# (288 épocas sobre el mismo fichero) sobreajustaba desde la época ~10.
STAGE_SPECS = {
	"preschool": {"n": 3000, "max_depth": 2, "mol_slice": 10, "binary": False, "unary": False, "group": False},
	"primary": {"n": 3600, "max_depth": 2, "mol_slice": 20, "binary": False, "unary": True, "group": True},
	"secondary": {"n": 4200, "max_depth": 3, "mol_slice": None, "binary": True, "unary": True, "group": True},
}


def build_tree(rng: random.Random, spec: dict, molecules: list[str], depth: int = 1) -> list:
	entities = ENTITY_PRIMES + molecules

	def atom() -> int | str:
		return rng.choice(entities)

	def clause_or_atom() -> list | int | str:
		if depth < spec["max_depth"] and rng.random() < 0.35:
			return build_tree(rng, spec, molecules, depth + 1)
		return atom()

	roll = rng.random()
	if spec["binary"] and depth < spec["max_depth"] and roll < 0.20:
		head = rng.choice(BINARY)
		return [head, build_tree(rng, spec, molecules, depth + 1), build_tree(rng, spec, molecules, depth + 1)]
	if spec["unary"] and depth < spec["max_depth"] and roll < 0.35:
		head = rng.choice(UNARY)
		return [head, build_tree(rng, spec, molecules, depth + 1)]
	if spec["group"] and roll < 0.42:
		head = rng.choice(EVALUATORS)
		members = rng.sample(entities, k=rng.randint(1, 2))
		return [head, ["G", *members]] if rng.random() < 0.5 else [head, rng.choice(entities)]
	if roll < 0.60:
		head = rng.choice(EVALUATORS)
		return [head, atom()]
	head = rng.choice(list(PREDICATES))
	low, high = PREDICATES[head]
	n_args = rng.randint(low, high)
	args = [clause_or_atom() for _ in range(n_args)]
	return [head, *args]


def tree_to_text(tree: list | int | str) -> str:
	if isinstance(tree, list):
		return "[" + " ".join(tree_to_text(t) for t in tree) + "]"
	return str(tree)


def collect_pairs(tree: list) -> set[tuple]:
	"""Pares (cabeza, primer argumento) de TODAS las cláusulas del árbol, anidadas incluidas."""
	pairs = set()
	if not isinstance(tree, list) or not tree:
		return pairs
	head = tree[0]
	first = tree[1] if len(tree) > 1 else None
	pairs.add((head, first[0] if isinstance(first, list) and first else first))
	for child in tree[1:]:
		pairs |= collect_pairs(child)
	return pairs


def main() -> None:
	parser = argparse.ArgumentParser(description="Generador composicional de corpus K-65P")
	parser.add_argument("--seed", type=int, default=770)
	args = parser.parse_args()

	rng = random.Random(args.seed)
	lexicon = load_lexicon()
	all_molecules = sorted(molecule_names())
	lexicon_set = set(lexicon)

	OUT_DIR.mkdir(parents=True, exist_ok=True)
	ood_records = []
	report = {}
	# Dedup GLOBAL entre etapas: una expresión corta de secondary pudo generarse
	# también en preschool → leakage train(etapa A)/val(etapa B). Prohibido.
	seen: set[str] = set()

	for stage, spec in STAGE_SPECS.items():
		molecules = all_molecules[: spec["mol_slice"]] if spec["mol_slice"] else all_molecules
		records = []
		attempts = 0
		while len(records) < spec["n"] and attempts < spec["n"] * 60:
			attempts += 1
			tree = build_tree(rng, spec, molecules)
			text = tree_to_text(tree)
			try:
				canonical = linearize(text)
			except Exception:
				continue
			if canonical in seen or not is_valid(canonical, lexicon=lexicon_set):
				continue
			seen.add(canonical)
			record = {
				"k65p_canonical": canonical,
				"stage": stage,
				"provenance": f"generator_v1(seed={args.seed})",
				"hash": hashlib.sha256(canonical.encode()).hexdigest()[:16],
			}
			hit = collect_pairs(tree) & HOLDOUT_PAIRS
			if hit:
				record["ood_reason"] = f"holdout_pairs={sorted(hit)}"
				ood_records.append(record)
			else:
				records.append(record)

		out_path = OUT_DIR / f"{stage}.jsonl"
		with open(out_path, "w", encoding="utf-8") as f:
			for r in records:
				f.write(json.dumps(r, ensure_ascii=False) + "\n")
		report[stage] = len(records)
		print(f"✓ {stage}: {len(records)} expresiones únicas y válidas → {out_path.name}")

	ood_path = OUT_DIR / "ood_holdout.jsonl"
	with open(ood_path, "w", encoding="utf-8") as f:
		for r in ood_records:
			f.write(json.dumps(r, ensure_ascii=False) + "\n")
	print(f"✓ OOD holdout: {len(ood_records)} expresiones (pares {sorted(HOLDOUT_PAIRS)}) → {ood_path.name}")

	summary = {"seed": args.seed, "stages": report, "ood": len(ood_records), "holdout_pairs": sorted(map(list, HOLDOUT_PAIRS))}
	(OUT_DIR / "generation_manifest.json").write_text(json.dumps(summary, indent=4, ensure_ascii=False), encoding="utf-8")
	print("✓ Manifiesto escrito: generation_manifest.json")


if __name__ == "__main__":
	main()
