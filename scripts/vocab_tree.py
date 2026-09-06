"""El árbol del vocabulario v2: el camino inverso de Wierzbicka, desplegado.

Wierzbicka descompone (palabra → primos, hacia abajo). Este script despliega la
dirección inversa (primos → palabras, hacia arriba): el grafo de derivación —
qué molécula referencia a cuál en sus cláusulas — como árbol de texto y como
grafo DOT (Graphviz) para visualización.

Uso:
	PYTHONPATH=.:../k65p/src .venv/bin/python scripts/vocab_tree.py [--dot]
"""

import re
import sys
from pathlib import Path

base = Path(__file__).resolve().parents[1]
sys.path.append(str(base))

TOKEN_RE = re.compile(r"\[|\]|-?\d+|[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_]+")


def clause_refs(clauses: list[str], anchor: str, molecules: set[str], primes: set[str]) -> dict:
	"""Referencias de unas cláusulas: {molécula: n} + {primo: n} usadas."""
	from src.bitnet.vocab.structured_explication import _normalize_prime
	refs: dict = {}
	for c in clauses:
		for t in TOKEN_RE.findall(c):
			tl = t.casefold()
			if tl in ("[", "]", "g", "G"):
				continue
			if tl == anchor.casefold():
				continue
			if tl in molecules:
				refs[("mol", tl)] = refs.get(("mol", tl), 0) + 1
				continue
			norm = _normalize_prime(tl)
			if norm in primes:
				refs[("pri", norm)] = refs.get(("pri", norm), 0) + 1
	return refs


def main() -> None:
	import json
	from src.bitnet.vocab.structured_explication import PRIME_IDX

	mol = json.loads((base / "configs" / "k65p_v2" / "moleculas.json").read_text(encoding="utf-8"))
	primes = {p.casefold() for p in PRIME_IDX} | {str(i) for i in range(65)}
	names = set(mol)

	dot = "--dot" in sys.argv
	if dot:
		print("digraph vocab_v2 {")
		print('\tnode [shape=box, style=rounded];')
		print('\tprimos [shape=ellipse, label="65 primos"];')
		for name, m in mol.items():
			refs = clause_refs(m["clauses"] + m.get("drags", []), name, names, primes)
			for (kind, ref) in refs:
				if kind == "mol":
					print(f'\t"{ref}" -> "{name}";')
		print("}")
		return

	# árbol de texto: por molécula, sus dependencias directas (mallas, no DAG puro:
	# se marca lo ya visitado para no repetir ramas)
	print("── EL ÁRBOL (primos → moléculas: el inverso de Wierzbicka) ──\n")
	for name, m in mol.items():
		all_clauses = m["clauses"] + m.get("drags", [])
		refs = clause_refs(all_clauses, name, names, primes)
		mols = sorted({r for (k, r) in refs if k == "mol"})
		pris = sorted({r for (k, r) in refs if k == "pri"})
		g = m["glyph"]
		n = sum(1 for x in g if x != 0)
		print(f"{name} [{n} trits]")
		all_clauses = m["clauses"] + m.get("drags", [])
		refs = clause_refs(all_clauses, name, names, primes)
		mols = sorted({r for (k, r) in refs if k == "mol"})
		pris = sorted({r for (k, r) in refs if k == "pri"})
		if mols:
			print(f"  └─ moléculas: {', '.join(mols)}")
		if pris:
			from src.bitnet.vocab.structured_explication import _normalize_prime
			pn = sorted({_normalize_prime(p) for p in pris})
			print(f"  └─ primos: {', '.join(pn[:12])}{'...' if len(pn) > 12 else ''}")


if __name__ == "__main__":
	main()
