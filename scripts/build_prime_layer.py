"""Capa 0 del vocabulario K-65P refundado (DL-016): los 65 primos desde cero.

Genera `configs/k65p_v2/primos.json`: la tabla fundacional — id canónico,
nombres (en/es/zh/fr/de), glifo identidad (one-hot), categoría sintáctica
(predicado/evaluador/unario/binario/átomo) y valencias del validador.

Los primos no reciben explicación: son indefinibles por definición (NSM).
Toda molécula nueva nacerá COMPUSTA desde esta tabla (idea fuente → cláusulas
→ glifo proyectado).

Uso:
	PYTHONPATH=.:../k65p/src .venv/bin/python scripts/build_prime_layer.py
"""

import json
import sys
from pathlib import Path

import numpy as np

base_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(base_dir))

k65p_src = base_dir.parent / "k65p" / "src"
if k65p_src.exists():
	sys.path.append(str(k65p_src))

from k65p.primes import GROUP, N_PRIMES, PRIMES  # noqa: E402
from k65p.validator import BINARY, EVALUATORS, PREDICATES, UNARY  # noqa: E402

# Marcos de rol por predicado (fuente: validator.py — la semántica posicional)
ROLE_NAMES = {
	21: ["agent", "action", "patient"],   # DO
	22: ["theme", "experiencer"],          # HAPPEN
	23: ["theme", "origin", "destination"],  # MOVE
	24: ["agent", "patient"],              # TOUCH
	12: ["experiencer", "content"],        # THINK
	13: ["experiencer", "content"],        # KNOW
	14: ["experiencer", "content"],        # WANT
	15: ["experiencer", "state"],          # FEEL
	16: ["experiencer", "theme"],          # SEE
	17: ["experiencer", "theme"],          # HEAR
	18: ["agent", "content", "addressee"], # SAY
	25: ["theme", "place"],                # EXIST
	27: ["theme", "place"],                # LIVE
	28: ["theme"],                         # DIE
}

ROLE_FOR_OPERATOR = {
	**{i: "predicate" for i in PREDICATES},
	**{i: "evaluator" for i in EVALUATORS},   # atribución unaria: [eval tema]
	**{i: "unary_operator" for i in UNARY},   # opera sobre cláusulas
	**{i: "binary_connector" for i in BINARY},  # foreground-first
}


def build() -> dict:
	primos = []
	for pid, row in enumerate(PRIMES):
		en, es, zh, fr, de = row
		role = ROLE_FOR_OPERATOR.get(pid, "atom")  # los primos "cosa" son átomos puros
		entry = {
			"id": pid,
			"names": {"en": en.lower(), "es": es, "zh": zh, "fr": fr, "de": de},
			"kind": role,
			"glyph": [1 if j == pid else 0 for j in range(N_PRIMES)],
		}
		if role == "predicate":
			entry["valency"] = {"args": PREDICATES[pid], "roles": ROLE_NAMES.get(pid)}
		primos.append(entry)

	# el marcador estructural G (grupo): no es primo, es sintaxis
	table = {
		"version": "v2-refundada",
		"date": "2026-09-02",
		"doctrine": "DL-016: vocabulario refundado desde los 65 primos; sin legado",
		"n_primes": N_PRIMES,
		"group_marker": {"id": "G", "note": "marcador de grupo: [G a b c] — miembros átomo, no cláusulas"},
		"primes": primos,
	}
	return table


def main() -> None:
	table = build()
	out = base_dir / "configs" / "k65p_v2"
	out.mkdir(parents=True, exist_ok=True)
	path = out / "primos.json"
	path.write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")

	kinds = {}
	for p in table["primes"]:
		kinds[p["kind"]] = kinds.get(p["kind"], 0) + 1
	print(f"✓ Capa 0: {table['n_primes']} primos → {path}")
	print(f"  categorías: {kinds}")
	print("  ejemplos:")
	for i in (0, 14, 8, 44, 48, 62):
		p = table["primes"][i]
		print(f"    [{i}] {p['names']['en']:12s} {p['kind']}"
			+ (f" valencia {p['valency']['args']} roles {p['valency']['roles']}" if "valency" in p else ""))


if __name__ == "__main__":
	main()
