"""KB demo DL-021: nombres propios como símbolos sobre la base de hechos.

El caso 1 del operador: hechos-relación entre nombres que se memorizan y se
cumplen "siempre" (hasta revisión) — {x: París, y: Francia} sobre la fórmula
"X es la capital de Y". La fórmula es el predicado (capital/2); las tuplas
son hechos de la KB. El caso 2 (Juan/perro/Jacky — episódico) usa la MISMA
sintaxis: la distinción hecho/episodio es de uso estadístico, no de marcaje.

Uso:
	cd ../k65p && PYTHONPATH=src ../frankenswarm/.venv/bin/python \
		../frankenswarm/scripts/kb_names_demo.py
"""

from pathlib import Path

from k65p.bridge import to_prolog
from k65p.validator import validate

# ── La base de hechos: fórmulas × tuplas (revisable: creencia, no eternidad) ──
FORMULAS = {
	"capital": "X es la capital de Y — 'la ciudad donde está el gobierno del país'",
	"author": "X es el autor de Y",
	"have": "X tiene un Y",
	"dog-name": "el perro de X se llama Y",
	"same-person": "X y Y son la misma entidad (alias — revisable)",
}

FACTS = [
	"[capital [N paris] [N france]]",
	"[capital [N madrid] [N spain]]",
	"[capital [N lisboa] [N portugal]]",
	"[capital [N roma] [N italia]]",
	"[author [N homero] [N iliada]]",
	"[author [N homero] [N odisea]]",
	"[author [N julio-verne] [N veinte-mil-leguas]]",
	"[have [N juan] dog]",
	"[dog-name dog [N jacky]]",
	"[same-person [N jacky] [N jack]]",
]


def main() -> None:
	print("── Fórmulas (predicados = moléculas explicables en primos) ──")
	for pred, gloss in FORMULAS.items():
		print(f"  {pred:12s} {gloss}")

	print("\n── Hechos: K-65P → Prolog ──")
	pl_lines = ["% KB DL-021 — hechos con nombres propios (revisable)"]
	for f in FACTS:
		errs = validate(f)
		assert not errs, f"{f}: {errs}"
		p = to_prolog(f)
		pl_lines.append(p)
		print(f"  {f:48s} → {p}")

	out = Path(__file__).resolve().parents[1] / "configs" / "k65p_v2" / "facts" / "kb_demo.pl"
	out.parent.mkdir(parents=True, exist_ok=True)
	out.write_text("\n".join(pl_lines) + "\n", encoding="utf-8")
	print(f"\n→ {out}")

	# La regla doctrinal que la gramática protege
	print("\n── La gramática protege la doctrina ──")
	print("  [[N juan] have dog]  →", validate("[[N juan] have dog]") or "VÁLIDA")
	print("    ↑ rechazada: el nombre jamás encabeza — la relación es la cabeza")
	print("  [have [N juan] dog]  →", validate("[have [N juan] dog]") or "VÁLIDA")


if __name__ == "__main__":
	main()
