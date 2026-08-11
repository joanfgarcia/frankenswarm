"""Examen gen_true — evalúa generaciones contra la KB vía Prolog.

A diferencia de gen_valid (sintaxis: is_valid), gen_true convierte la
continuación generada a Prolog vía el puente y comprueba si se DEMUESTRA
a partir de la KB compilada con swipl. Si no es demostrable, la generación
es sintácticamente válida pero no verdadera (→ reasoning fallido).

Métrica separada para teoremas held-out (nunca en corpus) vs recall (en corpus).

Uso: PYTHONPATH=.:../k65p/src .venv/bin/python scripts/exam_gen_true.py [--state_dir DIR]
"""

import argparse, json, subprocess, sys, tempfile
from pathlib import Path

base_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(base_dir))
k65p_src = base_dir.parent / "k65p" / "src"
if k65p_src.exists():
	sys.path.append(str(k65p_src))

from k65p.bridge import BridgeError, to_prolog
from k65p.lexicon import molecule_names
from k65p.validator import is_valid

LANG = "en"
KB_DIR = base_dir / "storage" / "curriculum" / "factory_semantic"
HELDOUT_SET = set()


def load_kb() -> list[str]:
	"""Carga la KB de verdad (kb.pl, sin negaciones contrastivas)."""
	kb_path = KB_DIR / "kb.pl"
	if kb_path.exists():
		return [l.strip() for l in kb_path.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("%")]
	# fallback: cargar de jsonl (viejos)
	clauses = []
	for f in sorted(KB_DIR.glob("*.jsonl")):
		if f.name == "ood_holdout.jsonl":
			continue
		for line in f.read_text(encoding="utf-8").splitlines():
			if not line.strip():
				continue
			data = json.loads(line)
			expr = data.get("k65p_canonical", "")
			if not expr or expr.startswith("[44 "):
				continue
			try:
				clauses.append(to_prolog(expr, executable=True))
				clauses.append(to_prolog(expr, executable=False))
			except BridgeError:
				continue
	return clauses


def check_gen_true(generated_text: str, clauses: list[str] | None = None) -> dict:
	"""Verifica si una expresión generada es demostrable desde la KB."""
	en_lex = molecule_names(lang=LANG, path=k65p_src / ".." / "data" / "lexicon.json")
	if not is_valid(generated_text, lexicon=en_lex):
		return {"valid": False, "provable": None, "error": "syntax"}
	try:
		goal = to_prolog(generated_text, executable=False)
	except BridgeError:
		return {"valid": True, "provable": False, "error": "bridge"}
	if clauses is None:
		clauses = load_kb()
	with tempfile.NamedTemporaryFile(mode="w", suffix=".pl", delete=False) as tmp:
		tmp.write("\n".join(clauses) + "\n")
		tmp.flush()
		query = goal.rstrip(".")
		cmd = ["swipl", "-q", "-s", tmp.name, "-g", f"({query} -> writeln(proved) ; writeln(not_proved)), halt"]
		res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
		provable = res.stdout.strip() == "proved"
	return {"valid": True, "provable": provable, "output": res.stdout[:200] if provable else res.stderr[:200]}


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Examinador gen_true (verdad vía Prolog)")
	parser.add_argument("--expr", type=str, help="Expresión K-65P a verificar")
	args = parser.parse_args()

	if args.expr:
		r = check_gen_true(args.expr)
		print(json.dumps(r, indent=2))
	else:
		# demo: verificar algunas expresiones de la KB y algunas falsas
		print("═" * 60)
		print("  🧪 EXAMEN GEN_TRUE — verdad vía Prolog")
		print(f"  KB cargada: {len(load_kb())} cláusulas")
		tests = [
			("[60 fire]", "hot(fire) — debe ser TRUE (está en la KB)"),
			("[8 fire]", "good(fire) — debe ser FALSE (fire es bad)"),
			("[48 [24 2 fire] [22 [G 4 9] 2]]", "if-touch-bad — TRUE (está en la KB)"),
			("[44 [9 fire]]", "not(bad(fire)) — FALSE (fire es bad)"),
		]
		for expr, desc in tests:
			r = check_gen_true(expr)
			icon = "✅" if r.get("provable") else "❌"
			print(f"  {icon} {desc}")
			print(f"     {expr} → provable={r.get('provable')}")
		print("═" * 60)
