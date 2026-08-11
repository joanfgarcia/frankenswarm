"""K-65P Corpus Validator — Audit K-65P JSONL datasets against the K-65P specification.

Checks:
- JSONL line structure
- K-65P expression syntax & S-expression parsing
- Lexicon molecule membership (gold + derived)
- Duplication / Hash uniqueness
- Summary metrics for Gate G3 acceptance

Usage:
	PYTHONPATH=.:../k65p/src .venv/bin/python scripts/validate_k65p_corpus.py \
		--file storage/curriculum/factory_k65p/preschool.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent dir & k65p src to sys.path
base_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(base_dir))

k65p_src = base_dir.parent / "k65p" / "src"
if k65p_src.exists():
	sys.path.append(str(k65p_src))

try:
	from k65p.lexicon import molecule_names
	from k65p.validator import validate
except ImportError:
	print("ERROR: k65p module not found in sys.path. Run with PYTHONPATH=.:../k65p/src")
	sys.exit(1)


def audit_corpus_file(file_path: Path) -> dict:
	if not file_path.exists():
		print(f"Error: file {file_path} does not exist.")
		sys.exit(1)

	lex_words = molecule_names()
	total, valid_count, invalid_count = 0, 0, 0
	seen_hashes = set()
	duplicates = 0
	error_samples = []

	with open(file_path, encoding="utf-8") as f:
		for idx, line in enumerate(f, start=1):
			line = line.strip()
			if not line:
				continue
			total += 1
			try:
				data = json.loads(line)
			except json.JSONDecodeError as exc:
				invalid_count += 1
				error_samples.append((idx, f"JSON parse error: {exc}"))
				continue

			k65p_expr = data.get("k65p_raw") or data.get("k65p_canonical")
			if not k65p_expr:
				invalid_count += 1
				error_samples.append((idx, "Missing 'k65p_raw' or 'k65p_canonical' field"))
				continue

			h = data.get("hash")
			if h:
				if h in seen_hashes:
					duplicates += 1
				seen_hashes.add(h)

			errs = validate(k65p_expr, lexicon=lex_words)
			if errs:
				invalid_count += 1
				error_samples.append((idx, "; ".join(errs)))
			else:
				valid_count += 1

	compliance_rate = (valid_count / total * 100) if total > 0 else 0.0

	return {
		"file": str(file_path),
		"total_records": total,
		"valid_records": valid_count,
		"invalid_records": invalid_count,
		"duplicates": duplicates,
		"compliance_rate": compliance_rate,
		"error_samples": error_samples[:5],
	}


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--file", type=str, required=True, help="Path to K-65P JSONL corpus file")
	args = ap.parse_args()

	res = audit_corpus_file(Path(args.file))

	print("=== K-65P CORPUS AUDIT REPORT ===")
	print(f"Archivo: {res['file']}")
	print(f"Total registros: {res['total_records']}")
	print(f"Registros válidos (K-65P): {res['valid_records']}")
	print(f"Registros inválidos: {res['invalid_records']}")
	print(f"Duplicados: {res['duplicates']}")
	print(f"Tasa de conformidad: {res['compliance_rate']:.2f}%")

	if res["error_samples"]:
		print("\nMuestras de errores (máx 5):")
		for line_num, err in res["error_samples"]:
			print(f"  Línea {line_num}: {err}")

	if res["compliance_rate"] == 100.0 and res["total_records"] > 0:
		print("\n✅ GATE G3 VERIFICATION PASS: Corpus 100% conforme con la especificación K-65P.")
		sys.exit(0)
	else:
		print("\n❌ GATE G3 VERIFICATION FAIL: El corpus contiene errores o está vacío.")
		sys.exit(1)


if __name__ == "__main__":
	main()
