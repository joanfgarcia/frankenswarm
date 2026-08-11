"""Samantha Story Factory (K-65P) — Controlled bilingual synthetic corpus for Formación v2.

Emits paired synthetic data: (text_es, k65p_canonical_sequence).
Every emitted K-65P sequence is checked against the mechanical k65p validator
and the gold/derived lexicon before being written to JSONL.

Usage:
	PYTHONPATH=.:../k65p/src .venv/bin/python scripts/samantha_story_factory_k65p.py \
		--stage preschool --count 100 --mock
"""

import argparse
import hashlib
import json
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

# Add parent dir & k65p src to sys.path
base_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(base_dir))

k65p_src = base_dir.parent / "k65p" / "src"
if k65p_src.exists():
	sys.path.append(str(k65p_src))

try:
	from k65p.lexicon import load_lexicon, molecule_names
	from k65p.validator import is_valid, linearize, validate
except ImportError:
	print("ERROR: k65p module not found in sys.path. Run with PYTHONPATH=.:../k65p/src")
	sys.exit(1)


STAGE_SPECS = {
	"preschool": {
		"age": "3 a 5 años",
		"topics": ["la familia", "los animales", "la comida", "el sol y la lluvia", "el cuerpo"],
	},
	"primary": {
		"age": "6 a 8 años",
		"topics": ["la escuela", "los amigos", "ayudar en casa", "un animal perdido"],
	},
	"secondary": {
		"age": "9 a 12 años",
		"topics": ["una aventura en el bosque", "un invento", "una promesa"],
	},
}

SYSTEM_PROMPT = (
	"Eres una maestra de escuela experta en K-65P. Generas pares bilingües de historias sencillas "
	"en español y su correspondiente traducción formal canónica a expresiones K-65P. "
	"Responde en formato JSONL estricto con campos 'text_es' y 'k65p'."
)


def mock_generate_pairs(stage: str, rng: random.Random) -> list[dict[str, str]]:
	"""Mock paired generator for testing pipeline without active LLM backend."""
	preschool_seeds = [
		{
			"text_es": "el fuego caliente es peligroso",
			"k65p": "[si [TOUCH alguien fuego] [HAPPEN [G BAD] alguien]]",
		},
		{
			"text_es": "yo como comida",
			"k65p": "[DO yo_palabra comer comida]",
		},
		{
			"text_es": "alguien bebe agua",
			"k65p": "[DO alguien beber agua]",
		},
		{
			"text_es": "el agua es buena",
			"k65p": "[GOOD agua]",
		},
		{
			"text_es": "el sol existe",
			"k65p": "[EXIST sol]",
		},
	]

	primary_seeds = [
		{
			"text_es": "si tocas el fuego te haces daño",
			"k65p": "[si [TOUCH alguien fuego] [HAPPEN [G BAD] alguien]]",
		},
		{
			"text_es": "alguien bebe agua",
			"k65p": "[DO alguien beber agua]",
		},
	]
	secondary_seeds = [
		{
			"text_es": "si el sol existe la tierra es buena",
			"k65p": "[si [EXIST sol] [GOOD tierra]]",
		},
	]

	seeds = preschool_seeds if stage == "preschool" else (primary_seeds if stage == "primary" else secondary_seeds)
	return seeds




def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--stage", choices=list(STAGE_SPECS), required=True)
	ap.add_argument("--count", type=int, default=50, help="historias objetivo en esta ejecución")
	ap.add_argument("--seed", type=int, default=770)
	ap.add_argument("--mock", action="store_true", help="sin LLM, para probar el pipeline")
	args = ap.parse_args()

	rng = random.Random(args.seed)
	lex_words = molecule_names()

	out_dir = base_dir / "storage" / "curriculum" / "factory_k65p"
	out_dir.mkdir(parents=True, exist_ok=True)
	out_path = out_dir / f"{args.stage}.jsonl"

	seen = set()
	if out_path.exists():
		with open(out_path, encoding="utf-8") as f:
			for line in f:
				try:
					seen.add(json.loads(line)["hash"])
				except (json.JSONDecodeError, KeyError):
					continue

	print(f"Fábrica K-65P: stage={args.stage}, objetivo={args.count}, ya en disco={len(seen)}, moléculas_lexicon={len(lex_words)}")

	accepted, rejected_val, rejected_dup = 0, 0, 0
	with open(out_path, "a", encoding="utf-8") as out:
		while accepted < args.count:
			pairs = mock_generate_pairs(args.stage, rng)
			for p in pairs:
				text_es = p["text_es"].strip()
				k65p_raw = p["k65p"].strip()

				# Validate K-65P tree & lexicon compliance
				errs = validate(k65p_raw, lexicon=lex_words)
				if errs:
					rejected_val += 1
					continue

				canon_k65p = linearize(k65p_raw)
				h = hashlib.sha256(canon_k65p.encode()).hexdigest()[:16]

				if h in seen:
					rejected_dup += 1
					continue

				seen.add(h)
				accepted += 1
				record = {
					"text_es": text_es,
					"k65p_raw": k65p_raw,
					"k65p_canonical": canon_k65p,
					"stage": args.stage,
					"hash": h,
					"generated": datetime.now(UTC).astimezone().isoformat(),
					"generator": "mock" if args.mock else "samantha_k65p",
				}
				out.write(json.dumps(record, ensure_ascii=False) + "\n")
				if accepted >= args.count:
					break
			if args.mock:
				break

	print(f"✅ aceptadas={accepted} | duplicadas={rejected_dup} | inválidas_k65p={rejected_val} | fichero={out_path}")


if __name__ == "__main__":
	main()
