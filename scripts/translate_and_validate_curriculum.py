"""Translate & Validate Curriculum (1:1 Bit v1 → Bit v2 K-65P).

Reads the exact curriculum dataset used for Bit v1 (`configs/school_curriculum.json`),
translates each sentence line-by-line into K-65P canonical S-expressions, and executes
a 3-Layer Validation Suite:

1. Layer 1: Mechanical Syntax & Lexicon Validation (`k65p.validator.validate`)
2. Layer 2: Round-Trip Inverse Rendering & Convergence Invariant (`render` -> re-parse -> match)
3. Layer 3: Horn Bridge Symbolics (`to_prolog` -> `from_prolog` -> match)

Outputs JSONL datasets to `storage/curriculum/factory_k65p/<stage>.jsonl`.

Usage:
	PYTHONPATH=.:../k65p/src .venv/bin/python scripts/translate_and_validate_curriculum.py [--limit 100]
"""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

# Add parent dir & k65p src to sys.path
base_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(base_dir))

k65p_src = base_dir.parent / "k65p" / "src"
if k65p_src.exists():
	sys.path.append(str(k65p_src))

try:
	from k65p.bridge import from_prolog, to_prolog
	from k65p.lexicon import load_lexicon, molecule_names
	from k65p.validator import is_valid, linearize, render, validate
except ImportError:
	print("ERROR: k65p module not found in sys.path. Run with PYTHONPATH=.:../k65p/src")
	sys.exit(1)


# Dictionary mapping spanish terms/lemmas to K-65P lexicon molecules or prime symbols
LEMMA_MAP = {
	# Molecules
	"gato": "yo_palabra",
	"oso": "yo_palabra",
	"conejo": "yo_palabra",
	"pato": "yo_palabra",
	"vaca": "yo_palabra",
	"pez": "yo_palabra",
	"pájaro": "yo_palabra",
	"perro": "yo_palabra",
	"niño": "alguien",
	"niña": "alguien",
	"mamá": "madre",
	"papá": "padre",
	"familia": "familia",
	"amigo": "amigo",
	"escuela": "escuela",
	"sol": "sol",
	"luna": "sol",
	"noche": "noche",
	"cueva": "cueva",
	"fuego": "fuego",
	"agua": "agua",
	"comida": "comida",
	"miel": "comida",
	"pan": "pan",
	"tierra": "tierra",
	"río": "río",
	"árbol": "árbol",
	"bosque": "bosque",
	"lluvia": "lluvia",
	"tormenta": "tormenta",
	"peligro": "peligro",
	"miedo": "miedo",
	"verdad": "verdad",
	"promesa": "promesa",
	"viaje": "viaje",
	"piedra": "piedra",
	"herida": "herida",
	"depredador": "depredador",

	# Verbs -> Primos / Lexicon Actions
	"come": "comer", "comer": "comer",
	"bebe": "beber", "beber": "beber",
	"duerme": "dormir", "dormir": "dormir",
	"soñador": "soñar", "sueña": "soñar",
	"mueve": "mover_accion", "correr": "mover_accion", "corre": "mover_accion", "salta": "mover_accion",
	"ve": "ver_accion", "veo": "ver_accion", "vista": "ver_accion", "mirar": "ver_accion",
	"aprende": "aprender", "enseña": "enseñar",
}


def parse_spanish_to_k65p(text: str) -> str | None:
	"""Translates a simple school Spanish sentence into a canonical K-65P S-expression."""
	words = [w.lower() for w in re.findall(r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_]+", text)]
	if not words:
		return None

	# Check conditional
	is_conditional = "si" in words

	# Extract subject, verb, object
	verb, subject, obj = None, None, None

	for w in words:
		lemma = LEMMA_MAP.get(w, w)
		if lemma in {"comer", "beber", "dormir", "soñar", "mover_accion", "ver_accion", "aprender", "enseñar"}:
			if verb is None:
				verb = lemma
		elif lemma in molecule_names() or lemma in {"alguien", "yo_palabra"}:
			if subject is None:
				subject = lemma
			elif obj is None:
				obj = lemma

	if not subject:
		subject = "alguien"

	# Build S-expression according to valency frames
	if verb:
		if verb in {"dormir", "soñar"}:
			tree = f"[DO {subject} {verb}]"
		elif obj:
			tree = f"[DO {subject} {verb} {obj}]"
		else:
			tree = f"[DO {subject} {verb}]"
	elif "bueno" in words or "feliz" in words or "bien" in words:
		tree = f"[GOOD {subject}]"
	elif "malo" in words or "peligroso" in words or "daño" in words:
		tree = f"[BAD {subject}]"
	else:
		tree = f"[EXIST {subject}]"

	if is_conditional:
		tree = f"[si [EXIST sol] {tree}]"

	return tree


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--limit", type=int, default=0, help="Límite de frases por etapa (0 = todas)")
	args = ap.parse_args()

	curriculum_path = base_dir / "configs" / "school_curriculum.json"
	if not curriculum_path.exists():
		print(f"Error: {curriculum_path} no existe.")
		sys.exit(1)

	with open(curriculum_path, encoding="utf-8") as f:
		school_data = json.load(f)

	lex_words = molecule_names()
	out_dir = base_dir / "storage" / "curriculum" / "factory_k65p"
	out_dir.mkdir(parents=True, exist_ok=True)

	print("=== TRADUCCIÓN E INSPECCIÓN 3-CAPAS DEL CURRÍCULO DE BIT V1 ===")
	print(f"Vocabulario Léxico K-65P: {len(lex_words)} moléculas registradas.")

	total_all, valid_layer1, valid_layer2, valid_layer3 = 0, 0, 0, 0

	for stage, sentences in school_data.items():
		if args.limit > 0:
			sentences = sentences[:args.limit]

		out_path = out_dir / f"{stage}.jsonl"
		print(f"\n🎒 Procesando Etapa: [{stage.upper()}] ({len(sentences)} frases)...")

		st_total, st_l1, st_l2, st_l3 = 0, 0, 0, 0
		seen = set()

		with open(out_path, "w", encoding="utf-8") as out:
			for text_es in sentences:
				st_total += 1
				total_all += 1

				k65p_tree = parse_spanish_to_k65p(text_es)
				if not k65p_tree:
					continue

				# Layer 1: Mechanical Syntax & Lexicon Validation
				errs = validate(k65p_tree, lexicon=lex_words)
				if errs:
					continue
				st_l1 += 1
				valid_layer1 += 1

				canon_k65p = linearize(k65p_tree)

				# Layer 2: Round-Trip Inverse Rendering & Convergence Invariant
				rendered_en = render(k65p_tree, "en")
				try:
					recompiled_tree = linearize(rendered_en)
					if recompiled_tree != canon_k65p:
						continue
					st_l2 += 1
					valid_layer2 += 1
				except Exception:
					continue

				# Layer 3: Horn Bridge Symbolics
				try:
					prolog_clause = to_prolog(k65p_tree)
					re_k65p = from_prolog(prolog_clause)
					if linearize(re_k65p) != canon_k65p:
						continue
					st_l3 += 1
					valid_layer3 += 1
				except Exception:
					continue

				h = hashlib.sha256(canon_k65p.encode()).hexdigest()[:16]
				if h in seen:
					continue
				seen.add(h)

				record = {
					"text_es": text_es,
					"k65p_raw": k65p_tree,
					"k65p_canonical": canon_k65p,
					"rendered_en": rendered_en,
					"prolog": prolog_clause,
					"stage": stage,
					"hash": h,
				}
				out.write(json.dumps(record, ensure_ascii=False) + "\n")

		l1_pct = (st_l1 / st_total * 100) if st_total > 0 else 0
		l2_pct = (st_l2 / st_total * 100) if st_total > 0 else 0
		l3_pct = (st_l3 / st_total * 100) if st_total > 0 else 0
		print(f"  ✓ [{stage}] Procesadas: {st_total} | L1 Validez: {l1_pct:.1f}% | L2 Round-Trip: {l2_pct:.1f}% | L3 Prolog: {l3_pct:.1f}% | Únicas escritas: {len(seen)}")

	overall_pct = (valid_layer3 / total_all * 100) if total_all > 0 else 0
	print("\n==================================================")
	print(f"📊 INFORME FINAL DE TRADUCCIÓN Y AUDITORÍA 3-CAPAS:")
	print(f"   Total Frases Procesadas: {total_all}")
	print(f"   Capa 1 (Sintaxis RFC-002 + Léxico): {valid_layer1} / {total_all}")
	print(f"   Capa 2 (Convergencia Round-Trip Inversa): {valid_layer2} / {total_all}")
	print(f"   Capa 3 (Consistencia Horn Bridge Prolog): {valid_layer3} / {total_all}")
	print(f"   Tasa de Conformidad Global 3-Capas: {overall_pct:.2f}%")
	print("==================================================")


if __name__ == "__main__":
	main()
