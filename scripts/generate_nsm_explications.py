"""Generador de explicaciones NSM por lotes (DL-014, F5 del plan DL-015).

Sobre las palabras pending/molecule del diccionario K-65P:
1. **Patrones del mundo de Bit** (determinístico): animales → rasgos de agente
   vivo; objetos → rasgos de cosa; sonidos → oír/decir. Estado → `drafted`.
2. **Lote LLM** (el resto): emitido a revisión con la plantilla de prompt —
   el LLM generador rellena, el operador cura, y la curación promueve a
   `explicated`.

Validación: consistencia Jaccard entre palabras de la misma familia.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.bitnet.vocab.k65p_dictionary import K65PDictionary  # noqa: E402
from src.bitnet.vocab.nsm_explication import consistency_report, explication_to_glyph  # noqa: E402

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Rasgos NSM por familia (el juicio semántico del patrón)
PAT_AGENT = {"alguien": 1, "cuerpo": 1, "sentir": 1, "mover": 1, "vivir": 1, "querer": 1}
PAT_THING = {"cosa": 1, "cuerpo": 1, "tocar": 1, "vivir": -1, "sentir": -1, "mover": -1}
PAT_SOUND = {"oír": 1, "decir": 1}

FAMILIA_ANIMAL = ("dog", "cat", "bird", "horse", "fish", "wolf", "bear", "mouse",
	"rabbit", "fox", "snake", "cow", "pig", "elephant", "lion", "tiger", "monkey",
	"owl", "duck", "eagle", "whale", "shark", "goat", "deer")
FAMILIA_OBJETO = ("chair", "table", "bed", "cup", "knife", "door", "house",
	"bridge", "road", "roof", "hammer", "basket", "ball", "book")
FAMILIA_SONIDO = ("bark", "meow", "roar", "howl", "chirp", "growl", "hiss",
	"moo", "oink", "hoot", "quack", "caw", "croak", "buzz", "trumpet", "chatter")


def pattern_explications(targets: list[str], gate_words: set) -> tuple[dict, list]:
	"""Explicaciones por patrón para las palabras de las familias curadas.
	Devuelve ({palabra: features}, [palabras sin patrón aplicable])."""
	drafted, no_pattern = {}, []
	for w in targets:
		if w in FAMILIA_ANIMAL:
			drafted[w] = dict(PAT_AGENT)
		elif w in FAMILIA_OBJETO:
			drafted[w] = dict(PAT_THING)
		elif w in FAMILIA_SONIDO:
			drafted[w] = dict(PAT_SOUND)
		else:
			no_pattern.append(w)
	return drafted, no_pattern


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--db", default="storage/k65p_dictionary.db")
	ap.add_argument("--batch", type=int, default=100)
	args = ap.parse_args()
	d = K65PDictionary(os.path.join(BASE, args.db))

	words = json.load(open(os.path.join(BASE, "configs", "expanded_glyphs.json")))["words"]
	masks = json.load(open(os.path.join(BASE, "storage", "checkpoints", "bit003_glyph_v4_x1", "stage_gate_masks.json")))
	gate7 = {words[i] for i in masks["stage_allowed"][7]}

	targets = sorted(w for w in d.words_by_status(("pending", "molecule")) if w in gate7)
	print(f"objetivos (pending + molecule) en el universo gateado: {len(targets)}")

	drafted, no_pattern = pattern_explications(targets, gate7)

	# codificar + validar consistencia entre familias
	glyphs = {w: explication_to_glyph(f) for w, f in drafted.items()}
	familias = [(a, b, e) for a, b, e in [
		("dog", "cat", 0.5), ("dog", "wolf", 0.5), ("cat", "lion", 0.5),
		("chair", "table", 0.6), ("bark", "meow", 0.6)] if a in glyphs and b in glyphs]
	errs = consistency_report(glyphs, familias)
	for a, b, msg in errs:
		print(f"  ⚠️ inconsistencia: {a} ~ {b} → {msg}")

	# persistir: el glifo gana el ADN NSM (pending/molecule → drafted)
	# force=True: el programa NSM es el camino INTENCIONAL de upgrade — las
	# moléculas son provisionales (Ridge) y su destino es esta descomposición.
	for w, feats in drafted.items():
		d.upsert_word(w, glyph=explication_to_glyph(feats), status="drafted",
			source="nsm_pattern_v0", curated_by="aleth", force=True)

	# el resto → lote LLM (revisión del operador)
	llm_batch = no_pattern[: args.batch]
	review_path = os.path.join(BASE, "configs", "battery_v3", "nsm_llm_batch.json")
	json.dump({"instructions": "Para cada palabra: juicios NSM por primo (+afirma/-contrasta/0 silencio). El operador cura antes de promover a explicated.",
		"words": llm_batch}, open(review_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

	st = d.stats()
	print(f"\n── F5 ── drafted por patrón: {len(drafted)} (consistencia: {len(errs)} avisos) "
		f"| lote LLM: {len(llm_batch)} → {review_path}")
	print(f"    stats: {st}")


if __name__ == "__main__":
	main()
