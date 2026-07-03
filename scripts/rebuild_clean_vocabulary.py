"""Rebuild Bit's vocabulary from CLEAN sources (School v3 route change, 2026-07-03).

Root cause being fixed: the 3k→15k expansion imported a generic Spanish frequency
list (subtitle-derived), leaving ~80% of expanded_glyphs.json as ghost words never
seen in training but with live glyphs and logits ("interpol", "hitler"...). Free
generation sampled straight into that contaminated tail.

This script:
1. Censuses every word actually used by the clean corpus sources.
2. Diffs against the current vocabulary and reports the ghost ratio.
3. Emits configs/clean_vocabulary_words.json — the wordlist that
   src/bitnet/expand_vocabulary.py must consume as FINAL_VOCAB to re-derive glyphs.

It does NOT derive glyphs itself (fastembed download + Ridge projection stay in
expand_vocabulary.py). Next step is printed at the end.
"""

import json
import os
import re
import sys
from collections import Counter
from datetime import UTC, datetime

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONTROL_TOKENS = ["<pad>", "<unk>"]
DIALOGUE_TOKENS = ["yo", "tú"]

CLEAN_SOURCES = [
	"configs/childes_pre_school.json",
	"configs/nsm_physics_pre_school.json",
	"configs/school_curriculum_structured.json",
	"configs/school_exams.json",
	"configs/tiny_dialogues_large.json",
]

FACTORY_SOURCES_GLOB = "storage/curriculum/factory"


def words_of(text: str) -> list[str]:
	return re.findall(r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ\-]+", str(text).lower())


def texts_from(path: str) -> list[str]:
	with open(path, encoding="utf-8") as f:
		data = json.load(f)
	if isinstance(data, list):
		out = []
		for item in data:
			out.extend(item if isinstance(item, list) else [str(item)])
		return out
	if isinstance(data, dict) and "curriculum" in data:
		return [it["text"] for stage in data["curriculum"].values() for it in stage]
	texts = []

	def walk(node):
		if isinstance(node, str):
			texts.append(node)
		elif isinstance(node, list):
			for x in node:
				walk(x)
		elif isinstance(node, dict):
			for x in node.values():
				walk(x)

	walk(data)
	return texts


def main():
	freq: Counter = Counter()
	print("── CENSO DE FUENTES LIMPIAS ──")
	for rel in CLEAN_SOURCES:
		path = os.path.join(base_dir, rel)
		if not os.path.exists(path):
			print(f"  ⚠️ {rel}: no existe, se omite")
			continue
		n_before = sum(freq.values())
		for t in texts_from(path):
			freq.update(words_of(t))
		print(f"  {rel}: +{sum(freq.values()) - n_before:,} tokens")

	factory_dir = os.path.join(base_dir, FACTORY_SOURCES_GLOB)
	if os.path.isdir(factory_dir):
		for fn in sorted(os.listdir(factory_dir)):
			if fn.endswith(".jsonl"):
				with open(os.path.join(factory_dir, fn), encoding="utf-8") as f:
					for line in f:
						try:
							freq.update(words_of(json.loads(line).get("text", "")))
						except json.JSONDecodeError:
							continue
				print(f"  factory/{fn}: incorporado")

	min_count = int(sys.argv[sys.argv.index("--min-count") + 1]) if "--min-count" in sys.argv else 1
	clean_words = sorted(w for w, c in freq.items() if c >= min_count)
	vocab = CONTROL_TOKENS + DIALOGUE_TOKENS + [w for w in clean_words if w not in DIALOGUE_TOKENS]

	print(f"\n  Palabras únicas en fuentes limpias (freq ≥ {min_count}): {len(clean_words):,}")

	current_path = os.path.join(base_dir, "configs/expanded_glyphs.json")
	if os.path.exists(current_path):
		with open(current_path, encoding="utf-8") as f:
			current_words = set(json.load(f)["words"])
		used = current_words & set(vocab)
		ghosts = current_words - set(vocab)
		print("\n── DIFF CONTRA VOCABULARIO ACTUAL ──")
		print(f"  vocabulario actual: {len(current_words):,} palabras")
		print(f"  usadas por el corpus: {len(used):,} ({len(used) / len(current_words) * 100:.1f}%)")
		print(f"  FANTASMAS (nunca vistas en entrenamiento): {len(ghosts):,} ({len(ghosts) / len(current_words) * 100:.1f}%)")
		sample = [w for w in ("interpol", "hitler", "sexualmente", "guarida", "contrólate") if w in ghosts]
		if sample:
			print(f"  contaminación confirmada entre fantasmas: {sample}")

	out = {
		"metadata": {
			"generated": datetime.now(UTC).astimezone().isoformat(),
			"sources": CLEAN_SOURCES + [FACTORY_SOURCES_GLOB + "/*.jsonl"],
			"min_count": min_count,
			"note": "School v3 clean vocabulary. Feed as FINAL_VOCAB to expand_vocabulary.calibrate_and_project().",
		},
		"words": vocab,
	}
	out_path = os.path.join(base_dir, "configs/clean_vocabulary_words.json")
	with open(out_path, "w", encoding="utf-8") as f:
		json.dump(out, f, indent=2, ensure_ascii=False)
	print(f"\n✅ Escrito {out_path} ({len(vocab):,} palabras)")
	print("\nSIGUIENTE PASO (deriva los glifos limpios):")
	print("  1. Apuntar FINAL_VOCAB en src/bitnet/expand_vocabulary.py a configs/clean_vocabulary_words.json")
	print("  2. PYTHONPATH=. .venv/bin/python -c 'from src.bitnet.expand_vocabulary import calibrate_and_project; calibrate_and_project()'")


if __name__ == "__main__":
	main()
