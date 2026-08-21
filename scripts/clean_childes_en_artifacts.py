"""Limpia artefactos CHAT del corpus CHILDES-en (BIT-003, fix B5).

xxx/yyy/www son marcadores de transcripción CHAT (habla ininteligible), no
palabras: 'xxx' aparecía 50.894 veces (rank ~27 del censo) y E0 habría
entrenado a Bit a producirlo. Se eliminan como TOKEN completo (no subcadena),
se descartan las oraciones que queden con <2 tokens y se re-deduplica.
Idempotente: correrlo dos veces no cambia nada.
"""
import json
import os

ARTIFACTS = {"xxx", "yyy", "www", "xx", "yy"}


def main() -> None:
	base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
	path = os.path.join(base_dir, "configs", "childes_pre_school_en.json")
	with open(path, encoding="utf-8") as f:
		sentences = json.load(f)

	cleaned = []
	dropped_tokens = 0
	dropped_sentences = 0
	for s in sentences:
		tokens = s.split()
		keep = [t for t in tokens if t not in ARTIFACTS]
		dropped_tokens += len(tokens) - len(keep)
		if len(keep) < 2:
			dropped_sentences += 1
			continue
		cleaned.append(" ".join(keep))

	seen = set()
	unique = []
	for s in cleaned:
		if s not in seen:
			seen.add(s)
			unique.append(s)

	with open(path, "w", encoding="utf-8") as f:
		json.dump(unique, f, indent=4, ensure_ascii=False)
	print(f"✓ tokens artefacto eliminados: {dropped_tokens:,} | oraciones descartadas: {dropped_sentences:,} | únicas finales: {len(unique):,}")


if __name__ == "__main__":
	main()
