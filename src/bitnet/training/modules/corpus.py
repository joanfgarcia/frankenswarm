import hashlib
import json
import os


def compute_corpus_hash(base_dir: str, n_stories: int) -> str:
	"""Hash de los inputs del corpus para invalidar caché si cambian."""
	h = hashlib.sha256()
	# school_exams_en.json entra en el hash porque las secuencias de examen se
	# mezclan en el dataset de etapa (×300) y viven en la caché de etapa.
	for path in [
		os.path.join(base_dir, "configs", "expanded_glyphs.json"),
		os.path.join(base_dir, "configs", "tiny_dialogues_large_en.json"),
		os.path.join(base_dir, "configs", "school_curriculum_structured_en.json"),
		os.path.join(base_dir, "configs", "school_exams_en.json"),
	]:
		with open(path, "rb") as f:
			h.update(f.read())
	h.update(str(n_stories).encode())
	return h.hexdigest()[:16]


def load_tokenized_cache(cache_path: str, expected_hash: str) -> dict | None:
	"""Carga caché tokenizado si existe y el hash coincide."""
	if not os.path.exists(cache_path):
		return None
	with open(cache_path) as f:
		data = json.load(f)
	if data.get("hash") != expected_hash:
		return None
	return data


def save_tokenized_cache(cache_path: str, data: dict) -> None:
	"""Guarda caché tokenizado a disco."""
	with open(cache_path, "w") as f:
		json.dump(data, f)
