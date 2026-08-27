import hashlib
import json
import os

import numpy as np


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
		os.path.join(base_dir, "configs", "childes_pre_school_en.json"),
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


def save_tokenized_store(store_path: str, expected_hash: str, flat: np.ndarray, offsets: np.ndarray, n_ts: int, n_dial: int) -> None:
	"""Guarda el corpus como CSR (flat int32 + offsets int64) en un .npz.

	Sustituye al cache JSON: el json.load de 2.48 GB construía ~18-20 GB de
	objetos Python (oom-kill del scope de 16G). Los arrays se cargan de golpe
	en ~1.9 GB y el acceso por secuencia es una vista flat[a:b].
	"""
	np.savez(
		store_path,
		flat=flat,
		offsets=offsets,
		n_ts=np.int64(n_ts),
		n_dial10=np.int64(n_dial),
		meta=np.array(expected_hash),
	)


def load_tokenized_store(store_path: str, expected_hash: str) -> dict | None:
	"""Carga el store CSR si existe y el hash coincide. Devuelve
	{flat, offsets, n_ts, n_dial} o None."""
	if not os.path.exists(store_path):
		return None
	data = np.load(store_path)
	if str(data["meta"]) != expected_hash:
		return None
	return {
		"flat": data["flat"],
		"offsets": data["offsets"],
		"n_ts": int(data["n_ts"]),
		"n_dial10": int(data["n_dial10"]),
	}


def store_len(offsets: np.ndarray) -> int:
	return len(offsets) - 1
