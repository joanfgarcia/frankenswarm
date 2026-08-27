"""Gateo de vocabulario por etapa curricular (BIT-003).

Determina qué tokens el modelo PUEDE PRODUCIR en cada etapa (logit_mask):
solo el vocabulario de su edad. El input sigue viendo el corpus completo en
contexto — como un niño que escucha palabras que aún no sabe decir. La pérdida
se excluye en posiciones cuyo target está fuera del gateo (no se aprende a
predecir lo prohibido).

PROPIEDAD DE TESIS (glyph vs standard): el gateo se puede AMPLIAR EN CALIENTE.
Con glifos una palabra nueva se compone de primos ya aprendidos → basta
desbloquear su token en la máscara. Con standard (tabla one-hot congelada) su
columna de embedding no ha sido entrenada → no se puede añadir sin reentrenar.
"""
from collections import Counter

import torch

from src.bitnet.training.modules.tokenization import words_of

# Vocabulario de referencia EN (los 28 glifos de V 0-7, mapeados por VOCAB_MAP)
REFERENCE_WORDS_EN = {
	"water", "food", "fire", "sun", "night", "cave", "i", "danger", "eat",
	"drink", "move", "see", "sleep", "give", "teach", "learn", "forest",
	"river", "stone", "tree", "earth", "rain", "predator", "storm", "wound",
	"safe", "full", "group",
}

# Palabras base de función (lenguaje del día a día infantil)
SAFE_WORDS_EN = {
	"no", "you", "he", "she", "we", "they", "my", "your", "his", "her", "its",
	"our", "their", "me", "him", "us", "them", "it", "this", "that", "these",
	"those", "a", "an", "the", "to", "of", "in", "for", "on", "with",
	"without", "about", "and", "or", "but", "if", "because", "is", "are",
	"was", "were", "be", "been", "have", "has", "had", "do", "does", "did",
	"want", "can", "say", "see", "go", "give", "know", "eat", "drink", "meow",
	"bark", "hurt", "hello", "fine", "good", "dad", "mom", "baby", "kid",
	"kiss", "take", "more", "sleep", "runs", "much", "very",
}

# Tokens del vocab gateados por etapa: top-N palabras de CHILDES-en (habla
# infantil real por frecuencia de adquisición) acumulado en las 8 etapas.
TOP_N_BY_STAGE = [200, 800, 3000, 5000, 8000, 10000, 15000, 20095]

PAD_TOKEN = 0
UNK_TOKEN = 1


def curriculum_words_by_stage(curriculum_data: dict, stage_idx: int, word_to_idx: dict) -> set:
	"""Palabras del currículo acumulado de la etapa (asegura que los targets de
	examen estén siempre dentro del gateo de su etapa)."""
	stages = []
	if stage_idx <= 3:
		stages = ["preschool"]
	elif stage_idx <= 5:
		stages = ["preschool", "primary"]
	else:
		stages = ["preschool", "primary", "secondary"]

	words = set()
	for stage in stages:
		for text in curriculum_data.get(stage, []):
			for w in words_of(text):
				if w in word_to_idx:
					words.add(w)
	return words


def build_stage_logit_mask(
	stage_idx: int,
	vocab_size: int,
	word_to_idx: dict,
	childes_freq: Counter | None,
	curriculum_words: set,
	exam_words: set | None = None,
) -> torch.Tensor:
	"""Máscara de logits (0 = permitido, -inf = vetado) para la etapa dada."""
	mask = torch.full((vocab_size,), float("-inf"))
	mask[PAD_TOKEN] = 0.0
	# <unk> vetado: no se aprende a emitirlo (ruido puro en generación)

	allowed = set(REFERENCE_WORDS_EN) | SAFE_WORDS_EN
	if childes_freq is not None:
		n = TOP_N_BY_STAGE[stage_idx]
		allowed.update(w for w, _ in childes_freq.most_common(n))
	allowed.update(curriculum_words)
	# Respuestas de examen de la etapa: todo target examinado debe ser
	# producible en la etapa que lo examina (BIT-003 S5, DL-009).
	allowed.update(exam_words or set())

	for w in allowed:
		idx = word_to_idx.get(w)
		if idx is not None:
			mask[idx] = 0.0
	return mask


def build_all_stage_masks(
	vocab_size: int,
	word_to_idx: dict,
	childes_freq: Counter,
	curriculum_data: dict,
	exam_words_by_stage: list | None = None,
) -> list:
	exam_words_by_stage = exam_words_by_stage or [set()] * 8
	masks = [
		build_stage_logit_mask(i, vocab_size, word_to_idx, childes_freq, curriculum_words_by_stage(curriculum_data, i, word_to_idx), exam_words_by_stage[i])
		for i in range(8)
	]
	# La etapa final cubre TODO el vocabulario real: toda frase del corpus debe
	# ser clasificable (ninguna se pierde, solo se reordena por etapa). <unk>
	# sigue vetado también en E7 (salvaguarda DL-008: ruido de generación).
	final_mask = torch.zeros(vocab_size)
	final_mask[UNK_TOKEN] = float("-inf")
	masks[-1] = final_mask
	return masks


def token_min_stage(masks: list, vocab_size: int) -> list:
	"""Etapa mínima de cada token (la menor etapa cuyo gateo lo permite)."""
	import numpy as np

	arr = [None] * vocab_size
	for t in range(vocab_size):
		for st, m in enumerate(masks):
			if m[t] == 0.0:
				arr[t] = st
				break
	return arr


def classify_store_by_gate(
	flat,
	offsets,
	token_min_stage_arr: list,
	threshold: float = 0.95,
	num_stages: int = 8,
) -> list:
	"""Versión CSR de classify_sequences_by_gate: itera las vistas del store
	(listas materializadas por secuencia) y devuelve 8 listas de índices.
	La lógica por frase es IDÉNTICA a la versión sobre listas — los conteos
	deben reproducir exactamente los de la clasificación original."""
	import math

	groups: list = [[] for _ in range(num_stages)]
	tms = token_min_stage_arr
	for i in range(len(offsets) - 1):
		seq = flat[offsets[i] : offsets[i + 1]].tolist()
		known = [tms[t] for t in seq if tms[t] is not None]
		if not known or len(known) / len(seq) < threshold:
			groups[-1].append(i)
			continue
		stages = sorted(known)
		k = min(len(stages) - 1, math.ceil(len(seq) * threshold) - 1)
		groups[stages[k]].append(i)
	return groups


def build_stage_pools(
	flat,
	offsets,
	stage_masks: list,
	vocab_size: int,
	pools_path: str,
	corpus_hash: str,
	threshold: float = 0.95,
) -> list:
	"""Pools de índices por etapa del gateo (E0..E7), con caché en disco.

	Los pools son deterministas dado el store y el corpus_hash (que cubre
	vocab/currículo/exámenes de los que dependen las máscaras), así que se
	persisten para no repetir la clasificación (~4 min) en cada arranque."""
	import numpy as np
	import os

	if os.path.exists(pools_path):
		data = np.load(pools_path)
		if str(data["meta"]) == corpus_hash:
			return [data[f"g{i}"] for i in range(8)]

	tms = token_min_stage(stage_masks, vocab_size)
	groups = classify_store_by_gate(flat, offsets, tms, threshold=threshold)
	pools = [np.array(g, dtype=np.int64) for g in groups]
	np.savez(pools_path, meta=np.array(corpus_hash), **{f"g{i}": p for i, p in enumerate(pools)})
	return pools


def materialize_sequences(flat, offsets, indices, base: int = 0) -> list:
	"""Materializa como listas las secuencias indicadas del store (transitorio:
	solo lo muestreado por época vive como listas de Python)."""
	out: list = []
	for i in indices:
		j = int(i) + base
		out.append(flat[offsets[j] : offsets[j + 1]].tolist())
	return out


def classify_sequences_by_gate(
	sequences: list,
	token_min_stage_arr: list,
	threshold: float = 0.95,
	num_stages: int = 8,
) -> list:
	"""Clasifica cada frase en su etapa mínima según el gateo (umbral de
	cobertura de tokens). La frase entra en la etapa más temprana que cubre
	≥threshold de sus tokens. Las frases con <umbral de tokens conocidos (o
	con vocabulario tardío) se asignan a la última etapa."""
	import math

	groups: list = [[] for _ in range(num_stages)]
	for seq in sequences:
		known = [token_min_stage_arr[t] for t in seq if token_min_stage_arr[t] is not None]
		if not known or len(known) / len(seq) < threshold:
			groups[-1].append(seq)
			continue
		stages = sorted(known)
		k = min(len(stages) - 1, math.ceil(len(seq) * threshold) - 1)
		groups[stages[k]].append(seq)
	return groups


def apply_stage_gate(logits: torch.Tensor, stage_mask: torch.Tensor) -> torch.Tensor:
	"""Veta en los logits la emisión de palabras fuera del gateo de la etapa."""
	return logits + stage_mask.view(1, 1, -1)


def loss_mask_for_gate(mask: torch.Tensor, targets: torch.Tensor, stage_mask: torch.Tensor) -> torch.Tensor:
	"""Excluye de la pérdida las posiciones cuyo target está vetado en la etapa
	(cross_entropy con logit -inf en la posición objetivo → NaN; se evita)."""
	gate_ok = stage_mask[targets.clamp(min=0)] == 0.0
	return mask * gate_ok.to(mask.dtype)