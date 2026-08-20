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
import re
from collections import Counter

import torch

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
			for w in re.findall(r"[a-zA-Z']+", text.lower()):
				if w in word_to_idx:
					words.add(w)
	return words


def build_stage_logit_mask(
	stage_idx: int,
	vocab_size: int,
	word_to_idx: dict,
	childes_freq: Counter | None,
	curriculum_words: set,
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
) -> list:
	return [
		build_stage_logit_mask(i, vocab_size, word_to_idx, childes_freq, curriculum_words_by_stage(curriculum_data, i, word_to_idx))
		for i in range(8)
	]


def apply_stage_gate(logits: torch.Tensor, stage_mask: torch.Tensor) -> torch.Tensor:
	"""Veta en los logits la emisión de palabras fuera del gateo de la etapa."""
	return logits + stage_mask.view(1, 1, -1)


def loss_mask_for_gate(mask: torch.Tensor, targets: torch.Tensor, stage_mask: torch.Tensor) -> torch.Tensor:
	"""Excluye de la pérdida las posiciones cuyo target está vetado en la etapa
	(cross_entropy con logit -inf en la posición objetivo → NaN; se evita)."""
	gate_ok = stage_mask[targets.clamp(min=0)] == 0.0
	return mask * gate_ok.to(mask.dtype)