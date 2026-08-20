import torch


def partition_corpus_by_mlu(sequences: list[list[int]]) -> tuple:
	stage_0_1 = []
	stage_1_2 = []
	stage_2_3 = []
	stage_3_4 = []
	for seq in sequences:
		length = len(seq)
		if length <= 3:
			stage_0_1.append(seq)
		elif length == 4:
			stage_1_2.append(seq)
		elif 5 <= length <= 6:
			stage_2_3.append(seq)
		else:
			stage_3_4.append(seq)
	return stage_0_1, stage_1_2, stage_2_3, stage_3_4


def compile_stage_dataset(base_data: list[list[int]], seq_len: int | None = None) -> tuple:
	# seq_len=None → sin padding (secuencias crudas); el bucketing por longitud
	# en entrenamiento evita pagar 128 tokens por oración de ~9 (RFC-VRAM-001/BIT-003).
	if seq_len is None:
		items = list(base_data)
	else:
		items = [seq[:seq_len] + [0] * (seq_len - len(seq)) for seq in base_data]

	import random
	rng = random.Random(42)
	rng.shuffle(items)

	val_size = int(len(items) * 0.1)
	train_seqs = items[val_size:]
	val_seqs = items[:val_size]

	return train_seqs, val_seqs


BUCKET_EDGES = (2, 8, 16, 32, 64)


def bucketize_batches(
	sequences: list[list[int]],
	batch_size: int = 64,
	pad_token: int = 0,
	seed: int = 42,
	edges: tuple = BUCKET_EDGES,
) -> list:
	"""Agrupa secuencias por buckets de longitud y devuelve una lista de batches.

	Cada batch es un tensor (B, L) donde L es la longitud máxima DENTRO del batch
	(padding dinámico intra-bucket). Elimina el ~90% de padding que pagaba el
	esquema anterior (todo padded a 128) — hace viable TinyStories completo sin
	tocar la semántica del entrenamiento (la máscara de pérdida sigue
	enmascarando el padding final de cada fila).
	"""
	import random

	buckets = [[] for _ in range(len(edges) - 1)]
	for seq in sequences:
		L = len(seq)
		for bi in range(len(edges) - 1):
			if edges[bi] <= L <= edges[bi + 1]:
				buckets[bi].append(seq)
				break

	rng = random.Random(seed)
	batches: list = []
	for bucket in buckets:
		rng.shuffle(bucket)
		for i in range(0, len(bucket), batch_size):
			chunk = bucket[i : i + batch_size]
			max_len = max(len(s) for s in chunk)
			padded = [s + [pad_token] * (max_len - len(s)) for s in chunk]
			batches.append(torch.tensor(padded, dtype=torch.long))
	rng.shuffle(batches)
	return batches


def oversample_curriculum_for_stage(general_data, curriculum_data, target_ratio=0.20):
	if len(curriculum_data) == 0:
		return general_data
	target_size = int(len(general_data) * target_ratio / (1 - target_ratio))
	multiplier = target_size // len(curriculum_data)
	remainder = target_size % len(curriculum_data)
	oversampled = curriculum_data * multiplier + curriculum_data[:remainder]
	return general_data + oversampled
