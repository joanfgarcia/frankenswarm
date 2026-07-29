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


def compile_stage_dataset(base_data: list[list[int]], seq_len: int = 128) -> tuple:
	padded = []
	for seq in base_data:
		if len(seq) < seq_len:
			seq_padded = seq + [0] * (seq_len - len(seq))
		else:
			seq_padded = seq[:seq_len]
		padded.append(seq_padded)

	import random
	rng = random.Random(42)
	rng.shuffle(padded)

	val_size = int(len(padded) * 0.1)
	train_seqs = padded[val_size:]
	val_seqs = padded[:val_size]

	return train_seqs, val_seqs


def oversample_curriculum_for_stage(general_data, curriculum_data, target_ratio=0.20):
	if len(curriculum_data) == 0:
		return general_data
	target_size = int(len(general_data) * target_ratio / (1 - target_ratio))
	multiplier = target_size // len(curriculum_data)
	remainder = target_size % len(curriculum_data)
	oversampled = curriculum_data * multiplier + curriculum_data[:remainder]
	return general_data + oversampled
