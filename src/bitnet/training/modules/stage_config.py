def get_stage_config(base_epochs=64, stage_scale=0.5):
	stages = [
		{"name": "0-1", "dim": 128, "age": None},
		{"name": "1-2", "dim": 256, "age": 2},
		{"name": "2-3", "dim": 384, "age": 3},
		{"name": "3-4", "dim": 512, "age": 4},
		{"name": "primary_5", "dim": 640, "age": 5},
		{"name": "primary_6", "dim": 768, "age": 6},
		{"name": "secondary_7", "dim": 896, "age": 7},
		{"name": "secondary_8", "dim": 1024, "age": 8},
	]
	start_epoch = 1
	config = []
	for i, stage in enumerate(stages):
		epochs_in_stage = int(base_epochs * (1 + i * stage_scale))
		end_epoch = start_epoch + epochs_in_stage - 1
		config.append({
			"stage_idx": i,
			"name": stage["name"],
			"dim": stage["dim"],
			"age": stage["age"],
			"start_epoch": start_epoch,
			"end_epoch": end_epoch,
			"epochs": epochs_in_stage
		})
		start_epoch = end_epoch + 1
	return config


def get_next_dim(current_dim, stage_config, current_epoch=None):
	"""Return the next neurogenesis target dim, or None if already at max.

	Finds the smallest dim in stage_config that is strictly larger than
	current_dim. If current_epoch is given, respects the dim ceiling of
	the current stage (the model shouldn't grow beyond its stage's max).
	"""
	all_dims = sorted({c["dim"] for c in stage_config})

	# If epoch provided, cap at the current stage's dim
	max_allowed = all_dims[-1]
	if current_epoch is not None:
		for config in stage_config:
			if config["start_epoch"] <= current_epoch <= config["end_epoch"]:
				max_allowed = config["dim"]
				break

	for d in all_dims:
		if d > current_dim and d <= max_allowed:
			return d
	return None


def get_stage_info(epoch, stage_config):
	for config in stage_config:
		if config["start_epoch"] <= epoch <= config["end_epoch"]:
			return config["stage_idx"], config["name"]
	return stage_config[-1]["stage_idx"], stage_config[-1]["name"]
