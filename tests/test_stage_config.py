"""Tests para el módulo de configuración de etapas."""


from src.bitnet.training.modules.stage_config import (
	get_next_dim,
	get_stage_config,
	get_stage_info,
)


class TestGetStageConfig:
	"""Tests para get_stage_config."""

	def test_default_config(self):
		config = get_stage_config()
		assert len(config) == 8
		assert config[0]["dim"] == 128
		assert config[-1]["dim"] == 1024

	def test_custom_base_epochs(self):
		config = get_stage_config(base_epochs=128)
		# All stages should have more epochs
		for stage in config:
			assert stage["epochs"] > 64

	def test_stage_epochs_increasing(self):
		config = get_stage_config()
		for i in range(1, len(config)):
			assert config[i]["epochs"] >= config[i-1]["epochs"]


class TestGetNextDim:
	"""Tests para get_next_dim."""

	def test_next_dim_from_128(self):
		config = get_stage_config()
		next_dim = get_next_dim(128, config)
		assert next_dim == 256

	def test_next_dim_at_max(self):
		config = get_stage_config()
		next_dim = get_next_dim(1024, config)
		assert next_dim is None

	def test_next_dim_caps_at_stage_dim(self):
		config = get_stage_config()
		# At epoch 1 (stage 0, dim 128), can't grow beyond 128
		next_dim = get_next_dim(128, config, current_epoch=1)
		assert next_dim is None

	def test_progression_without_epoch(self):
		config = get_stage_config()
		dims = []
		current_dim = 128
		while True:
			next_dim = get_next_dim(current_dim, config)
			if next_dim is None:
				break
			dims.append(next_dim)
			current_dim = next_dim
		assert dims == [256, 384, 512, 640, 768, 896, 1024]


class TestGetStageInfo:
	"""Tests para get_stage_info."""

	def test_get_stage_info(self):
		config = get_stage_config()
		stage_idx, stage_name = get_stage_info(1, config)
		assert stage_idx == 0
		assert stage_name == "0-1"

	def test_get_stage_info_later_epoch(self):
		config = get_stage_config()
		# Find the start epoch of stage 3
		stage_3_start = config[3]["start_epoch"]
		stage_idx, stage_name = get_stage_info(stage_3_start, config)
		assert stage_idx == 3
		assert stage_name == "3-4"
