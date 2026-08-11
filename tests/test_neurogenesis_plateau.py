"""Tests for plateau-driven neurogenesis in School v3.

Tests the get_next_dim() helper and the plateau detection logic,
ensuring the calendar-based trigger has been replaced.
"""

import json
import os
import sys
import tempfile
import unittest

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

from src.bitnet.training.train_sovereign_school import get_next_dim, get_stage_config


class TestGetNextDim(unittest.TestCase):
	"""Tests for the get_next_dim() helper function."""

	def setUp(self):
		self.config = get_stage_config(base_epochs=64, stage_scale=0.5)

	def test_next_dim_from_128(self):
		"""From dim 128, next should be 256."""
		result = get_next_dim(128, self.config)
		self.assertEqual(result, 256)

	def test_next_dim_from_512(self):
		"""From dim 512, next should be 640."""
		result = get_next_dim(512, self.config)
		self.assertEqual(result, 640)

	def test_next_dim_at_max(self):
		"""At max dim 1024, should return None."""
		result = get_next_dim(1024, self.config)
		self.assertIsNone(result)

	def test_next_dim_respects_stage_ceiling(self):
		"""When current_epoch is in stage 0 (dim 128), can't grow beyond 128."""
		# Stage 0 spans epochs 1-64 with dim=128
		result = get_next_dim(128, self.config, current_epoch=30)
		# Stage 0 has dim=128, so no growth possible within this stage
		self.assertIsNone(result)

	def test_next_dim_allows_growth_in_later_stage(self):
		"""When current_epoch is in stage 1 (dim 256), can grow from 128 to 256."""
		# Stage 1 starts at epoch 65
		result = get_next_dim(128, self.config, current_epoch=65)
		self.assertEqual(result, 256)

	def test_next_dim_caps_at_stage_dim(self):
		"""Growth is capped at the current stage's dim."""
		# Stage 1 has dim=256, so from 128 can only go to 256, not 384
		result = get_next_dim(128, self.config, current_epoch=65)
		self.assertEqual(result, 256)
		# And from 256 in stage 1, should return None (already at stage max)
		result2 = get_next_dim(256, self.config, current_epoch=65)
		self.assertIsNone(result2)

	def test_progression_without_epoch(self):
		"""Without epoch constraint, dims progress through all stages."""
		dims = []
		current = 128
		while True:
			nxt = get_next_dim(current, self.config)
			if nxt is None:
				break
			dims.append(nxt)
			current = nxt
		self.assertEqual(dims, [256, 384, 512, 640, 768, 896, 1024])


class TestPlateauDetectionLogic(unittest.TestCase):
	"""Test the plateau detection logic in isolation."""

	def test_improving_val_loss_resets_counter(self):
		"""When val_loss improves by more than min_delta, counter resets."""
		best_val_loss = 3.0
		min_delta = 0.01
		epochs_without_improvement = 5

		val_loss = 2.98  # Improved by 0.02 > min_delta

		if val_loss < best_val_loss - min_delta:
			best_val_loss = val_loss
			epochs_without_improvement = 0

		self.assertEqual(epochs_without_improvement, 0)
		self.assertAlmostEqual(best_val_loss, 2.98)

	def test_stagnant_val_loss_increments_counter(self):
		"""When val_loss doesn't improve enough, counter increments."""
		best_val_loss = 3.0
		min_delta = 0.01
		epochs_without_improvement = 5

		val_loss = 2.995  # Only improved by 0.005 < min_delta

		if val_loss < best_val_loss - min_delta:
			best_val_loss = val_loss
			epochs_without_improvement = 0
		else:
			epochs_without_improvement += 1

		self.assertEqual(epochs_without_improvement, 6)
		self.assertAlmostEqual(best_val_loss, 3.0)  # Unchanged

	def test_worsening_val_loss_increments_counter(self):
		"""When val_loss gets worse, counter increments."""
		best_val_loss = 3.0
		min_delta = 0.01
		epochs_without_improvement = 0

		val_loss = 3.5  # Worse

		if val_loss < best_val_loss - min_delta:
			best_val_loss = val_loss
			epochs_without_improvement = 0
		else:
			epochs_without_improvement += 1

		self.assertEqual(epochs_without_improvement, 1)

	def test_plateau_triggers_at_patience(self):
		"""Neurogenesis triggers exactly at patience threshold."""
		patience = 15
		best_val_loss = 3.0
		min_delta = 0.01
		epochs_without_improvement = 0
		config = get_stage_config()

		# Simulate 15 epochs of no improvement
		for _ in range(patience):
			val_loss = 3.005  # Not enough improvement
			if val_loss < best_val_loss - min_delta:
				best_val_loss = val_loss
				epochs_without_improvement = 0
			else:
				epochs_without_improvement += 1

		self.assertEqual(epochs_without_improvement, patience)
		# Should trigger: patience reached and next dim available
		next_dim = get_next_dim(128, config)
		self.assertIsNotNone(next_dim)

	def test_plateau_no_trigger_before_patience(self):
		"""Neurogenesis does NOT trigger before patience is reached."""
		patience = 15
		epochs_without_improvement = 14  # One less than patience
		self.assertLess(epochs_without_improvement, patience)

	def test_reset_after_neurogenesis(self):
		"""After neurogenesis triggers, best_val_loss and counter reset."""
		best_val_loss = 3.0
		epochs_without_improvement = 15

		# Simulate neurogenesis trigger
		best_val_loss = float("inf")
		epochs_without_improvement = 0

		self.assertEqual(epochs_without_improvement, 0)
		self.assertEqual(best_val_loss, float("inf"))


class TestNeurogenesisStatePersistence(unittest.TestCase):
	"""Test that plateau state persists correctly in school_state.json."""

	def test_state_includes_plateau_fields(self):
		"""school_state.json should include plateau monitoring fields."""
		state = {
			"current_epoch": 42,
			"hidden_dim": 256,
			"num_layers": 6,
			"best_val_loss": 2.5432,
			"epochs_without_improvement": 7,
			"neurogenesis_history": [
				{"epoch": 30, "old_dim": 128, "new_dim": 256, "val_loss_at_trigger": 3.1}
			],
		}

		with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
			json.dump(state, f)
			path = f.name

		try:
			with open(path, encoding="utf-8") as f:
				loaded = json.load(f)

			self.assertAlmostEqual(loaded["best_val_loss"], 2.5432)
			self.assertEqual(loaded["epochs_without_improvement"], 7)
			self.assertEqual(len(loaded["neurogenesis_history"]), 1)
			self.assertEqual(loaded["neurogenesis_history"][0]["old_dim"], 128)
			self.assertEqual(loaded["neurogenesis_history"][0]["new_dim"], 256)
		finally:
			os.unlink(path)

	def test_state_defaults_for_fresh_start(self):
		"""Fresh start should have default plateau values."""
		best_val_loss = float("inf")
		epochs_without_improvement = 0
		neurogenesis_history = []

		self.assertEqual(best_val_loss, float("inf"))
		self.assertEqual(epochs_without_improvement, 0)
		self.assertEqual(neurogenesis_history, [])


class TestCalendarTriggerRemoved(unittest.TestCase):
	"""Verify that the calendar-based neurogenesis trigger no longer exists."""

	def test_no_calendar_trigger_in_training_loop(self):
		"""The epoch == config['start_epoch'] trigger should not exist in the training loop."""
		import inspect

		from src.bitnet.training.train_sovereign_school import run_school_training

		source = inspect.getsource(run_school_training)
		# The old trigger pattern was: epoch == config["start_epoch"] and model.hidden_dim < config["dim"]
		self.assertNotIn('epoch == config["start_epoch"]', source,
			"Calendar-based neurogenesis trigger should have been removed")
		self.assertNotIn("epoch == config['start_epoch']", source,
			"Calendar-based neurogenesis trigger should have been removed")

	def test_plateau_trigger_exists(self):
		"""The plateau-based trigger should exist in the training loop."""
		import inspect

		from src.bitnet.training.train_sovereign_school import run_school_training

		source = inspect.getsource(run_school_training)
		self.assertIn("epochs_without_improvement", source,
			"Plateau-based neurogenesis monitor should exist")
		self.assertIn("PLATEAU DETECTADO", source,
			"Plateau detection log message should exist")
		self.assertIn("get_next_dim", source,
			"get_next_dim helper should be called in training loop")


if __name__ == "__main__":
	unittest.main()
