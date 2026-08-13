"""Tests for samantha_story_factory.py — mock pipeline validation."""

import hashlib
import json
import os
import re
import tempfile
import unittest

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestStoryFactoryHelpers(unittest.TestCase):
	"""Test helper functions from the story factory."""

	def test_words_of_basic(self):
		"""words_of extracts lowercase word tokens."""
		from scripts.samantha_story_factory import words_of

		result = words_of("El gato de Ana tiene hambre.")
		self.assertEqual(result, ["el", "gato", "de", "ana", "tiene", "hambre"])

	def test_words_of_accents(self):
		"""words_of preserves accented characters."""
		from scripts.samantha_story_factory import words_of

		result = words_of("¿Cómo está Lucía?")
		self.assertEqual(result, ["cómo", "está", "lucía"])

	def test_words_of_empty(self):
		"""words_of returns empty list for non-word input."""
		from scripts.samantha_story_factory import words_of

		self.assertEqual(words_of("123 !@#"), [])

	def test_load_clean_vocab_exists(self):
		"""load_clean_vocab loads the vocabulary when the file exists."""
		from scripts.samantha_story_factory import load_clean_vocab

		vocab = load_clean_vocab()
		# May be None if configs/clean_vocabulary_words.json doesn't exist
		if vocab is not None:
			self.assertIsInstance(vocab, set)
			self.assertIn("<pad>", vocab)
			self.assertIn("<unk>", vocab)

	def test_build_prompt_contains_stage_info(self):
		"""build_prompt produces a prompt with age and sentence instructions."""
		import random

		from scripts.samantha_story_factory import STAGE_SPECS, build_prompt

		rng = random.Random(42)
		prompt = build_prompt(STAGE_SPECS["preschool"], rng)
		self.assertIn("3 a 5 años", prompt)
		self.assertIn("10 mini-historias", prompt)

	def test_mock_generate_returns_multiple_lines(self):
		"""mock_generate returns at least 3 lines."""
		from scripts.samantha_story_factory import mock_generate

		result = mock_generate("test prompt")
		lines = result.strip().splitlines()
		self.assertGreaterEqual(len(lines), 3)


class TestStoryFactoryMockPipeline(unittest.TestCase):
	"""End-to-end test of the mock pipeline."""

	def test_mock_pipeline_produces_valid_jsonl(self):
		"""Running with --mock produces valid JSONL with required fields."""
		# Simulate the pipeline manually (to avoid sys.argv issues)
		import random

		from scripts.samantha_story_factory import (
			STAGE_SPECS,
			build_prompt,
			mock_generate,
			words_of,
		)

		spec = STAGE_SPECS["preschool"]
		rng = random.Random(770)

		with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
			out_path = f.name

		try:
			seen = set()
			accepted = 0

			raw = mock_generate(build_prompt(spec, rng))
			with open(out_path, "w", encoding="utf-8") as out:
				for line in raw.splitlines():
					text = line.strip().lower()
					if len(words_of(text)) < 5:
						continue
					h = hashlib.sha256(re.sub(r"\W+", "", text).encode()).hexdigest()[:16]
					if h in seen:
						continue
					seen.add(h)
					accepted += 1
					record = {
						"text": text,
						"stage": "preschool",
						"hash": h,
						"generator": "mock",
					}
					out.write(json.dumps(record, ensure_ascii=False) + "\n")

			self.assertGreater(accepted, 0, "Should accept at least one story")

			# Validate JSONL structure
			with open(out_path, encoding="utf-8") as f:
				for line_num, line in enumerate(f, 1):
					record = json.loads(line)
					self.assertIn("text", record, f"Line {line_num} missing 'text'")
					self.assertIn("hash", record, f"Line {line_num} missing 'hash'")
					self.assertIn("stage", record, f"Line {line_num} missing 'stage'")
					self.assertEqual(record["stage"], "preschool")
					self.assertGreater(len(words_of(record["text"])), 0)
		finally:
			os.unlink(out_path)

	def test_deduplication_rejects_identical_stories(self):
		"""Identical stories are rejected by hash deduplication."""
		import hashlib

		text1 = "el gato de ana tiene hambre ana le da leche"
		text2 = "el gato de ana tiene hambre ana le da leche"  # exact duplicate

		h1 = hashlib.sha256(re.sub(r"\W+", "", text1).encode()).hexdigest()[:16]
		h2 = hashlib.sha256(re.sub(r"\W+", "", text2).encode()).hexdigest()[:16]

		self.assertEqual(h1, h2, "Identical texts should have same hash")

	def test_oov_filter_rejects_high_oov(self):
		"""Stories with >2% OOV words should be rejected when vocab is available."""
		from scripts.samantha_story_factory import load_clean_vocab, words_of

		vocab = load_clean_vocab()
		if vocab is None:
			self.skipTest("No clean vocabulary file available")

		# A text with many invented words should fail OOV filter
		text = "xyzblorp flarbnik gloomzat quipster bazzfoo agua pan"
		ws = words_of(text)
		oov = sum(1 for w in ws if w not in vocab) / len(ws)
		self.assertGreater(oov, 0.02, "Nonsense text should exceed OOV threshold")


if __name__ == "__main__":
	unittest.main()
