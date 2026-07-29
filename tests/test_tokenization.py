"""Tests para el módulo de tokenización."""

import pytest

from src.bitnet.training.modules.tokenization import (
	format_and_tokenize_dialogue,
	generate_question_variations,
	tokenize,
)


class TestTokenize:
	"""Tests para la función tokenize."""

	def test_basic_tokenization(self):
		word_to_idx = {"hello": 0, "world": 1}
		tokens = tokenize("hello world", word_to_idx)
		assert tokens == [0, 1]

	def test_unknown_word_returns_1(self):
		word_to_idx = {"hello": 0}
		tokens = tokenize("hello unknown", word_to_idx)
		assert tokens == [0, 1]  # 1 is <unk>

	def test_empty_text(self):
		word_to_idx = {"hello": 0}
		tokens = tokenize("", word_to_idx)
		assert tokens == []

	def test_special_characters(self):
		word_to_idx = {"hello": 0}
		tokens = tokenize("hello!", word_to_idx)
		assert tokens == [0]


class TestGenerateQuestionVariations:
	"""Tests para generate_question_variations."""

	def test_basic_variation(self):
		variations = generate_question_variations("el gato")
		assert len(variations) >= 2
		assert "el gato" in variations

	def test_determinant_change(self):
		variations = generate_question_variations("el gato")
		# Should have "un gato" as a variation
		assert any("un gato" in v for v in variations)

	def test_remove_accents(self):
		variations = generate_question_variations("corazón")
		# Should have "corazon" as a variation
		assert any("corazon" in v for v in variations)


class TestFormatAndTokenizeDialogue:
	"""Tests para format_and_tokenize_dialogue."""

	def test_spanish_dialogue(self):
		word_to_idx = {"yo": 0, "hola": 1, "tú": 2, "adiós": 3}
		dialogue = ["yo: hola", "tú: adiós"]
		tokens = format_and_tokenize_dialogue(dialogue, word_to_idx)
		# Should include speaker tokens
		assert len(tokens) > 0

	def test_english_dialogue(self):
		word_to_idx = {"me": 0, "hello": 1, "you": 2, "goodbye": 3}
		dialogue = ["me: hello", "you: goodbye"]
		tokens = format_and_tokenize_dialogue(dialogue, word_to_idx)
		assert len(tokens) > 0

	def test_empty_dialogue(self):
		word_to_idx = {"hello": 0}
		tokens = format_and_tokenize_dialogue([], word_to_idx)
		assert tokens == []
