import json
import os
import unittest

import numpy as np
import torch

from src.bitnet.dictionary_tool import SovereignDictionary
from src.bitnet.modeling_bitnet import BitNet4LayerModel


class TestDictionaryLearning(unittest.TestCase):
	def setUp(self):
		self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
		self.expanded_glyphs_path = os.path.join(self.base_dir, "configs", "expanded_glyphs.json")
		
		# Cargar vocabulario base para verificar pertenencia
		with open(self.expanded_glyphs_path, encoding="utf-8") as f:
			data = json.load(f)
			self.base_vocab = set(data["words"])
			self.glyphs = np.array(data["glyphs"], dtype=np.float32)
			
		self.dictionary = SovereignDictionary(self.expanded_glyphs_path)

	def test_dictionary_only_uses_base_vocab(self):
		"""Verificar que la definición obtenida usa exclusivamente palabras del vocabulario base."""
		# Test con computadora
		definition = self.dictionary.buscar("computadora")
		self.assertIsNotNone(definition)
		self.assertGreater(len(definition), 0)
		
		# Cada palabra de la definición debe estar en el vocabulario base
		def_words = definition.split()
		for w in def_words:
			self.assertIn(w, self.base_vocab, f"La palabra '{w}' de la definición no está en el vocabulario base!")

	def test_dynamic_word_registration(self):
		"""Verificar que register_new_word expande correctamente la tabla de glifos en BitNet4LayerModel."""
		model = BitNet4LayerModel(
			use_glyphs=True,
			glyph_table=self.glyphs,
			hidden_dim=256,
			num_layers=4,
			use_pos_embedding=True
		)
		
		orig_vocab_size = model.vocab_size
		orig_table_shape = model.glyph_embedding.glyph_table.shape
		
		# Crear un glifo ficticio de 65 trits
		new_glyph = torch.zeros(65)
		new_glyph[10] = 1.0
		new_glyph[20] = -1.0
		
		# Registrar palabra
		model.register_new_word("computadora", new_glyph)
		
		# Verificar expansión
		self.assertEqual(model.vocab_size, orig_vocab_size + 1)
		self.assertEqual(model.glyph_embedding.glyph_table.shape[0], orig_table_shape[0] + 1)
		self.assertEqual(model.glyph_embedding.glyph_table.shape[1], 65)
		
		# Verificar que el último elemento es idéntico a nuestro new_glyph
		last_registered = model.glyph_embedding.glyph_table[-1].cpu()
		self.assertTrue(torch.equal(last_registered, new_glyph))

	def test_projection_head_shapes(self):
		"""Verificar que la cabeza de proyección genera logits de forma coherente (N_batch, 65)."""
		model = BitNet4LayerModel(
			use_glyphs=True,
			glyph_table=self.glyphs,
			hidden_dim=256,
			num_layers=4,
			use_pos_embedding=True
		)
		
		# Entrada ficticia de definición: batch=2, seq_len=8
		dummy_input = torch.zeros((2, 256)) # hidden_dim = 256
		
		proj = model.glyph_projection_head(dummy_input)
		self.assertEqual(proj.shape, (2, 65))
		
		# Las salidas de Tanh deben estar en [-1, 1]
		self.assertTrue((proj >= -1.0).all())
		self.assertTrue((proj <= 1.0).all())

if __name__ == "__main__":
	unittest.main()
