import unittest
import copy
import torch
import torch.nn as nn
from src.bitnet.cooperative_world import CooperativeWorld, CoopAgentState
from src.bitnet.modeling_bitnet import BitNet4LayerModel
from src.bitnet.glyph_vocabulary import N_EMOTIONS
from src.bitnet.genetic import project_model_to_width, svd_crossover, recombine_parents

class TestReproduction(unittest.TestCase):
	def setUp(self):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.world = CooperativeWorld(seed=42)

	def test_biological_costs_and_restrictions(self):
		# Configurar padres en la cueva con suficiente metabolismo
		self.world.agent_a.location = "cueva"
		self.world.agent_b.location = "cueva"
		self.world.agent_a.hambre = 80.0
		self.world.agent_a.sed = 80.0
		self.world.agent_b.hambre = 80.0
		self.world.agent_b.sed = 80.0
		self.world.agent_d.alive = False
		self.world.agent_d_was_alive = False

		# Ejecutar apareamiento
		result_a, result_b, result_c, world_info = self.world.step(
			"reproducir", "reproducir", "ver", "ver"
		)

		# Verificaciones
		self.assertTrue(self.world.agent_d.alive, "Domi debería haber nacido vivo")
		self.assertEqual(self.world.agent_d.location, "cueva")
		
		# 80 - 35 (apareamiento) - 3.5 (tasa basal) = 41.5
		self.assertAlmostEqual(self.world.agent_a.hambre, 41.5)
		self.assertAlmostEqual(self.world.agent_b.hambre, 41.5)

	def test_mating_outside_shelter_fails(self):
		self.world.agent_a.location = "lago"
		self.world.agent_b.location = "lago"
		self.world.agent_a.hambre = 80.0
		self.world.agent_a.sed = 80.0
		self.world.agent_b.hambre = 80.0
		self.world.agent_b.sed = 80.0
		self.world.agent_d.alive = False
		self.world.agent_d_was_alive = False

		result_a, result_b, result_c, world_info = self.world.step(
			"reproducir", "reproducir", "ver", "ver"
		)
		self.assertFalse(self.world.agent_d.alive, "No debería haber nacido Domi fuera del refugio")

	def test_mating_with_low_metabolism_fails(self):
		self.world.agent_a.location = "cueva"
		self.world.agent_b.location = "cueva"
		self.world.agent_a.hambre = 30.0 # por debajo del límite de 40.0
		self.world.agent_a.sed = 80.0
		self.world.agent_b.hambre = 80.0
		self.world.agent_b.sed = 80.0
		self.world.agent_d.alive = False
		self.world.agent_d_was_alive = False

		result_a, result_b, result_c, world_info = self.world.step(
			"reproducir", "reproducir", "ver", "ver"
		)
		self.assertFalse(self.world.agent_d.alive, "No debería haber reproducción con hambre < 40")

	def test_genetic_skill_inheritance(self):
		self.world.agent_a.learned_skills = ["comida", "caza"]
		self.world.agent_b.learned_skills = ["agua"]
		
		self.world.agent_a.location = "cueva"
		self.world.agent_b.location = "cueva"
		self.world.agent_a.hambre = 85.0
		self.world.agent_a.sed = 85.0
		self.world.agent_b.hambre = 85.0
		self.world.agent_b.sed = 85.0
		self.world.agent_d.alive = False
		self.world.agent_d_was_alive = False

		self.world.step("reproducir", "reproducir", "ver", "ver")

		self.assertTrue(self.world.agent_d.alive)
		# Debería heredar exactamente 2 de la unión {"comida", "caza", "agua"}
		self.assertEqual(len(self.world.agent_d.learned_skills), 2)
		for skill in self.world.agent_d.learned_skills:
			self.assertIn(skill, ["comida", "caza", "agua"])

	def test_svd_crossover_and_net2net_surgery(self):
		hidden_dim = 64
		num_layers = 2
		
		model_a = BitNet4LayerModel(
			use_glyphs=True,
			hidden_dim=hidden_dim,
			num_layers=num_layers,
			n_emotions=N_EMOTIONS,
			emotion_dim=16,
			max_seq_len=6,
		).to(self.device)
		model_a.action_head = nn.Sequential(
			nn.Linear(hidden_dim, 128),
			nn.GELU(),
			nn.Linear(128, 11),
		).to(self.device)
		model_a.value_head = nn.Sequential(
			nn.Linear(hidden_dim, 128),
			nn.GELU(),
			nn.Linear(128, 1),
		).to(self.device)

		model_b = BitNet4LayerModel(
			use_glyphs=True,
			hidden_dim=hidden_dim,
			num_layers=num_layers,
			n_emotions=N_EMOTIONS,
			emotion_dim=16,
			max_seq_len=6,
		).to(self.device)
		model_b.action_head = nn.Sequential(
			nn.Linear(hidden_dim, 192),
			nn.GELU(),
			nn.Linear(192, 11),
		).to(self.device)
		model_b.value_head = nn.Sequential(
			nn.Linear(hidden_dim, 192),
			nn.GELU(),
			nn.Linear(192, 1),
		).to(self.device)

		child_width = 160
		
		model_child = BitNet4LayerModel(
			use_glyphs=True,
			hidden_dim=hidden_dim,
			num_layers=num_layers,
			n_emotions=N_EMOTIONS,
			emotion_dim=16,
			max_seq_len=6,
		).to(self.device)
		model_child.action_head = nn.Sequential(
			nn.Linear(hidden_dim, child_width),
			nn.GELU(),
			nn.Linear(child_width, 11),
		).to(self.device)
		model_child.value_head = nn.Sequential(
			nn.Linear(hidden_dim, child_width),
			nn.GELU(),
			nn.Linear(child_width, 1),
		).to(self.device)

		# Crossover con Surgery y Net2Net
		recombine_parents(model_a, model_b, model_child, child_width, self.device)

		self.assertEqual(model_child.action_head[0].out_features, 160)
		self.assertEqual(model_child.value_head[0].out_features, 160)

		# Forward pass para asegurar coherencia matemática de la red resultante
		test_input = torch.randint(0, 5, (1, 6), dtype=torch.long, device=self.device)
		test_emotion = torch.zeros(1, dtype=torch.long, device=self.device)
		
		res, meta = model_child.forward_resonance(
			test_input, n_steps=2, pos_mode="clock", emotion_ids=test_emotion
		)
		hidden = meta["hidden"][:, 2, :]
		logits = model_child.action_head(hidden)
		
		self.assertEqual(logits.shape, (1, 11))

if __name__ == "__main__":
	unittest.main()
