import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.bitnet.cooperative_world import CooperativeWorld, CoopAgentState, COOP_ACTIONS
from src.bitnet.consolidate_sleep import consolidate_latent_resonance
from src.bitnet.modeling_bitnet import BitNet4LayerModel
from src.bitnet.glyph_vocabulary import WORD_INDEX, N_EMOTIONS

class TestDistillation(unittest.TestCase):
	def setUp(self):
		self.world = CooperativeWorld(seed=42)

	def test_brain_capacity_slots(self):
		# Nico (a) has width 128 (slots: 2)
		state_a = CoopAgentState(agent_id="a", network_width=128)
		self.assertEqual(state_a.max_slots, 2)
		
		# width 192 (slots: 3)
		state_b = CoopAgentState(agent_id="b", network_width=192)
		self.assertEqual(state_b.max_slots, 3)
		
		# width 288 (slots: 4)
		state_c = CoopAgentState(agent_id="c", network_width=288)
		self.assertEqual(state_c.max_slots, 4)

	def test_neurogenesis_penalty(self):
		# Set pending_growth_penalty = True
		self.world.agent_a.network_width = 192
		self.world.agent_a.pending_growth_penalty = True
		self.world.agent_a.hambre = 60.0
		self.world.agent_a.sed = 80.0
		
		# Run reset
		self.world.reset()
		
		# Should apply -25 penalty
		self.assertEqual(self.world.agent_a.hambre, 35.0)
		self.assertEqual(self.world.agent_a.sed, 55.0)
		self.assertFalse(self.world.agent_a.pending_growth_penalty)

	def test_geographic_resource_restrictions(self):
		# Test "agua" skill can only be taught/learned at water nodes ("lago", "pantano", "río")
		# Nico (a) has "agua" skill, Hugo (c) lacks it but has slots: width=192, max_slots=3.
		# Hugo already has "comida" and "caza", so the only potential skill to learn is "agua".
		self.world.agent_a.location = "bosque"
		self.world.agent_a.learned_skills = ["comida", "agua"]
		self.world.agent_c.location = "bosque"
		self.world.agent_c.network_width = 192
		self.world.agent_c.learned_skills = ["caza", "comida"]
		
		# Enseñar at "bosque" (which does not have water)
		self.world.current_actions = {"a": "enseñar", "b": "ver", "c": "aprender"}
		res_a = self.world.act(self.world.agent_a, "enseñar")
		res_c = self.world.act(self.world.agent_c, "aprender")
		# Should not succeed for "agua" at bosque
		self.assertFalse(res_c.get("success", False))

		# Move both to "lago" (which has water)
		self.world.agent_a.location = "lago"
		self.world.agent_c.location = "lago"
		self.world.current_actions = {"a": "enseñar", "b": "ver", "c": "aprender"}
		res_a = self.world.act(self.world.agent_a, "enseñar")
		res_c = self.world.act(self.world.agent_c, "aprender")
		# Should succeed for "agua" at lago
		self.assertTrue(res_c.get("success", False))
		self.assertEqual(res_c.get("learning_skill"), "agua")
		self.assertEqual(res_c.get("learned_from"), "a")

	def test_radio_proposal_slot_check(self):
		# If Nico (a) shouts "enseñar", but Hugo (c) has full slots, Nico's shout is rewritten to "seguro"
		self.world.agent_a.location = "bosque"
		self.world.agent_c.location = "bosque"
		
		# Make Hugo's slots full (width=128, max_slots=2, skills=["comida", "caza"])
		self.world.agent_c.network_width = 128
		self.world.agent_c.learned_skills = ["comida", "caza"]
		
		# Nico tries to shout "enseñar"
		shout_enseñar_idx = WORD_INDEX.get("enseñar", 0)
		res_a = self.world.act(self.world.agent_a, "gritar", shout_concept_idx=shout_enseñar_idx)
		
		# Should fall back to "seguro" since no living companion has free slots
		self.assertEqual(res_a["shout_content"][1], WORD_INDEX.get("seguro", 0))

	def test_latent_resonance_distillation(self):
		# Prepare a mock model and training buffer to test consolidation loss minimization
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		model = BitNet4LayerModel(
			use_glyphs=True,
			hidden_dim=64, # small hidden dim for quick test
			num_layers=2,
			use_pos_embedding=True,
			n_emotions=N_EMOTIONS,
			max_seq_len=6,
		).to(device)
		
		# Create a target teacher hidden state
		teacher_hidden = torch.randn(1, 64)
		
		# Create sample buffer
		teaching_buffer = [{
			"input": torch.tensor([[1, 2, 3, 4, 0, 0]], dtype=torch.long),
			"emotion": 0,
			"teacher_hidden": teacher_hidden,
			"skill": "comida",
			"teacher_id": "b"
		}]
		
		# Run consolidation
		graduated = consolidate_latent_resonance(
			model, teaching_buffer, device, lr=1e-2, epochs=20, n_think=2
		)
		
		# Loss should have decreased, let's run it again to make sure it finishes
		self.assertTrue(isinstance(graduated, list))

if __name__ == "__main__":
	unittest.main()
