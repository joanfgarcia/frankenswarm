import torch
import torch.nn as nn

from src.bitnet.data.dojo_populora import generate_sequential_dojo_batch, train_sequential_dojo_step
from src.bitnet.model.modeling_bitnet import BitNet4LayerModel
from src.bitnet.worlds.cooperative_world import COOP_N_ACTIONS


def test_sequential_dojo_generation():
	# Test generating a batch
	batch_size = 8
	seq_len = 4
	device = "cpu"
	inputs, emotions, target_actions, target_shouts, agent_ids = generate_sequential_dojo_batch(
		batch_size=batch_size, seq_len=seq_len, device=device
	)
	
	assert inputs.shape == (batch_size, seq_len, 6)
	assert emotions.shape == (batch_size, seq_len)
	assert target_actions.shape == (batch_size, seq_len)
	assert target_shouts.shape == (batch_size, seq_len)
	assert len(agent_ids) == batch_size
	assert all(aid in ["a", "b", "c"] for aid in agent_ids)

def test_sequential_dojo_training():
	# Test training step
	import random
	random.seed(42)
	torch.manual_seed(42)
	device = "cpu"
	hidden_dim = 64
	
	# Instantiate models for agents
	agent_a = BitNet4LayerModel(
		use_glyphs=True,
		hidden_dim=hidden_dim,
		num_layers=2,
		use_pos_embedding=True,
		max_resonance_steps=5,
		n_emotions=6,
		emotion_dim=16,
		emotion_mode="first_only",
		max_seq_len=6,
	)
	
	# Resize action head to match COOP_N_ACTIONS
	if agent_a.action_head[2].out_features != COOP_N_ACTIONS:
		old_width = agent_a.action_head[0].out_features
		agent_a.action_head = nn.Sequential(
			agent_a.action_head[0],
			agent_a.action_head[1],
			nn.Linear(old_width, COOP_N_ACTIONS)
		)
		
	agent_b = BitNet4LayerModel(
		use_glyphs=True,
		hidden_dim=hidden_dim,
		num_layers=2,
		use_pos_embedding=True,
		max_resonance_steps=5,
		n_emotions=6,
		emotion_dim=16,
		emotion_mode="first_only",
		max_seq_len=6,
	)
	if agent_b.action_head[2].out_features != COOP_N_ACTIONS:
		old_width = agent_b.action_head[0].out_features
		agent_b.action_head = nn.Sequential(
			agent_b.action_head[0],
			agent_b.action_head[1],
			nn.Linear(old_width, COOP_N_ACTIONS)
		)

	agents_dict = {"a": agent_a, "b": agent_b, "c": None}
	
	losses = train_sequential_dojo_step(
		agents_dict=agents_dict,
		device=device,
		batch_size=32,
		seq_len=4,
		lr=1e-4,
		epochs=2,
		n_think=2
	)
	
	assert "a" in losses
	assert "b" in losses
	assert "c" not in losses
	assert isinstance(losses["a"], float)
	assert isinstance(losses["b"], float)
