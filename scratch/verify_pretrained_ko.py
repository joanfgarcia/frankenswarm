import json
import os
import sys

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bitnet.cooperative_world import COOP_ACTIONS, COOP_N_ACTIONS, CooperativeWorld
from src.bitnet.glyph_vocabulary import N_EMOTIONS
from src.bitnet.modeling_bitnet import BitNet4LayerModel
from src.bitnet.train_arena_ppo import get_masked_probs, perception_to_input_coop

device = torch.device("cpu")
with open("configs/experiments/EXP_073_easy_train.json") as f:
	config = json.load(f)

model_cfg = config.get("model", {})
emotion_cfg = config.get("emotion", {})

def load_m(path):
	m = BitNet4LayerModel(
		use_glyphs=True,
		hidden_dim=model_cfg.get("hidden_dim", 256),
		num_layers=model_cfg.get("num_layers", 3),
		use_pos_embedding=model_cfg.get("use_pos_embedding", True),
		max_resonance_steps=5,
		n_emotions=N_EMOTIONS,
		emotion_dim=emotion_cfg.get("dim", 64),
		emotion_mode=emotion_cfg.get("mode", "first_only"),
		max_seq_len=6,
	)
	if os.path.exists(path):
		sd = torch.load(path, map_location=device, weights_only=True)
		if "glyph_embedding.glyph_table" in sd:
			del sd["glyph_embedding.glyph_table"]
		if "action_head.0.weight" in sd and "action_head.2.weight" in sd:
			ckpt_action_width = sd["action_head.0.weight"].shape[0]
			ckpt_out_features = sd["action_head.2.weight"].shape[0]
			if ckpt_action_width != m.action_head[0].out_features or ckpt_out_features != m.action_head[2].out_features:
				m.action_head = nn.Sequential(
					nn.Linear(model_cfg.get("hidden_dim", 256), ckpt_action_width),
					nn.GELU(),
					nn.Linear(ckpt_action_width, ckpt_out_features),
				).to(device)
		m.load_state_dict(sd, strict=False)

	# Preserving weights resize
	old_width = m.action_head[0].out_features
	old_out = m.action_head[2].out_features
	if old_out != COOP_N_ACTIONS:
		old_first = m.action_head[0]
		old_gelu = m.action_head[1]
		old_linear = m.action_head[2]
		new_linear = nn.Linear(old_width, COOP_N_ACTIONS)
		with torch.no_grad():
			copy_limit = min(old_out, COOP_N_ACTIONS)
			new_linear.weight[:copy_limit] = old_linear.weight[:copy_limit].clone()
			new_linear.bias[:copy_limit] = old_linear.bias[:copy_limit].clone()
		m.action_head = nn.Sequential(old_first, old_gelu, new_linear).to(device)
	return m

agent_a = load_m("storage/experiments/EXP_073_easy_train/best_agent_a.pt")
agent_b = load_m("storage/experiments/EXP_073_easy_train/best_agent_b.pt")
agent_c = load_m("storage/experiments/EXP_073_easy_train/best_agent_c.pt")

# We use the world config from easy_train
world_cfg = config.get("world", {})
world = CooperativeWorld(
	seed=999,
	food_interval=world_cfg.get("food_interval", 12),
	food_duration=world_cfg.get("food_duration", 8),
	hunger_rate=world_cfg.get("hunger_rate", 1.5),
	resource_capacity=world_cfg.get("resource_capacity", 10.0),
	resource_recovery=world_cfg.get("resource_recovery", 0.5),
	predator_chance_multiplier=world_cfg.get("predator_chance_multiplier", 0.1),
	predator_damage_multiplier=world_cfg.get("predator_damage_multiplier", 0.1),
	storm_chance_multiplier=world_cfg.get("storm_chance_multiplier", 0.1),
	storm_damage_multiplier=world_cfg.get("storm_damage_multiplier", 0.1),
	prey_spawn_interval=world_cfg.get("prey_spawn_interval", 15)
)

state_a, state_b, state_c = world.reset(seed=999)

ko_count_a = 0
ko_count_b = 0
ko_count_c = 0
consecutive_healthy = 0
max_consecutive_healthy = 0

for tick in range(1, 501):
	# Choose actions using policy (stochastic or argmax)
	# Here we do the same sampling as in train_arena_ppo.py with explore_rate = 1.0 (Episode 1)
	explore_rate = 1.0
	
	# Nico (A)
	perc_a = world.perceive(state_a)
	input_a = perception_to_input_coop(perc_a, device)
	emo_a = torch.tensor([state_a.emotion_id], device=device)
	with torch.no_grad():
		_, meta_a = agent_a.forward_resonance(input_a, n_steps=2, pos_mode="clock", emotion_ids=emo_a)
		hidden_a = meta_a["hidden"][:, 2, :].clone()
		logits_a = agent_a.action_head(hidden_a)
		mask_a = world.get_valid_actions_mask(state_a)
		probs_a = get_masked_probs(logits_a, 0, 2000, True, mask_a)
		action_a = COOP_ACTIONS[torch.argmax(probs_a, dim=-1).item()]

	# Sofy (B)
	perc_b = world.perceive(state_b)
	input_b = perception_to_input_coop(perc_b, device)
	emo_b = torch.tensor([state_b.emotion_id], device=device)
	with torch.no_grad():
		_, meta_b = agent_b.forward_resonance(input_b, n_steps=2, pos_mode="clock", emotion_ids=emo_b)
		hidden_b = meta_b["hidden"][:, 2, :].clone()
		logits_b = agent_b.action_head(hidden_b)
		mask_b = world.get_valid_actions_mask(state_b)
		probs_b = get_masked_probs(logits_b, 0, 2000, True, mask_b)
		action_b = COOP_ACTIONS[torch.argmax(probs_b, dim=-1).item()]

	# Hugo (C)
	perc_c = world.perceive(state_c)
	input_c = perception_to_input_coop(perc_c, device)
	emo_c = torch.tensor([state_c.emotion_id], device=device)
	with torch.no_grad():
		_, meta_c = agent_c.forward_resonance(input_c, n_steps=2, pos_mode="clock", emotion_ids=emo_c)
		hidden_c = meta_c["hidden"][:, 2, :].clone()
		logits_c = agent_c.action_head(hidden_c)
		mask_c = world.get_valid_actions_mask(state_c)
		probs_c = get_masked_probs(logits_c, 0, 2000, True, mask_c)
		action_c = COOP_ACTIONS[torch.argmax(probs_c, dim=-1).item()]

	ko_before_a = getattr(state_a, "debuff_inconsciente_ticks", 0) > 0
	ko_before_b = getattr(state_b, "debuff_inconsciente_ticks", 0) > 0
	ko_before_c = getattr(state_c, "debuff_inconsciente_ticks", 0) > 0

	res_a, res_b, res_c, info = world.step(action_a, action_b, action_c)

	if getattr(state_a, "debuff_inconsciente_ticks", 0) > 0 and not ko_before_a:
		ko_count_a += 1
		print(f"Tick {tick:03d} | [KO EVENT] Agent A (Nico) went KO! Thirst: {state_a.sed:.1f}, Hunger: {state_a.hambre:.1f}")
	if getattr(state_b, "debuff_inconsciente_ticks", 0) > 0 and not ko_before_b:
		ko_count_b += 1
		print(f"Tick {tick:03d} | [KO EVENT] Agent B (Sofy) went KO! Thirst: {state_b.sed:.1f}, Hunger: {state_b.hambre:.1f}")
	if getattr(state_c, "debuff_inconsciente_ticks", 0) > 0 and not ko_before_c:
		ko_count_c += 1
		print(f"Tick {tick:03d} | [KO EVENT] Agent C (Hugo) went KO! Thirst: {state_c.sed:.1f}, Hunger: {state_c.hambre:.1f}")

	is_any_ko = (
		getattr(state_a, "debuff_inconsciente_ticks", 0) > 0 or
		getattr(state_b, "debuff_inconsciente_ticks", 0) > 0 or
		getattr(state_c, "debuff_inconsciente_ticks", 0) > 0
	)

	if is_any_ko:
		consecutive_healthy = 0
	else:
		consecutive_healthy += 1
		if consecutive_healthy > max_consecutive_healthy:
			max_consecutive_healthy = consecutive_healthy

print("\nVerification finished.")
print(f"Total KOs: Nico={ko_count_a}, Sofy={ko_count_b}, Hugo={ko_count_c}")
print(f"Max consecutive healthy ticks: {max_consecutive_healthy}")
