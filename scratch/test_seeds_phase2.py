import json
import os
import sys

import torch
import torch.nn as nn

sys.path.append('/home/joan/Documents/IA/sharing/src')
from src.bitnet.cooperative_world import (
	COOP_ACTIONS,
	COOP_N_ACTIONS,
	CooperativeWorld,
)
from src.bitnet.glyph_vocabulary import N_EMOTIONS, WORD_NAMES
from src.bitnet.modeling_bitnet import BitNet4LayerModel


def load_agent(checkpoint_path: str, model_cfg: dict, emotion_cfg: dict, max_res_steps: int, device: torch.device) -> nn.Module:
	m = BitNet4LayerModel(
		use_glyphs=True,
		hidden_dim=model_cfg.get("hidden_dim", 256),
		num_layers=model_cfg.get("num_layers", 3),
		use_pos_embedding=model_cfg.get("use_pos_embedding", True),
		max_resonance_steps=max_res_steps,
		n_emotions=N_EMOTIONS,
		emotion_dim=emotion_cfg.get("dim", 64),
		emotion_mode=emotion_cfg.get("mode", "first_only"),
		max_seq_len=6,
	).to(device)

	if os.path.exists(checkpoint_path):
		sd = torch.load(checkpoint_path, map_location=device, weights_only=True)
		if "action_head.0.weight" in sd and "action_head.2.weight" in sd:
			ckpt_action_width = sd["action_head.0.weight"].shape[0]
			ckpt_out_features = sd["action_head.2.weight"].shape[0]
			if ckpt_action_width != m.action_head[0].out_features or ckpt_out_features != m.action_head[2].out_features:
				m.action_head = nn.Sequential(
					nn.Linear(model_cfg.get("hidden_dim", 256), ckpt_action_width),
					nn.GELU(),
					nn.Linear(ckpt_action_width, ckpt_out_features),
				).to(device)
		if "value_head.0.weight" in sd and "value_head.2.weight" in sd:
			ckpt_value_width = sd["value_head.0.weight"].shape[0]
			ckpt_val_out = sd["value_head.2.weight"].shape[0]
			if ckpt_value_width != m.value_head[0].out_features or ckpt_val_out != m.value_head[2].out_features:
				m.value_head = nn.Sequential(
					nn.Linear(model_cfg.get("hidden_dim", 256), ckpt_value_width),
					nn.GELU(),
					nn.Linear(ckpt_value_width, ckpt_val_out),
				).to(device)
		m.load_state_dict(sd, strict=False)

	if m.action_head[2].out_features != COOP_N_ACTIONS:
		old_first = m.action_head[0]
		old_gelu = m.action_head[1]
		old_linear = m.action_head[2]
		old_width = old_first.out_features
		old_out = old_linear.out_features
		new_linear = nn.Linear(old_width, COOP_N_ACTIONS)
		with torch.no_grad():
			copy_limit = min(old_out, COOP_N_ACTIONS)
			new_linear.weight[:copy_limit] = old_linear.weight[:copy_limit].clone()
			new_linear.bias[:copy_limit] = old_linear.bias[:copy_limit].clone()
			if old_out < COOP_N_ACTIONS:
				new_linear.weight[old_out:] = torch.randn(COOP_N_ACTIONS - old_out, old_width) * 0.01
				new_linear.bias[old_out:] = 0.0
		m.action_head = nn.Sequential(
			old_first,
			old_gelu,
			new_linear,
		).to(device)

	return m

def get_glyph_name(idx: int) -> str:
	if 0 <= idx < len(WORD_NAMES):
		return WORD_NAMES[idx]
	return "desconocido"

def test_seeds():
	base_dir = "/home/joan/Documents/IA/frankenswarm"
	config_path = os.path.join(base_dir, "configs/experiments/EXP_050_arena_listener.json")
	with open(config_path, encoding="utf-8") as f:
		config = json.load(f)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model_cfg = config.get("model", {})
	emotion_cfg = config.get("emotion", {})
	max_res_steps = config.get("resonance", {}).get("max_resonance_steps", 5)
	n_think = config.get("resonance", {}).get("n_think", 2)

	exp_dir = os.path.join(base_dir, "storage", "experiments", "EXP_050_arena_listener")
	path_c = os.path.join(exp_dir, "best_agent_c.pt")
	if not os.path.exists(path_c) and os.path.exists(os.path.join(exp_dir, "best_agent_a.pt")):
		path_c = os.path.join(exp_dir, "best_agent_a.pt")

	agent_a = load_agent(os.path.join(exp_dir, "best_agent_a.pt"), model_cfg, emotion_cfg, max_res_steps, device)
	agent_b = load_agent(os.path.join(exp_dir, "best_agent_b.pt"), model_cfg, emotion_cfg, max_res_steps, device)
	agent_c = load_agent(path_c, model_cfg, emotion_cfg, max_res_steps, device)
	agent_a.eval()
	agent_b.eval()
	agent_c.eval()

	results = []
	for seed in range(100):
		world = CooperativeWorld(
			seed=seed,
			food_interval=config["world"]["food_interval"],
			food_duration=config["world"]["food_duration"],
			hunger_rate=config["world"]["hunger_rate"],
		)
		state_a, state_b, state_c = world.reset()
		ticks = 0
		shouts = []
		while state_a.alive and state_b.alive and state_c.alive and ticks < 100:
			perc_a = world.perceive(state_a)
			perc_b = world.perceive(state_b)
			perc_c = world.perceive(state_c)
			input_a = torch.tensor([perc_a[:6]], dtype=torch.long, device=device)
			input_b = torch.tensor([perc_b[:6]], dtype=torch.long, device=device)
			input_c = torch.tensor([perc_c[:6]], dtype=torch.long, device=device)
			emo_a = torch.tensor([state_a.emotion_id], device=device)
			emo_b = torch.tensor([state_b.emotion_id], device=device)
			emo_c = torch.tensor([state_c.emotion_id], device=device)

			with torch.no_grad():
				_, meta_a = agent_a.forward_resonance(input_a, n_steps=n_think, pos_mode="clock", emotion_ids=emo_a)
				_, meta_b = agent_b.forward_resonance(input_b, n_steps=n_think, pos_mode="clock", emotion_ids=emo_b)
				_, meta_c = agent_c.forward_resonance(input_c, n_steps=n_think, pos_mode="clock", emotion_ids=emo_c)
				h_a = meta_a["hidden"][:, 2, :].clone()
				h_b = meta_b["hidden"][:, 2, :].clone()
				h_c = meta_c["hidden"][:, 2, :].clone()
				act_a = torch.argmax(agent_a.action_head(h_a), dim=-1).item()
				act_b = torch.argmax(agent_b.action_head(h_b), dim=-1).item()
				act_c = torch.argmax(agent_c.action_head(h_c), dim=-1).item()

				shout_concept_a = torch.argmax(agent_a._decode_hidden(h_a), dim=-1).item()
				shout_concept_b = torch.argmax(agent_b._decode_hidden(h_b), dim=-1).item()
				shout_concept_c = torch.argmax(agent_c._decode_hidden(h_c), dim=-1).item()

			action_a = COOP_ACTIONS[act_a]
			action_b = COOP_ACTIONS[act_b]
			action_c = COOP_ACTIONS[act_c]

			s_concept_a = shout_concept_a if act_a == 6 else None
			s_concept_b = shout_concept_b if act_b == 6 else None
			s_concept_c = shout_concept_c if act_c == 6 else None

			res_a, res_b, res_c, w_info = world.step(
				action_a, action_b, action_c,
				shout_concept_a=s_concept_a, shout_concept_b=s_concept_b, shout_concept_c=s_concept_c
			)
			if res_a.get("shouted"):
				shouts.append(("A", state_a.location, get_glyph_name(res_a["shout_content"][1])))
			if res_b.get("shouted"):
				shouts.append(("B", state_b.location, get_glyph_name(res_b["shout_content"][1])))
			if res_c.get("shouted"):
				shouts.append(("C", state_c.location, get_glyph_name(res_c["shout_content"][1])))
			ticks += 1

		results.append((seed, ticks, shouts))

	# Imprimir semillas donde sobrevivieron y gritaron
	print("Seeds with communication activity:")
	results_with_shouts = [r for r in results if len(r[2]) > 0]
	results_with_shouts.sort(key=lambda x: x[1], reverse=True)
	for seed, ticks, shouts in results_with_shouts[:10]:
		print(f"Seed {seed:03d} | Survival: {ticks:2d} ticks | Shouts: {len(shouts)}")
		for agent, loc, concept in shouts[:3]:
			print(f"   - {agent} en {loc} gritó '{concept}'")

if __name__ == "__main__":
	test_seeds()
