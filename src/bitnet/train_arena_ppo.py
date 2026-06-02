"""
train_arena_ppo.py — EXP_073_v5: Arena PPO Training for 3 Agents.

Entrenamiento cooperativo multi-agente en la Arena (CooperativeWorld)
para 3 agentes (Nico, Sofi y Hugo) utilizando PPO con GAE y critic dedicado.

Origen: Aleth & Joan, 2026-06-02
"""

import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.bitnet.cooperative_world import (
	COOP_ACTIONS,
	COOP_N_ACTIONS,
	SILENCE_GLYPH,
	CooperativeWorld,
)
from src.bitnet.glyph_vocabulary import (
	N_EMOTIONS,
)
from src.bitnet.modeling_bitnet import BitNet4LayerModel
from src.bitnet.net2net import grow_action_head, net2wider_linear
from src.bitnet.telemetry import ExperimentLogger


def perception_to_input_coop(perception: list[int], device: torch.device) -> torch.Tensor:
	"""Convertir percepción del mundo cooperativo a tensor."""
	indices = perception[:6]
	while len(indices) < 6:
		indices.append(SILENCE_GLYPH)
	return torch.tensor([indices], dtype=torch.long, device=device)


def grow_value_head(model: nn.Module, growth_factor: float = 1.5, noise_std: float = 0.01):
	"""Hacer crecer el value_head del modelo preservando la función (Net2WiderNet)."""
	head = model.value_head
	layer_in = head[0]
	layer_out = head[2]

	old_width = layer_in.out_features
	new_width = int(old_width * growth_factor)

	new_layer_in, new_layer_out = net2wider_linear(
		layer_in, layer_out, new_width, noise_std=noise_std
	)

	device = next(model.parameters()).device
	model.value_head[0] = new_layer_in.to(device)
	model.value_head[2] = new_layer_out.to(device)


def run_arena_ppo_training():
	parser = argparse.ArgumentParser(description="Arena 3-Agent PPO Training")
	parser.add_argument("--config", type=str, required=True)
	args, _ = parser.parse_known_args()

	base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	config_path = args.config if os.path.isabs(args.config) else os.path.join(base_dir, args.config)
	with open(config_path, encoding="utf-8") as f:
		config = json.load(f)

	experiment_id = config.get("experiment_id", "EXP_073_arena_ppo_3way")
	seed = config.get("seed", 42)
	torch.manual_seed(seed)
	np.random.seed(seed)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	exp_dir = os.path.join(base_dir, "storage", "experiments", experiment_id)
	os.makedirs(exp_dir, exist_ok=True)
	with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf-8") as f:
		json.dump(config, f, ensure_ascii=False, indent=4)

	model_cfg = config.get("model", {})
	emotion_cfg = config.get("emotion", {})

	# ═══ Build agents ═══
	def make_agent(agent_label: str):
		m = BitNet4LayerModel(
			use_glyphs=True,
			hidden_dim=model_cfg.get("hidden_dim", 256),
			num_layers=model_cfg.get("num_layers", 3),
			use_pos_embedding=model_cfg.get("use_pos_embedding", True),
			max_resonance_steps=config.get("resonance", {}).get("max_resonance_steps", 5),
			n_emotions=N_EMOTIONS,
			emotion_dim=emotion_cfg.get("dim", 64),
			emotion_mode=emotion_cfg.get("mode", "first_only"),
			max_seq_len=6, # Arena uses length 6 perception
		).to(device)

		# Determinar pesos pre-entrenados a cargar
		p = None
		load_label = "a" if agent_label == "c" else agent_label
		pretrained_from = config.get("pretrained_from")
		if pretrained_from:
			p_base = pretrained_from if os.path.isabs(pretrained_from) else os.path.join(base_dir, pretrained_from)
			if os.path.isdir(p_base):
				p = os.path.join(p_base, f"best_agent_{load_label}.pt")
			else:
				p_dir = os.path.dirname(p_base)
				p_spec = os.path.join(p_dir, f"best_agent_{load_label}.pt")
				if os.path.exists(p_spec):
					p = p_spec
				elif os.path.exists(p_base):
					p = p_base

		if p and os.path.exists(p):
			sd = torch.load(p, map_location=device, weights_only=True)
			if "action_head.0.weight" in sd and "action_head.2.weight" in sd:
				ckpt_width = sd["action_head.0.weight"].shape[0]
				ckpt_out = sd["action_head.2.weight"].shape[0]
				if ckpt_width != m.action_head[0].out_features or ckpt_out != m.action_head[2].out_features:
					m.action_head = nn.Sequential(
						nn.Linear(model_cfg.get("hidden_dim", 256), ckpt_width),
						nn.GELU(),
						nn.Linear(ckpt_width, ckpt_out),
					).to(device)
			if "value_head.0.weight" in sd and "value_head.2.weight" in sd:
				ckpt_val_width = sd["value_head.0.weight"].shape[0]
				ckpt_val_out = sd["value_head.2.weight"].shape[0]
				if ckpt_val_width != m.value_head[0].out_features or ckpt_val_out != m.value_head[2].out_features:
					m.value_head = nn.Sequential(
						nn.Linear(model_cfg.get("hidden_dim", 256), ckpt_val_width),
						nn.GELU(),
						nn.Linear(ckpt_val_width, ckpt_val_out),
					).to(device)
			m.load_state_dict(sd, strict=False)
			print(f"Loaded {agent_label.upper()} weights from: {p}")
		else:
			print(f"No custom checkpoint found for {agent_label.upper()} at {p}. Using random init or standard fallback.")

		# Resize action head for 8 actions (7 original + dar) preserving pre-trained weights
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
				if COOP_N_ACTIONS > old_out:
					new_linear.weight[old_out:] = torch.randn(COOP_N_ACTIONS - old_out, old_width) * 0.01
					new_linear.bias[old_out:] = 0.0
			m.action_head = nn.Sequential(
				old_first,
				old_gelu,
				new_linear,
			).to(device)
			print(f"Resize action head (PRESERVING WEIGHTS): {old_out} -> {COOP_N_ACTIONS} actions (width={old_width})")

		# Configurar gradientes
		unfreeze = config.get("unfreeze_backbone", True)
		if not unfreeze:
			for name, param in m.named_parameters():
				if "action_head" not in name and "value_head" not in name:
					param.requires_grad = False
			print(f"🔒 {agent_label.upper()} Backbone frozen")
		else:
			for name, param in m.named_parameters():
				# Aseguramos que la tabla de glifos no sea entrenable
				if "glyph_table" in name:
					param.requires_grad = False
				else:
					param.requires_grad = True
			print(f"🔓 {agent_label.upper()} Backbone unfrozen — full plasticity")

		return m

	agent_a = make_agent("a")
	agent_b = make_agent("b")
	agent_c = make_agent("c")

	communicate = config.get("communication", {}).get("enabled", True)
	comm_str = "📡 COMUNICACIÓN ON" if communicate else "🔇 SILENCIO"

	print(f"═══ 🏟️ Arena 3-Agent PPO — {experiment_id} ═══")
	print(f"  {comm_str}")

	# PPO Hyperparameters
	train_cfg = config.get("training", {})
	lr = train_cfg.get("lr", 5e-4)
	gamma = train_cfg.get("gamma", 0.97)
	lmbda = train_cfg.get("lmbda", 0.95)
	ppo_epochs = train_cfg.get("ppo_epochs", 4)
	clip_eps = train_cfg.get("clip_eps", 0.2)
	entropy_bonus = train_cfg.get("entropy_bonus", 0.05)
	c1 = train_cfg.get("value_loss_coeff", 0.5)
	grad_clip = train_cfg.get("grad_clip", 1.0)
	n_episodes = train_cfg.get("episodes", 500)
	max_ticks = config.get("world", {}).get("max_ticks", 200)
	n_think = config.get("resonance", {}).get("n_think", 2)

	# World config
	world_cfg = config.get("world", {})
	food_interval = world_cfg.get("food_interval", 10)
	food_duration = world_cfg.get("food_duration", 5)
	hunger_rate = world_cfg.get("hunger_rate", 5.0)
	resource_capacity = world_cfg.get("resource_capacity", 5.0)
	resource_recovery = world_cfg.get("resource_recovery", 0.2)
	predator_chance_multiplier = world_cfg.get("predator_chance_multiplier", 1.0)
	predator_damage_multiplier = world_cfg.get("predator_damage_multiplier", 1.0)
	storm_chance_multiplier = world_cfg.get("storm_chance_multiplier", 1.0)
	storm_damage_multiplier = world_cfg.get("storm_damage_multiplier", 1.0)
	prey_spawn_interval = world_cfg.get("prey_spawn_interval", 15)

	# Growth config
	growth_cfg = config.get("growth", {})
	growth_factor = growth_cfg.get("factor", 1.5)
	max_width = growth_cfg.get("max_width", 648)

	def make_optimizer(model):
		return torch.optim.AdamW(
			filter(lambda p: p.requires_grad, model.parameters()),
			lr=lr, weight_decay=0.01
		)

	opt_a = make_optimizer(agent_a)
	opt_b = make_optimizer(agent_b)
	opt_c = make_optimizer(agent_c)

	logger = ExperimentLogger(
		experiment_id,
		{**config, "device": str(device), "mode": "ppo_arena_3agent"},
		log_path=os.path.join(exp_dir, "telemetry.jsonl"),
	)

	best_combined = 0.0
	survival_hist = []
	total_growths_a = 0
	total_growths_b = 0
	total_growths_c = 0
	total_shouts_a = 0
	total_shouts_b = 0
	total_shouts_c = 0

	for episode in range(n_episodes):
		world = CooperativeWorld(
			seed=seed + episode,
			food_interval=food_interval,
			food_duration=food_duration,
			hunger_rate=hunger_rate,
			resource_capacity=resource_capacity,
			resource_recovery=resource_recovery,
			predator_chance_multiplier=predator_chance_multiplier,
			predator_damage_multiplier=predator_damage_multiplier,
			storm_chance_multiplier=storm_chance_multiplier,
			storm_damage_multiplier=storm_damage_multiplier,
			prey_spawn_interval=prey_spawn_interval,
		)
		state_a, state_b, state_c = world.reset()

		# Trajectory buffers
		buf_a = {"inputs": [], "emotions": [], "actions": [], "shout_concepts": [], "log_probs": [], "values": [], "rewards": []}
		buf_b = {"inputs": [], "emotions": [], "actions": [], "shout_concepts": [], "log_probs": [], "values": [], "rewards": []}
		buf_c = {"inputs": [], "emotions": [], "actions": [], "shout_concepts": [], "log_probs": [], "values": [], "rewards": []}

		agent_a.eval()
		agent_b.eval()
		agent_c.eval()

		ep_shouts_a = 0
		ep_shouts_b = 0
		ep_shouts_c = 0

		for tick in range(max_ticks):
			if not state_a.alive or not state_b.alive or not state_c.alive:
				break

			# ── 1. PERCIBIR (fog of war) ──
			perc_a = world.perceive(state_a)
			perc_b = world.perceive(state_b)
			perc_c = world.perceive(state_c)

			if not communicate:
				perc_a[4] = SILENCE_GLYPH
				perc_a[5] = SILENCE_GLYPH
				perc_b[4] = SILENCE_GLYPH
				perc_b[5] = SILENCE_GLYPH
				perc_c[4] = SILENCE_GLYPH
				perc_c[5] = SILENCE_GLYPH

			input_a = perception_to_input_coop(perc_a, device)
			input_b = perception_to_input_coop(perc_b, device)
			input_c = perception_to_input_coop(perc_c, device)

			emo_a = torch.tensor([state_a.emotion_id], device=device)
			emo_b = torch.tensor([state_b.emotion_id], device=device)
			emo_c = torch.tensor([state_c.emotion_id], device=device)

			# ── 2. PENSAR (forward resonance) ──
			with torch.no_grad():
				_, meta_a = agent_a.forward_resonance(
					input_a, n_steps=n_think, pos_mode="clock", emotion_ids=emo_a
				)
				_, meta_b = agent_b.forward_resonance(
					input_b, n_steps=n_think, pos_mode="clock", emotion_ids=emo_b
				)
				_, meta_c = agent_c.forward_resonance(
					input_c, n_steps=n_think, pos_mode="clock", emotion_ids=emo_c
				)

				hidden_a = meta_a["hidden"][:, 2, :].clone()
				hidden_b = meta_b["hidden"][:, 2, :].clone()
				hidden_c = meta_c["hidden"][:, 2, :].clone()

				# Predicciones de política y valor
				action_logits_a = agent_a.action_head(hidden_a)
				action_logits_b = agent_b.action_head(hidden_b)
				action_logits_c = agent_c.action_head(hidden_c)

				shout_logits_a = agent_a._decode_hidden(hidden_a)
				shout_logits_b = agent_b._decode_hidden(hidden_b)
				shout_logits_c = agent_c._decode_hidden(hidden_c)

				val_a = agent_a.value_head(hidden_a).item()
				val_b = agent_b.value_head(hidden_b).item()
				val_c = agent_c.value_head(hidden_c).item()

			probs_a = F.softmax(action_logits_a, dim=-1)
			probs_b = F.softmax(action_logits_b, dim=-1)
			probs_c = F.softmax(action_logits_c, dim=-1)

			shout_probs_a = F.softmax(shout_logits_a, dim=-1)
			shout_probs_b = F.softmax(shout_logits_b, dim=-1)
			shout_probs_c = F.softmax(shout_logits_c, dim=-1)

			# Enmascarar GRITAR si no hay comunicación
			if not communicate:
				probs_a = probs_a.clone()
				probs_a[0, 6] = 0.0
				probs_a = probs_a / probs_a.sum()
				probs_b = probs_b.clone()
				probs_b[0, 6] = 0.0
				probs_b = probs_b / probs_b.sum()
				probs_c = probs_c.clone()
				probs_c[0, 6] = 0.0
				probs_c = probs_c / probs_c.sum()

			# Muestreo con exploración decreciente
			explore_rate = max(0.02, 1.0 - episode / (n_episodes * 0.40))

			# Decidir acciones y conceptos (A)
			dist_shout_a = torch.distributions.Categorical(shout_probs_a)
			if np.random.rand() < explore_rate:
				shout_concept_a_val = np.random.randint(0, shout_probs_a.shape[-1])
				lp_shout_a = torch.log(shout_probs_a[0, shout_concept_a_val] + 1e-8).item()
			else:
				shout_concept_a_val = dist_shout_a.sample().item()
				lp_shout_a = dist_shout_a.log_prob(torch.tensor(shout_concept_a_val, device=device)).item()

			if np.random.rand() < explore_rate:
				if communicate:
					idx_a = np.random.randint(0, COOP_N_ACTIONS)
				else:
					idx_a = np.random.choice([0, 1, 2, 3, 4, 5, 7])
				lp_a = torch.log(probs_a[0, idx_a] + 1e-8).item()
			else:
				dist_a = torch.distributions.Categorical(probs_a)
				idx_a = dist_a.sample().item()
				lp_a = dist_a.log_prob(torch.tensor(idx_a, device=device)).item()

			if idx_a == 6:
				lp_a = lp_a + lp_shout_a

			# Decidir acciones y conceptos (B)
			dist_shout_b = torch.distributions.Categorical(shout_probs_b)
			if np.random.rand() < explore_rate:
				shout_concept_b_val = np.random.randint(0, shout_probs_b.shape[-1])
				lp_shout_b = torch.log(shout_probs_b[0, shout_concept_b_val] + 1e-8).item()
			else:
				shout_concept_b_val = dist_shout_b.sample().item()
				lp_shout_b = dist_shout_b.log_prob(torch.tensor(shout_concept_b_val, device=device)).item()

			if np.random.rand() < explore_rate:
				if communicate:
					idx_b = np.random.randint(0, COOP_N_ACTIONS)
				else:
					idx_b = np.random.choice([0, 1, 2, 3, 4, 5, 7])
				lp_b = torch.log(probs_b[0, idx_b] + 1e-8).item()
			else:
				dist_b = torch.distributions.Categorical(probs_b)
				idx_b = dist_b.sample().item()
				lp_b = dist_b.log_prob(torch.tensor(idx_b, device=device)).item()

			if idx_b == 6:
				lp_b = lp_b + lp_shout_b

			# Decidir acciones y conceptos (C)
			dist_shout_c = torch.distributions.Categorical(shout_probs_c)
			if np.random.rand() < explore_rate:
				shout_concept_c_val = np.random.randint(0, shout_probs_c.shape[-1])
				lp_shout_c = torch.log(shout_probs_c[0, shout_concept_c_val] + 1e-8).item()
			else:
				shout_concept_c_val = dist_shout_c.sample().item()
				lp_shout_c = dist_shout_c.log_prob(torch.tensor(shout_concept_c_val, device=device)).item()

			if np.random.rand() < explore_rate:
				if communicate:
					idx_c = np.random.randint(0, COOP_N_ACTIONS)
				else:
					idx_c = np.random.choice([0, 1, 2, 3, 4, 5, 7])
				lp_c = torch.log(probs_c[0, idx_c] + 1e-8).item()
			else:
				dist_c = torch.distributions.Categorical(probs_c)
				idx_c = dist_c.sample().item()
				lp_c = dist_c.log_prob(torch.tensor(idx_c, device=device)).item()

			if idx_c == 6:
				lp_c = lp_c + lp_shout_c

			# ── 3. ACTUAR ──
			action_a = COOP_ACTIONS[idx_a]
			action_b = COOP_ACTIONS[idx_b]
			action_c = COOP_ACTIONS[idx_c]

			shout_concept_a = shout_concept_a_val if idx_a == 6 else None
			shout_concept_b = shout_concept_b_val if idx_b == 6 else None
			shout_concept_c = shout_concept_c_val if idx_c == 6 else None

			result_a, result_b, result_c, world_info = world.step(
				action_a, action_b, action_c,
				shout_concept_a=shout_concept_a, shout_concept_b=shout_concept_b, shout_concept_c=shout_concept_c
			)

			if result_a.get("shouted"):
				ep_shouts_a += 1
			if result_b.get("shouted"):
				ep_shouts_b += 1
			if result_c.get("shouted"):
				ep_shouts_c += 1

			reward_a = world.get_reward(state_a, result_a)
			reward_b = world.get_reward(state_b, result_b)
			reward_c = world.get_reward(state_c, result_c)

			# Recompensa cooperativa
			if communicate:
				reward_a += world_info.get("coop_bonus_a", 0.0)
				reward_b += world_info.get("coop_bonus_b", 0.0)
				reward_c += world_info.get("coop_bonus_c", 0.0)

			# Almacenar en buffers
			buf_a["inputs"].append(input_a.squeeze(0).cpu())
			buf_a["emotions"].append(state_a.emotion_id)
			buf_a["actions"].append(idx_a)
			buf_a["shout_concepts"].append(shout_concept_a_val)
			buf_a["log_probs"].append(lp_a)
			buf_a["values"].append(val_a)
			buf_a["rewards"].append(reward_a)

			buf_b["inputs"].append(input_b.squeeze(0).cpu())
			buf_b["emotions"].append(state_b.emotion_id)
			buf_b["actions"].append(idx_b)
			buf_b["shout_concepts"].append(shout_concept_b_val)
			buf_b["log_probs"].append(lp_b)
			buf_b["values"].append(val_b)
			buf_b["rewards"].append(reward_b)

			buf_c["inputs"].append(input_c.squeeze(0).cpu())
			buf_c["emotions"].append(state_c.emotion_id)
			buf_c["actions"].append(idx_c)
			buf_c["shout_concepts"].append(shout_concept_c_val)
			buf_c["log_probs"].append(lp_c)
			buf_c["values"].append(val_c)
			buf_c["rewards"].append(reward_c)

		# ── 4. ACTUALIZACIÓN PPO ──
		loss_a_val, loss_b_val, loss_c_val = 0.0, 0.0, 0.0

		for agent, opt, buf, state, ep_rewards, ep_values in [
			(agent_a, opt_a, buf_a, state_a, buf_a["rewards"], buf_a["values"]),
			(agent_b, opt_b, buf_b, state_b, buf_b["rewards"], buf_b["values"]),
			(agent_c, opt_c, buf_c, state_c, buf_c["rewards"], buf_c["values"])
		]:
			if not ep_rewards:
				continue

			agent.train()

			# Calcular el valor del siguiente estado
			next_val = 0.0
			if state.alive:
				with torch.no_grad():
					perc = world.perceive(state)
					if not communicate:
						perc[4] = SILENCE_GLYPH
						perc[5] = SILENCE_GLYPH
					inp = perception_to_input_coop(perc, device)
					emo = torch.tensor([state.emotion_id], device=device)
					_, m_next = agent.forward_resonance(
						inp, n_steps=n_think, pos_mode="clock", emotion_ids=emo
					)
					h_next = m_next["hidden"][:, 2, :]
					next_val = agent.value_head(h_next).item()

			# Computar ventajas usando GAE
			values = ep_values + [next_val]
			advantages = []
			gae = 0.0
			for t in reversed(range(len(ep_rewards))):
				delta = ep_rewards[t] + gamma * values[t + 1] - values[t]
				gae = delta + gamma * lmbda * gae
				advantages.insert(0, gae)

			advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=device)
			returns_tensor = advantages_tensor + torch.tensor(ep_values, dtype=torch.float32, device=device)

			if len(advantages_tensor) > 1:
				advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

			# Stackear buffers
			inputs_tensor = torch.stack(buf["inputs"], dim=0).to(device) # (T, 6)
			emotions_tensor = torch.tensor(buf["emotions"], dtype=torch.long, device=device) # (T,)
			actions_tensor = torch.tensor(buf["actions"], dtype=torch.long, device=device) # (T,)
			shout_concepts_tensor = torch.tensor(buf["shout_concepts"], dtype=torch.long, device=device) # (T,)
			old_log_probs_tensor = torch.tensor(buf["log_probs"], dtype=torch.float32, device=device) # (T,)

			# Optimizar en K épocas locales
			losses = []
			for _ in range(ppo_epochs):
				# Forward pass
				_, m_epoch = agent.forward_resonance(
					inputs_tensor, n_steps=n_think, pos_mode="clock", emotion_ids=emotions_tensor
				)
				h_epoch = m_epoch["hidden"][:, 2, :]

				new_action_logits = agent.action_head(h_epoch)
				new_values = agent.value_head(h_epoch).squeeze(-1)

				new_probs = F.softmax(new_action_logits, dim=-1)
				if not communicate:
					new_probs = new_probs.clone()
					new_probs[:, 6] = 0.0
					new_probs = new_probs / new_probs.sum(dim=-1, keepdim=True)

				dist = torch.distributions.Categorical(new_probs)
				new_log_probs = dist.log_prob(actions_tensor)
				entropy = dist.entropy()

				# Computar política de gritos
				new_shout_logits = agent._decode_hidden(h_epoch)
				new_shout_probs = F.softmax(new_shout_logits, dim=-1)
				dist_shout = torch.distributions.Categorical(new_shout_probs)
				new_log_probs_shout = dist_shout.log_prob(shout_concepts_tensor)
				entropy_shout = dist_shout.entropy()

				shout_mask = (actions_tensor == 6)

				# Sumar log probs y entropías
				new_log_probs = new_log_probs + shout_mask.float() * new_log_probs_shout
				entropy = entropy + shout_mask.float() * entropy_shout

				# Ratio r_t
				ratios = torch.exp(new_log_probs - old_log_probs_tensor)

				# Surrogate objectives
				surr1 = ratios * advantages_tensor
				surr2 = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * advantages_tensor
				policy_loss = -torch.min(surr1, surr2).mean()

				# Value Loss
				value_loss = F.mse_loss(new_values, returns_tensor)

				# Total loss
				loss = policy_loss + c1 * value_loss - entropy_bonus * entropy.mean()

				opt.zero_grad()
				loss.backward()
				torch.nn.utils.clip_grad_norm_(
					filter(lambda p: p.requires_grad, agent.parameters()),
					max_norm=grad_clip
				)
				opt.step()
				losses.append(loss.item())

			if agent == agent_a:
				loss_a_val = np.mean(losses)
			elif agent == agent_b:
				loss_b_val = np.mean(losses)
			else:
				loss_c_val = np.mean(losses)

		# ── 5. MÉTRICAS Y NEUROGENÉSIS ──
		combined = min(state_a.tick, state_b.tick, state_c.tick)
		survival_hist.append(combined)
		total_shouts_a += ep_shouts_a
		total_shouts_b += ep_shouts_b
		total_shouts_c += ep_shouts_c

		if combined > best_combined:
			best_combined = combined
			torch.save(agent_a.state_dict(), os.path.join(exp_dir, "best_agent_a.pt"))
			torch.save(agent_b.state_dict(), os.path.join(exp_dir, "best_agent_b.pt"))
			torch.save(agent_c.state_dict(), os.path.join(exp_dir, "best_agent_c.pt"))

		# Neurogénesis
		for label, model, opt_ref in [("A", agent_a, "opt_a"), ("B", agent_b, "opt_b"), ("C", agent_c, "opt_c")]:
			curr_w = model.action_head[0].out_features
			if len(survival_hist) >= 10 and curr_w < max_width:
				avg_s = np.mean(survival_hist[-10:])
				if avg_s < max_ticks * 0.10: # threshold adaptativo
					grow_action_head(model, growth_factor=growth_factor)
					grow_value_head(model, growth_factor=growth_factor)
					if label == "A":
						total_growths_a += 1
						opt_a = make_optimizer(agent_a)
					elif label == "B":
						total_growths_b += 1
						opt_b = make_optimizer(agent_b)
					else:
						total_growths_c += 1
						opt_c = make_optimizer(agent_c)
					print(f"  🧬 {label} NEUROGENÉSIS | width: {curr_w} -> {model.action_head[0].out_features}")

		# Logging periódico
		if episode % 10 == 0 or episode == n_episodes - 1:
			avg = np.mean(survival_hist[-50:]) if survival_hist else 0
			w_a = agent_a.action_head[0].out_features
			w_b = agent_b.action_head[0].out_features
			w_c = agent_c.action_head[0].out_features
			food_str = f"🍖{world_info['food_location'] or 'none'}"
			prey_str = f"🎯{world.prey_location or 'none'}"
			print(
				f"Ep {episode+1:4d}/{n_episodes} | "
				f"A:{state_a.tick:3d} B:{state_b.tick:3d} C:{state_c.tick:3d} | "
				f"Best:{best_combined:.0f} Avg50:{avg:.1f} | "
				f"Loss A:{loss_a_val:.3f} B:{loss_b_val:.3f} C:{loss_c_val:.3f} | "
				f"{'📡' if communicate else '🔇'} shouts:{ep_shouts_a+ep_shouts_b+ep_shouts_c} | "
				f"G_A:{total_growths_a} G_B:{total_growths_b} G_C:{total_growths_c} | {food_str} {prey_str}"
			)

			logger.log_epoch(
				epoch=episode + 1,
				loss_avg=(loss_a_val + loss_b_val + loss_c_val) / 3,
				acc_concept=combined,
				acc_emotion=ticks_to_survival_ratio(combined, max_ticks),
				acc_joint=combined / max_ticks * 100,
				fitness=[state_a.tick, state_b.tick, state_c.tick],
				worst_agent=0,
				parent_a=-1,
				parent_b=-1,
				acc_homeostasis=np.mean(buf_a["rewards"] + buf_b["rewards"] + buf_c["rewards"]) if buf_a["rewards"] else 0,
			)

	# Guardar checkpoints finales
	torch.save(agent_a.state_dict(), os.path.join(exp_dir, "final_agent_a.pt"))
	torch.save(agent_b.state_dict(), os.path.join(exp_dir, "final_agent_b.pt"))
	torch.save(agent_c.state_dict(), os.path.join(exp_dir, "final_agent_c.pt"))

	avg_final = np.mean(survival_hist[-100:])
	print(f"\n{'═'*60}")
	print(f"📊 Arena 3-Agent PPO Results — {experiment_id}")
	print(f"   {'📡 Communication ON' if communicate else '🔇 Silent'}")
	print(f"   Best combined: {best_combined:.0f} ticks")
	print(f"   Avg (last 100): {avg_final:.1f} ticks")
	print(f"   Total shouts: A={total_shouts_a}, B={total_shouts_b}, C={total_shouts_c}")
	print(f"   Final width: A={agent_a.action_head[0].out_features}, B={agent_b.action_head[0].out_features}, C={agent_c.action_head[0].out_features}")
	print(f"   Growths: A={total_growths_a}, B={total_growths_b}, C={total_growths_c}")
	print(f"{'═'*60}")

	logger.close()


def ticks_to_survival_ratio(ticks, max_ticks):
	return (ticks / max_ticks) * 100


if __name__ == "__main__":
	run_arena_ppo_training()
