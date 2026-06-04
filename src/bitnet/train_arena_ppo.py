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


def get_masked_probs(logits: torch.Tensor, episode: int, n_episodes: int, communicate: bool, state_mask = None) -> torch.Tensor:
	"""Enmascarar acciones no permitidas progresivamente (curriculum de tutorial) y dinámicamente (contexto)."""
	probs = F.softmax(logits, dim=-1)
	
	# Clonar para evitar modificar in-place
	probs = probs.clone()
	
	# Crear una máscara para las acciones permitidas
	mask = torch.ones_like(probs)
	
	# Si la comunicación está desactivada, enmascarar GRITAR (6)
	if not communicate:
		mask[..., 6] = 0.0
		
	# Desenmascaramiento progresivo (Tutorial de Juego):
	progress = episode / n_episodes
	if progress < 0.20:
		# Fase 0: Supervivencia y Cooperación Básica (Comer, Beber, Dormir, Mover, Ver, Dar, Enseñar, Aprender)
		# Bloquear: Luchar (5), Gritar (6), Reproducir (10), Fabricar (11), Construir (12), Encender (13)
		mask[..., 5] = 0.0
		mask[..., 6] = 0.0
		mask[..., 10:14] = 0.0
	elif progress < 0.45:
		# Fase 1: Defensa y Herramientas (Luchar, Fabricar, Encender)
		# Bloquear: Gritar (6), Reproducir (10), Construir (12)
		mask[..., 6] = 0.0
		mask[..., 10] = 0.0
		mask[..., 12] = 0.0
	elif progress < 0.70:
		# Fase 2: Comunicación y Coordinación (Gritar)
		# Bloquear: Reproducir (10), Construir (12)
		mask[..., 10] = 0.0
		mask[..., 12] = 0.0
	elif progress < 0.85:
		# Fase 3: Asentamientos y Construcción (Construir)
		# Bloquear: Reproducir (10)
		mask[..., 10] = 0.0
	# Fase 4 (progress >= 0.85): Reproducción y Evolución (Todo desbloqueado)
		
	# Enmascaramiento contextual dinámico:
	if state_mask is not None:
		if not isinstance(state_mask, torch.Tensor):
			state_mask = torch.tensor(state_mask, dtype=torch.float, device=logits.device)
		mask = mask * state_mask
		
	masked_probs = probs * mask
	sum_probs = masked_probs.sum(dim=-1, keepdim=True)
	masked_probs = masked_probs / torch.clamp(sum_probs, min=1e-8)
	return masked_probs


def run_eval_episode(world, agent_a, agent_b, agent_c, agent_d, device, communicate, n_think=2) -> bool:
	"""Ejecuta un episodio de evaluación determinista (greedy) para verificar maestría."""
	state_a, state_b, state_c = world.reset(seed=999)
	state_d = world.agent_d
	consecutive_healthy = 0
	
	agent_a.eval()
	agent_b.eval()
	agent_c.eval()
	if agent_d is not None:
		agent_d.eval()

	for tick in range(1, 501):
		if not state_a.alive and not state_b.alive and not state_c.alive:
			break
			
		action_a = "ver"
		if state_a.alive:
			perc_a = world.perceive(state_a)
			if not communicate:
				perc_a[4] = SILENCE_GLYPH
				perc_a[5] = SILENCE_GLYPH
			inp_a = perception_to_input_coop(perc_a, device)
			emo_a = torch.tensor([state_a.emotion_id], device=device)
			with torch.no_grad():
				_, meta_a = agent_a.forward_resonance(inp_a, n_steps=n_think, pos_mode="clock", emotion_ids=emo_a)
				hidden_a = meta_a["hidden"][:, 2, :].clone()
				logits_a = agent_a.action_head(hidden_a)
				mask_a = world.get_valid_actions_mask(state_a)
				probs_a = get_masked_probs(logits_a, 1, 1, communicate, mask_a)
				action_a = COOP_ACTIONS[torch.argmax(probs_a, dim=-1).item()]

		action_b = "ver"
		if state_b.alive:
			perc_b = world.perceive(state_b)
			if not communicate:
				perc_b[4] = SILENCE_GLYPH
				perc_b[5] = SILENCE_GLYPH
			inp_b = perception_to_input_coop(perc_b, device)
			emo_b = torch.tensor([state_b.emotion_id], device=device)
			with torch.no_grad():
				_, meta_b = agent_b.forward_resonance(inp_b, n_steps=n_think, pos_mode="clock", emotion_ids=emo_b)
				hidden_b = meta_b["hidden"][:, 2, :].clone()
				logits_b = agent_b.action_head(hidden_b)
				mask_b = world.get_valid_actions_mask(state_b)
				probs_b = get_masked_probs(logits_b, 1, 1, communicate, mask_b)
				action_b = COOP_ACTIONS[torch.argmax(probs_b, dim=-1).item()]

		action_c = "ver"
		if state_c.alive:
			perc_c = world.perceive(state_c)
			if not communicate:
				perc_c[4] = SILENCE_GLYPH
				perc_c[5] = SILENCE_GLYPH
			inp_c = perception_to_input_coop(perc_c, device)
			emo_c = torch.tensor([state_c.emotion_id], device=device)
			with torch.no_grad():
				_, meta_c = agent_c.forward_resonance(inp_c, n_steps=n_think, pos_mode="clock", emotion_ids=emo_c)
				hidden_c = meta_c["hidden"][:, 2, :].clone()
				logits_c = agent_c.action_head(hidden_c)
				mask_c = world.get_valid_actions_mask(state_c)
				probs_c = get_masked_probs(logits_c, 1, 1, communicate, mask_c)
				action_c = COOP_ACTIONS[torch.argmax(probs_c, dim=-1).item()]

		action_d = "ver"
		if state_d.alive and agent_d is not None:
			perc_d = world.perceive(state_d)
			if not communicate:
				perc_d[4] = SILENCE_GLYPH
				perc_d[5] = SILENCE_GLYPH
			inp_d = perception_to_input_coop(perc_d, device)
			emo_d = torch.tensor([state_d.emotion_id], device=device)
			with torch.no_grad():
				_, meta_d = agent_d.forward_resonance(inp_d, n_steps=n_think, pos_mode="clock", emotion_ids=emo_d)
				hidden_d = meta_d["hidden"][:, 2, :].clone()
				logits_d = agent_d.action_head(hidden_d)
				mask_d = world.get_valid_actions_mask(state_d)
				probs_d = get_masked_probs(logits_d, 1, 1, communicate, mask_d)
				action_d = COOP_ACTIONS[torch.argmax(probs_d, dim=-1).item()]

		_, _, _, world_info = world.step(action_a, action_b, action_c, action_d)

		is_any_ko = (
			getattr(state_a, "debuff_inconsciente_ticks", 0) > 0 or
			getattr(state_b, "debuff_inconsciente_ticks", 0) > 0 or
			getattr(state_c, "debuff_inconsciente_ticks", 0) > 0
		)
		if is_any_ko:
			consecutive_healthy = 0
		else:
			consecutive_healthy += 1
			if consecutive_healthy >= 200:
				return True
	return False


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
		load_label = agent_label
		pretrained_from = config.get("pretrained_from")
		if pretrained_from:
			p_base = pretrained_from if os.path.isabs(pretrained_from) else os.path.join(base_dir, pretrained_from)
			if os.path.isdir(p_base):
				p = os.path.join(p_base, f"best_agent_{load_label}.pt")
				if agent_label == "c" and not os.path.exists(p):
					p = os.path.join(p_base, "best_agent_a.pt")
			else:
				p_dir = os.path.dirname(p_base)
				p_spec = os.path.join(p_dir, f"best_agent_{load_label}.pt")
				if os.path.exists(p_spec):
					p = p_spec
				elif agent_label == "c" and os.path.exists(os.path.join(p_dir, "best_agent_a.pt")):
					p = os.path.join(p_dir, "best_agent_a.pt")
				elif os.path.exists(p_base):
					p = p_base

		if p and os.path.exists(p):
			sd = torch.load(p, map_location=device, weights_only=True)
			if "glyph_embedding.glyph_table" in sd:
				del sd["glyph_embedding.glyph_table"]
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
	replenish_cooldown_ticks = world_cfg.get("replenish_cooldown_ticks", 0)

	# Growth config
	growth_cfg = config.get("growth", {})
	growth_factor = growth_cfg.get("factor", 1.5)
	max_width = growth_cfg.get("max_width", 648)

	def make_optimizer(model):
		return torch.optim.AdamW(
			filter(lambda p: p.requires_grad, model.parameters()),
			lr=lr, weight_decay=0.01
		)

	def make_child_model(width: int):
		m = BitNet4LayerModel(
			use_glyphs=True,
			hidden_dim=model_cfg.get("hidden_dim", 256),
			num_layers=model_cfg.get("num_layers", 3),
			use_pos_embedding=model_cfg.get("use_pos_embedding", True),
			max_resonance_steps=config.get("resonance", {}).get("max_resonance_steps", 5),
			n_emotions=N_EMOTIONS,
			emotion_dim=emotion_cfg.get("dim", 64),
			emotion_mode=emotion_cfg.get("mode", "first_only"),
			max_seq_len=6,
		).to(device)

		unfreeze = config.get("unfreeze_backbone", True)
		for name, param in m.named_parameters():
			if "glyph_table" in name:
				param.requires_grad = False
			elif not unfreeze and "action_head" not in name and "value_head" not in name:
				param.requires_grad = False
			else:
				param.requires_grad = True

		m.action_head = nn.Sequential(
			nn.Linear(model_cfg.get("hidden_dim", 256), width),
			nn.GELU(),
			nn.Linear(width, COOP_N_ACTIONS),
		).to(device)

		m.value_head = nn.Sequential(
			nn.Linear(model_cfg.get("hidden_dim", 256), width),
			nn.GELU(),
			nn.Linear(width, 1),
		).to(device)

		return m

	opt_a = make_optimizer(agent_a)
	opt_b = make_optimizer(agent_b)
	opt_c = make_optimizer(agent_c)

	agent_d = None
	opt_d = None

	logger = ExperimentLogger(
		experiment_id,
		{**config, "device": str(device), "mode": "ppo_arena_3agent"},
		log_path=os.path.join(exp_dir, "telemetry.jsonl"),
	)

	best_combined = 0.0
	survival_hist = []
	success_mastery = False
	total_growths_a = 0
	total_growths_b = 0
	total_growths_c = 0
	total_growths_d = 0
	total_shouts_a = 0
	total_shouts_b = 0
	total_shouts_c = 0
	total_shouts_d = 0

	world = CooperativeWorld(
		seed=seed,
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
		replenish_cooldown_ticks=replenish_cooldown_ticks,
	)

	# ── 0. DOJO PRE-TRAINING (Bootstrapping) ──
	dojo_pretrain_steps = config.get("dojo", {}).get("pretrain_steps", 50)
	dojo_pretrain_epochs = config.get("dojo", {}).get("pretrain_epochs", 10)
	if dojo_pretrain_steps > 0:
		print(f"\n🎓 [DOJO] Iniciando pre-entrenamiento del Manual de Instrucciones en el Dojo...")
		agents_dict = {"a": agent_a, "b": agent_b, "c": agent_c}
		if agent_d is not None:
			agents_dict["d"] = agent_d
		
		from src.bitnet.dojo_populora import train_dojo_step
		for step in range(dojo_pretrain_steps):
			losses = train_dojo_step(agents_dict, device, batch_size=256, lr=1e-4, epochs=dojo_pretrain_epochs, n_think=n_think)
			if (step + 1) % 10 == 0 or step == 0:
				loss_str = ", ".join(f"{k.upper()}:{v:.4f}" for k, v in losses.items())
				print(f"  [Dojo Pretrain Step {step+1}/{dojo_pretrain_steps}] Pérdidas: {loss_str}")
		print("🎓 [DOJO] Pre-entrenamiento completado con éxito. Pesos iniciales alineados con las reglas expertas.\n")

	for episode in range(n_episodes):
		state_a, state_b, state_c = world.reset(seed=seed)
		state_d = world.agent_d
		consecutive_healthy_ticks = 0

		# Trajectory buffers
		buf_a = {"inputs": [], "emotions": [], "actions": [], "shout_concepts": [], "state_masks": [], "log_probs": [], "values": [], "rewards": []}
		buf_b = {"inputs": [], "emotions": [], "actions": [], "shout_concepts": [], "state_masks": [], "log_probs": [], "values": [], "rewards": []}
		buf_c = {"inputs": [], "emotions": [], "actions": [], "shout_concepts": [], "state_masks": [], "log_probs": [], "values": [], "rewards": []}
		buf_d = {"inputs": [], "emotions": [], "actions": [], "shout_concepts": [], "state_masks": [], "log_probs": [], "values": [], "rewards": []}

		agent_a.eval()
		agent_b.eval()
		agent_c.eval()
		if agent_d is not None:
			agent_d.eval()

		ep_shouts_a = 0
		ep_shouts_b = 0
		ep_shouts_c = 0
		ep_shouts_d = 0

		agent_d_episode = agent_d

		for tick in range(max_ticks):
			# Terminar solo si todos los fundadores activos están muertos
			if not state_a.alive and not state_b.alive and not state_c.alive:
				break

			# Capturar estado de vida al inicio del tick
			state_a_alive_at_start = state_a.alive
			state_b_alive_at_start = state_b.alive
			state_c_alive_at_start = state_c.alive
			state_d_alive_at_start = state_d.alive

			# Muestreo con exploración decreciente
			explore_rate = max(0.02, 1.0 - episode / (n_episodes * 0.40))

			# Inicializar variables por defecto para evitar NameError
			input_a, input_b, input_c, input_d = None, None, None, None
			hidden_a, hidden_b, hidden_c, hidden_d = None, None, None, None
			idx_a, idx_b, idx_c, idx_d = 4, 4, 4, 4
			shout_concept_a_val, shout_concept_b_val, shout_concept_c_val, shout_concept_d_val = SILENCE_GLYPH, SILENCE_GLYPH, SILENCE_GLYPH, SILENCE_GLYPH
			lp_a, lp_b, lp_c, lp_d = 0.0, 0.0, 0.0, 0.0
			val_a, val_b, val_c, val_d = 0.0, 0.0, 0.0, 0.0
			action_a, action_b, action_c, action_d = "ver", "ver", "ver", "ver"
			shout_concept_a, shout_concept_b, shout_concept_c, shout_concept_d = None, None, None, None
			state_mask_a, state_mask_b, state_mask_c, state_mask_d = None, None, None, None

			# ── Nico (A) ──
			if state_a_alive_at_start:
				perc_a = world.perceive(state_a)
				if not communicate:
					perc_a[4] = SILENCE_GLYPH
					perc_a[5] = SILENCE_GLYPH
				input_a = perception_to_input_coop(perc_a, device)
				emo_a_id = state_a.emotion_id
				emo_a = torch.tensor([emo_a_id], device=device)

				with torch.no_grad():
					_, meta_a = agent_a.forward_resonance(
						input_a, n_steps=n_think, pos_mode="clock", emotion_ids=emo_a
					)
					hidden_a = meta_a["hidden"][:, 2, :].clone()
					action_logits_a = agent_a.action_head(hidden_a)
					shout_logits_a = agent_a._decode_hidden(hidden_a)
					val_a = agent_a.value_head(hidden_a).item()

				state_mask_a = world.get_valid_actions_mask(state_a)
				probs_a = get_masked_probs(action_logits_a, episode, n_episodes, communicate, state_mask_a)
				shout_probs_a = F.softmax(shout_logits_a, dim=-1)

				dist_shout_a = torch.distributions.Categorical(shout_probs_a)
				if np.random.rand() < explore_rate:
					shout_concept_a_val = np.random.randint(0, shout_probs_a.shape[-1])
					lp_shout_a = torch.log(shout_probs_a[0, shout_concept_a_val] + 1e-8).item()
				else:
					shout_concept_a_val = dist_shout_a.sample().item()
					lp_shout_a = dist_shout_a.log_prob(torch.tensor(shout_concept_a_val, device=device)).item()

				allowed_indices_a = torch.where(probs_a[0] > 0.0)[0].cpu().numpy()
				if np.random.rand() < explore_rate:
					idx_a = np.random.choice(allowed_indices_a)
					lp_a = torch.log(probs_a[0, idx_a] + 1e-8).item()
				else:
					dist_a = torch.distributions.Categorical(probs_a)
					idx_a = dist_a.sample().item()
					lp_a = dist_a.log_prob(torch.tensor(idx_a, device=device)).item()

				if idx_a == 6:
					lp_a = lp_a + lp_shout_a
				action_a = COOP_ACTIONS[idx_a]
				shout_concept_a = shout_concept_a_val if idx_a == 6 else None

			# ── Sofy (B) ──
			if state_b_alive_at_start:
				perc_b = world.perceive(state_b)
				if not communicate:
					perc_b[4] = SILENCE_GLYPH
					perc_b[5] = SILENCE_GLYPH
				input_b = perception_to_input_coop(perc_b, device)
				emo_b_id = state_b.emotion_id
				emo_b = torch.tensor([emo_b_id], device=device)

				with torch.no_grad():
					_, meta_b = agent_b.forward_resonance(
						input_b, n_steps=n_think, pos_mode="clock", emotion_ids=emo_b
					)
					hidden_b = meta_b["hidden"][:, 2, :].clone()
					action_logits_b = agent_b.action_head(hidden_b)
					shout_logits_b = agent_b._decode_hidden(hidden_b)
					val_b = agent_b.value_head(hidden_b).item()

				state_mask_b = world.get_valid_actions_mask(state_b)
				probs_b = get_masked_probs(action_logits_b, episode, n_episodes, communicate, state_mask_b)
				shout_probs_b = F.softmax(shout_logits_b, dim=-1)

				dist_shout_b = torch.distributions.Categorical(shout_probs_b)
				if np.random.rand() < explore_rate:
					shout_concept_b_val = np.random.randint(0, shout_probs_b.shape[-1])
					lp_shout_b = torch.log(shout_probs_b[0, shout_concept_b_val] + 1e-8).item()
				else:
					shout_concept_b_val = dist_shout_b.sample().item()
					lp_shout_b = dist_shout_b.log_prob(torch.tensor(shout_concept_b_val, device=device)).item()

				allowed_indices_b = torch.where(probs_b[0] > 0.0)[0].cpu().numpy()
				if np.random.rand() < explore_rate:
					idx_b = np.random.choice(allowed_indices_b)
					lp_b = torch.log(probs_b[0, idx_b] + 1e-8).item()
				else:
					dist_b = torch.distributions.Categorical(probs_b)
					idx_b = dist_b.sample().item()
					lp_b = dist_b.log_prob(torch.tensor(idx_b, device=device)).item()

				if idx_b == 6:
					lp_b = lp_b + lp_shout_b
				action_b = COOP_ACTIONS[idx_b]
				shout_concept_b = shout_concept_b_val if idx_b == 6 else None

			# ── Hugo (C) ──
			if state_c_alive_at_start:
				perc_c = world.perceive(state_c)
				if not communicate:
					perc_c[4] = SILENCE_GLYPH
					perc_c[5] = SILENCE_GLYPH
				input_c = perception_to_input_coop(perc_c, device)
				emo_c_id = state_c.emotion_id
				emo_c = torch.tensor([emo_c_id], device=device)

				with torch.no_grad():
					_, meta_c = agent_c.forward_resonance(
						input_c, n_steps=n_think, pos_mode="clock", emotion_ids=emo_c
					)
					hidden_c = meta_c["hidden"][:, 2, :].clone()
					action_logits_c = agent_c.action_head(hidden_c)
					shout_logits_c = agent_c._decode_hidden(hidden_c)
					val_c = agent_c.value_head(hidden_c).item()

				state_mask_c = world.get_valid_actions_mask(state_c)
				probs_c = get_masked_probs(action_logits_c, episode, n_episodes, communicate, state_mask_c)
				shout_probs_c = F.softmax(shout_logits_c, dim=-1)

				dist_shout_c = torch.distributions.Categorical(shout_probs_c)
				if np.random.rand() < explore_rate:
					shout_concept_c_val = np.random.randint(0, shout_probs_c.shape[-1])
					lp_shout_c = torch.log(shout_probs_c[0, shout_concept_c_val] + 1e-8).item()
				else:
					shout_concept_c_val = dist_shout_c.sample().item()
					lp_shout_c = dist_shout_c.log_prob(torch.tensor(shout_concept_c_val, device=device)).item()

				allowed_indices_c = torch.where(probs_c[0] > 0.0)[0].cpu().numpy()
				if np.random.rand() < explore_rate:
					idx_c = np.random.choice(allowed_indices_c)
					lp_c = torch.log(probs_c[0, idx_c] + 1e-8).item()
				else:
					dist_c = torch.distributions.Categorical(probs_c)
					idx_c = dist_c.sample().item()
					lp_c = dist_c.log_prob(torch.tensor(idx_c, device=device)).item()

				if idx_c == 6:
					lp_c = lp_c + lp_shout_c
				action_c = COOP_ACTIONS[idx_c]
				shout_concept_c = shout_concept_c_val if idx_c == 6 else None

			# ── Domi (D) ──
			if state_d_alive_at_start and agent_d_episode is not None:
				perc_d = world.perceive(state_d)
				if not communicate:
					perc_d[4] = SILENCE_GLYPH
					perc_d[5] = SILENCE_GLYPH
				input_d = perception_to_input_coop(perc_d, device)
				emo_d_id = state_d.emotion_id
				emo_d = torch.tensor([emo_d_id], device=device)

				with torch.no_grad():
					_, meta_d = agent_d_episode.forward_resonance(
						input_d, n_steps=n_think, pos_mode="clock", emotion_ids=emo_d
					)
					hidden_d = meta_d["hidden"][:, 2, :].clone()
					action_logits_d = agent_d_episode.action_head(hidden_d)
					shout_logits_d = agent_d_episode._decode_hidden(hidden_d)
					val_d = agent_d_episode.value_head(hidden_d).item()

				state_mask_d = world.get_valid_actions_mask(state_d)
				probs_d = get_masked_probs(action_logits_d, episode, n_episodes, communicate, state_mask_d)
				shout_probs_d = F.softmax(shout_logits_d, dim=-1)

				dist_shout_d = torch.distributions.Categorical(shout_probs_d)
				if np.random.rand() < explore_rate:
					shout_concept_d_val = np.random.randint(0, shout_probs_d.shape[-1])
					lp_shout_d = torch.log(shout_probs_d[0, shout_concept_d_val] + 1e-8).item()
				else:
					shout_concept_d_val = dist_shout_d.sample().item()
					lp_shout_d = dist_shout_d.log_prob(torch.tensor(shout_concept_d_val, device=device)).item()

				allowed_indices_d = torch.where(probs_d[0] > 0.0)[0].cpu().numpy()
				if np.random.rand() < explore_rate:
					idx_d = np.random.choice(allowed_indices_d)
					lp_d = torch.log(probs_d[0, idx_d] + 1e-8).item()
				else:
					dist_d = torch.distributions.Categorical(probs_d)
					idx_d = dist_d.sample().item()
					lp_d = dist_d.log_prob(torch.tensor(idx_d, device=device)).item()

				if idx_d == 6:
					lp_d = lp_d + lp_shout_d
				action_d = COOP_ACTIONS[idx_d]
				shout_concept_d = shout_concept_d_val if idx_d == 6 else None

			# ── 3. ACTUAR ──
			action_a = COOP_ACTIONS[idx_a]
			action_b = COOP_ACTIONS[idx_b]
			action_c = COOP_ACTIONS[idx_c]
			action_d = COOP_ACTIONS[idx_d]

			shout_concept_a = shout_concept_a_val if idx_a == 6 else None
			shout_concept_b = shout_concept_b_val if idx_b == 6 else None
			shout_concept_c = shout_concept_c_val if idx_c == 6 else None
			shout_concept_d = shout_concept_d_val if idx_d == 6 else None

			result_a, result_b, result_c, world_info = world.step(
				action_a, action_b, action_c, action_d,
				shout_concept_a=shout_concept_a, shout_concept_b=shout_concept_b, shout_concept_c=shout_concept_c, shout_concept_d=shout_concept_d
			)
			result_d = world_info.get("result_d")

			# Si nace Domi, instanciar dinámicamente su modelo, optimizador y buffer
			if "born_child" in result_a and agent_d_episode is None:
				child_info = result_a["born_child"]
				parent_a_id = child_info["parent_a"]
				parent_b_id = child_info["parent_b"]
				child_width = child_info["width"]

				p_models = {
					"a": agent_a,
					"b": agent_b,
					"c": agent_c
				}
				parent_model_a = p_models[parent_a_id]
				parent_model_b = p_models[parent_b_id]

				# Instanciar modelo de Domi
				agent_d = make_child_model(child_width)

				# Cruzar pesos SVD
				from src.bitnet.genetic import recombine_parents
				recombine_parents(parent_model_a, parent_model_b, agent_d, child_width, device)

				opt_d = make_optimizer(agent_d)
				buf_d = {"inputs": [], "emotions": [], "actions": [], "shout_concepts": [], "log_probs": [], "values": [], "rewards": []}
				agent_d.eval()
				agent_d_episode = agent_d

			# Registrar en el teaching_buffer para destilación latente (Fase 6)
			agents_to_check = []
			if state_a_alive_at_start:
				agents_to_check.append(("a", result_a, state_a, input_a))
			if state_b_alive_at_start:
				agents_to_check.append(("b", result_b, state_b, input_b))
			if state_c_alive_at_start:
				agents_to_check.append(("c", result_c, state_c, input_c))
			if world.agent_d_was_alive and result_d is not None and state_d_alive_at_start:
				agents_to_check.append(("d", result_d, state_d, input_d))

			for label, result_x, state_x, input_x in agents_to_check:
				if result_x.get("success") and result_x.get("learning_skill") and result_x.get("learned_from"):
					skill = result_x["learning_skill"]
					teacher_id = result_x["learned_from"]
					teacher_hidden = None
					if teacher_id == "a":
						teacher_hidden = hidden_a
					elif teacher_id == "b":
						teacher_hidden = hidden_b
					elif teacher_id == "c":
						teacher_hidden = hidden_c
					elif teacher_id == "d" and hidden_d is not None:
						teacher_hidden = hidden_d

					if teacher_hidden is not None:
						state_x.teaching_buffer.append({
							"input": input_x.clone().cpu(),
							"emotion": state_x.emotion_id,
							"teacher_hidden": teacher_hidden.clone().cpu(),
							"skill": skill,
							"teacher_id": teacher_id
						})

			if state_a_alive_at_start and result_a.get("shouted"):
				ep_shouts_a += 1
			if state_b_alive_at_start and result_b.get("shouted"):
				ep_shouts_b += 1
			if state_c_alive_at_start and result_c.get("shouted"):
				ep_shouts_c += 1
			if state_d_alive_at_start and result_d is not None and result_d.get("shouted"):
				ep_shouts_d += 1

			reward_a = world.get_reward(state_a, result_a)
			reward_b = world.get_reward(state_b, result_b)
			reward_c = world.get_reward(state_c, result_c)

			# Recompensa cooperativa
			if communicate:
				reward_a += world_info.get("coop_bonus_a", 0.0)
				reward_b += world_info.get("coop_bonus_b", 0.0)
				reward_c += world_info.get("coop_bonus_c", 0.0)

			# Almacenar en buffers
			if state_a_alive_at_start:
				buf_a["inputs"].append(input_a.squeeze(0).cpu())
				buf_a["emotions"].append(emo_a_id)
				buf_a["actions"].append(idx_a)
				buf_a["shout_concepts"].append(shout_concept_a_val)
				buf_a["state_masks"].append(state_mask_a if state_mask_a is not None else [1.0]*COOP_N_ACTIONS)
				buf_a["log_probs"].append(lp_a)
				buf_a["values"].append(val_a)
				buf_a["rewards"].append(reward_a)

			if state_b_alive_at_start:
				buf_b["inputs"].append(input_b.squeeze(0).cpu())
				buf_b["emotions"].append(emo_b_id)
				buf_b["actions"].append(idx_b)
				buf_b["shout_concepts"].append(shout_concept_b_val)
				buf_b["state_masks"].append(state_mask_b if state_mask_b is not None else [1.0]*COOP_N_ACTIONS)
				buf_b["log_probs"].append(lp_b)
				buf_b["values"].append(val_b)
				buf_b["rewards"].append(reward_b)

			if state_c_alive_at_start:
				buf_c["inputs"].append(input_c.squeeze(0).cpu())
				buf_c["emotions"].append(emo_c_id)
				buf_c["actions"].append(idx_c)
				buf_c["shout_concepts"].append(shout_concept_c_val)
				buf_c["state_masks"].append(state_mask_c if state_mask_c is not None else [1.0]*COOP_N_ACTIONS)
				buf_c["log_probs"].append(lp_c)
				buf_c["values"].append(val_c)
				buf_c["rewards"].append(reward_c)

			if world.agent_d_was_alive and result_d is not None and agent_d_episode is not None and state_d_alive_at_start:
				reward_d = world.get_reward(state_d, result_d)
				if communicate:
					reward_d += world_info.get("coop_bonus_d", 0.0)

				buf_d["inputs"].append(input_d.squeeze(0).cpu())
				buf_d["emotions"].append(emo_d_id)
				buf_d["actions"].append(idx_d)
				buf_d["shout_concepts"].append(shout_concept_d_val)
				buf_d["state_masks"].append(state_mask_d if state_mask_d is not None else [1.0]*COOP_N_ACTIONS)
				buf_d["log_probs"].append(lp_d)
				buf_d["values"].append(val_d)
				buf_d["rewards"].append(reward_d)

			# Verificar si alguno de los tres fundadores activos está en K.O. en este tick
			is_any_ko = (
				getattr(state_a, "debuff_inconsciente_ticks", 0) > 0 or
				getattr(state_b, "debuff_inconsciente_ticks", 0) > 0 or
				getattr(state_c, "debuff_inconsciente_ticks", 0) > 0
			)
			if is_any_ko:
				consecutive_healthy_ticks = 0
			else:
				consecutive_healthy_ticks += 1

		# ── Sueño y Destilación de Resonancia Latente (Fase 6) ──
		from src.bitnet.consolidate_sleep import consolidate_latent_resonance

		sleep_list = [("a", agent_a, state_a), ("b", agent_b, state_b), ("c", agent_c, state_c)]
		if agent_d_episode is not None:
			sleep_list.append(("d", agent_d_episode, state_d))

		for label, model, state in sleep_list:
			if state.teaching_buffer:
				graduated = consolidate_latent_resonance(model, state.teaching_buffer, device, n_think=n_think)

				# Aplicar las habilidades aprendidas permanentemente en el estado y recompensar maestros
				for skill in graduated:
					if skill not in state.learned_skills:
						state.learned_skills.append(skill)

						# Buscar el maestro
						teacher_id = None
						for sample in state.teaching_buffer:
							if sample["skill"] == skill:
								teacher_id = sample.get("teacher_id")
								break

						if teacher_id:
							# Recompensar al maestro con el Bono de Graduación Altruista (+15.0)
							if teacher_id == "a" and buf_a["rewards"]:
								buf_a["rewards"][-1] += 15.0
								print(f"  🏆 Maestro A (NICO) recibe +15.0 de bono altruista por graduar a {label.upper()} en '{skill}'.")
							elif teacher_id == "b" and buf_b["rewards"]:
								buf_b["rewards"][-1] += 15.0
								print(f"  🏆 Maestro B (SOFY) recibe +15.0 de bono altruista por graduar a {label.upper()} en '{skill}'.")
							elif teacher_id == "c" and buf_c["rewards"]:
								buf_c["rewards"][-1] += 15.0
								print(f"  🏆 Maestro C (HUGO) recibe +15.0 de bono altruista por graduar a {label.upper()} en '{skill}'.")
							elif teacher_id == "d" and buf_d["rewards"]:
								buf_d["rewards"][-1] += 15.0
								print(f"  🏆 Maestro D (DOMI) recibe +15.0 de bono altruista por graduar a {label.upper()} en '{skill}'.")

				# Limpiar buffer de aprendizaje una vez procesado el sueño
				state.teaching_buffer = []

		# ── Dojo de PopuLoRA (Fase 6 - Curriculum) ──
		dojo_interval = config.get("dojo", {}).get("interval", 1)
		dojo_sleep_epochs = config.get("dojo", {}).get("sleep_epochs", 5)
		if dojo_sleep_epochs > 0 and (episode + 1) % dojo_interval == 0:
			print(f"  💤 [Dojo] Nico, Sofy y Hugo visitan el Dojo de PopuLoRA con el profesor...")
			agents_dict = {"a": agent_a, "b": agent_b, "c": agent_c}
			if agent_d_episode is not None:
				agents_dict["d"] = agent_d_episode
			
			from src.bitnet.dojo_populora import train_dojo_step
			dojo_losses = train_dojo_step(agents_dict, device, batch_size=256, lr=1e-4, epochs=dojo_sleep_epochs, n_think=n_think)
			loss_str = ", ".join(f"{k.upper()}:{v:.4f}" for k, v in dojo_losses.items())
			print(f"    [Dojo] Pérdidas consolidadas: {loss_str}")

		# ── 4. ACTUALIZACIÓN PPO ──
		loss_a_val, loss_b_val, loss_c_val, loss_d_val = 0.0, 0.0, 0.0, 0.0

		agents_to_update = [
			(agent_a, opt_a, buf_a, state_a, buf_a["rewards"], buf_a["values"], "a"),
			(agent_b, opt_b, buf_b, state_b, buf_b["rewards"], buf_b["values"], "b"),
			(agent_c, opt_c, buf_c, state_c, buf_c["rewards"], buf_c["values"], "c")
		]
		if agent_d_episode is not None and len(buf_d["rewards"]) > 0:
			agents_to_update.append((agent_d_episode, opt_d, buf_d, state_d, buf_d["rewards"], buf_d["values"], "d"))

		for agent, opt, buf, state, ep_rewards, ep_values, label in agents_to_update:
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
			state_masks_tensor = torch.tensor(buf["state_masks"], dtype=torch.float32, device=device) # (T, 14)
			old_log_probs_tensor = torch.tensor(buf["log_probs"], dtype=torch.float32, device=device) # (T,)

			# Optimizar en K épocas locales
			losses = []
			
			# Congelar temporalmente el backbone para estabilidad de PPO
			orig_grads = {name: param.requires_grad for name, param in agent.named_parameters()}
			for name, param in agent.named_parameters():
				if "action_head" not in name and "value_head" not in name:
					param.requires_grad = False
					
			for _ in range(ppo_epochs):
				# Forward pass
				_, m_epoch = agent.forward_resonance(
					inputs_tensor, n_steps=n_think, pos_mode="clock", emotion_ids=emotions_tensor
				)
				h_epoch = m_epoch["hidden"][:, 2, :]

				new_action_logits = agent.action_head(h_epoch)
				new_values = agent.value_head(h_epoch).squeeze(-1)

				new_probs = get_masked_probs(new_action_logits, episode, n_episodes, communicate, state_mask=state_masks_tensor)

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

				# Ratio r_t (clamped to prevent exponential explosion on divergence)
				ratios = torch.exp(new_log_probs - old_log_probs_tensor).clamp(max=10.0)

				# Surrogate objectives
				surr1 = ratios * advantages_tensor
				surr2 = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * advantages_tensor
				policy_loss = -torch.min(surr1, surr2).mean()

				# Value Loss (using Huber loss to prevent quadratic gradient explosions on high errors)
				value_loss = F.huber_loss(new_values, returns_tensor, delta=5.0)

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

			# Restaurar estado original de requires_grad
			for name, param in agent.named_parameters():
				param.requires_grad = orig_grads[name]

			if label == "a":
				loss_a_val = np.mean(losses)
			elif label == "b":
				loss_b_val = np.mean(losses)
			elif label == "c":
				loss_c_val = np.mean(losses)
			elif label == "d":
				loss_d_val = np.mean(losses)

		# ── 5. MÉTRICAS Y NEUROGENÉSIS ──
		combined = min(state_a.tick, state_b.tick, state_c.tick)
		survival_hist.append(combined)
		total_shouts_a += ep_shouts_a
		total_shouts_b += ep_shouts_b
		total_shouts_c += ep_shouts_c
		total_shouts_d += ep_shouts_d

		if combined > best_combined:
			best_combined = combined
			torch.save(agent_a.state_dict(), os.path.join(exp_dir, "best_agent_a.pt"))
			torch.save(agent_b.state_dict(), os.path.join(exp_dir, "best_agent_b.pt"))
			torch.save(agent_c.state_dict(), os.path.join(exp_dir, "best_agent_c.pt"))
			if agent_d_episode is not None:
				torch.save(agent_d_episode.state_dict(), os.path.join(exp_dir, "best_agent_d.pt"))

		# Neurogénesis
		neuro_list = [
			("A", agent_a, state_a, "opt_a"),
			("B", agent_b, state_b, "opt_b"),
			("C", agent_c, state_c, "opt_c")
		]
		if agent_d_episode is not None:
			neuro_list.append(("D", agent_d_episode, state_d, "opt_d"))

		for label, model, state, opt_ref in neuro_list:
			curr_w = model.action_head[0].out_features
			if len(survival_hist) >= 10 and curr_w < max_width:
				avg_s = np.mean(survival_hist[-10:])
				if avg_s < max_ticks * 0.10: # threshold adaptativo
					grow_action_head(model, growth_factor=growth_factor)
					grow_value_head(model, growth_factor=growth_factor)
					
					# Actualizar el ancho del modelo y aplicar penalización metabólica diferida
					state.network_width = model.action_head[0].out_features
					state.pending_growth_penalty = True
					
					if label == "A":
						total_growths_a += 1
						opt_a = make_optimizer(agent_a)
					elif label == "B":
						total_growths_b += 1
						opt_b = make_optimizer(agent_b)
					elif label == "C":
						total_growths_c += 1
						opt_c = make_optimizer(agent_c)
					elif label == "D":
						total_growths_d += 1
						opt_d = make_optimizer(agent_d_episode)
					print(f"  🧬 {label} NEUROGENÉSIS | width: {curr_w} -> {model.action_head[0].out_features}")

		# Logging periódico
		if episode % 10 == 0 or episode == n_episodes - 1:
			avg = np.mean(survival_hist[-50:]) if survival_hist else 0
			w_a = agent_a.action_head[0].out_features
			w_b = agent_b.action_head[0].out_features
			w_c = agent_c.action_head[0].out_features
			food_str = f"🍖{world_info['food_location'] or 'none'}"
			prey_str = f"🎯{world.prey_location or 'none'}"
			d_ticks_str = f" D:{state_d.tick:3d}" if state_d.alive else " D: --"
			print(
				f"Ep {episode+1:4d}/{n_episodes} | "
				f"A:{state_a.tick:3d} B:{state_b.tick:3d} C:{state_c.tick:3d}{d_ticks_str} | "
				f"Best:{best_combined:.0f} Avg50:{avg:.1f} | "
				f"Loss A:{loss_a_val:.3f} B:{loss_b_val:.3f} C:{loss_c_val:.3f} D:{loss_d_val:.3f} | "
				f"{'📡' if communicate else '🔇'} shouts:{ep_shouts_a+ep_shouts_b+ep_shouts_c+ep_shouts_d} | "
				f"G_A:{total_growths_a} G_B:{total_growths_b} G_C:{total_growths_c} G_D:{total_growths_d} | {food_str} {prey_str}"
			)

			logger.log_epoch(
				epoch=episode + 1,
				loss_avg=(loss_a_val + loss_b_val + loss_c_val + loss_d_val) / (4 if agent_d_episode is not None else 3),
				acc_concept=combined,
				acc_emotion=ticks_to_survival_ratio(combined, max_ticks),
				acc_joint=combined / max_ticks * 100,
				fitness=[state_a.tick, state_b.tick, state_c.tick] + ([state_d.tick] if state_d.alive else []),
				worst_agent=0,
				parent_a=-1,
				parent_b=-1,
			)

		# Evaluar maestría determinista
		eval_success = run_eval_episode(world, agent_a, agent_b, agent_c, agent_d_episode, device, communicate, n_think=n_think)
		if eval_success:
			print(f"\n{'🏆'*30}")
			print(f"🥇 ¡DOMINIO ALCANZADO! Los 3 agentes han sobrevivido 200 ticks deterministas consecutivos sin entrar en K.O.")
			print(f"🥇 Deteniendo simulación de entrenamiento con éxito en el episodio {episode+1}.")
			print(f"{'🏆'*30}\n")
			torch.save(agent_a.state_dict(), os.path.join(exp_dir, "best_agent_a.pt"))
			torch.save(agent_b.state_dict(), os.path.join(exp_dir, "best_agent_b.pt"))
			torch.save(agent_c.state_dict(), os.path.join(exp_dir, "best_agent_c.pt"))
			if agent_d_episode is not None:
				torch.save(agent_d_episode.state_dict(), os.path.join(exp_dir, "best_agent_d.pt"))
			break

	# Guardar checkpoints finales
	torch.save(agent_a.state_dict(), os.path.join(exp_dir, "final_agent_a.pt"))
	torch.save(agent_b.state_dict(), os.path.join(exp_dir, "final_agent_b.pt"))
	torch.save(agent_c.state_dict(), os.path.join(exp_dir, "final_agent_c.pt"))
	if agent_d is not None:
		torch.save(agent_d.state_dict(), os.path.join(exp_dir, "final_agent_d.pt"))

	avg_final = np.mean(survival_hist[-100:])
	print(f"\n{'═'*60}")
	print(f"📊 Arena 3-Agent PPO Results — {experiment_id}")
	print(f"   {'📡 Communication ON' if communicate else '🔇 Silent'}")
	print(f"   Best combined: {best_combined:.0f} ticks")
	print(f"   Avg (last 100): {avg_final:.1f} ticks")
	print(f"   Total shouts: A={total_shouts_a}, B={total_shouts_b}, C={total_shouts_c}, D={total_shouts_d}")
	print(f"   Final width: A={agent_a.action_head[0].out_features}, B={agent_b.action_head[0].out_features}, C={agent_c.action_head[0].out_features}" + (f", D={agent_d.action_head[0].out_features}" if agent_d is not None else ""))
	print(f"   Growths: A={total_growths_a}, B={total_growths_b}, C={total_growths_c}, D={total_growths_d}")
	print(f"{'═'*60}")

	logger.close()


def ticks_to_survival_ratio(ticks, max_ticks):
	return (ticks / max_ticks) * 100


if __name__ == "__main__":
	run_arena_ppo_training()
