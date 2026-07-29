"""
EXP_040: Neurogenésis — Aprendizaje continuo con crecimiento neural.

El modelo aprende EN VIVO (online, cada N ticks) y CRECE cuando
su convergencia cae persistentemente (Net2WiderNet).

Bucle:
  1. Percibir → Sentir → Pensar → Actuar (como EXP_039)
  2. Cada update_interval ticks: actualizar action head (online REINFORCE)
  3. Si convergencia < growth_threshold durante growth_patience ticks:
     → Net2WiderNet: action head crece
     → Modelo tiene MÁS capacidad para problemas nuevos

Hipótesis:
  H₁: El modelo crece solo cuando lo necesita
  H₂: Tras crecer, convergencia sube (novedad resuelta)
  H₃: Conocimiento previo se preserva (no forgetting)
  H₄: Tamaño final ∝ complejidad del entorno

Origen: Joan Garcia — "Net2Net en caliente?" — 2026-05-31
"""

import argparse
import json
import os
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from src.bitnet.worlds.complex_world import (
	COMPLEX_ACTIONS,
	ComplexWorld,
)
from src.bitnet.vocab.glyph_vocabulary import (
	N_EMOTIONS,
)
from src.bitnet.worlds.minimal_world import (
	MinimalWorld,
)
from src.bitnet.model.modeling_bitnet import BitNet4LayerModel
from src.bitnet.growth.net2net import grow_action_head
from src.bitnet.telemetry.telemetry import ExperimentLogger
from lab.experiments.train_survival import perception_to_input


def run_neurogenesis_training():
	parser = argparse.ArgumentParser(description="EXP_040: Neurogenesis")
	parser.add_argument("--config", type=str, required=True)
	args, _ = parser.parse_known_args()

	base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	config_path = args.config if os.path.isabs(args.config) else os.path.join(base_dir, args.config)
	with open(config_path, encoding="utf-8") as f:
		config = json.load(f)

	experiment_id = config["experiment_id"]
	print(f"═══ 🧬 Neurogenesis — {experiment_id} ═══")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	seed = config.get("seed", 42)
	np.random.seed(seed)
	torch.manual_seed(seed)

	exp_dir = os.path.join(base_dir, "storage", "experiments", experiment_id)
	os.makedirs(exp_dir, exist_ok=True)

	# ═══ Modelo ═══
	model_cfg = config.get("model", {})
	emotion_cfg = config.get("emotion", {})

	model = BitNet4LayerModel(
		use_glyphs=True,
		hidden_dim=model_cfg.get("hidden_dim", 256),
		num_layers=model_cfg.get("num_layers", 3),
		use_pos_embedding=model_cfg.get("use_pos_embedding", True),
		max_resonance_steps=config.get("resonance", {}).get("max_resonance_steps", 5),
		n_emotions=N_EMOTIONS,
		emotion_dim=emotion_cfg.get("dim", 64),
		emotion_mode=emotion_cfg.get("mode", "first_only"),
	).to(device)

	pretrained_from = config.get("pretrained_from")
	if pretrained_from:
		p = pretrained_from if os.path.isabs(pretrained_from) else os.path.join(base_dir, pretrained_from)
		if os.path.exists(p):
			sd = torch.load(p, map_location=device, weights_only=True)
			# Reconstruct action head if checkpoint has different shape (grown model)
			if "action_head.0.weight" in sd:
				ckpt_width = sd["action_head.0.weight"].shape[0]
				ckpt_out = sd["action_head.2.weight"].shape[0]
				if ckpt_width != model.action_head[0].out_features:
					import torch.nn as tnn
					model.action_head = tnn.Sequential(
						tnn.Linear(model_cfg.get("hidden_dim", 256), ckpt_width),
						tnn.GELU(),
						tnn.Linear(ckpt_width, ckpt_out),
					).to(device)
			model.load_state_dict(sd, strict=False)
			print(f"🧒→🧑 Pre-trained: {p} (head width={model.action_head[0].out_features})")

	# Resize action head if config specifies a different width
	init_width = config.get("growth", {}).get("init_width", None)
	if init_width and init_width != model.action_head[0].out_features:
		import torch.nn as tnn
		model.action_head = tnn.Sequential(
			tnn.Linear(model_cfg.get("hidden_dim", 256), init_width),
			tnn.GELU(),
			tnn.Linear(init_width, 6),
		).to(device)
		print(f"🔧 Action head resized to width={init_width}")

	# Freeze base, train action head
	for name, param in model.named_parameters():
		if "action_head" not in name:
			param.requires_grad = False

	train_cfg = config.get("training", {})
	lr = train_cfg.get("lr", 5e-4)
	optimizer = torch.optim.AdamW(model.action_head.parameters(), lr=lr, weight_decay=0.01)

	# ═══ World selection ═══
	world_type = config.get("world", {}).get("type", "simple")
	if world_type == "complex":
		action_names = COMPLEX_ACTIONS
		n_actions = len(COMPLEX_ACTIONS)
		print(f"🌍 Mundo COMPLEJO: 15 loc, {n_actions} acciones")
	else:
		action_names = ["comer", "beber", "dormir", "mover", "ver", "piedra"]
		n_actions = 6
		print(f"🌍 Mundo simple: 5 loc, {n_actions} acciones")

	# Resize action head output if needed (preserve learned first layer for curriculum)
	if model.action_head[2].out_features != n_actions:
		import torch.nn as tnn
		old_width = model.action_head[0].out_features
		old_first_layer = model.action_head[0]  # preserve learned weights
		old_gelu = model.action_head[1]
		model.action_head = tnn.Sequential(
			old_first_layer,          # KEEP: carries learned representations
			old_gelu,
			tnn.Linear(old_width, n_actions),  # NEW: output for new action space
		).to(device)
		print(f"🔧 Action head output resized: {model.action_head[2].out_features} actions (width={old_width} preserved)")

	n_think = config.get("resonance", {}).get("n_think", 2)
	n_verify = config.get("resonance", {}).get("n_verify", 2)

	# ═══ Neurogenesis config ═══
	growth_cfg = config.get("growth", {})
	growth_threshold = growth_cfg.get("convergence_threshold", 0.80)
	growth_patience = growth_cfg.get("patience", 50)  # ticks bajo umbral
	growth_factor = growth_cfg.get("factor", 1.5)
	max_width = growth_cfg.get("max_width", 512)
	update_interval = train_cfg.get("update_interval", 16)  # online update cada N ticks
	gamma = train_cfg.get("gamma", 0.97)
	entropy_bonus = train_cfg.get("entropy_bonus", 0.05)
	grad_clip = train_cfg.get("grad_clip", 1.0)

	n_episodes = train_cfg.get("episodes", 500)
	base_max_ticks = config.get("world", {}).get("max_ticks", 200)
	base_width = 128  # reference width for lifespan scaling
	lifespan_scaling = growth_cfg.get("lifespan_scaling", True)
	plateau_window = growth_cfg.get("plateau_window", 200)  # episodes without improvement

	head_width = model.action_head[0].out_features
	head_params = sum(p.numel() for p in model.action_head.parameters())
	print(f"🧠 Action head: width={head_width}, params={head_params:,}")
	print(f"🧬 Growth: threshold={growth_threshold}, patience={growth_patience}, factor={growth_factor}x, max={max_width}")
	print(f"📡 Online update: every {update_interval} ticks")
	if lifespan_scaling:
		print(f"🕐 Lifespan scaling: ON (base={base_max_ticks} ticks at width={base_width})")

	logger = ExperimentLogger(
		experiment_id,
		{**config, "device": str(device), "mode": "neurogenesis"},
		log_path=os.path.join(exp_dir, "telemetry.jsonl"),
	)

	# ═══ Estado de neurogenésis ═══
	growth_events = []
	convergence_history = deque(maxlen=growth_patience)
	reward_history = deque(maxlen=growth_patience)
	total_growths = 0
	best_survival = 0.0
	best_survival_episode = 0  # for plateau detection
	survival_history = []

	for episode in range(n_episodes):
		world = ComplexWorld(seed=seed + episode) if world_type == "complex" else MinimalWorld(seed=seed + episode)
		state = world.reset()

		# Buffers para online update
		buffer_log_probs = []
		buffer_rewards = []
		buffer_entropies = []

		episode_ticks = 0
		episode_rethinks = 0
		episode_convergences = []

		model.train()

		# ═══ Dynamic lifespan: more brain = longer life ═══
		current_width = model.action_head[0].out_features
		max_ticks = int(base_max_ticks * (current_width / base_width) ** 0.5) if lifespan_scaling else base_max_ticks

		for tick in range(max_ticks):
			if not state.alive:
				break

			# 1. PERCIBIR
			perception = world.perceive()
			x = perception_to_input(perception, state.location, device)

			# 2. SENTIR
			emotion_id = torch.tensor([state.emotion_id], dtype=torch.long, device=device)

			# 3. PENSAR
			logits, meta = model.forward_deep_think(
				x, n_think=n_think, n_verify=n_verify,
				pos_mode="clock", emotion_ids=emotion_id,
				convergence_threshold=0.85,
			)
			convergence = meta["convergence"]
			hidden = meta.get("hidden", None)
			if meta["n_rethinks"] > 1:
				episode_rethinks += 1

			# 4. DECIDIR
			if hidden is not None:
				h_action = hidden[:, 2, :]
			else:
				h_action = logits[:, 2, :].unsqueeze(0)  # fallback

			action_logits = model.action_head(h_action)
			probs = F.softmax(action_logits, dim=-1)

			explore_rate = max(0.05, 1.0 - episode / (n_episodes * 0.4))
			if torch.rand(1).item() < explore_rate:
				action_idx = torch.randint(0, n_actions, (1,)).item()
				log_prob = torch.log(probs[0, action_idx] + 1e-8).squeeze()
			else:
				dist = torch.distributions.Categorical(probs)
				action_idx = dist.sample().item()
				log_prob = dist.log_prob(torch.tensor(action_idx, device=device))

			entropy = torch.distributions.Categorical(probs).entropy().mean()

			# 5. ACTUAR
			result = world.act(action_names[action_idx])
			reward = world.get_reward(result)

			buffer_log_probs.append(log_prob)
			buffer_rewards.append(reward)
			buffer_entropies.append(entropy)
			episode_convergences.append(convergence)
			convergence_history.append(convergence)
			reward_history.append(reward)

			episode_ticks += 1

			# ═══ ONLINE UPDATE (cada N ticks) ═══
			if len(buffer_log_probs) >= update_interval:
				# Mini REINFORCE
				returns = []
				G = 0
				for r in reversed(buffer_rewards):
					G = r + gamma * G
					returns.insert(0, G)
				returns = torch.tensor(returns, dtype=torch.float32, device=device)
				if len(returns) > 1:
					returns = (returns - returns.mean()) / (returns.std() + 1e-8)

				policy_loss = torch.tensor(0.0, device=device)
				for i in range(len(buffer_log_probs)):
					lp = buffer_log_probs[i].squeeze()
					policy_loss = policy_loss - lp * returns[i] - entropy_bonus * buffer_entropies[i]
				policy_loss = policy_loss / len(buffer_log_probs)

				optimizer.zero_grad()
				policy_loss.backward()
				torch.nn.utils.clip_grad_norm_(model.action_head.parameters(), max_norm=grad_clip)
				optimizer.step()

				# Limpiar buffers
				buffer_log_probs = []
				buffer_rewards = []
				buffer_entropies = []

			# ═══ NEUROGENÉSIS CHECK ═══
			# Dual signal: convergencia baja (cerebro confuso) O dolor persistente (mundo duele)
			if len(convergence_history) >= growth_patience:
				avg_conv = np.mean(list(convergence_history))
				avg_reward = np.mean(list(reward_history)) if len(reward_history) >= growth_patience else 0
				current_width = model.action_head[0].out_features

				# Crecer si: cerebro confuso O sufriendo mucho
				needs_growth = (avg_conv < growth_threshold) or (avg_reward < -8.0)

				if needs_growth and current_width < max_width:
					# ¡CRECER!
					growth_info = grow_action_head(model, growth_factor=growth_factor)
					total_growths += 1

					# Recrear optimizer para nuevos params
					optimizer = torch.optim.AdamW(
						model.action_head.parameters(), lr=lr, weight_decay=0.01
					)

					growth_events.append({
						"episode": episode + 1,
						"tick": tick,
						"avg_convergence": avg_conv,
						"avg_reward": avg_reward,
						"trigger": "convergence" if avg_conv < growth_threshold else "pain",
						**growth_info,
					})

					print(
						f"  🧬 NEUROGENÉSIS #{total_growths} | "
						f"width: {growth_info['old_width']}→{growth_info['new_width']} | "
						f"params: {growth_info['old_params']:,}→{growth_info['new_params']:,} | "
						f"conv={avg_conv:.3f}"
					)

					# Reset convergence history
					convergence_history.clear()
					reward_history.clear()

					# Limpiar buffers (gradients del head viejo ya no sirven)
					buffer_log_probs = []
					buffer_rewards = []
					buffer_entropies = []

		# ═══ Flush remaining buffer ═══
		if buffer_log_probs and len(buffer_log_probs) > 1:
			returns = []
			G = 0
			for r in reversed(buffer_rewards):
				G = r + gamma * G
				returns.insert(0, G)
			returns = torch.tensor(returns, dtype=torch.float32, device=device)
			if len(returns) > 1:
				returns = (returns - returns.mean()) / (returns.std() + 1e-8)

			policy_loss = torch.tensor(0.0, device=device)
			for i in range(len(buffer_log_probs)):
				lp = buffer_log_probs[i].squeeze()
				policy_loss = policy_loss - lp * returns[i] - entropy_bonus * buffer_entropies[i]
			policy_loss = policy_loss / len(buffer_log_probs)

			optimizer.zero_grad()
			policy_loss.backward()
			torch.nn.utils.clip_grad_norm_(model.action_head.parameters(), max_norm=grad_clip)
			optimizer.step()

		# ═══ Métricas ═══
		ticks_survived = state.tick
		avg_conv = np.mean(episode_convergences) if episode_convergences else 0
		survival_history.append(ticks_survived)

		if ticks_survived > best_survival:
			best_survival = ticks_survived
			best_survival_episode = episode
			torch.save(model.state_dict(), os.path.join(exp_dir, "best_agent.pt"))

		# ═══ POST-EPISODE GROWTH CHECK ═══
		# Plateau-based: if no improvement for N episodes → brain has hit its ceiling
		current_width = model.action_head[0].out_features
		episodes_since_improvement = episode - best_survival_episode
		if episode >= 20 and current_width < max_width and episodes_since_improvement >= plateau_window:
			# Has the agent plateaued? Check if avg is stable (not still learning)
			if len(survival_history) >= plateau_window:
				avg_first_half = np.mean(survival_history[-plateau_window:-plateau_window//2])
				avg_second_half = np.mean(survival_history[-plateau_window//2:])
				improvement = (avg_second_half - avg_first_half) / max(avg_first_half, 1)
				# If less than 5% improvement in the window → plateau confirmed
				if improvement < 0.05:
					growth_info = grow_action_head(model, growth_factor=growth_factor)
					total_growths += 1
					optimizer = torch.optim.AdamW(
						model.action_head.parameters(), lr=lr, weight_decay=0.01
					)
					# Update lifespan after growth
					new_width = model.action_head[0].out_features
					new_max_ticks = int(base_max_ticks * (new_width / base_width) ** 0.5) if lifespan_scaling else base_max_ticks
					growth_events.append({
						"episode": episode + 1,
						"tick": -1,
						"avg_convergence": avg_conv,
						"avg_survival": np.mean(survival_history[-10:]),
						"trigger": "plateau",
						"episodes_since_improvement": episodes_since_improvement,
						"new_max_ticks": new_max_ticks,
						**growth_info,
					})
					print(
						f"  🧬 NEUROGENÉSIS #{total_growths} (plateau) | "
						f"width: {growth_info['old_width']}→{growth_info['new_width']} | "
						f"stalled {episodes_since_improvement} ep | "
						f"lifespan: {max_ticks}→{new_max_ticks} ticks"
					)
					best_survival_episode = episode  # reset plateau counter
					convergence_history.clear()
					reward_history.clear()

		current_width = model.action_head[0].out_features
		current_params = sum(p.numel() for p in model.action_head.parameters())

		if episode % 10 == 0 or episode == n_episodes - 1:
			avg50 = np.mean(survival_history[-50:]) if len(survival_history) >= 50 else np.mean(survival_history)
			cause = "alive" if state.alive else f"muerto({state.emotion_name})"
			print(
				f"Ep {episode+1:4d}/{n_episodes} | "
				f"Ticks: {ticks_survived:3d} | Best: {best_survival:.0f} | "
				f"Avg50: {avg50:.1f} | Conv: {avg_conv:.3f} | "
				f"🧬 w={current_width} p={current_params:,} | "
				f"Growths: {total_growths} | "
				f"{cause} | 📍{state.location}"
			)

		logger.log_epoch(
			epoch=episode + 1,
			loss_avg=0,
			acc_concept=ticks_survived,
			acc_emotion=avg_conv * 100,
			acc_joint=ticks_survived / max_ticks * 100,
			fitness=[ticks_survived, current_width, current_params],
			worst_agent=0,
			parent_a=-1,
			parent_b=-1,
			acc_homeostasis=total_growths,
		)

	# ═══ Final ═══
	print(f"\n{'═'*60}")
	print("📊 Resultados finales — Neurogenésis")
	print(f"   Best: {best_survival:.0f} ticks")
	print(f"   Avg (últimos 100): {np.mean(survival_history[-100:]):.1f}")
	print(f"   Growth events: {total_growths}")
	print(f"   Final width: {model.action_head[0].out_features}")
	print(f"   Final params: {sum(p.numel() for p in model.action_head.parameters()):,}")
	print(f"   Inicio → Final: {head_params:,} → {sum(p.numel() for p in model.action_head.parameters()):,}")
	if growth_events:
		print("\n   📈 Growth timeline:")
		for g in growth_events:
			print(f"      Ep {g['episode']:4d} | {g['old_width']}→{g['new_width']} | conv={g['avg_convergence']:.3f}")
	print(f"{'═'*60}")

	torch.save(model.state_dict(), os.path.join(exp_dir, "final_agent.pt"))
	with open(os.path.join(exp_dir, "growth_events.json"), "w") as f:
		json.dump(growth_events, f, indent=2)

	logger.close()
	return best_survival


if __name__ == "__main__":
	run_neurogenesis_training()
