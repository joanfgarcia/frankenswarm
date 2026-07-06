"""
EXP_050: PPO (Proximal Policy Optimization) Training for Survival.

Optimiza la política del agente utilizando PPO con Generalized Advantage Estimation (GAE)
y una cabeza de Valor (Critic) dedicada, solucionando el cuello de botella de asignación
de crédito temporal de REINFORCE.

Origen: "vamos a seguir jugando" — Joan Garcia & Aleth, 2026-06-01
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from src.bitnet.vocab.glyph_vocabulary import (
	N_EMOTIONS,
	WORD_INDEX,
)
from src.bitnet.worlds.minimal_world import (
	LOCATION_GLYPHS,
	MinimalWorld,
)
from src.bitnet.model.modeling_bitnet import BitNet4LayerModel
from src.bitnet.telemetry.telemetry import ExperimentLogger


def perception_to_input(perception: list[str], location: str, device: torch.device) -> torch.Tensor:
	"""Convertir percepción del mundo a tensor de input para el modelo."""
	loc_glyph = LOCATION_GLYPHS.get(location, "tierra")
	indices = [WORD_INDEX.get(loc_glyph, 0)]
	for word in perception:
		if word in WORD_INDEX:
			indices.append(WORD_INDEX[word])
		elif word == "montaña":
			indices.append(WORD_INDEX["piedra"])
	# Pad to 4 tokens
	while len(indices) < 4:
		indices.append(indices[-1] if indices else 0)
	indices = indices[:4]
	return torch.tensor([indices], dtype=torch.long, device=device)


def run_ppo_training():
	parser = argparse.ArgumentParser(description="EXP_050: PPO Survival")
	parser.add_argument("--config", type=str, required=True)
	args, _ = parser.parse_known_args()

	base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	config_path = args.config if os.path.isabs(args.config) else os.path.join(base_dir, args.config)
	with open(config_path, encoding="utf-8") as f:
		config = json.load(f)

	experiment_id = config["experiment_id"]
	print(f"═══ 🌍 PPO Survival — {experiment_id} ═══")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"[Device]: {device}")

	seed = config.get("seed", 42)
	np.random.seed(seed)
	torch.manual_seed(seed)

	exp_dir = os.path.join(base_dir, "storage", "experiments", experiment_id)
	os.makedirs(exp_dir, exist_ok=True)
	with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf-8") as f:
		json.dump(config, f, ensure_ascii=False, indent=4)

	# ═══ Modelo ═══
	model_cfg = config.get("model", {})
	emotion_cfg = config.get("emotion", {})
	emotion_mode = emotion_cfg.get("mode", "first_only")
	growth_cfg = config.get("growth", {})

	model = BitNet4LayerModel(
		use_glyphs=True,
		hidden_dim=model_cfg.get("hidden_dim", 256),
		num_layers=model_cfg.get("num_layers", 3),
		use_pos_embedding=model_cfg.get("use_pos_embedding", True),
		max_resonance_steps=config.get("resonance", {}).get("max_resonance_steps", 5),
		n_emotions=N_EMOTIONS,
		emotion_dim=emotion_cfg.get("dim", 64),
		emotion_mode=emotion_mode,
		action_head_width=growth_cfg.get("init_width", None),
	).to(device)

	# Pre-entrenar
	pretrained_from = config.get("pretrained_from")
	if pretrained_from:
		pretrained_path = pretrained_from if os.path.isabs(pretrained_from) else os.path.join(base_dir, pretrained_from)
		if os.path.exists(pretrained_path):
			state_dict = torch.load(pretrained_path, map_location=device, weights_only=True)
			model.load_state_dict(state_dict, strict=False)  # strict=False: heads nuevos
			print(f"🧒→🧑 Pre-trained: {pretrained_path}")

	n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f"📊 Params: {n_params:,}")

	# ═══ Freeze base model, train only action head and value head ═══
	for name, param in model.named_parameters():
		if "action_head" not in name and "value_head" not in name:
			param.requires_grad = False

	train_cfg = config.get("training", {})
	lr = train_cfg.get("lr", 5e-4)
	optimizer = torch.optim.AdamW(
		list(model.action_head.parameters()) + list(model.value_head.parameters()),
		lr=lr, weight_decay=train_cfg.get("weight_decay", 0.01),
	)

	n_trainable_action = sum(p.numel() for p in model.action_head.parameters())
	n_trainable_value = sum(p.numel() for p in model.value_head.parameters())
	print(f"🧠→🦶 Actor params: {n_trainable_action:,} | Critic params: {n_trainable_value:,} (base CONGELADA)")

	action_names = ["comer", "beber", "dormir", "mover", "ver", "piedra"]

	# ═══ PPO Hyperparameters ═══
	n_episodes = train_cfg.get("episodes", 500)
	max_ticks = config.get("world", {}).get("max_ticks", 200)
	n_think = config.get("resonance", {}).get("n_think", 2)
	n_verify = config.get("resonance", {}).get("n_verify", 2)
	use_metacognition = config.get("metacognition", {}).get("enabled", True)
	convergence_threshold = config.get("metacognition", {}).get("threshold", 0.85)
	grad_clip = train_cfg.get("grad_clip", 1.0)
	gamma = train_cfg.get("gamma", 0.99)
	lmbda = train_cfg.get("lmbda", 0.95)
	ppo_epochs = train_cfg.get("ppo_epochs", 4)
	clip_eps = train_cfg.get("clip_eps", 0.2)
	entropy_bonus = train_cfg.get("entropy_bonus", 0.01)
	c1 = train_cfg.get("value_loss_coeff", 0.5)

	print(f"🌍 Episodios: {n_episodes} | Max ticks: {max_ticks}")
	print(f"🎲 PPO Epochs: {ppo_epochs} | Clip: {clip_eps} | Gamma: {gamma} | Lambda: {lmbda}")

	logger = ExperimentLogger(
		experiment_id,
		{**config, "device": str(device), "mode": "ppo_survival"},
		log_path=os.path.join(exp_dir, "telemetry.jsonl"),
	)

	best_survival = 0.0
	survival_history = []

	for episode in range(n_episodes):
		world = MinimalWorld(seed=seed + episode)
		state = world.reset()

		# Buffer de la trayectoria del episodio
		episode_h_actions = []
		episode_action_idxs = []
		episode_log_probs = []
		episode_rewards = []
		episode_values = []
		episode_convergences = []
		episode_rethinks = 0

		model.eval()

		for _tick in range(max_ticks):
			if not state.alive:
				break

			# 1. PERCIBIR
			perception = world.perceive()
			x = perception_to_input(perception, state.location, device)
			emotion_id = torch.tensor([state.emotion_id], dtype=torch.long, device=device)

			# 2. PENSAR (con o sin metacognición)
			with torch.no_grad():
				if use_metacognition:
					logits, meta = model.forward_deep_think(
						x, n_think=n_think, n_verify=n_verify,
						pos_mode="clock", emotion_ids=emotion_id,
						convergence_threshold=convergence_threshold,
					)
					convergence = meta["convergence"]
					hidden = meta.get("hidden", None)
					if meta["n_rethinks"] > 1:
						episode_rethinks += 1
				else:
					logits, meta = model.forward_resonance(
						x, n_steps=n_think, pos_mode="clock",
						emotion_ids=emotion_id,
					)
					convergence = 1.0
					hidden = meta.get("hidden", None)

				if hidden is not None:
					h_action = hidden[:, 2, :].clone()
				else:
					_, meta_fallback = model.forward_resonance(
						x, n_steps=n_think, pos_mode="clock",
						emotion_ids=emotion_id,
					)
					h_action = meta_fallback["hidden"][:, 2, :].clone()

				# Predecir logits de acción y valor del estado
				action_logits = model.action_head(h_action)
				state_value = model.value_head(h_action).item()

			probs = F.softmax(action_logits, dim=-1)

			# Muestrear acción
			explore_rate = max(0.02, 1.0 - episode / (n_episodes * 0.4))
			if np.random.rand() < explore_rate:
				# Exploración uniforme
				action_idx = np.random.randint(0, 6)
				log_prob = torch.log(probs[0, action_idx] + 1e-8).item()
			else:
				dist = torch.distributions.Categorical(probs)
				action_idx = dist.sample().item()
				log_prob = dist.log_prob(torch.tensor(action_idx, device=device)).item()

			action_name = action_names[action_idx]

			# 3. ACTUAR
			result = world.act(action_name)
			reward = world.get_reward(result)

			# Guardar en buffer
			episode_h_actions.append(h_action)
			episode_action_idxs.append(action_idx)
			episode_log_probs.append(log_prob)
			episode_rewards.append(reward)
			episode_values.append(state_value)
			episode_convergences.append(convergence if isinstance(convergence, float) else convergence)

		# ═══ PPO Update ═══
		if episode_rewards:
			model.train()

			# Calcular el valor del siguiente estado (0 si ha muerto)
			next_value = 0.0
			if state.alive:
				with torch.no_grad():
					perception = world.perceive()
					x = perception_to_input(perception, state.location, device)
					emotion_id = torch.tensor([state.emotion_id], dtype=torch.long, device=device)
					_, meta_next = model.forward_resonance(
						x, n_steps=n_think, pos_mode="clock",
						emotion_ids=emotion_id,
					)
					h_next = meta_next["hidden"][:, 2, :]
					next_value = model.value_head(h_next).item()

			# Calcular ventajas usando GAE
			values = episode_values + [next_value]
			advantages = []
			gae = 0.0
			for t in reversed(range(len(episode_rewards))):
				delta = episode_rewards[t] + gamma * values[t + 1] - values[t]
				gae = delta + gamma * lmbda * gae
				advantages.insert(0, gae)

			advantages_batch = torch.tensor(advantages, dtype=torch.float32, device=device)
			returns_batch = advantages_batch + torch.tensor(episode_values, dtype=torch.float32, device=device)

			# Normalizar ventajas
			if len(advantages_batch) > 1:
				advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

			# Stackear hidden states (detached para optimizar)
			h_action_batch = torch.cat(episode_h_actions, dim=0).detach()  # (T, hidden_dim)
			action_idx_batch = torch.tensor(episode_action_idxs, dtype=torch.long, device=device)
			old_log_probs_batch = torch.tensor(episode_log_probs, dtype=torch.float32, device=device)

			# Optimizar Actor-Critic en K épocas
			epoch_losses = []
			for _ in range(ppo_epochs):
				# Forward rápido solo en las cabezas lineales
				new_action_logits = model.action_head(h_action_batch)
				new_values = model.value_head(h_action_batch).squeeze(-1)

				new_probs = F.softmax(new_action_logits, dim=-1)
				dist = torch.distributions.Categorical(new_probs)
				new_log_probs = dist.log_prob(action_idx_batch)
				entropy = dist.entropy()

				# Ratio r_t(theta)
				ratios = torch.exp(new_log_probs - old_log_probs_batch)

				# Clipped surrogate objective
				surr1 = ratios * advantages_batch
				surr2 = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * advantages_batch
				policy_loss = -torch.min(surr1, surr2).mean()

				# Value loss
				value_loss = F.mse_loss(new_values, returns_batch)

				# Total loss
				loss = policy_loss + c1 * value_loss - entropy_bonus * entropy.mean()

				optimizer.zero_grad()
				loss.backward()
				torch.nn.utils.clip_grad_norm_(
					list(model.action_head.parameters()) + list(model.value_head.parameters()),
					max_norm=grad_clip
				)
				optimizer.step()
				epoch_losses.append(loss.item())

			mean_loss = np.mean(epoch_losses)
		else:
			mean_loss = 0.0

		# ═══ Métricas ═══
		ticks_survived = state.tick
		avg_reward = np.mean(episode_rewards) if episode_rewards else 0
		avg_conv = np.mean([c if isinstance(c, float) else 0.0 for c in episode_convergences]) if episode_convergences else 0

		survival_history.append(ticks_survived)

		if ticks_survived > best_survival:
			best_survival = ticks_survived
			save_path = os.path.join(exp_dir, "best_agent.pt")
			torch.save(model.state_dict(), save_path)

		# Log cada 10 episodios
		if episode % 10 == 0 or episode == n_episodes - 1:
			avg_last_50 = np.mean(survival_history[-50:]) if len(survival_history) >= 50 else np.mean(survival_history)
			cause = "alive" if state.alive else f"muerto({state.emotion_name})"
			print(
				f"Ep {episode+1:4d}/{n_episodes} | "
				f"Ticks: {ticks_survived:3d} | Best: {best_survival:.0f} | "
				f"Avg50: {avg_last_50:.1f} | "
				f"Loss: {mean_loss:.4f} | Reward: {avg_reward:.3f} | "
				f"{cause} | 📍{state.location}"
			)

		logger.log_epoch(
			epoch=episode + 1,
			loss_avg=mean_loss,
			acc_concept=ticks_survived,
			acc_emotion=avg_conv * 100,
			acc_joint=ticks_survived / max_ticks * 100,
			fitness=[ticks_survived],
			worst_agent=0,
			parent_a=-1,
			parent_b=-1,
			acc_homeostasis=avg_reward * 100,
		)

	# Final
	avg_survival = np.mean(survival_history[-100:])
	print(f"\n{'═'*60}")
	print("📊 Resultados finales PPO")
	print(f"   Best: {best_survival:.0f} ticks")
	print(f"   Avg (últimos 100): {avg_survival:.1f} ticks")
	print(f"   Max posible: {max_ticks} ticks")
	print(f"{'═'*60}")

	save_path = os.path.join(exp_dir, "final_agent.pt")
	torch.save(model.state_dict(), save_path)
	print(f"💾 Agente guardado en {save_path}")

	logger.close()
	return avg_survival


if __name__ == "__main__":
	run_ppo_training()
