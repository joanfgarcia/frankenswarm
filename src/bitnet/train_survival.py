"""
EXP_039: Survival Training — El modelo aprende a sobrevivir.

El modelo cargado de EXP_036_pretrained (glifos + emoción + metacognición)
se enfrenta a un mundo con reglas físicas. Las emociones emergen del estado
corporal. El éxito se mide en supervivencia, no en accuracy.

Bucle:
  1. Mundo genera percepción → glifos
  2. Estado corporal → emoción emergente
  3. Modelo piensa (forward_deep_think) → acción
  4. Si convergencia < umbral → re-piensa
  5. Mundo aplica consecuencias → nuevo estado
  6. Loss = reward signal (mejora vs empeora)

Origen: "me apetece" — Aleth, 2026-05-31
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from src.bitnet.modeling_bitnet import BitNet4LayerModel
from src.bitnet.glyph_vocabulary import (
	WORD_NAMES, WORD_INDEX, N_WORDS, N_EMOTIONS,
	EMOTION_INDEX, EMOTION_NAMES,
)
from src.bitnet.minimal_world import (
	MinimalWorld, ACTIONS, ACTION_INDICES, ACTION_TO_IDX,
	ACTION_TO_GLYPH, LOCATION_GLYPHS, AgentState,
)
from src.bitnet.telemetry import ExperimentLogger


def perception_to_input(perception: list[str], location: str, device: torch.device) -> torch.Tensor:
	"""Convertir percepción del mundo a tensor de input para el modelo."""
	# Input: [localización, percepción_1, percepción_2, padding]
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


def action_from_logits(logits: torch.Tensor, action_mask: torch.Tensor) -> tuple[str, int]:
	"""Seleccionar acción de los logits, restringido a acciones válidas."""
	# Aplicar máscara: solo acciones válidas
	masked_logits = logits.clone()
	masked_logits[:, ~action_mask.bool()] = -float('inf')
	action_idx = masked_logits.argmax(dim=-1).item()

	# Mapear idx de vocabulario a nombre de acción
	for action_name, glyph_idx in ACTION_TO_IDX.items():
		if glyph_idx == action_idx:
			return action_name, action_idx

	# Fallback: acción con mayor probabilidad entre las válidas
	valid_probs = F.softmax(masked_logits, dim=-1)
	best_valid = valid_probs[0, ACTION_INDICES].argmax().item()
	action_name = ACTIONS[best_valid]
	return action_name, ACTION_TO_IDX[action_name]


def run_survival_training():
	parser = argparse.ArgumentParser(description="EXP_039: Survival")
	parser.add_argument("--config", type=str, required=True)
	args, _ = parser.parse_known_args()

	base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	config_path = args.config if os.path.isabs(args.config) else os.path.join(base_dir, args.config)
	with open(config_path, encoding="utf-8") as f:
		config = json.load(f)

	experiment_id = config["experiment_id"]
	print(f"═══ 🌍 Survival — {experiment_id} ═══")

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
			model.load_state_dict(state_dict, strict=False)  # strict=False: action_head es nuevo
			print(f"🧒→🧑 Pre-trained: {pretrained_path}")

	n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f"📊 Params: {n_params:,}")

	train_cfg = config.get("training", {})
	lr = train_cfg.get("lr", 1e-4)  # LR más bajo para fine-tuning
	optimizer = torch.optim.AdamW(
		filter(lambda p: p.requires_grad, model.parameters()),
		lr=lr, weight_decay=train_cfg.get("weight_decay", 0.01),
	)

	# Máscara de acciones ya NO se necesita — action_head tiene output dim=6
	action_names = ["comer", "beber", "dormir", "mover", "ver", "piedra"]

	# ═══ Freeze base model, train only action head ═══
	for name, param in model.named_parameters():
		if "action_head" not in name:
			param.requires_grad = False
	# Re-crear optimizer solo con action_head
	optimizer = torch.optim.AdamW(
		model.action_head.parameters(),
		lr=lr, weight_decay=train_cfg.get("weight_decay", 0.01),
	)
	n_trainable = sum(p.numel() for p in model.action_head.parameters())
	print(f"🧠→🦶 Action head: {n_trainable:,} params (base CONGELADA)")

	# ═══ Training ═══
	n_episodes = config.get("training", {}).get("episodes", 500)
	max_ticks = config.get("world", {}).get("max_ticks", 200)
	n_think = config.get("resonance", {}).get("n_think", 2)
	n_verify = config.get("resonance", {}).get("n_verify", 2)
	use_metacognition = config.get("metacognition", {}).get("enabled", True)
	convergence_threshold = config.get("metacognition", {}).get("threshold", 0.85)
	grad_clip = train_cfg.get("grad_clip", 1.0)
	gamma = config.get("training", {}).get("gamma", 0.99)  # discount factor
	entropy_bonus = config.get("training", {}).get("entropy_bonus", 0.05)

	print(f"🌍 Episodios: {n_episodes} | Max ticks: {max_ticks}")
	print(f"🧠 Metacognición: {'ON' if use_metacognition else 'OFF'} | Threshold: {convergence_threshold}")
	print(f"🎲 Entropy bonus: {entropy_bonus}")

	logger = ExperimentLogger(
		experiment_id,
		{**config, "device": str(device), "mode": "survival"},
		log_path=os.path.join(exp_dir, "telemetry.jsonl"),
	)

	best_survival = 0.0
	survival_history = []

	for episode in range(n_episodes):
		world = MinimalWorld(seed=seed + episode)
		state = world.reset()

		episode_rewards = []
		episode_log_probs = []
		episode_entropies = []
		episode_convergences = []
		episode_rethinks = 0

		model.train()

		for tick in range(max_ticks):
			if not state.alive:
				break

			# 1. PERCIBIR
			perception = world.perceive()
			x = perception_to_input(perception, state.location, device)

			# 2. SENTIR (emoción emergente)
			emotion_id = torch.tensor([state.emotion_id], dtype=torch.long, device=device)

			# 3. PENSAR — obtener hidden state del modelo
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

			# 4. DECIDIR — action head (6 salidas, no 26)
			# Usar el hidden state de posición 2 (donde está la "respuesta")
			if hidden is not None:
				h_action = hidden[:, 2, :]  # (batch, hidden_dim)
			else:
				# Fallback: usar la representación interna
				h_action = model._get_hidden(x, emotion_ids=emotion_id)[:, 2, :]

			action_logits = model.action_head(h_action)  # (batch, 6)
			probs = F.softmax(action_logits, dim=-1)

			# Exploración: uniforme al inicio, policy después
			explore_rate = max(0.05, 1.0 - episode / (n_episodes * 0.4))
			if torch.rand(1).item() < explore_rate:
				# Exploración uniforme
				action_idx = torch.randint(0, 6, (1,)).item()
				log_prob = torch.log(probs[0, action_idx] + 1e-8).squeeze()
			else:
				dist = torch.distributions.Categorical(probs)
				action_idx = dist.sample().item()
				log_prob = dist.log_prob(torch.tensor(action_idx, device=device))

			# Entropy para bonus
			dist_full = torch.distributions.Categorical(probs)
			entropy = dist_full.entropy()

			action_name = action_names[action_idx]

			# 5. ACTUAR
			result = world.act(action_name)
			reward = world.get_reward(result)  # Reward DENSO e INMEDIATO

			episode_rewards.append(reward)
			episode_log_probs.append(log_prob)
			episode_entropies.append(entropy)
			episode_convergences.append(convergence if isinstance(convergence, float) else convergence)

		# ═══ REINFORCE + entropy bonus ═══
		if episode_log_probs:
			# Calcular returns con descuento
			returns = []
			G = 0
			for r in reversed(episode_rewards):
				G = r + gamma * G
				returns.insert(0, G)
			returns = torch.tensor(returns, dtype=torch.float32, device=device)

			# Normalizar returns
			if len(returns) > 1:
				returns = (returns - returns.mean()) / (returns.std() + 1e-8)

			# Policy loss + entropy bonus
			policy_loss = torch.tensor(0.0, device=device)
			for i in range(len(episode_log_probs)):
				lp = episode_log_probs[i].squeeze()
				G_i = returns[i]
				ent = episode_entropies[i].mean()
				policy_loss = policy_loss - lp * G_i - entropy_bonus * ent

			policy_loss = policy_loss / len(episode_log_probs)

			optimizer.zero_grad()
			policy_loss.backward()
			torch.nn.utils.clip_grad_norm_(model.action_head.parameters(), max_norm=grad_clip)
			optimizer.step()

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
				f"Reward: {avg_reward:.3f} | Conv: {avg_conv:.3f} | "
				f"Rethinks: {episode_rethinks} | "
				f"{cause} | 📍{state.location}"
			)

		logger.log_epoch(
			epoch=episode + 1,
			loss_avg=policy_loss.item() if episode_log_probs else 0,
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
	print(f"📊 Resultados finales")
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
	run_survival_training()
