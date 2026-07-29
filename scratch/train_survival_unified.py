import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Añadir src al path
base_dir = "/home/joan/Documents/IA/frankenswarm"
sys.path.append(os.path.join(base_dir, "src"))

from bitnet.glyph_vocabulary import N_EMOTIONS, EMOTION_INDEX
from bitnet.minimal_world import MinimalWorld, LOCATION_GLYPHS, ACTIONS
from bitnet.modeling_bitnet import BitNet4LayerModel
from bitnet.telemetry import ExperimentLogger

def perception_to_input_unified(perception: list[str], location: str, word_to_idx: dict, device: torch.device) -> torch.Tensor:
	"""Convertir percepción del mundo a tensor de input para el modelo (vocabulario 15k)."""
	loc_glyph = LOCATION_GLYPHS.get(location, "tierra")
	indices = [word_to_idx.get(loc_glyph, 0)]
	for word in perception:
		if word in word_to_idx:
			indices.append(word_to_idx[word])
		elif word == "montaña":
			indices.append(word_to_idx.get("piedra", 0))
	# Pad to 4 tokens
	while len(indices) < 4:
		indices.append(indices[-1] if indices else 0)
	indices = indices[:4]
	return torch.tensor([indices], dtype=torch.long, device=device)

def forward_resonance_dispatch(model_a, model_b, x, emotion_ids, n_steps, pos_mode):
	"""
	Forward pass híbrido con Despacho de Tensores Intra-Forward en el bucle latente.
	Re-embebemos a través del espacio semántico del vocabulario común.
	"""
	# ── Entrada: Capas 1→2 (una sola vez en A) ──
	h = model_a._embed_input(x)
	
	# Preparar vector emocional en A
	emo_vec_a = None
	if emotion_ids is not None and model_a.emotion_embeddings is not None:
		emo_emb_a = model_a.emotion_embeddings(emotion_ids)
		emo_vec_a = model_a.emotion_proj(emo_emb_a).unsqueeze(1)

	# Bucle latente de resonancia
	trajectory_norms = []
	for step in range(n_steps):
		# Clock pos embedding en A
		if pos_mode == "clock" and getattr(model_a, "resonance_clock", None) is not None:
			h = h + model_a.resonance_clock[:, step, :].unsqueeze(1)

		# Ingestión emocional en A
		if emo_vec_a is not None:
			if model_a.emotion_mode == "first_only" and step == 0:
				h = h + emo_vec_a
			elif model_a.emotion_mode == "additive":
				h = h + emo_vec_a

		# A: Capas 0, 1, 2
		for i in range(3):
			h = model_a.core_layers[i](h)
		h = model_a.norm(h)

		# ── DESPACHO: Decodificar en A a 15k logits ──
		logits_a = model_a._decode_hidden(h)
		probs_a = F.softmax(logits_a / 1.0, dim=-1)

		# Proyectar a B: Multiplicar por embeddings de B
		word_embeds_b = model_b.glyph_embedding.get_word_embeddings()
		h_b = torch.matmul(probs_a, word_embeds_b) # (batch, seq, 384)

		# B: Capas 3, 4 (procesamiento cognitivo del experto)
		for i in range(3, 5):
			h_b = model_b.core_layers[i](h_b)
		h_b = model_b.norm(h_b)

		# ── RETORNO: Decodificar en B a 15k logits ──
		logits_b = model_b._decode_hidden(h_b)
		probs_b = F.softmax(logits_b / 1.0, dim=-1)

		# Proyectar de vuelta a A: Multiplicar por embeddings de A
		word_embeds_a = model_a.glyph_embedding.get_word_embeddings()
		h = torch.matmul(probs_b, word_embeds_a) # (batch, seq, 256)

		# A: Capas 3, 4, 5
		for i in range(3, 6):
			h = model_a.core_layers[i](h)
		h = model_a.norm(h)

		trajectory_norms.append(h.norm(dim=-1).mean().item())

	metadata = {
		"trajectory_norms": trajectory_norms,
		"hidden": h,
	}
	# El retorno principal en supervivencia son las acciones del action_head,
	# no necesitamos calcular los logits finales de vocabulario en cada paso.
	return None, metadata

def run_survival_training():
	parser = argparse.ArgumentParser(description="Survival Training with 15k Unified Vocab")
	parser.add_argument("--config", type=str, required=True)
	parser.add_argument("--episodes", type=int, default=None)
	parser.add_argument("--dispatch", action="store_true", help="Enable Intra-Forward Tensor Dispatch to Model B")
	args, _ = parser.parse_known_args()

	# Cargar configuración
	config_path = args.config if os.path.isabs(args.config) else os.path.join(base_dir, args.config)
	with open(config_path, encoding="utf-8") as f:
		config = json.load(f)

	experiment_id = config["experiment_id"]
	if args.dispatch:
		experiment_id += "_dispatch"
	print(f"═══ 🌍 Survival Unified — {experiment_id} ═══")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"[Device]: {device}")

	seed = config.get("seed", 42)
	np.random.seed(seed)
	torch.manual_seed(seed)

	exp_dir = os.path.join(base_dir, "storage", "experiments", experiment_id)
	os.makedirs(exp_dir, exist_ok=True)
	with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf-8") as f:
		json.dump(config, f, ensure_ascii=False, indent=4)

	# 1. Cargar vocabulario de 15,005
	expanded_glyphs_path = os.path.join(base_dir, "configs", "expanded_glyphs.json")
	with open(expanded_glyphs_path, encoding="utf-8") as f:
		vocab_data = json.load(f)
		words = vocab_data["words"]
		glyphs = np.array(vocab_data["glyphs"], dtype=np.float32)
	word_to_idx = {w: i for i, w in enumerate(words)}

	# 2. Inicializar Modelo A (Tronco Local / Cliente, dim 256, layers 6)
	print("Inicializando Modelo A (256-dim)...")
	model_a = BitNet4LayerModel(
		use_glyphs=True,
		glyph_table=glyphs,
		hidden_dim=256,
		num_layers=6,
		use_pos_embedding=True,
		max_resonance_steps=config.get("resonance", {}).get("max_resonance_steps", 5),
		n_emotions=N_EMOTIONS,
		emotion_dim=64,
		emotion_mode="first_only",
	).to(device)

	# Cargar checkpoint pre-entrenado de la escuela (2 años)
	path_a = os.path.join(base_dir, "storage/checkpoints/sovereign_school/model_milestone_2_years.pt")
	if os.path.exists(path_a):
		sd_a = torch.load(path_a, map_location=device, weights_only=True)
		model_a.load_state_dict(sd_a, strict=False) # strict=False para inicializar action_head aleatoriamente
		print(f"Cargado checkpoint Modelo A: {path_a}")
	else:
		print(f"⚠️ Checkpoint Modelo A no encontrado en {path_a}!")

	# 3. Si despacho activo, cargar Modelo B (Experto, dim 384, layers 6)
	model_b = None
	if args.dispatch:
		print("Inicializando Modelo B para despacho (384-dim)...")
		model_b = BitNet4LayerModel(
			use_glyphs=True,
			glyph_table=glyphs,
			hidden_dim=384,
			num_layers=6,
			use_pos_embedding=True,
			max_resonance_steps=config.get("resonance", {}).get("max_resonance_steps", 5),
			n_emotions=N_EMOTIONS,
			emotion_dim=64,
			emotion_mode="first_only",
		).to(device)

		path_b = os.path.join(base_dir, "storage/checkpoints/sovereign_school/model_milestone_3_years.pt")
		if os.path.exists(path_b):
			sd_b = torch.load(path_b, map_location=device, weights_only=True)
			model_b.load_state_dict(sd_b, strict=False)
			print(f"Cargado checkpoint Modelo B: {path_b}")
		else:
			print(f"⚠️ Checkpoint Modelo B no encontrado en {path_b}!")

		# Congelar todo el Modelo B y poner en eval
		model_b.eval()
		for param in model_b.parameters():
			param.requires_grad = False

	# 4. Congelar backbone de Model A, entrenar sólo action_head
	for name, param in model_a.named_parameters():
		if "action_head" not in name:
			param.requires_grad = False

	# Optimizar sólo action_head de A
	train_cfg = config.get("training", {})
	lr = train_cfg.get("lr", 1e-3)
	optimizer = torch.optim.AdamW(
		model_a.action_head.parameters(),
		lr=lr, weight_decay=train_cfg.get("weight_decay", 0.01)
	)

	n_trainable = sum(p.numel() for p in model_a.action_head.parameters() if p.requires_grad)
	print(f"Cabeza de Acción: {n_trainable:,} parámetros a entrenar (backbone congelado)")

	# Parámetros del bucle
	n_episodes = args.episodes if args.episodes is not None else train_cfg.get("episodes", 500)
	max_ticks = config.get("world", {}).get("max_ticks", 200)
	n_think = config.get("resonance", {}).get("n_think", 2)
	gamma = train_cfg.get("gamma", 0.99)
	entropy_bonus = train_cfg.get("entropy_bonus", 0.05)
	grad_clip = train_cfg.get("grad_clip", 1.0)

	print(f"Entrenamiento: {n_episodes} episodios, max ticks={max_ticks}")

	logger = ExperimentLogger(
		experiment_id,
		{**config, "device": str(device), "mode": "survival_unified", "dispatch": args.dispatch},
		log_path=os.path.join(exp_dir, "telemetry.jsonl"),
	)

	action_names = ["comer", "beber", "dormir", "mover", "ver", "piedra"]
	best_survival_avg = 0.0
	survival_history = []
	rewards_history = []

	for episode in range(n_episodes):
		world = MinimalWorld(seed=seed + episode)
		state = world.reset()

		episode_rewards = []
		episode_log_probs = []
		episode_entropies = []
		episode_ticks = 0

		model_a.train()

		for _tick in range(max_ticks):
			if not state.alive:
				break

			# 1. Percibir
			perception = world.perceive()
			x_input = perception_to_input_unified(perception, state.location, word_to_idx, device)
			emotion_id = torch.tensor([state.emotion_id], dtype=torch.long, device=device)

			# 2. Forward pass (híbrido o puro)
			if args.dispatch and model_b is not None:
				_, meta = forward_resonance_dispatch(
					model_a, model_b, x_input, emotion_id,
					n_steps=n_think, pos_mode="clock"
				)
				hidden = meta["hidden"]
			else:
				_, meta = model_a.forward_resonance(
					x_input, n_steps=n_think, pos_mode="clock",
					emotion_ids=emotion_id
				)
				hidden = meta["hidden"]

			# 3. Decidir acción (extraer token del índice 2)
			h_action = hidden[:, 2, :] # (batch, 256)
			action_logits = model_a.action_head(h_action)
			probs = F.softmax(action_logits, dim=-1)

			# Exploración epsilon-greedy decayente
			explore_rate = max(0.05, 1.0 - episode / (n_episodes * 0.4))
			if np.random.rand() < explore_rate:
				action_idx = np.random.randint(0, 6)
				log_prob = torch.log(probs[0, action_idx] + 1e-8).squeeze()
			else:
				dist = torch.distributions.Categorical(probs)
				action_idx = dist.sample().item()
				log_prob = dist.log_prob(torch.tensor(action_idx, device=device)).squeeze()

			dist_full = torch.distributions.Categorical(probs)
			entropy = dist_full.entropy().squeeze()

			action_name = action_names[action_idx]

			# 4. Actuar en el mundo
			result = world.act(action_name)
			reward = world.get_reward(result)

			episode_rewards.append(reward)
			episode_log_probs.append(log_prob)
			episode_entropies.append(entropy)
			episode_ticks += 1

		# 5. Policy gradient update al final del episodio
		if len(episode_rewards) > 0:
			returns = []
			G = 0
			for r in reversed(episode_rewards):
				G = r + gamma * G
				returns.insert(0, G)
			returns = torch.tensor(returns, dtype=torch.float32, device=device)
			if len(returns) > 1:
				returns = (returns - returns.mean()) / (returns.std() + 1e-8)

			loss = torch.tensor(0.0, device=device)
			for lp, g, ent in zip(episode_log_probs, returns, episode_entropies, strict=False):
				loss -= lp * g
				loss -= entropy_bonus * ent

			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model_a.action_head.parameters(), grad_clip)
			optimizer.step()

		survival_history.append(episode_ticks)
		rewards_history.append(float(sum(episode_rewards)))

		# Loguear telemetría
		loss_val = float(loss.item()) if len(episode_rewards) > 0 else 0.0
		logger.log_epoch(
			epoch=episode + 1,
			loss_avg=loss_val,
			acc_concept=float(episode_ticks),
			acc_emotion=0.0,
			acc_joint=episode_ticks / max_ticks * 100,
			fitness=[float(episode_ticks)],
			worst_agent=0,
			parent_a=-1,
			parent_b=-1,
			acc_homeostasis=float(sum(episode_rewards)),
		)

		# Imprimir progreso cada 50 episodios
		if episode % 50 == 0 or episode == n_episodes - 1:
			avg_surv = np.mean(survival_history[-50:])
			avg_rew = np.mean(rewards_history[-50:])
			print(f"Ep {episode+1:4d}/{n_episodes} | Ticks: {episode_ticks:3d} (Avg50: {avg_surv:.1f}) | Reward: {sum(episode_rewards):.1f} (Avg50: {avg_rew:.1f})")

			# Guardar el mejor modelo
			if avg_surv > best_survival_avg:
				best_survival_avg = avg_surv
				save_path = os.path.join(exp_dir, "best_agent.pt")
				torch.save(model_a.state_dict(), save_path)
				print(f"  💾 Guardado mejor agente con supervivencia avg: {best_survival_avg:.1f} en {save_path}")

	# Guardar modelo final
	final_path = os.path.join(exp_dir, "final_agent.pt")
	torch.save(model_a.state_dict(), final_path)
	print(f"🏁 Entrenamiento completado. Agente guardado en {final_path}")
	logger.close()

if __name__ == "__main__":
	run_survival_training()
