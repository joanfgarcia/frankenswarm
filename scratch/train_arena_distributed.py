import os
import sys
import json
import socket
import struct
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Añadir src al path
base_dir = "/home/joan/Documents/IA/frankenswarm"
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, "src"))

# Monkeypatch del CooperativeWorld con el vocabulario de 15k antes de instanciar nada
import bitnet.cooperative_world as coop_world

expanded_glyphs_path = os.path.join(base_dir, "configs", "expanded_glyphs.json")
with open(expanded_glyphs_path, encoding="utf-8") as f:
	vocab_data = json.load(f)
	words = vocab_data["words"]
	glyphs = np.array(vocab_data["glyphs"], dtype=np.float32)

word_to_idx_15k = {w: i for i, w in enumerate(words)}
coop_world.WORD_INDEX = word_to_idx_15k
coop_world.SILENCE_GLYPH = word_to_idx_15k.get("noche", 0)

from bitnet.glyph_vocabulary import N_EMOTIONS
from bitnet.cooperative_world import CooperativeWorld, COOP_ACTIONS, COOP_N_ACTIONS, SILENCE_GLYPH
from bitnet.modeling_bitnet import BitNet4LayerModel
from bitnet.telemetry import ExperimentLogger

class SocketCognitiveClient:
	def __init__(self, socket_path="/tmp/bit_cognitive.sock"):
		self.socket_path = socket_path

	def dispatch(self, probs_a: torch.Tensor) -> torch.Tensor:
		# Convertir a numpy cpu float32
		probs_a_np = probs_a.detach().cpu().numpy().astype(np.float32)
		B, S, D = probs_a_np.shape

		# Conectar a socket UNIX
		s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
		try:
			s.connect(self.socket_path)
			# Enviar cabecera (12 bytes: B, S, D)
			header = struct.pack('!3i', B, S, D)
			s.sendall(header)
			# Enviar tensor serializado
			s.sendall(probs_a_np.tobytes())

			# Leer cabecera de respuesta
			resp_header = s.recv(12)
			if not resp_header or len(resp_header) < 12:
				raise RuntimeError("Error leyendo cabecera de respuesta del servidor")
			B_out, S_out, D_out = struct.unpack('!3i', resp_header)

			# Leer payload de respuesta
			payload_len = B_out * S_out * D_out * 4
			data = b''
			while len(data) < payload_len:
				chunk = s.recv(payload_len - len(data))
				if not chunk:
					raise RuntimeError("Desconexión inesperada del servidor")
				data += chunk

			# Reconstruir tensor
			probs_b_np = np.frombuffer(data, dtype=np.float32).reshape(B_out, S_out, D_out).copy()
			return torch.from_numpy(probs_b_np).to(probs_a.device)
		except Exception as e:
			print(f"⚠️ [Error de Conexión al Servidor Cognitivo]: {e}")
			raise e
		finally:
			s.close()

def perception_to_input_coop(perception: list[int], device: torch.device) -> torch.Tensor:
	"""Convertir percepción del mundo cooperativo (ya con índices 15k) a tensor de input de 6 tokens."""
	indices = perception[:6]
	while len(indices) < 6:
		indices.append(SILENCE_GLYPH)
	return torch.tensor([indices], dtype=torch.long, device=device)

def forward_resonance_distributed_dispatch(model_a, socket_client, x, emotion_ids, n_steps, pos_mode):
	"""Forward pass híbrido con Despacho de Tensores por Socket UNIX."""
	h = model_a._embed_input(x)
	
	emo_vec_a = None
	if emotion_ids is not None and model_a.emotion_embeddings is not None:
		emo_emb_a = model_a.emotion_embeddings(emotion_ids)
		emo_vec_a = model_a.emotion_proj(emo_emb_a).unsqueeze(1)

	trajectory_norms = []
	for step in range(n_steps):
		if pos_mode == "clock" and getattr(model_a, "resonance_clock", None) is not None:
			h = h + model_a.resonance_clock[:, step, :].unsqueeze(1)

		if emo_vec_a is not None:
			if model_a.emotion_mode == "first_only" and step == 0:
				h = h + emo_vec_a
			elif model_a.emotion_mode == "additive":
				h = h + emo_vec_a

		# A: Capas 0-2
		for i in range(3):
			h = model_a.core_layers[i](h)
		h = model_a.norm(h)

		# Despacho
		logits_a = model_a._decode_hidden(h)
		probs_a = F.softmax(logits_a / 1.0, dim=-1)

		# Delegación distribuida por Socket UNIX
		probs_b = socket_client.dispatch(probs_a)

		# Re-embeber en el espacio oculto de A (256-dim)
		word_embeds_a = model_a.glyph_embedding.get_word_embeddings()
		h = torch.matmul(probs_b, word_embeds_a)

		# A: Capas 3-5
		for i in range(3, 6):
			h = model_a.core_layers[i](h)
		h = model_a.norm(h)

		trajectory_norms.append(h.norm(dim=-1).mean().item())

	metadata = {
		"trajectory_norms": trajectory_norms,
		"hidden": h,
	}
	return None, metadata

def run_arena_training_distributed():
	parser = argparse.ArgumentParser(description="Distributed Multi-Agent Arena Training")
	parser.add_argument("--config", type=str, required=True)
	parser.add_argument("--episodes", type=int, default=None)
	parser.add_argument("--socket", type=str, default="/tmp/bit_cognitive.sock", help="Path to UNIX socket server")
	args, _ = parser.parse_known_args()

	config_path = args.config if os.path.isabs(args.config) else os.path.join(base_dir, args.config)
	with open(config_path, encoding="utf-8") as f:
		config = json.load(f)

	experiment_id = config.get("experiment_id", "EXP_047_arena_comm") + "_distributed"

	print(f"═══ 🏟️ Arena Distributed Client — {experiment_id} ═══")
	print(f"📡 Conectando al socket: {args.socket}")

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

	# Instanciar agentes Nico y Sofy (256-dim)
	path_a = os.path.join(base_dir, "storage/checkpoints/sovereign_school/model_milestone_2_years.pt")

	def make_agent():
		m = BitNet4LayerModel(
			use_glyphs=True,
			glyph_table=glyphs,
			hidden_dim=256,
			num_layers=6,
			use_pos_embedding=True,
			max_resonance_steps=config.get("resonance", {}).get("max_resonance_steps", 5),
			n_emotions=N_EMOTIONS,
			emotion_dim=64,
			emotion_mode=emotion_cfg.get("mode", "first_only"),
		).to(device)

		if os.path.exists(path_a):
			sd = torch.load(path_a, map_location=device, weights_only=True)
			m.load_state_dict(sd, strict=False)
			print(f"Cargado Modelo A de la escuela: {path_a}")
		else:
			print(f"⚠️ Checkpoint A no encontrado en {path_a}!")

		# Redimensionar action head
		old_width = m.action_head[0].out_features
		old_gelu = m.action_head[1]
		m.action_head = nn.Sequential(
			nn.Linear(256, old_width),
			old_gelu,
			nn.Linear(old_width, COOP_N_ACTIONS),
		).to(device)

		for name, param in m.named_parameters():
			if "action_head" not in name:
				param.requires_grad = False
		return m

	agent_a = make_agent()
	agent_b = make_agent()

	# Inicializar cliente de socket
	socket_client = SocketCognitiveClient(socket_path=args.socket)

	communicate = config.get("communication", {}).get("enabled", True)
	comm_str = "📡 COMUNICACIÓN ON" if communicate else "🔇 SILENCIO"
	print(f"Configuración: {comm_str} | Despacho: SÍ (Socket Distribuido)")

	# Optimizers
	train_cfg = config.get("training", {})
	lr = train_cfg.get("lr", 5e-4)
	opt_a = torch.optim.AdamW(agent_a.action_head.parameters(), lr=lr, weight_decay=0.01)
	opt_b = torch.optim.AdamW(agent_b.action_head.parameters(), lr=lr, weight_decay=0.01)

	n_episodes = args.episodes if args.episodes is not None else train_cfg.get("episodes", 500)
	max_ticks = config.get("world", {}).get("max_ticks", 200)
	n_think = config.get("resonance", {}).get("n_think", 2)
	gamma = train_cfg.get("gamma", 0.97)
	entropy_bonus = train_cfg.get("entropy_bonus", 0.05)
	grad_clip = train_cfg.get("grad_clip", 1.0)

	world_cfg = config.get("world", {})
	food_interval = world_cfg.get("food_interval", 10)
	food_duration = world_cfg.get("food_duration", 5)
	hunger_rate = world_cfg.get("hunger_rate", 5.0)

	logger = ExperimentLogger(
		experiment_id,
		{**config, "device": str(device), "mode": "arena_distributed", "dispatch": True},
		log_path=os.path.join(exp_dir, "telemetry.jsonl"),
	)

	best_combined = 0.0
	survival_history = []
	rewards_a_history = []
	rewards_b_history = []

	for episode in range(n_episodes):
		world = CooperativeWorld(
			seed=seed + episode,
			food_interval=food_interval,
			food_duration=food_duration,
			hunger_rate=hunger_rate,
		)
		state_a, state_b, state_c = world.reset()

		buf_a = {"log_probs": [], "rewards": [], "entropies": []}
		buf_b = {"log_probs": [], "rewards": [], "entropies": []}

		agent_a.train()
		agent_b.train()

		ep_shouts = 0

		for tick in range(max_ticks):
			if not state_a.alive or not state_b.alive:
				break

			perc_a = world.perceive(state_a)
			perc_b = world.perceive(state_b)

			if not communicate:
				perc_a[4] = SILENCE_GLYPH
				perc_a[5] = SILENCE_GLYPH
				perc_b[4] = SILENCE_GLYPH
				perc_b[5] = SILENCE_GLYPH

			input_a = perception_to_input_coop(perc_a, device)
			input_b = perception_to_input_coop(perc_b, device)

			emo_a = torch.tensor([state_a.emotion_id], device=device)
			emo_b = torch.tensor([state_b.emotion_id], device=device)

			# Forward Pass Híbrido por Socket UNIX
			try:
				_, meta_a = forward_resonance_distributed_dispatch(agent_a, socket_client, input_a, emo_a, n_think, "clock")
				_, meta_b = forward_resonance_distributed_dispatch(agent_b, socket_client, input_b, emo_b, n_think, "clock")
			except Exception as e:
				print(f"❌ Error crítico en el dispatch distribuido: {e}. Abortando entrenamiento.")
				sys.exit(1)

			hidden_a = meta_a["hidden"][:, 2, :]
			hidden_b = meta_b["hidden"][:, 2, :]

			action_logits_a = agent_a.action_head(hidden_a)
			action_logits_b = agent_b.action_head(hidden_b)
			probs_a = F.softmax(action_logits_a, dim=-1)
			probs_b = F.softmax(action_logits_b, dim=-1)

			explore_rate = max(0.05, 1.0 - episode / (n_episodes * 0.4))
			if not communicate:
				probs_a = probs_a.clone()
				probs_a[0, 6] = 0.0
				probs_a = probs_a / probs_a.sum()
				probs_b = probs_b.clone()
				probs_b[0, 6] = 0.0
				probs_b = probs_b / probs_b.sum()

			if np.random.rand() < explore_rate:
				n_act = COOP_N_ACTIONS if communicate else COOP_N_ACTIONS - 1
				idx_a = np.random.randint(0, n_act)
				lp_a = torch.log(probs_a[0, idx_a] + 1e-8).squeeze()
			else:
				dist_a = torch.distributions.Categorical(probs_a)
				idx_a = dist_a.sample().item()
				lp_a = dist_a.log_prob(torch.tensor(idx_a, device=device)).squeeze()

			if np.random.rand() < explore_rate:
				n_act = COOP_N_ACTIONS if communicate else COOP_N_ACTIONS - 1
				idx_b = np.random.randint(0, n_act)
				lp_b = torch.log(probs_b[0, idx_b] + 1e-8).squeeze()
			else:
				dist_b = torch.distributions.Categorical(probs_b)
				idx_b = dist_b.sample().item()
				lp_b = dist_b.log_prob(torch.tensor(idx_b, device=device)).squeeze()

			ent_a = -(probs_a * (probs_a + 1e-8).log()).sum().squeeze()
			ent_b = -(probs_b * (probs_b + 1e-8).log()).sum().squeeze()

			action_a = COOP_ACTIONS[idx_a]
			action_b = COOP_ACTIONS[idx_b]

			result_a, result_b, result_c, world_info = world.step(action_a, action_b)

			if result_a.get("shouted") or result_b.get("shouted"):
				ep_shouts += 1

			reward_a = world.get_reward(state_a, result_a)
			reward_b = world.get_reward(state_b, result_b)

			if communicate:
				if world_info.get("coop_bonus_a", 0.0) > 0:
					reward_a += 5.0
					reward_b += 5.0
				if world_info.get("coop_bonus_b", 0.0) > 0:
					reward_b += 5.0
					reward_a += 5.0

			buf_a["log_probs"].append(lp_a)
			buf_a["rewards"].append(reward_a)
			buf_a["entropies"].append(ent_a)

			buf_b["log_probs"].append(lp_b)
			buf_b["rewards"].append(reward_b)
			buf_b["entropies"].append(ent_b)

		# Update REINFORCE
		for buf, opt, model in [(buf_a, opt_a, agent_a), (buf_b, opt_b, agent_b)]:
			if len(buf["log_probs"]) > 0:
				returns = []
				G = 0
				for r in reversed(buf["rewards"]):
					G = r + gamma * G
					returns.insert(0, G)
				returns = torch.tensor(returns, dtype=torch.float32, device=device)
				if len(returns) > 1:
					returns = (returns - returns.mean()) / (returns.std() + 1e-8)

				loss = torch.tensor(0.0, device=device)
				for lp, g, ent in zip(buf["log_probs"], returns, buf["entropies"], strict=False):
					loss -= lp * g
					loss -= entropy_bonus * ent

				opt.zero_grad()
				loss.backward()
				torch.nn.utils.clip_grad_norm_(model.action_head.parameters(), grad_clip)
				opt.step()

		combined = min(state_a.tick, state_b.tick)
		survival_history.append(combined)
		rewards_a_history.append(float(sum(buf_a["rewards"])))
		rewards_b_history.append(float(sum(buf_b["rewards"])))

		logger.log_epoch(
			epoch=episode + 1,
			loss_avg=float(loss.item()) if buf_a["log_probs"] else 0.0,
			acc_concept=float(combined),
			acc_emotion=float(ep_shouts),
			acc_joint=combined / max_ticks * 100,
			fitness=[float(state_a.tick), float(state_b.tick)],
			worst_agent=0,
			parent_a=-1,
			parent_b=-1,
			acc_homeostasis=float(sum(buf_a["rewards"]) + sum(buf_b["rewards"])) / 2.0,
		)

		if episode % 10 == 0 or episode == n_episodes - 1:
			avg_surv = np.mean(survival_history[-50:])
			avg_rew_a = np.mean(rewards_a_history[-50:])
			avg_rew_b = np.mean(rewards_b_history[-50:])
			print(f"Ep {episode+1:4d}/{n_episodes} | Ticks combined: {combined:3d} (Avg50: {avg_surv:.1f}) | Rew_A: {rewards_a_history[-1]:.1f} (Avg50: {avg_rew_a:.1f}) | Rew_B: {rewards_b_history[-1]:.1f} (Avg50: {avg_rew_b:.1f}) | Shouts: {ep_shouts}")

			if avg_surv > best_combined:
				best_combined = avg_surv
				torch.save(agent_a.state_dict(), os.path.join(exp_dir, "best_agent_a.pt"))
				torch.save(agent_b.state_dict(), os.path.join(exp_dir, "best_agent_b.pt"))
				print(f"  💾 Guardado mejor agente cooperativo (Avg50 Ticks: {best_combined:.1f})")

	torch.save(agent_a.state_dict(), os.path.join(exp_dir, "final_agent_a.pt"))
	torch.save(agent_b.state_dict(), os.path.join(exp_dir, "final_agent_b.pt"))
	print(f"🏁 Arena Distributed completada. Modelos guardados en {exp_dir}")
	logger.close()

if __name__ == "__main__":
	run_arena_training_distributed()
