"""
EXP_043: Dos Bits — Comunicación emergente entre agentes.

Dos agentes en el mismo mundo. Cada uno tiene su propio cerebro,
sus emociones y su action head. Se comunican "pensando en voz alta":
cada agente recibe como percepción adicional lo que el otro piensa
(watcher output del paso anterior).

Hipótesis: dos agentes que se comunican sobreviven más que uno solo,
porque comparten información sobre dónde hay comida y peligro.

Origen: Joan Garcia — "si en vez de uno tenemos a 2 que hablan entre ellos?"
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from src.bitnet.glyph_vocabulary import (
	N_EMOTIONS,
	WORD_INDEX,
)
from src.bitnet.minimal_world import (
	LOCATION_GLYPHS,
	MinimalWorld,
)
from src.bitnet.modeling_bitnet import BitNet4LayerModel
from src.bitnet.net2net import grow_action_head


class DualWorld:
	"""Un mundo compartido por dos agentes con estados independientes."""

	def __init__(self, seed: int = 42):
		# UN solo mundo — los dos viven aquí
		self.world = MinimalWorld(seed=seed)
		self.seed = seed

	def reset(self):
		# Resetear mundo y crear dos estados independientes
		self.world.reset()
		state_a = self.world.state  # agent A usa el state del mundo
		# Agent B: copia independiente, empieza en la misma cueva
		from src.bitnet.minimal_world import AgentState
		state_b = AgentState()
		self._state_b = state_b
		return state_a, state_b

	def perceive_for(self, agent_label: str):
		"""Percepción desde la perspectiva de un agente."""
		if agent_label == "A":
			self.world.state = self._get_state("A")
		else:
			old = self.world.state
			self.world.state = self._state_b
			result = self.world.perceive()
			self.world.state = old
			return result
		return self.world.perceive()

	def act_for(self, agent_label: str, action: str):
		"""Ejecutar acción para un agente específico."""
		if agent_label == "A":
			return self.world.act(action)
		else:
			# Swap state, act, swap back
			old = self.world.state
			self.world.state = self._state_b
			result = self.world.act(action)
			self._state_b = self.world.state
			self.world.state = old
			return result

	def get_reward_for(self, agent_label: str, result: dict):
		"""Reward para un agente."""
		if agent_label == "A":
			return self.world.get_reward(result)
		else:
			old = self.world.state
			self.world.state = self._state_b
			r = self.world.get_reward(result)
			self.world.state = old
			return r

	def _get_state(self, label):
		if label == "A":
			return self.world.state
		return self._state_b

	@property
	def state_a(self):
		return self.world.state

	@property
	def state_b(self):
		return self._state_b


def build_input_from_perception(perception: list[str], state, device, partner_thought: list[int] = None) -> torch.Tensor:
	"""Construye input tensor.
	Sin comunicación: 4 tokens [loc, perc1, perc2, perc3]
	Con comunicación: 6 tokens [loc, perc1, perc2, perc3, msg1, msg2]
	Canal separado — la percepción NUNCA se sacrifica por comunicación.
	"""
	from src.bitnet.minimal_world import LOCATION_GLYPHS

	loc_glyph = LOCATION_GLYPHS.get(state.location, "tierra")
	indices = [WORD_INDEX.get(loc_glyph, 0)]

	# Percepciones del mundo (strings → indices)
	for word in perception:
		if word in WORD_INDEX:
			indices.append(WORD_INDEX[word])
		elif word == "montaña":
			indices.append(WORD_INDEX.get("piedra", 0))

	# Pad percepción a 4 tokens
	while len(indices) < 4:
		indices.append(indices[-1] if indices else 0)
	indices = indices[:4]

	# Canal de comunicación: 2 tokens extra del compañero
	if partner_thought is not None and len(partner_thought) > 0:
		for t in partner_thought[:2]:
			indices.append(t)
		# Pad comunicación a 2 tokens
		while len(indices) < 6:
			indices.append(indices[-1])

	return torch.tensor([indices], dtype=torch.long, device=device)


def run_dual_agent():
	parser = argparse.ArgumentParser(description="EXP_043: Dual agents")
	parser.add_argument("--config", type=str, required=True)
	args, _ = parser.parse_known_args()

	base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	config_path = args.config if os.path.isabs(args.config) else os.path.join(base_dir, args.config)
	with open(config_path, encoding="utf-8") as f:
		config = json.load(f)

	experiment_id = config.get("experiment_id", "EXP_043_dual")
	seed = config.get("seed", 42)
	torch.manual_seed(seed)
	np.random.seed(seed)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	exp_dir = os.path.join(base_dir, "storage", "experiments", experiment_id)
	os.makedirs(exp_dir, exist_ok=True)

	model_cfg = config.get("model", {})
	emotion_cfg = config.get("emotion", {})

	# ═══ Dos agentes con el mismo cerebro pero action heads independientes ═══
	def make_agent():
		m = BitNet4LayerModel(
			use_glyphs=True,
			hidden_dim=model_cfg.get("hidden_dim", 256),
			num_layers=model_cfg.get("num_layers", 3),
			use_pos_embedding=model_cfg.get("use_pos_embedding", True),
			max_resonance_steps=config.get("resonance", {}).get("max_resonance_steps", 5),
			n_emotions=N_EMOTIONS,
			emotion_dim=emotion_cfg.get("dim", 64),
			emotion_mode=emotion_cfg.get("mode", "first_only"),
		).to(device)

		# Cargar pretrained
		pretrained_from = config.get("pretrained_from")
		if pretrained_from:
			p = pretrained_from if os.path.isabs(pretrained_from) else os.path.join(base_dir, pretrained_from)
			if os.path.exists(p):
				sd = torch.load(p, map_location=device, weights_only=True)
				if "action_head.0.weight" in sd:
					ckpt_width = sd["action_head.0.weight"].shape[0]
					if ckpt_width != m.action_head[0].out_features:
						import torch.nn as tnn
						m.action_head = tnn.Sequential(
							tnn.Linear(model_cfg.get("hidden_dim", 256), ckpt_width),
							tnn.GELU(),
							tnn.Linear(ckpt_width, 6),
						).to(device)
				m.load_state_dict(sd, strict=False)

		# Freeze base (unless config says to unfreeze)
		unfreeze = config.get("unfreeze_backbone", False)
		if not unfreeze:
			for name, param in m.named_parameters():
				if "action_head" not in name:
					param.requires_grad = False
		else:
			print("🔓 Backbone DESCONGELADO — todo entrena")
		return m

	agent_a = make_agent()
	agent_b = make_agent()

	print(f"═══ 👥 Dual Agents — {experiment_id} ═══")
	head_w = agent_a.action_head[0].out_features
	head_p = sum(p.numel() for p in agent_a.action_head.parameters())
	print(f"🧠 Agent A: width={head_w}, params={head_p:,}")
	print(f"🧠 Agent B: width={head_w}, params={head_p:,}")

	train_cfg = config.get("training", {})
	lr = train_cfg.get("lr", 5e-4)
	gamma = train_cfg.get("gamma", 0.97)
	entropy_bonus = train_cfg.get("entropy_bonus", 0.05)
	grad_clip = train_cfg.get("grad_clip", 1.0)
	update_interval = train_cfg.get("update_interval", 16)
	n_episodes = train_cfg.get("episodes", 500)
	max_ticks = config.get("world", {}).get("max_ticks", 200)
	n_think = config.get("resonance", {}).get("n_think", 2)
	n_verify = config.get("resonance", {}).get("n_verify", 2)
	communicate = config.get("communication", {}).get("enabled", True)

	opt_a = torch.optim.AdamW(agent_a.action_head.parameters(), lr=lr, weight_decay=0.01)
	opt_b = torch.optim.AdamW(agent_b.action_head.parameters(), lr=lr, weight_decay=0.01)

	growth_cfg = config.get("growth", {})
	growth_factor = growth_cfg.get("factor", 1.5)
	max_width = growth_cfg.get("max_width", 512)
	growth_cfg.get("patience", 50)

	# Track
	best_combined = 0.0
	survival_a_hist = []
	survival_b_hist = []
	total_growths_a = 0
	total_growths_b = 0
	action_names = ["comer", "beber", "dormir", "mover", "ver", "piedra"]

	for episode in range(n_episodes):
		dual = DualWorld(seed=seed + episode)
		state_a, state_b = dual.reset()

		buf_a = {"log_probs": [], "rewards": [], "entropies": []}
		buf_b = {"log_probs": [], "rewards": [], "entropies": []}

		# Pensamientos del compañero (inicialmente vacíos)
		thought_a = []  # lo que A piensa (para B)
		thought_b = []  # lo que B piensa (para A)

		tick = 0
		agent_a.train()
		agent_b.train()

		while state_a.alive and state_b.alive and tick < max_ticks:
			tick += 1

			# ── Percepciones: mundo + mensaje del otro ──
			perception_a = dual.perceive_for("A")
			perception_b = dual.perceive_for("B")
			if communicate:
				input_a = build_input_from_perception(perception_a, state_a, device, thought_b)
				input_b = build_input_from_perception(perception_b, state_b, device, thought_a)
			else:
				input_a = build_input_from_perception(perception_a, state_a, device)
				input_b = build_input_from_perception(perception_b, state_b, device)

			emo_a = torch.tensor([state_a.emotion_id], device=device)
			emo_b = torch.tensor([state_b.emotion_id], device=device)

			# ── Forward: Pensar + Verificar ──
			with torch.no_grad():
				logits_a, meta_a = agent_a.forward_deep_think(
					input_a, n_think=n_think, n_verify=n_verify,
					pos_mode="clock", emotion_ids=emo_a,
				)
				logits_b, meta_b = agent_b.forward_deep_think(
					input_b, n_think=n_think, n_verify=n_verify,
					pos_mode="clock", emotion_ids=emo_b,
				)



			# ── Acciones ──
			hidden_a = meta_a["hidden"][:, 2, :]
			hidden_b = meta_b["hidden"][:, 2, :]
			action_logits_a = agent_a.action_head(hidden_a)
			action_logits_b = agent_b.action_head(hidden_b)
			probs_a = F.softmax(action_logits_a, dim=-1)
			probs_b = F.softmax(action_logits_b, dim=-1)

			explore_rate = max(0.05, 1.0 - episode / (n_episodes * 0.4))

			# Agent A
			if torch.rand(1).item() < explore_rate:
				idx_a = torch.randint(0, 6, (1,)).item()
				lp_a = torch.log(probs_a[0, idx_a] + 1e-8).squeeze()
			else:
				dist_a = torch.distributions.Categorical(probs_a)
				idx_a = dist_a.sample().item()
				lp_a = dist_a.log_prob(torch.tensor(idx_a, device=device))

			# Agent B
			if torch.rand(1).item() < explore_rate:
				idx_b = torch.randint(0, 6, (1,)).item()
				lp_b = torch.log(probs_b[0, idx_b] + 1e-8).squeeze()
			else:
				dist_b = torch.distributions.Categorical(probs_b)
				idx_b = dist_b.sample().item()
				lp_b = dist_b.log_prob(torch.tensor(idx_b, device=device))

			ent_a = -(probs_a * (probs_a + 1e-8).log()).sum()
			ent_b = -(probs_b * (probs_b + 1e-8).log()).sum()

			# ── Ejecutar acciones ──
			result_a = dual.act_for("A", action_names[idx_a])
			result_b = dual.act_for("B", action_names[idx_b])
			reward_a = dual.get_reward_for("A", result_a)
			reward_b = dual.get_reward_for("B", result_b)

			# ── Comunicación por EVENTOS (no constante) ──
			# Solo se comunica cuando pasa algo importante:
			#   - Comió → señal [localización, comida]
			#   - Herido → señal [localización, peligro]
			#   - Nada especial → silencio (thought = [])
			if communicate:
				loc_a_idx = WORD_INDEX.get(LOCATION_GLYPHS.get(state_a.location, "tierra"), 0)
				loc_b_idx = WORD_INDEX.get(LOCATION_GLYPHS.get(state_b.location, "tierra"), 0)

				# A comunica
				if result_a.get("delta_hambre", 0) > 0:
					thought_a = [loc_a_idx, WORD_INDEX.get("comida", 0)]  # "¡comida aquí!"
				elif result_a.get("delta_salud", 0) < -5:
					thought_a = [loc_a_idx, WORD_INDEX.get("peligro", 0)]  # "¡peligro!"
				else:
					thought_a = []  # silencio

				# B comunica
				if result_b.get("delta_hambre", 0) > 0:
					thought_b = [loc_b_idx, WORD_INDEX.get("comida", 0)]
				elif result_b.get("delta_salud", 0) < -5:
					thought_b = [loc_b_idx, WORD_INDEX.get("peligro", 0)]
				else:
					thought_b = []  # silencio

			buf_a["log_probs"].append(lp_a)
			buf_a["rewards"].append(reward_a)
			buf_a["entropies"].append(ent_a)
			buf_b["log_probs"].append(lp_b)
			buf_b["rewards"].append(reward_b)
			buf_b["entropies"].append(ent_b)

			# ── Online update ──
			if tick % update_interval == 0:
				for buf, opt, model in [(buf_a, opt_a, agent_a), (buf_b, opt_b, agent_b)]:
					if len(buf["log_probs"]) > 1:
						returns = []
						G = 0
						for r in reversed(buf["rewards"]):
							G = r + gamma * G
							returns.insert(0, G)
						returns = torch.tensor(returns, dtype=torch.float32, device=device)
						if len(returns) > 1:
							returns = (returns - returns.mean()) / (returns.std() + 1e-8)
						loss = torch.tensor(0.0, device=device)
						for i in range(len(buf["log_probs"])):
							lp = buf["log_probs"][i].squeeze()
							ret = returns[i].squeeze()
							loss -= lp * ret
							loss -= entropy_bonus * buf["entropies"][i]
						loss = loss / len(buf["log_probs"])
						opt.zero_grad()
						loss.backward()
						torch.nn.utils.clip_grad_norm_(model.action_head.parameters(), max_norm=grad_clip)
						opt.step()
					buf["log_probs"].clear()
					buf["rewards"].clear()
					buf["entropies"].clear()

		# ── Métricas ──
		ticks_a = state_a.tick if not state_a.alive else tick
		ticks_b = state_b.tick if not state_b.alive else tick
		combined = min(ticks_a, ticks_b)  # ambos deben sobrevivir
		survival_a_hist.append(ticks_a)
		survival_b_hist.append(ticks_b)

		if combined > best_combined:
			best_combined = combined
			torch.save(agent_a.state_dict(), os.path.join(exp_dir, "best_agent_a.pt"))
			torch.save(agent_b.state_dict(), os.path.join(exp_dir, "best_agent_b.pt"))

		# ── Post-episode growth ──
		for label, model, _opt_ref, hist, _growths in [
			("A", agent_a, "opt_a", survival_a_hist, "total_growths_a"),
			("B", agent_b, "opt_b", survival_b_hist, "total_growths_b"),
		]:
			curr_w = model.action_head[0].out_features
			if len(hist) >= 10 and curr_w < max_width:
				avg_s = np.mean(hist[-10:])
				if avg_s < max_ticks * 0.15:
					gi = grow_action_head(model, growth_factor=growth_factor)
					if label == "A":
						total_growths_a += 1
						opt_a = torch.optim.AdamW(model.action_head.parameters(), lr=lr, weight_decay=0.01)
					else:
						total_growths_b += 1
						opt_b = torch.optim.AdamW(model.action_head.parameters(), lr=lr, weight_decay=0.01)
					print(f"  🧬 {label} NEUROGENÉSIS | width: {gi['old_width']}→{gi['new_width']} | avg_surv={avg_s:.1f}")

		if episode % 10 == 0 or episode == n_episodes - 1:
			avg_a = np.mean(survival_a_hist[-50:])
			avg_b = np.mean(survival_b_hist[-50:])
			w_a = agent_a.action_head[0].out_features
			w_b = agent_b.action_head[0].out_features
			loc_a = state_a.location if hasattr(state_a, "location") else "?"
			loc_b = state_b.location if hasattr(state_b, "location") else "?"
			comm_str = "📡" if communicate else "🔇"
			print(
				f"Ep {episode+1:4d}/{n_episodes} | "
				f"A:{ticks_a:3d}t 📍{loc_a:8s} w={w_a} | "
				f"B:{ticks_b:3d}t 📍{loc_b:8s} w={w_b} | "
				f"Best:{best_combined:.0f} | AvgA:{avg_a:.1f} AvgB:{avg_b:.1f} | "
				f"{comm_str} | G_A:{total_growths_a} G_B:{total_growths_b}"
			)

	# ═══ Final ═══
	torch.save(agent_a.state_dict(), os.path.join(exp_dir, "final_agent_a.pt"))
	torch.save(agent_b.state_dict(), os.path.join(exp_dir, "final_agent_b.pt"))

	print(f"\n{'═'*60}")
	print(f"📊 Resultados finales — Dual Agents {'(con comunicación)' if communicate else '(sin comunicación)'}")
	print(f"   Best combined: {best_combined:.0f} ticks")
	print(f"   Agent A — Avg100: {np.mean(survival_a_hist[-100:]):.1f}, Growths: {total_growths_a}, Width: {agent_a.action_head[0].out_features}")
	print(f"   Agent B — Avg100: {np.mean(survival_b_hist[-100:]):.1f}, Growths: {total_growths_b}, Width: {agent_b.action_head[0].out_features}")
	print(f"{'═'*60}")


if __name__ == "__main__":
	run_dual_agent()
