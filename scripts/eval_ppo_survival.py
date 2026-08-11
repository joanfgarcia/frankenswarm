"""
eval_ppo_survival.py

Script para evaluar y comparar determinísticamente el agente PPO (EXP_050)
frente al baseline REINFORCE (EXP_039) en 100 episodios.
"""

import os

import numpy as np
import torch
import torch.nn.functional as F

from src.bitnet.model.modeling_bitnet import BitNet4LayerModel
from src.bitnet.vocab.glyph_vocabulary import N_EMOTIONS, WORD_INDEX
from src.bitnet.worlds.minimal_world import LOCATION_GLYPHS, MinimalWorld


def perception_to_input(perception: list[str], location: str, device: torch.device) -> torch.Tensor:
	loc_glyph = LOCATION_GLYPHS.get(location, "tierra")
	indices = [WORD_INDEX.get(loc_glyph, 0)]
	for word in perception:
		if word in WORD_INDEX:
			indices.append(WORD_INDEX[word])
		elif word == "montaña":
			indices.append(WORD_INDEX["piedra"])
	while len(indices) < 4:
		indices.append(indices[-1] if indices else 0)
	indices = indices[:4]
	return torch.tensor([indices], dtype=torch.long, device=device)

def evaluate_agent(checkpoint_path: str, device: torch.device, use_value_head: bool = False, n_episodes: int = 100) -> dict:
	# Cargar modelo con la arquitectura de 3 capas usada en EXP_039 y EXP_050
	model = BitNet4LayerModel(
		use_glyphs=True,
		hidden_dim=256,
		num_layers=3,
		use_pos_embedding=True,
		max_resonance_steps=5,
		n_emotions=N_EMOTIONS,
		emotion_dim=64,
		emotion_mode="first_only",
		action_head_width=128,
	).to(device)

	state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
	model.load_state_dict(state_dict, strict=False)
	model.eval()

	survival_ticks = []
	rewards_collected = []
	causes_of_death = {}
	max_ticks = 200

	action_names = ["comer", "beber", "dormir", "mover", "ver", "piedra"]

	for episode in range(n_episodes):
		world = MinimalWorld(seed=1000 + episode)  # Usar semillas distintas al entrenamiento
		state = world.reset()
		episode_reward = 0.0

		for _tick in range(max_ticks):
			if not state.alive:
				break

			perception = world.perceive()
			x = perception_to_input(perception, state.location, device)
			emotion_id = torch.tensor([state.emotion_id], dtype=torch.long, device=device)

			with torch.no_grad():
				logits, meta = model.forward_deep_think(
					x, n_think=2, n_verify=2,
					pos_mode="clock", emotion_ids=emotion_id,
					convergence_threshold=0.85,
				)
				hidden = meta.get("hidden", None)
				if hidden is not None:
					h_action = hidden[:, 2, :].clone()
				else:
					_, meta_fallback = model.forward_resonance(
						x, n_steps=2, pos_mode="clock",
						emotion_ids=emotion_id,
					)
					h_action = meta_fallback["hidden"][:, 2, :].clone()

				action_logits = model.action_head(h_action)
				probs = F.softmax(action_logits, dim=-1)
				# Selección determinista para evaluación
				action_idx = probs.argmax(dim=-1).item()

			action_name = action_names[action_idx]
			result = world.act(action_name)
			reward = world.get_reward(result)
			episode_reward += reward

		survival_ticks.append(state.tick)
		rewards_collected.append(episode_reward)

		if not state.alive:
			cause = state.emotion_name
			causes_of_death[cause] = causes_of_death.get(cause, 0) + 1
		else:
			causes_of_death["sobrevivió"] = causes_of_death.get("sobrevivió", 0) + 1

	return {
		"mean_ticks": np.mean(survival_ticks),
		"max_ticks": np.max(survival_ticks),
		"min_ticks": np.min(survival_ticks),
		"std_ticks": np.std(survival_ticks),
		"mean_reward": np.mean(rewards_collected),
		"deaths": causes_of_death
	}

def main():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

	ppo_path = os.path.join(base_dir, "storage", "experiments", "EXP_050_ppo_survival", "best_agent.pt")
	reinforce_path = os.path.join(base_dir, "storage", "experiments", "EXP_039_survival_full", "best_agent.pt")

	print("═══ 🔬 EVALUACIÓN COMPARATIVA: PPO vs REINFORCE ═══")
	print(f"Dispositivo: {device}\n")

	# Evaluar PPO
	if os.path.exists(ppo_path):
		print("🏃 Evaluando Agente PPO (EXP_050)...")
		ppo_results = evaluate_agent(ppo_path, device, use_value_head=True)
	else:
		ppo_results = None
		print(f"⚠️ Checkpoint PPO no encontrado en {ppo_path}")

	# Evaluar REINFORCE
	if os.path.exists(reinforce_path):
		print("🏃 Evaluando Agente REINFORCE (EXP_039)...")
		reinforce_results = evaluate_agent(reinforce_path, device, use_value_head=False)
	else:
		reinforce_results = None
		print(f"⚠️ Checkpoint REINFORCE no encontrado en {reinforce_path}")

	if ppo_results and reinforce_results:
		print("\n" + "═"*70)
		print(f"{'Métrica':<25} | {'REINFORCE (EXP_039)':<20} | {'PPO (EXP_050)':<20}")
		print("─"*70)
		print(f"{'Ticks Sobrevividos (Media)':<25} | {reinforce_results['mean_ticks']:20.2f} | {ppo_results['mean_ticks']:20.2f}")
		print(f"{'Desviación Estándar':<25} | {reinforce_results['std_ticks']:20.2f} | {ppo_results['std_ticks']:20.2f}")
		print(f"{'Máx Ticks Alcanzados':<25} | {reinforce_results['max_ticks']:20.0f} | {ppo_results['max_ticks']:20.0f}")
		print(f"{'Min Ticks Alcanzados':<25} | {reinforce_results['min_ticks']:20.0f} | {ppo_results['min_ticks']:20.0f}")
		print(f"{'Recompensa Media':<25} | {reinforce_results['mean_reward']:20.2f} | {ppo_results['mean_reward']:20.2f}")
		print("─"*70)
		print("Causas de Muerte / Sobrevivientes:")
		all_causes = set(list(reinforce_results["deaths"].keys()) + list(ppo_results["deaths"].keys()))
		for cause in all_causes:
			r_count = reinforce_results["deaths"].get(cause, 0)
			p_count = ppo_results["deaths"].get(cause, 0)
			print(f"  - {cause:<21} | {r_count:<20d} | {p_count:<20d}")
		print("═"*70)

if __name__ == "__main__":
	main()
