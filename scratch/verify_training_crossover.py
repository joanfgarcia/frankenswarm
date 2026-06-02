import torch
import torch.nn as nn
import torch.nn.functional as F
from src.bitnet.cooperative_world import CooperativeWorld, COOP_ACTIONS, COOP_N_ACTIONS, SILENCE_GLYPH
from src.bitnet.modeling_bitnet import BitNet4LayerModel
from src.bitnet.glyph_vocabulary import N_EMOTIONS
from src.bitnet.genetic import recombine_parents

def test_training_pipeline_with_birth():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	world = CooperativeWorld(seed=42)
	
	# Inicializar modelos de padres
	agent_a = BitNet4LayerModel(
		use_glyphs=True,
		hidden_dim=64,
		num_layers=2,
		n_emotions=N_EMOTIONS,
		emotion_dim=16,
		max_seq_len=6,
	).to(device)
	
	agent_b = BitNet4LayerModel(
		use_glyphs=True,
		hidden_dim=64,
		num_layers=2,
		n_emotions=N_EMOTIONS,
		emotion_dim=16,
		max_seq_len=6,
	).to(device)
	
	# Ajustar cabezales
	agent_a.action_head = nn.Sequential(
		nn.Linear(64, 128),
		nn.GELU(),
		nn.Linear(128, COOP_N_ACTIONS),
	).to(device)
	agent_a.value_head = nn.Sequential(
		nn.Linear(64, 128),
		nn.GELU(),
		nn.Linear(128, 1),
	).to(device)
	
	agent_b.action_head = nn.Sequential(
		nn.Linear(64, 192),
		nn.GELU(),
		nn.Linear(192, COOP_N_ACTIONS),
	).to(device)
	agent_b.value_head = nn.Sequential(
		nn.Linear(64, 192),
		nn.GELU(),
		nn.Linear(192, 1),
	).to(device)
	
	# Configurar estado inicial para inducir apareamiento
	state_a, state_b, state_c = world.reset(seed=42)
	state_d = world.agent_d
	
	state_a.network_width = 128
	state_b.network_width = 192
	
	state_a.location = "cueva"
	state_b.location = "cueva"
	state_a.hambre = 80.0
	state_a.sed = 80.0
	state_b.hambre = 80.0
	state_b.sed = 80.0
	
	agent_d_model = None
	opt_d = None
	buf_d = {"inputs": [], "emotions": [], "actions": [], "shout_concepts": [], "log_probs": [], "values": [], "rewards": []}
	
	# TIPO DE ACCIONES
	action_a = "reproducir"
	action_b = "reproducir"
	action_c = "ver"
	action_d = "ver"
	
	# PASO 1: Apareamiento
	result_a, result_b, result_c, world_info = world.step(
		action_a, action_b, action_c, action_d
	)
	
	print(f"Paso 1 (Apareamiento) - Domi vivo: {state_d.alive}")
	assert state_d.alive, "Domi debería estar vivo tras apareamiento"
	assert "born_child" in result_a, "Debería haber evento de nacimiento en result_a"
	
	# PASO 2: Crossover en caliente en el entrenamiento
	if "born_child" in result_a and agent_d_model is None:
		child_info = result_a["born_child"]
		child_width = child_info["width"]
		
		# Crear modelo del hijo Domi
		agent_d_model = BitNet4LayerModel(
			use_glyphs=True,
			hidden_dim=64,
			num_layers=2,
			n_emotions=N_EMOTIONS,
			emotion_dim=16,
			max_seq_len=6,
		).to(device)
		agent_d_model.action_head = nn.Sequential(
			nn.Linear(64, child_width),
			nn.GELU(),
			nn.Linear(child_width, COOP_N_ACTIONS),
		).to(device)
		agent_d_model.value_head = nn.Sequential(
			nn.Linear(64, child_width),
			nn.GELU(),
			nn.Linear(child_width, 1),
		).to(device)
		
		# Cruzar pesos de Nico y Sofy
		recombine_parents(agent_a, agent_b, agent_d_model, child_width, device)
		
		opt_d = torch.optim.AdamW(agent_d_model.parameters(), lr=1e-3)
		agent_d_model.eval()
		
		print(f"Modelo Domi inicializado vía SVD Crossover. Ancho: {child_width}")
		assert agent_d_model.action_head[0].out_features == 160, "El ancho del hijo debería ser 160"
		
	# PASO 3: Domi percibe y actúa en el siguiente tick
	perc_d = world.perceive(state_d)
	input_d = torch.tensor([perc_d[:6]], dtype=torch.long, device=device)
	emo_d = torch.tensor([state_d.emotion_id], device=device)
	
	with torch.no_grad():
		_, meta_d = agent_d_model.forward_resonance(
			input_d, n_steps=2, pos_mode="clock", emotion_ids=emo_d
		)
		hidden_d = meta_d["hidden"][:, 2, :].clone()
		action_logits_d = agent_d_model.action_head(hidden_d)
		val_d = agent_d_model.value_head(hidden_d).item()
		
	probs_d = F.softmax(action_logits_d, dim=-1)
	dist_d = torch.distributions.Categorical(probs_d)
	idx_d = dist_d.sample().item()
	lp_d = dist_d.log_prob(torch.tensor(idx_d, device=device)).item()
	
	action_d = COOP_ACTIONS[idx_d]
	print(f"Domi selecciona acción: {action_d} (Prob: {probs_d[0, idx_d]:.4f})")
	
	# Ejecutar siguiente tick
	result_a2, result_b2, result_c2, world_info2 = world.step(
		"ver", "ver", "ver", action_d
	)
	result_d2 = world_info2.get("result_d")
	
	# Guardar transición de Domi
	reward_d = world.get_reward(state_d, result_d2)
	buf_d["inputs"].append(input_d.squeeze(0).cpu())
	buf_d["emotions"].append(state_d.emotion_id)
	buf_d["actions"].append(idx_d)
	buf_d["shout_concepts"].append(SILENCE_GLYPH)
	buf_d["log_probs"].append(lp_d)
	buf_d["values"].append(val_d)
	buf_d["rewards"].append(reward_d)
	
	print(f"Paso 2 (Domi actúa) - Reward de Domi: {reward_d}")
	
	# PASO 4: Optimización PPO de Domi al final del episodio
	agent_d_model.train()
	
	inputs_tensor = torch.stack(buf_d["inputs"], dim=0).to(device)
	emotions_tensor = torch.tensor(buf_d["emotions"], dtype=torch.long, device=device)
	actions_tensor = torch.tensor(buf_d["actions"], dtype=torch.long, device=device)
	old_log_probs = torch.tensor(buf_d["log_probs"], dtype=torch.float32, device=device)
	returns = torch.tensor(buf_d["rewards"], dtype=torch.float32, device=device)
	advantages = returns - torch.tensor(buf_d["values"], dtype=torch.float32, device=device)
	
	# Simular PPO loss backward
	_, m_epoch = agent_d_model.forward_resonance(
		inputs_tensor, n_steps=2, pos_mode="clock", emotion_ids=emotions_tensor
	)
	h_epoch = m_epoch["hidden"][:, 2, :]
	new_action_logits = agent_d_model.action_head(h_epoch)
	new_values = agent_d_model.value_head(h_epoch).squeeze(-1)
	
	new_probs = F.softmax(new_action_logits, dim=-1)
	dist = torch.distributions.Categorical(new_probs)
	new_log_probs = dist.log_prob(actions_tensor)
	
	ratios = torch.exp(new_log_probs - old_log_probs)
	surr1 = ratios * advantages
	surr2 = torch.clamp(ratios, 0.8, 1.2) * advantages
	policy_loss = -torch.min(surr1, surr2).mean()
	value_loss = F.mse_loss(new_values, returns)
	loss = policy_loss + 0.5 * value_loss - 0.05 * dist.entropy().mean()
	
	opt_d.zero_grad()
	loss.backward()
	opt_d.step()
	
	print(f"PASO 4 (PPO Update) completado con éxito. Loss: {loss.item():.4f}")
	
	# PASO 5: Consolidación Sleep
	from src.bitnet.consolidate_sleep import consolidate_latent_resonance
	
	# Simular una lección recibida por Domi
	teacher_hidden = torch.randn(1, 64, device=device)
	state_d.teaching_buffer.append({
		"input": input_d.clone().cpu(),
		"emotion": state_d.emotion_id,
		"teacher_hidden": teacher_hidden.clone().cpu(),
		"skill": "caza",
		"teacher_id": "a"
	})
	
	graduated = consolidate_latent_resonance(agent_d_model, state_d.teaching_buffer, device, n_think=2)
	print(f"PASO 5 (Sleep Consolidate) completado con éxito. Habilidades graduadas: {graduated}")
	
	print("\n🎉 ¡TODOS LOS PASOS DE INTEGRACIÓN PPO EVOLUTIVA COMPLETADOS CORRECTAMENTE! 🎉")

if __name__ == "__main__":
	test_training_pipeline_with_birth()
