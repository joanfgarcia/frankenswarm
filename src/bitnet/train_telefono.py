"""
EXP_030: Teléfono Roto — Multi-Agente en Cadena y en Loop.

Dos modos:
  1. CADENA: Agent_0 → Agent_1 → Agent_2 (cada uno es un paso de razonamiento)
  2. LOOP:  Agent_0 → Agent_1 → Agent_0 → Agent_1 → ... (N iteraciones)

La salida Gumbel-Softmax de un agente es la entrada DIRECTA del siguiente.
Los gradientes fluyen a través de TODOS los agentes en la cadena.
Cada agente se convierte en un "paso de pensamiento" especializado.

El teléfono roto: ¿puede un agente que nunca vio la pregunta original
dar la respuesta correcta, solo a partir de lo que le pasó el anterior?

El loop: ¿pueden dos agentes, iterando entre ellos, refinar su
razonamiento hasta converger en la respuesta correcta?
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from src.bitnet.modeling_bitnet import BitNet4LayerModel
from src.bitnet.operators_logic import CONCEPT_NAMES, get_causal_graph
from src.bitnet.translator import SovereignTranslator


def svd_crossover(parent_a, parent_b, child, alpha=0.5, sigma=0.01):
	with torch.no_grad():
		for name, param in child.named_parameters():
			if not param.requires_grad:
				continue
			p_a = parent_a.state_dict()[name]
			p_b = parent_b.state_dict()[name]
			if p_a.ndim == 2:
				w_avg = alpha * p_a + (1.0 - alpha) * p_b
				try:
					u, s, vh = torch.linalg.svd(w_avg, full_matrices=False)
					s_perturbed = (s + torch.randn_like(s) * sigma).clamp_(min=0.0)
					param.copy_(u @ torch.diag(s_perturbed) @ vh)
				except Exception:
					param.copy_(w_avg)
			else:
				param.copy_(alpha * p_a + (1.0 - alpha) * p_b + torch.randn_like(p_a) * sigma)


def build_chain_dataset(translator, max_depth=3):
	graph = get_causal_graph()
	concept_tids = {}
	for name in CONCEPT_NAMES:
		tids = translator.encode(name)
		concept_tids[name] = tids[0] if tids else 0

	chains = []

	def dfs(start_name, current_name, path, depth, max_fear):
		if depth >= 2 and len(path) > 1:
			start_idx = CONCEPT_NAMES.index(start_name)
			chains.append(
				{
					"start": start_idx,
					"start_tid": concept_tids[start_name],
					"end": CONCEPT_NAMES.index(current_name),
					"end_tid": concept_tids[current_name],
					"intermediates": [CONCEPT_NAMES.index(p) for p in path[1:-1]],
					"intermediate_tids": [concept_tids[CONCEPT_NAMES[CONCEPT_NAMES.index(p)]] for p in path[1:-1]],
					"depth": len(path) - 1,
					"fear": max_fear,
					"path_names": list(path),
				}
			)
		if depth >= max_depth:
			return
		current_idx = CONCEPT_NAMES.index(current_name)
		for next_idx, fear in graph.get(current_idx, []):
			next_name = CONCEPT_NAMES[next_idx]
			if next_name not in path:
				dfs(start_name, next_name, path + [next_name], depth + 1, max(max_fear, fear))

	for name in CONCEPT_NAMES:
		idx = CONCEPT_NAMES.index(name)
		if idx in graph:
			for next_idx, fear in graph[idx]:
				dfs(name, CONCEPT_NAMES[next_idx], [name, CONCEPT_NAMES[next_idx]], 1, fear)
	return chains, concept_tids


def run_telefono_roto():
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", type=str, default=None)
	args, _ = parser.parse_known_args()

	base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	config_path = args.config or os.path.join(base_dir, "configs", "experiments", "EXP_030.json")
	if not os.path.isabs(config_path):
		config_path = os.path.join(base_dir, config_path)

	with open(config_path, encoding="utf-8") as f:
		config = json.load(f)

	experiment_id = config["experiment_id"]
	mode = config.get("chain_mode", "telefono")  # "telefono" o "loop"
	loop_iterations = config.get("loop_iterations", 2)

	print(f"=== 📞 Teléfono Roto — {experiment_id} ===")
	print(f"Modo: {mode} | Loop iterations: {loop_iterations}")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"[Device]: {device}")

	seed = config.get("seed", 42)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)

	exp_dir = os.path.join(base_dir, "storage", "experiments", experiment_id)
	os.makedirs(exp_dir, exist_ok=True)
	with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf-8") as f:
		json.dump(config, f, ensure_ascii=False, indent=4)

	translator = SovereignTranslator()
	vocab_embeddings = translator.get_concept_embeddings()

	chains, concept_tids = build_chain_dataset(translator, max_depth=config.get("max_chain_depth", 3))
	chains_2 = [c for c in chains if c["depth"] == 2]
	chains_3 = [c for c in chains if c["depth"] >= 3]
	print(f"🔗 Cadenas: {len(chains)} total | 2-paso: {len(chains_2)} | 3+paso: {len(chains_3)}")

	op_implica_tid = translator.encode("implica")[0]

	# ═══════════════════════════════════════════
	# CREAR AGENTES ESPECIALIZADOS
	# ═══════════════════════════════════════════
	# En modo teléfono: 2 agentes (pensador + decodificador)
	# En modo loop: 2 agentes que iteran entre sí
	num_agents = config.get("num_agents", 2)

	agents = [
		BitNet4LayerModel(
			vocab_embeddings=vocab_embeddings,
			hidden_dim=config["hidden_dim"],
			num_layers=config["num_layers"],
			use_pos_embedding=config.get("use_pos_embedding", True),
		).to(device)
		for _ in range(num_agents)
	]

	# Hotstart: cada agente arranca desde el checkpoint
	resume = config.get("resume_checkpoint")
	if resume:
		resume_path = resume if os.path.isabs(resume) else os.path.join(base_dir, resume)
		if os.path.exists(resume_path):
			print(f"💾 [HOTSTART] Cargando desde {resume_path}...")
			state_dict = torch.load(resume_path, map_location=device)
			for agent in agents:
				agent.load_state_dict(state_dict)
			print(f"✅ {num_agents} agentes inicializados.")

	lr = config["lr"]
	wd = config.get("weight_decay", 0.05)
	optimizers = [torch.optim.AdamW(filter(lambda p: p.requires_grad, a.parameters()), lr=lr, weight_decay=wd) for a in agents]

	# Logit mask
	logit_mask = None
	if config.get("use_logit_mask", True):
		logit_mask = torch.zeros(8192, dtype=torch.bool, device=device)
		for word in config["micro_vocab_words"]:
			tids = translator.encode(word)
			if tids:
				logit_mask[tids[0]] = True

	# Etiquetas
	agent_names = [f"🧠{i}" for i in range(num_agents)]

	epochs = config["epochs"]
	steps_per_epoch = config["steps_per_epoch"]
	batch_size = config["batch_size"]
	tau_start, tau_min = config["tau_start"], config["tau_min"]
	total_steps = epochs * steps_per_epoch
	nursery_end = config["nursery_end"]
	transition_end = config["transition_end"]
	tf_min = config["tf_min"]
	fear_amplifier = config.get("fear_amplifier", 2.5)

	current_step = 0

	for epoch in range(epochs):
		if epoch < nursery_end:
			tf_ratio, phase = 1.0, "🍼 Guardería"
		elif epoch < transition_end:
			progress = (epoch - nursery_end) / (transition_end - nursery_end)
			tf_ratio, phase = 1.0 - progress * (1.0 - tf_min), "🎮 Recreo"
		else:
			tf_ratio, phase = tf_min, "🦅 Autonomía"

		print(f"\n--- Época {epoch + 1}/{epochs} [{phase}] TF={tf_ratio:.0%} ---")

		epoch_losses = []
		epoch_final_correct = 0
		epoch_total = 0

		# Curriculum: solo 2-paso en guardería
		active_chains = chains_2 if epoch < nursery_end else chains
		if not active_chains:
			active_chains = chains

		for agent in agents:
			agent.train()

		for step in range(steps_per_epoch):
			tau = max(tau_min, tau_start * (1.0 - current_step / total_steps))

			batch_chains = [active_chains[np.random.randint(len(active_chains))] for _ in range(batch_size)]

			for opt in optimizers:
				opt.zero_grad()

			start_tids = torch.tensor([c["start_tid"] for c in batch_chains], dtype=torch.long, device=device)
			end_tids = torch.tensor([c["end_tid"] for c in batch_chains], dtype=torch.long, device=device)
			fears = torch.tensor([1.0 + c["fear"] * fear_amplifier for c in batch_chains], dtype=torch.float32, device=device)

			# ═══════════════════════════════════════════
			# MODO TELÉFONO ROTO / LOOP
			# ═══════════════════════════════════════════

			if mode == "loop":
				# LOOP: Agent_0 → Agent_1 → Agent_0 → Agent_1 → ...
				# El input inicial es [start, implica, 0, 0]
				current_soft = F.one_hot(start_tids, num_classes=8192).float()  # (batch, 8192)

				all_intermediate_logits = []

				for iteration in range(loop_iterations):
					agent_idx = iteration % num_agents
					agent = agents[agent_idx]

					# Construir input: [current_concept, implica, 0, 0]
					inp_soft = torch.zeros((batch_size, 4, 8192), device=device)
					inp_soft[:, 0, :] = current_soft
					inp_soft[:, 1, :] = F.one_hot(
						torch.full((batch_size,), op_implica_tid, dtype=torch.long, device=device), num_classes=8192
					).float()

					# Forward
					logits = agent(inp_soft, logit_mask=logit_mask)
					all_intermediate_logits.append(logits)

					# Teacher forcing: dar el paso intermedio correcto
					if torch.rand(1).item() < tf_ratio and iteration < len(batch_chains[0].get("intermediate_tids", [])):
						# Usar el intermedio correcto
						int_tids = torch.tensor(
							[c["intermediate_tids"][iteration] if iteration < len(c["intermediate_tids"]) else c["end_tid"] for c in batch_chains],
							dtype=torch.long,
							device=device,
						)
						current_soft = F.one_hot(int_tids, num_classes=8192).float()
					else:
						# El agente produce el siguiente concepto (posición 2)
						current_soft = F.gumbel_softmax(logits[:, 2, :], tau=tau, hard=False, dim=-1)

				# Loss final: el último output debe ser el destino
				final_logits = all_intermediate_logits[-1]
				loss_final = F.cross_entropy(final_logits[:, 2, :], end_tids, reduction="none")
				total_loss = (loss_final * fears).mean()

				# Loss intermedia: cada paso contribuye
				for i, logits_i in enumerate(all_intermediate_logits[:-1]):
					# Target intermedio
					int_target_tids = torch.tensor(
						[c["intermediate_tids"][i] if i < len(c["intermediate_tids"]) else c["end_tid"] for c in batch_chains],
						dtype=torch.long,
						device=device,
					)
					loss_int = F.cross_entropy(logits_i[:, 2, :], int_target_tids, reduction="none")
					total_loss = total_loss + (loss_int * fears * 0.3).mean()

			else:
				# TELÉFONO ROTO: Agent_0 → Agent_1 (cada uno un paso)
				# Agent_0: recibe la pregunta, produce un mensaje
				# Agent_1: recibe el mensaje de Agent_0, produce la respuesta final

				# Agent_0: [start, implica, 0, 0] → genera mensaje completo
				inp_0 = torch.zeros((batch_size, 4), dtype=torch.long, device=device)
				inp_0[:, 0] = start_tids
				inp_0[:, 1] = op_implica_tid

				logits_0 = agents[0](inp_0, logit_mask=logit_mask)
				msg_0 = F.gumbel_softmax(logits_0, tau=tau, hard=False, dim=-1)  # (batch, 4, 8192)

				# Teacher forcing en el mensaje
				if torch.rand(1).item() < tf_ratio:
					# Darle al Agent_1 el mensaje correcto (con intermedio)
					teacher_msg = torch.zeros((batch_size, 4), dtype=torch.long, device=device)
					teacher_msg[:, 0] = start_tids
					teacher_msg[:, 1] = op_implica_tid
					for b_idx in range(batch_size):
						c = batch_chains[b_idx]
						if c["intermediate_tids"]:
							teacher_msg[b_idx, 2] = c["intermediate_tids"][0]
						else:
							teacher_msg[b_idx, 2] = c["end_tid"]
					msg_to_1 = F.one_hot(teacher_msg, num_classes=8192).float()
				else:
					msg_to_1 = msg_0

				# Agent_1: recibe el mensaje de Agent_0 y produce la respuesta
				logits_1 = agents[1](msg_to_1, logit_mask=logit_mask)

				# Loss: Agent_1 debe decodificar el destino final
				loss_final = F.cross_entropy(logits_1[:, 2, :], end_tids, reduction="none")

				# Loss intermedia: Agent_0 debe producir el intermedio correcto
				int_targets = torch.tensor(
					[c["intermediate_tids"][0] if c["intermediate_tids"] else c["end_tid"] for c in batch_chains], dtype=torch.long, device=device
				)
				loss_intermediate = F.cross_entropy(logits_0[:, 2, :], int_targets, reduction="none")

				total_loss = (loss_final * fears).mean() + (loss_intermediate * fears * 0.5).mean()

			total_loss.backward()
			for agent in agents:
				torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
			for opt in optimizers:
				opt.step()

			epoch_losses.append(total_loss.item())

			# Métricas
			with torch.no_grad():
				if mode == "loop":
					pred_final = torch.argmax(all_intermediate_logits[-1][:, 2, :], dim=-1)
				else:
					pred_final = torch.argmax(logits_1[:, 2, :], dim=-1)
				final_ok = (pred_final == end_tids).sum().item()

			epoch_final_correct += final_ok
			epoch_total += batch_size

			if step % 50 == 0:
				c = batch_chains[0]
				path = " → ".join(c["path_names"])
				pred_name = translator.decode([pred_final[0].item()])
				target_name = CONCEPT_NAMES[c["end"]]
				ok = "✓" if pred_name == target_name else "✗"

				if mode == "loop":
					# Mostrar qué produjo cada agente en cada iteración
					preds_per_step = []
					for i, logits_i in enumerate(all_intermediate_logits):
						p = translator.decode([torch.argmax(logits_i[0, 2, :]).item()])
						preds_per_step.append(f"{agent_names[i % num_agents]}={p}")
					chain_str = " → ".join(preds_per_step)
					print(
						f"  step {step:3d} | τ={tau:.3f} TF={tf_ratio:.0%} | loss={total_loss.item():.3f} | "
						f"🔗 {path} | {chain_str} | final={pred_name}({ok})"
					)
				else:
					# Teléfono roto
					pred_int = translator.decode([torch.argmax(logits_0[0, 2, :]).item()])
					int_target = CONCEPT_NAMES[c["intermediates"][0]] if c["intermediates"] else "?"
					print(
						f"  step {step:3d} | τ={tau:.3f} TF={tf_ratio:.0%} | loss={total_loss.item():.3f} | "
						f"🔗 {path} | {agent_names[0]}→{pred_int}({'✓' if pred_int == int_target else '✗'}) "
						f"→ {agent_names[1]}→{pred_name}({ok})"
					)

			current_step += 1

		avg_loss = np.mean(epoch_losses)
		acc_final = (epoch_final_correct / epoch_total) * 100
		print(f"Loss: {avg_loss:.4f} | Cadena Completa: {acc_final:.2f}%")

	# Guardar todos los agentes
	for i, agent in enumerate(agents):
		save_path = os.path.join(exp_dir, f"agent_{i}.pt")
		torch.save(agent.state_dict(), save_path)
	print(f"💾 {num_agents} agentes guardados en {exp_dir}")


if __name__ == "__main__":
	run_telefono_roto()
