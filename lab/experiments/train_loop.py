"""
EXP_029: Entrenamiento en Loop — Razonamiento Multi-Paso Diferenciable.

El modelo se encadena CONSIGO MISMO. La salida del paso N se convierte
en la entrada del paso N+1, y la loss se aplica al final del loop.

El gradiente fluye a través de todo el loop gracias a Gumbel-Softmax.
Esto fuerza al modelo a producir pasos intermedios CORRECTOS para
llegar a la respuesta final.

Formato:
  Paso 1: [A, implica, 0, emo] → modelo produce B_soft en posición 2
  Paso 2: [B_soft, implica, 0, emo] → modelo produce C_soft en posición 2
  Loss: C_soft == C_target (el destino final de la cadena)

  Backprop: ∂Loss/∂weights fluye a través de paso 2 → paso 1

Esto NO es memorización. Es razonamiento en pasos.
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from src.bitnet.model.modeling_bitnet import BitNet4LayerModel
from src.bitnet.operators.operators_logic import CONCEPT_NAMES, get_causal_graph
from src.bitnet.telemetry.telemetry import ExperimentLogger
from src.bitnet.translation.translator import SovereignTranslator


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


def build_chain_dataset(translator: SovereignTranslator, max_depth: int = 3):
	"""
	Construye dataset de cadenas para entrenamiento en loop.

	Cada sample es: (concepto_inicio, pasos_intermedios, concepto_final, fear)

	Ejemplo: luna → agua → seguridad
	  start=luna, intermedios=[agua], end=seguridad, fear=0.2
	"""
	graph = get_causal_graph()
	concept_tids = {}
	for name in CONCEPT_NAMES:
		tids = translator.encode(name)
		concept_tids[name] = tids[0] if tids else 0

	chains = []

	def dfs(start_name, current_name, path, depth, max_fear):
		if depth >= 2 and len(path) > 1:
			# Tenemos una cadena válida de al menos 2 pasos
			start_idx = CONCEPT_NAMES.index(start_name)
			end_idx = CONCEPT_NAMES.index(current_name)
			intermediates = [CONCEPT_NAMES.index(p) for p in path[1:-1]]

			chains.append(
				{
					"start": start_idx,
					"start_tid": concept_tids[start_name],
					"end": end_idx,
					"end_tid": concept_tids[current_name],
					"intermediates": intermediates,
					"intermediate_tids": [concept_tids[CONCEPT_NAMES[i]] for i in intermediates],
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
			if next_name not in path:  # evitar ciclos
				dfs(start_name, next_name, path + [next_name], depth + 1, max(max_fear, fear))

	for start_name in CONCEPT_NAMES:
		start_idx = CONCEPT_NAMES.index(start_name)
		if start_idx in graph:
			for next_idx, fear in graph[start_idx]:
				next_name = CONCEPT_NAMES[next_idx]
				dfs(start_name, next_name, [start_name, next_name], 1, fear)

	return chains, concept_tids


def run_loop_training():
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", type=str, default=None)
	args, _ = parser.parse_known_args()

	base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

	config_path = args.config or os.path.join(base_dir, "configs", "experiments", "EXP_029.json")
	if not os.path.isabs(config_path):
		config_path = os.path.join(base_dir, config_path)

	with open(config_path, encoding="utf-8") as f:
		config = json.load(f)

	experiment_id = config["experiment_id"]
	print(f"=== 🔄 Arena de Razonamiento en Loop — {experiment_id} ===")
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

	# 1. Translator y dataset de cadenas
	translator = SovereignTranslator()
	vocab_embeddings = translator.get_concept_embeddings()

	chains, concept_tids = build_chain_dataset(translator, max_depth=config.get("max_chain_depth", 3))
	print(f"🔗 Cadenas descubiertas: {len(chains)}")
	for c in chains[:5]:
		print(f"   {' → '.join(c['path_names'])}  (depth={c['depth']}, fear={c['fear']:.2f})")

	# Separar por profundidad
	chains_2 = [c for c in chains if c["depth"] == 2]
	chains_3 = [c for c in chains if c["depth"] >= 3]
	print(f"   2 pasos: {len(chains_2)} | 3+ pasos: {len(chains_3)}")

	# Token IDs necesarios
	op_implica_tid = translator.encode("implica")[0]

	# 2. Población
	pop_size = config["pop_size"]
	population = [
		BitNet4LayerModel(
			vocab_embeddings=vocab_embeddings,
			hidden_dim=config["hidden_dim"],
			num_layers=config["num_layers"],
			use_pos_embedding=config.get("use_pos_embedding", True),
		).to(device)
		for _ in range(pop_size)
	]

	# Hotstart
	resume = config.get("resume_checkpoint")
	if resume:
		resume_path = resume if os.path.isabs(resume) else os.path.join(base_dir, resume)
		if os.path.exists(resume_path):
			print(f"💾 [HOTSTART] Cargando desde {resume_path}...")
			state_dict = torch.load(resume_path, map_location=device)
			for model in population:
				model.load_state_dict(state_dict)
			print("✅ Transferencia completada.")

	lr = config["lr"]
	wd = config.get("weight_decay", 0.05)
	optimizers = [torch.optim.AdamW(filter(lambda p: p.requires_grad, m.parameters()), lr=lr, weight_decay=wd) for m in population]

	# Logit mask
	logit_mask = None
	if config.get("use_logit_mask", True):
		logit_mask = torch.zeros(8192, dtype=torch.bool, device=device)
		for word in config["micro_vocab_words"]:
			tids = translator.encode(word)
			if tids:
				logit_mask[tids[0]] = True

	# 3. Loop de entrenamiento
	epochs = config["epochs"]
	steps_per_epoch = config["steps_per_epoch"]
	batch_size = config["batch_size"]
	tau_start = config["tau_start"]
	tau_min = config["tau_min"]
	total_steps = epochs * steps_per_epoch
	nursery_end = config["nursery_end"]
	transition_end = config["transition_end"]
	tf_min = config["tf_min"]
	fear_amplifier = config.get("fear_amplifier", 3.0)

	fitness = np.zeros(pop_size)
	logger = ExperimentLogger(
		experiment_id,
		{**config, "device": str(device)},
		log_path=os.path.join(exp_dir, "telemetry.jsonl"),
	)
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

		epoch_losses, epoch_correct, epoch_total = [], 0, 0
		epoch_intermediate_correct = 0
		interactions = np.zeros((pop_size, pop_size))
		successes = np.zeros((pop_size, pop_size))

		for model in population:
			model.train()

		# Decidir qué cadenas usar por curriculum
		if epoch < nursery_end:
			active_chains = chains_2  # Solo cadenas de 2 pasos en guardería
		else:
			active_chains = chains  # Todas

		if not active_chains:
			active_chains = chains

		for step in range(steps_per_epoch):
			tau = max(tau_min, tau_start * (1.0 - current_step / total_steps))

			# Seleccionar cadena aleatoria
			batch_chains = [active_chains[np.random.randint(len(active_chains))] for _ in range(batch_size)]

			# Selección de pareja
			idx_s = np.random.randint(0, pop_size)
			idx_l = np.random.randint(0, pop_size)
			while idx_l == idx_s:
				idx_l = np.random.randint(0, pop_size)

			speaker, listener = population[idx_s], population[idx_l]
			opt_s, opt_l = optimizers[idx_s], optimizers[idx_l]
			opt_s.zero_grad()
			opt_l.zero_grad()

			# ═══════════════════════════════════════════
			# EL LOOP DIFERENCIABLE
			# ═══════════════════════════════════════════
			# Para cadenas de 2 pasos (A→B→C):
			#   Paso 1: speaker([A, implica, 0, 0]) → genera mensaje → listener extrae B_soft
			#   Paso 2: speaker([B_soft, implica, 0, 0]) → genera mensaje → listener extrae C_soft
			#   Loss: C_soft debe ser C_target

			op_tid = op_implica_tid
			total_loss = torch.tensor(0.0, device=device)

			# Preparar targets
			start_tids = torch.tensor([c["start_tid"] for c in batch_chains], dtype=torch.long, device=device)
			end_tids = torch.tensor([c["end_tid"] for c in batch_chains], dtype=torch.long, device=device)
			fears = torch.tensor([1.0 + c["fear"] * fear_amplifier for c in batch_chains], dtype=torch.float32, device=device)

			# Paso 1: Input = [start, implica, 0, 0]
			current_input = torch.zeros((batch_size, 4), dtype=torch.long, device=device)
			current_input[:, 0] = start_tids
			current_input[:, 1] = op_tid

			# Obtener primer paso intermedio target (para loss intermedia)
			# Para cadenas de 2 pasos: intermediates[0] es el paso intermedio
			intermediate_tids = []
			for c in batch_chains:
				if c["intermediate_tids"]:
					intermediate_tids.append(c["intermediate_tids"][0])
				else:
					# Cadena de 1 paso directo, el intermedio es el end
					# Esto no debería pasar con depth>=2
					intermediate_tids.append(c["end_tid"])
			intermediate_target = torch.tensor(intermediate_tids, dtype=torch.long, device=device)

			# === PASO 1 ===
			speaker_logits_1 = speaker(current_input, logit_mask=logit_mask)
			speaker_msg_1 = speaker.generate_message(current_input, tau=tau, hard=False, logit_mask=logit_mask)

			# Teacher forcing para paso 1
			if torch.rand(1).item() < tf_ratio:
				# Teacher: dar el intermedio correcto
				teacher_1 = current_input.clone()
				teacher_1[:, 2] = intermediate_target
				msg_to_listener_1 = F.one_hot(teacher_1, num_classes=8192).float()
			else:
				msg_to_listener_1 = speaker_msg_1

			listener_logits_1 = listener(msg_to_listener_1, logit_mask=logit_mask)

			# Loss intermedia: el listener debe decodificar el paso intermedio en posición 2
			loss_intermediate = F.cross_entropy(listener_logits_1[:, 2, :], intermediate_target, reduction="none")
			total_loss = total_loss + (loss_intermediate * fears * 0.5).mean()

			# Extraer B_soft del speaker para el siguiente paso
			# Usamos la salida SUAVE del speaker en posición 2 (el concepto que "piensa")
			b_soft = F.gumbel_softmax(speaker_logits_1[:, 2, :], tau=tau, hard=False, dim=-1)

			# === PASO 2 ===
			# Input = [B_soft, implica, 0, 0]
			# B_soft es un vector continuo (8192,) — la ruta diferenciable
			step2_input_soft = torch.zeros((batch_size, 4, 8192), device=device)
			step2_input_soft[:, 0, :] = b_soft  # B_soft en posición 0
			step2_input_soft[:, 1, :] = F.one_hot(torch.full((batch_size,), op_tid, dtype=torch.long, device=device), num_classes=8192).float()
			# Posiciones 2 y 3 quedan en cero (el modelo debe llenarlas)

			speaker_logits_2 = speaker(step2_input_soft, logit_mask=logit_mask)
			speaker_msg_2 = (
				speaker.generate_message_from_logits(speaker_logits_2, tau=tau, logit_mask=logit_mask)
				if hasattr(speaker, "generate_message_from_logits")
				else F.gumbel_softmax(speaker_logits_2, tau=tau, hard=False, dim=-1)
			)

			# Teacher forcing para paso 2
			if torch.rand(1).item() < tf_ratio:
				teacher_2 = torch.zeros((batch_size, 4), dtype=torch.long, device=device)
				teacher_2[:, 0] = intermediate_target
				teacher_2[:, 1] = op_tid
				teacher_2[:, 2] = end_tids
				msg_to_listener_2 = F.one_hot(teacher_2, num_classes=8192).float()
			else:
				msg_to_listener_2 = speaker_msg_2

			listener_logits_2 = listener(msg_to_listener_2, logit_mask=logit_mask)

			# ═══════════════════════════════════════════
			# LOSS FINAL: ¿llegó al destino correcto?
			# ═══════════════════════════════════════════
			loss_final = F.cross_entropy(listener_logits_2[:, 2, :], end_tids, reduction="none")
			total_loss = total_loss + (loss_final * fears).mean()

			# Loss de consistencia del speaker
			loss_speaker_1 = F.cross_entropy(speaker_logits_1[:, 2, :], intermediate_target, reduction="none")
			loss_speaker_2 = F.cross_entropy(speaker_logits_2[:, 2, :], end_tids, reduction="none")
			total_loss = total_loss + ((loss_speaker_1 + loss_speaker_2) * fears * 0.3).mean()

			total_loss.backward()
			# Gradient clipping para estabilidad del loop
			torch.nn.utils.clip_grad_norm_(speaker.parameters(), max_norm=1.0)
			torch.nn.utils.clip_grad_norm_(listener.parameters(), max_norm=1.0)
			opt_s.step()
			opt_l.step()

			epoch_losses.append(total_loss.item())

			# Métricas
			with torch.no_grad():
				pred_intermediate = torch.argmax(speaker_logits_1[:, 2, :], dim=-1)
				pred_final = torch.argmax(listener_logits_2[:, 2, :], dim=-1)

				intermediate_ok = (pred_intermediate == intermediate_target).sum().item()
				(pred_final == end_tids).sum().item()
				# Joint: ambos pasos correctos
				joint_ok = ((pred_intermediate == intermediate_target) & (pred_final == end_tids)).sum().item()

			epoch_intermediate_correct += intermediate_ok
			epoch_correct += joint_ok
			epoch_total += batch_size
			interactions[idx_s, idx_l] += batch_size
			successes[idx_s, idx_l] += joint_ok

			if step % 50 == 0:
				c = batch_chains[0]
				path = " → ".join(c["path_names"])
				pred_int_name = translator.decode([pred_intermediate[0].item()])
				pred_fin_name = translator.decode([pred_final[0].item()])
				target_int_name = CONCEPT_NAMES[c["intermediates"][0]] if c["intermediates"] else "?"
				target_fin_name = CONCEPT_NAMES[c["end"]]

				print(
					f"  step {step:3d} | τ={tau:.3f} TF={tf_ratio:.0%} | loss={total_loss.item():.3f} | "
					f"🔗 {path} | "
					f"paso1={pred_int_name}({'✓' if pred_int_name == target_int_name else '✗'}) "
					f"paso2={pred_fin_name}({'✓' if pred_fin_name == target_fin_name else '✗'})"
				)

			current_step += 1

		# Fitness
		for i in range(pop_size):
			sent_total = interactions[i, :].sum() + interactions[:, i].sum()
			sent_correct = successes[i, :].sum() + successes[:, i].sum()
			fitness[i] = sent_correct / (sent_total + 1e-10)

		avg_loss = np.mean(epoch_losses)
		acc_intermediate = (epoch_intermediate_correct / epoch_total) * 100
		acc_joint = (epoch_correct / epoch_total) * 100

		print(f"Loss: {avg_loss:.4f} | Paso Intermedio: {acc_intermediate:.2f}% | Cadena Completa: {acc_joint:.2f}%")
		print(f"Fitness: {[f'Agent_{i}: {f * 100:.2f}%' for i, f in enumerate(fitness)]}")

		logger.log_epoch(
			epoch=epoch + 1,
			loss_avg=avg_loss,
			acc_concept=acc_intermediate,
			acc_emotion=0.0,
			acc_joint=acc_joint,
			fitness=fitness.tolist(),
			worst_agent=int(np.argmin(fitness)),
			parent_a=-1,
			parent_b=-1,
			acc_homeostasis=acc_joint,
		)

		# Evolución SVD
		svd_interval = config.get("svd_interval", 3)
		svd_phases = config.get("svd_phases", ["recreo", "autonomia"])
		current_phase = "guarderia" if epoch < nursery_end else ("recreo" if epoch < transition_end else "autonomia")

		if current_phase in svd_phases and epoch % svd_interval == 0:
			worst_idx = int(np.argmin(fitness))
			best = np.argsort(fitness)[-2:]
			print(f"[Evolución] Reemplazando Agent_{worst_idx} con hijo SVD de Agent_{best[1]} y Agent_{best[0]}")
			svd_crossover(population[best[1]], population[best[0]], population[worst_idx])
			optimizers[worst_idx] = torch.optim.AdamW(filter(lambda p: p.requires_grad, population[worst_idx].parameters()), lr=lr, weight_decay=wd)

	# Guardar
	best_idx = np.argsort(fitness)[-1]
	save_path = os.path.join(exp_dir, "best_agent.pt")
	torch.save(population[best_idx].state_dict(), save_path)
	print(f"💾 Mejor agente guardado en {save_path}")
	logger.close()


if __name__ == "__main__":
	run_loop_training()
