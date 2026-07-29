"""
EXP_025: Lógica Proposicional con Emoción y Miedo.

Usa el motor genérico (train_generic.py) con hooks IoC para:
1. Inyectar el PropositionalLogicBreeder en vez del aritmético
2. Modular la función de pérdida con fear_weights
3. Añadir el vector de emoción como contexto implícito

El miedo no es un canal separado — es un multiplicador de la pérdida.
Equivocarse en "fuego implica peligro" duele 2.8x más que
equivocarse en "árbol implica aire".

Esto crea presión selectiva asimétrica: los agentes que aprenden
las reglas de supervivencia primero son los que sobreviven la
evolución SVD.
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Importar operadores lógicos para que se auto-registren en el Registry
import src.bitnet.operators.operators_logic  # noqa: F401 — side-effect import
from src.bitnet.data.logic_breeder import PropositionalLogicBreeder
from src.bitnet.model.modeling_bitnet import BitNet4LayerModel
from src.bitnet.telemetry.telemetry import ExperimentLogger
from src.bitnet.translation.translator import SovereignTranslator


def svd_crossover(parent_a: nn.Module, parent_b: nn.Module, child: nn.Module, alpha: float = 0.5, sigma: float = 0.01):
	"""Aplica cruzamiento y perturbación SVD."""
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
					noise = torch.randn_like(s) * sigma
					s_perturbed = (s + noise).clamp_(min=0.0)
					param.copy_(u @ torch.diag(s_perturbed) @ vh)
				except Exception:
					param.copy_(w_avg)
			else:
				noise = torch.randn_like(p_a) * sigma
				param.copy_(alpha * p_a + (1.0 - alpha) * p_b + noise)


def load_config() -> dict:
	parser = argparse.ArgumentParser(description="Arena de Lógica Proposicional + Miedo")
	parser.add_argument("--config", type=str, default=None)
	args, _ = parser.parse_known_args()

	base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	default_path = os.path.join(base_dir, "configs", "experiments", "EXP_025.json")

	with open(default_path, encoding="utf-8") as f:
		config = json.load(f)

	if args.config:
		config_path = args.config if os.path.isabs(args.config) else os.path.join(base_dir, args.config)
		if os.path.exists(config_path):
			with open(config_path, encoding="utf-8") as f:
				config.update(json.load(f))

	return config


def evaluate_test_set(population, breeder, logit_mask, device, emotion_mode="none"):
	"""Evalúa generalización en test set con métricas de emoción y miedo."""
	for model in population:
		model.eval()

	test_eqs = breeder.test_equations
	if not test_eqs:
		return 0.0, 0.0, 0.0

	batch_size = len(test_eqs)
	(
		ca_targets,
		ca_tids,
		op_targets,
		op_tids,
		cb_targets,
		cb_tids,
		r_targets,
		r_tids,
		emo_tids,
		fear_weights,
	) = breeder.generate_batch(batch_size, mode="test")

	ca_tensor = torch.from_numpy(ca_tids).long().to(device)
	op_tensor = torch.from_numpy(op_tids).long().to(device)
	cb_tensor = torch.from_numpy(cb_tids).long().to(device)
	r_tensor = torch.from_numpy(r_tids).long().to(device)
	fear_tensor = torch.from_numpy(fear_weights).float().to(device)

	pop_size = len(population)
	correct_counts = []
	fear_correct = []  # Aciertos en ejemplos con miedo > 1.5

	with torch.no_grad():
		for i in range(pop_size):
			for j in range(pop_size):
				if i == j:
					continue
				speaker = population[i]
				listener = population[j]

				speaker_input = torch.zeros((batch_size, 4), dtype=torch.long, device=device)
				speaker_input[:, 0] = ca_tensor
				speaker_input[:, 1] = op_tensor
				speaker_input[:, 2] = cb_tensor
				if emotion_mode == "input":
					emo_tensor = torch.from_numpy(emo_tids).long().to(device)
					speaker_input[:, 3] = emo_tensor

				message = speaker.generate_message(speaker_input, tau=0.3, hard=True, logit_mask=logit_mask)
				logits = listener(message, logit_mask=logit_mask)

				preds_a = torch.argmax(logits[:, 0, :], dim=-1)
				preds_op = torch.argmax(logits[:, 1, :], dim=-1)
				preds_b = torch.argmax(logits[:, 2, :], dim=-1)
				preds_r = torch.argmax(logits[:, 3, :], dim=-1)

				joint_ok = (preds_a == ca_tensor) & (preds_op == op_tensor) & (preds_b == cb_tensor) & (preds_r == r_tensor)
				correct_counts.append(joint_ok.sum().item())

				# Métricas de supervivencia: ¿acierta en los peligrosos?
				high_fear_mask = fear_tensor > 1.5
				if high_fear_mask.any():
					fear_ok = (joint_ok & high_fear_mask).sum().item()
					fear_total = high_fear_mask.sum().item()
					fear_correct.append(fear_ok / fear_total * 100)

	avg_acc = np.mean(correct_counts) / batch_size * 100
	avg_fear_acc = np.mean(fear_correct) if fear_correct else 0.0

	return avg_acc, avg_fear_acc


def run_logic_arena():
	config = load_config()
	experiment_id = config["experiment_id"]
	emotion_mode = config.get("emotion_mode", "none")
	print(f"=== 🧠 Arena de Lógica Proposicional + Miedo — {experiment_id} ===")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"[Device]: {device}")

	seed = config.get("seed", 42)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)

	base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	exp_dir = os.path.join(base_dir, "storage", "experiments", experiment_id)
	os.makedirs(exp_dir, exist_ok=True)

	with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf-8") as f:
		json.dump(config, f, ensure_ascii=False, indent=4)

	# 1. Traductor
	translator = SovereignTranslator()
	vocab_embeddings = translator.get_concept_embeddings()

	# 2. Breeder con emoción y miedo
	operators_config = config.get("operators", {})
	enabled_operators = list(operators_config.keys())
	fear_amplifier = config.get("fear_amplifier", 2.0)

	breeder = PropositionalLogicBreeder(
		translator,
		enabled_operators=enabled_operators,
		split_ratio=config.get("split_ratio", 0.90),
		seed=seed,
		fear_amplifier=fear_amplifier,
	)
	print(f"📊 Partición Lógica: Train={len(breeder.train_equations)} | Test={len(breeder.test_equations)}")
	print(f"😱 Fear Amplifier: {fear_amplifier}x")
	print(f"💚 Emotion Mode: {emotion_mode}")

	# 3. Población
	pop_size = config["pop_size"]
	hidden_dim = config["hidden_dim"]
	num_layers = config["num_layers"]

	population = [
		BitNet4LayerModel(
			vocab_embeddings=vocab_embeddings,
			hidden_dim=hidden_dim,
			num_layers=num_layers,
			use_pos_embedding=config.get("use_pos_embedding", True),
		).to(device)
		for _ in range(pop_size)
	]

	# Hotstart desde checkpoint anterior (e.g., EXP_024 relacional)
	resume = config.get("resume_checkpoint")
	if resume:
		resume_path = resume if os.path.isabs(resume) else os.path.join(base_dir, resume)
		if os.path.exists(resume_path):
			print(f"💾 [HOTSTART] Cargando pesos desde {resume_path}...")
			try:
				state_dict = torch.load(resume_path, map_location=device)
				for model in population:
					model.load_state_dict(state_dict)
				print("✅ Transferencia de conocimiento completada.")
			except Exception as e:
				print(f"⚠️ Error: {e}. Iniciando desde cero.")

	lr = config["lr"]
	wd = config.get("weight_decay", 0.01)
	optimizers = [torch.optim.AdamW(filter(lambda p: p.requires_grad, m.parameters()), lr=lr, weight_decay=wd) for m in population]

	# 4. Logit Mask
	logit_mask = None
	if config.get("use_logit_mask", True):
		micro_vocab = config["micro_vocab_words"]
		logit_mask = torch.zeros(8192, dtype=torch.bool, device=device)
		for word in micro_vocab:
			tids = translator.encode(word)
			if tids:
				logit_mask[tids[0]] = True
		print(f"🔒 Logit Mask: {len(micro_vocab)} palabras permitidas.")

	# 5. Bucle de entrenamiento
	fitness = np.zeros(pop_size)
	epochs = config["epochs"]
	steps_per_epoch = config["steps_per_epoch"]
	batch_size = config["batch_size"]
	tau_start, tau_min = config["tau_start"], config["tau_min"]
	total_steps = epochs * steps_per_epoch
	nursery_end = config["nursery_end"]
	transition_end = config["transition_end"]
	tf_min = config["tf_min"]

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
		epoch_fear_losses = []
		interactions = np.zeros((pop_size, pop_size))
		successes = np.zeros((pop_size, pop_size))

		for model in population:
			model.train()

		for step in range(steps_per_epoch):
			tau = max(tau_min, tau_start * (1.0 - current_step / total_steps))

			# Lote con emoción y miedo
			(
				ca_targets,
				ca_tids,
				op_targets,
				op_tids,
				cb_targets,
				cb_tids,
				r_targets,
				r_tids,
				emo_tids,
				fear_weights,
			) = breeder.generate_batch(batch_size, mode="train")

			ca_t = torch.from_numpy(ca_tids).long().to(device)
			op_t = torch.from_numpy(op_tids).long().to(device)
			cb_t = torch.from_numpy(cb_tids).long().to(device)
			r_t = torch.from_numpy(r_tids).long().to(device)
			emo_t = torch.from_numpy(emo_tids).long().to(device)
			fear_t = torch.from_numpy(fear_weights).float().to(device)

			# Selección de pareja
			idx_s = np.random.randint(0, pop_size)
			idx_l = np.random.randint(0, pop_size)
			while idx_l == idx_s:
				idx_l = np.random.randint(0, pop_size)

			speaker, listener = population[idx_s], population[idx_l]
			opt_s, opt_l = optimizers[idx_s], optimizers[idx_l]
			opt_s.zero_grad()
			opt_l.zero_grad()

			# Input del Speaker: [concepto_a, operador, concepto_b, contexto]
			speaker_input = torch.zeros((batch_size, 4), dtype=torch.long, device=device)
			speaker_input[:, 0] = ca_t
			speaker_input[:, 1] = op_t
			speaker_input[:, 2] = cb_t
			if emotion_mode == "input":
				speaker_input[:, 3] = emo_t  # El chroma colorea el razonamiento

			speaker_logits = speaker(speaker_input, logit_mask=logit_mask)
			speaker_message = speaker.generate_message(speaker_input, tau=tau, hard=True, logit_mask=logit_mask)

			# Teacher input: [concepto_a, operador, concepto_b, resultado]
			teacher_input = torch.zeros((batch_size, 4), dtype=torch.long, device=device)
			teacher_input[:, 0] = ca_t
			teacher_input[:, 1] = op_t
			teacher_input[:, 2] = cb_t
			teacher_input[:, 3] = r_t

			teacher_onehot = F.one_hot(teacher_input, num_classes=8192).float()

			# Scheduled Teacher Forcing
			use_teacher = torch.rand(batch_size, device=device) < tf_ratio
			if use_teacher.all():
				message_input = teacher_onehot
			elif use_teacher.any():
				mask = use_teacher.view(-1, 1, 1).float()
				message_input = mask * teacher_onehot + (1 - mask) * speaker_message
			else:
				message_input = speaker_message

			# Listener decodifica
			logits = listener(message_input, logit_mask=logit_mask)

			# Pérdidas por muestra, moduladas por MIEDO
			loss_a = F.cross_entropy(logits[:, 0, :], ca_t, reduction="none")
			loss_op = F.cross_entropy(logits[:, 1, :], op_t, reduction="none")
			loss_b = F.cross_entropy(logits[:, 2, :], cb_t, reduction="none")
			loss_r = F.cross_entropy(logits[:, 3, :], r_t, reduction="none")

			# ═══════════════════════════════════════════
			# EL MIEDO MODULA LA PÉRDIDA
			# ═══════════════════════════════════════════
			# fear_t ya viene calculado por el breeder:
			#   - base 1.0 para conceptos neutros
			#   - hasta 1.0 + (1.0 * amplifier) = 3.0 para "peligro"
			# Esto significa que equivocarse en "fuego implica peligro"
			# produce 3x más gradiente que equivocarse en "gato implica seguridad"
			loss_listener = ((loss_a + loss_op + loss_b + loss_r) * fear_t).mean()

			# Consistencia del Speaker (también modulada por miedo)
			loss_speaker = (F.cross_entropy(speaker_logits[:, 3, :], r_t, reduction="none") * fear_t).mean()

			loss = loss_listener + loss_speaker

			# ═══════════════════════════════════════════
			# MODO TARGET: La emoción como objetivo adicional
			# ═══════════════════════════════════════════
			# El speaker debe aprender a SENTIR: predecir la emoción
			# apropiada a partir del contexto lógico.
			# "fuego implica peligro" → el speaker debe sentir MIEDO
			# "agua implica seguridad" → el speaker debe sentir ALEGRÍA
			if emotion_mode == "target":
				emo_loss_weight = config.get("emotion_loss_weight", 0.5)
				loss_emo = F.cross_entropy(speaker_logits[:, 0, :], emo_t, reduction="none")
				loss += (loss_emo * fear_t).mean() * emo_loss_weight
			loss.backward()
			opt_s.step()
			opt_l.step()

			epoch_losses.append(loss.item())
			epoch_fear_losses.append((loss_r * fear_t).mean().item())

			# Métricas
			preds = [torch.argmax(logits[:, i, :], dim=-1) for i in range(4)]
			joint_ok = ((preds[0] == ca_t) & (preds[1] == op_t) & (preds[2] == cb_t) & (preds[3] == r_t)).sum().item()

			epoch_correct += joint_ok
			epoch_total += batch_size
			interactions[idx_s, idx_l] += batch_size
			successes[idx_s, idx_l] += joint_ok

			if step % 40 == 0:
				sa = breeder.get_concept_name(ca_targets[0])
				sop = breeder.get_operator_name(op_targets[0])
				sb = breeder.get_concept_name(cb_targets[0])
				sr = breeder.get_result_name(r_targets[0])
				emo_idx = breeder._get_emotion_id(ca_targets[0])
				emo_name = breeder.get_emotion_name(emo_idx)
				fear_val = fear_weights[0]

				pa = translator.decode([preds[0][0].item()])
				pb = translator.decode([preds[2][0].item()])
				pr = translator.decode([preds[3][0].item()])

				# En modo target, mostrar predicción emocional del speaker
				emo_pred_str = ""
				if emotion_mode == "target":
					pred_emo_id = torch.argmax(speaker_logits[0, 0, :]).item()
					pred_emo_name = translator.decode([pred_emo_id])
					emo_pred_str = f" → siente:{pred_emo_name}"

				print(
					f"  step {step:3d} | τ={tau:.3f} TF={tf_ratio:.0%} | loss={loss.item():.3f} | "
					f"({sa} {sop} {sb} = {sr}) [😱{fear_val:.1f} 💚{emo_name}] "
					f"→ pred=({pa} _ {pb} = {pr}){emo_pred_str}"
				)

			current_step += 1

		# Fitness
		for i in range(pop_size):
			sent_total = interactions[i, :].sum() + interactions[:, i].sum()
			sent_correct = successes[i, :].sum() + successes[:, i].sum()
			fitness[i] = sent_correct / (sent_total + 1e-10)

		# Test
		test_acc, fear_acc = evaluate_test_set(population, breeder, logit_mask, device, emotion_mode=emotion_mode)

		avg_loss = np.mean(epoch_losses)
		avg_fear_loss = np.mean(epoch_fear_losses)
		train_acc = (epoch_correct / epoch_total) * 100

		print(
			f"Loss: {avg_loss:.4f} | Fear Loss: {avg_fear_loss:.4f} | Train: {train_acc:.2f}% | Test: {test_acc:.2f}% | 😱 Survival: {fear_acc:.2f}%"
		)
		print(f"Fitness: {[f'Agent_{i}: {f * 100:.2f}%' for i, f in enumerate(fitness)]}")

		logger.log_epoch(
			epoch=epoch + 1,
			loss_avg=avg_loss,
			acc_concept=train_acc,
			acc_emotion=fear_acc,
			acc_joint=train_acc,
			fitness=fitness.tolist(),
			worst_agent=int(np.argmin(fitness)),
			parent_a=-1,
			parent_b=-1,
			acc_homeostasis=test_acc,
		)

		# Evolución SVD
		svd_interval = config.get("svd_interval", 2)
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
	run_logic_arena()
