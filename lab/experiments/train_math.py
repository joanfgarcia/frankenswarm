import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.bitnet.data.dataset_breeder import MathDatasetBreeder
from src.bitnet.model.modeling_bitnet import BitNet4LayerModel
from src.bitnet.telemetry.telemetry import ExperimentLogger
from src.bitnet.translation.translator import SovereignTranslator


def svd_crossover(parent_a: nn.Module, parent_b: nn.Module, child: nn.Module, alpha: float = 0.5, sigma: float = 0.01):
	"""
	Aplica cruzamiento y perturbación SVD en el espacio de parámetros entrenables
	de parent_a y parent_b, escribiendo el resultado en child.
	"""
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
					s_perturbed = s + noise
					s_perturbed.clamp_(min=0.0)
					w_child = u @ torch.diag(s_perturbed) @ vh
					param.copy_(w_child)
				except Exception:
					param.copy_(w_avg)
			else:
				noise = torch.randn_like(p_a) * sigma
				param.copy_(alpha * p_a + (1.0 - alpha) * p_b + noise)


def load_config() -> dict:
	parser = argparse.ArgumentParser(description="Arena Aritmética Frankenswarm")
	parser.add_argument("--config", type=str, default=None, help="Ruta al archivo de configuración JSON del experimento")
	args, unknown = parser.parse_known_args()

	base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	default_path = os.path.join(base_dir, "configs", "experiments", "default.json")

	with open(default_path, encoding="utf-8") as f:
		config = json.load(f)

	if args.config:
		config_path = args.config
		if not os.path.isabs(config_path):
			config_path = os.path.join(base_dir, config_path)
		if os.path.exists(config_path):
			print(f"📖 Cargando configuración del experimento desde {config_path}...")
			with open(config_path, encoding="utf-8") as f:
				exp_config = json.load(f)
			config.update(exp_config)
		else:
			print(f"⚠️ Archivo de configuración no encontrado: {config_path}. Usando valores predeterminados.")

	return config


def run_math_arena():
	config = load_config()
	experiment_id = config["experiment_id"]
	print(f"=== 🧮 Iniciando Arena Aritmética de Grado 1 — {experiment_id} ===")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"[Device]: {device}")

	seed = config.get("seed")
	if seed is not None:
		print(f"🌱 [SEED] Fijando semilla aleatoria: {seed}")
		np.random.seed(seed)
		torch.manual_seed(seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed_all(seed)

	base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	exp_dir = os.path.join(base_dir, "storage", "experiments", experiment_id)
	os.makedirs(exp_dir, exist_ok=True)

	config_copy_path = os.path.join(exp_dir, "config.json")
	with open(config_copy_path, "w", encoding="utf-8") as f:
		json.dump(config, f, ensure_ascii=False, indent=4)
	print(f"📄 Parámetros del experimento guardados en {config_copy_path}")

	# 1. Cargar Traductor
	translator = SovereignTranslator()
	vocab_embeddings = translator.get_concept_embeddings()

	# 2. Inicializar Math Breeder
	breeder = MathDatasetBreeder(translator)

	# 3. Inicializar Población (sin hotstart para evitar colisión de IDs reestructurados)
	pop_size = config["pop_size"]
	hidden_dim = config["hidden_dim"]
	num_layers = config["num_layers"]
	population = [
		BitNet4LayerModel(vocab_embeddings=vocab_embeddings, hidden_dim=hidden_dim, num_layers=num_layers).to(device) for _ in range(pop_size)
	]

	resume_checkpoint = config.get("resume_checkpoint")
	if resume_checkpoint:
		if not os.path.isabs(resume_checkpoint):
			resume_checkpoint = os.path.join(base_dir, resume_checkpoint)
		if os.path.exists(resume_checkpoint):
			print(f"💾 [HOTSTART] Cargando pesos desde checkpoint {resume_checkpoint}...")
			try:
				state_dict = torch.load(resume_checkpoint, map_location=device)
				for _idx, model in enumerate(population):
					model.load_state_dict(state_dict)
				print("✅ Inicialización de población completada con éxito.")
			except Exception as e:
				print(f"⚠️ Error cargando checkpoint: {e}. Inicializando pesos aleatorios.")
		else:
			print(f"ℹ️ Checkpoint {resume_checkpoint} no encontrado. Inicializando con pesos aleatorios.")

	lr = config["lr"]
	optimizers = [torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr) for model in population]

	# 4. Logit Masking
	use_logit_mask = config["use_logit_mask"]
	logit_mask = None
	if use_logit_mask:
		micro_vocab_words = config["micro_vocab_words"]
		if not micro_vocab_words:
			micro_vocab_words = ["cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez", "suma", "resta"]
		micro_vocab_words = list(set(micro_vocab_words))
		logit_mask = torch.zeros(8192, dtype=torch.bool, device=device)
		for word in micro_vocab_words:
			tids = translator.encode(word)
			if tids:
				logit_mask[tids[0]] = True
		print(f"🔒 Logit Masking activado: {len(micro_vocab_words)} palabras permitidas.")

	fitness = np.zeros(pop_size)
	epochs = config["epochs"]
	steps_per_epoch = config["steps_per_epoch"]
	batch_size = config["batch_size"]

	tau_start = config["tau_start"]
	tau_min = config["tau_min"]
	total_steps = epochs * steps_per_epoch

	nursery_end = config["nursery_end"]
	transition_end = config["transition_end"]
	tf_min = config["tf_min"]

	telemetry_log_path = os.path.join(exp_dir, "telemetry.jsonl")
	params = dict(config)
	params["device"] = str(device)
	logger = ExperimentLogger(experiment_id, params, log_path=telemetry_log_path)

	current_step = 0

	for epoch in range(epochs):
		if epoch < nursery_end:
			tf_ratio = 1.0
			phase_name = "🍼 Guardería"
		elif epoch < transition_end:
			progress = (epoch - nursery_end) / (transition_end - nursery_end)
			tf_ratio = 1.0 - progress * (1.0 - tf_min)
			phase_name = "🎮 Recreo"
		else:
			tf_ratio = tf_min
			phase_name = "🦅 Autonomía"

		print(f"\n--- Época {epoch + 1}/{epochs} [{phase_name}] TF={tf_ratio:.0%} ---")
		epoch_losses = []
		epoch_correct = 0
		epoch_a_correct = 0
		epoch_op_correct = 0
		epoch_b_correct = 0
		epoch_r_correct = 0
		epoch_total = 0

		interactions = np.zeros((pop_size, pop_size))
		successes = np.zeros((pop_size, pop_size))

		for step in range(steps_per_epoch):
			tau = max(tau_min, tau_start * (1.0 - current_step / total_steps))

			# Lote aritmético: (A, op, B, R)
			op_a_targets, op_a_token_ids, operator_targets, operator_token_ids, op_b_targets, op_b_token_ids, result_targets, result_token_ids = (
				breeder.generate_batch(batch_size)
			)

			op_a_token_ids_tensor = torch.from_numpy(op_a_token_ids).long().to(device)
			operator_token_ids_tensor = torch.from_numpy(operator_token_ids).long().to(device)
			op_b_token_ids_tensor = torch.from_numpy(op_b_token_ids).long().to(device)
			result_token_ids_tensor = torch.from_numpy(result_token_ids).long().to(device)

			idx_speaker = np.random.randint(0, pop_size)
			idx_listener = np.random.randint(0, pop_size)
			while idx_listener == idx_speaker:
				idx_listener = np.random.randint(0, pop_size)

			speaker = population[idx_speaker]
			listener = population[idx_listener]

			opt_speaker = optimizers[idx_speaker]
			opt_listener = optimizers[idx_listener]

			opt_speaker.zero_grad()
			opt_listener.zero_grad()

			# 1. Entrada del Hablante: [A, op, B, 0] (no sabe R a nivel de tokens de entrada)
			speaker_input = torch.zeros((batch_size, 4), dtype=torch.long, device=device)
			speaker_input[:, 0] = op_a_token_ids_tensor
			speaker_input[:, 1] = operator_token_ids_tensor
			speaker_input[:, 2] = op_b_token_ids_tensor

			# Ejecución del Speaker para calcular logits del mensaje
			speaker_logits = speaker(speaker_input, logit_mask=logit_mask)  # (batch_size, 4, 8192)

			# Canal Diferenciable (Gumbel-Softmax)
			speaker_message = speaker.generate_message(speaker_input, tau=tau, hard=True, logit_mask=logit_mask)

			# 2. Entrada de Profesor (Teacher Message con R)
			teacher_input = torch.zeros((batch_size, 4), dtype=torch.long, device=device)
			teacher_input[:, 0] = op_a_token_ids_tensor
			teacher_input[:, 1] = operator_token_ids_tensor
			teacher_input[:, 2] = op_b_token_ids_tensor
			teacher_input[:, 3] = result_token_ids_tensor

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

			# 3. El Oyente decodifica la secuencia en los 4 pasos:
			# - Paso 0: Operando A (basado en m_0)
			# - Paso 1: Operador   (basado en m_0, m_1)
			# - Paso 2: Operando B (basado en m_0, m_1, m_2)
			# - Paso 3: Resultado  (basado en m_0, m_1, m_2, m_3)
			logits = listener(message_input, logit_mask=logit_mask)  # (batch_size, 4, 8192)
			pred_a_logits = logits[:, 0, :]
			pred_op_logits = logits[:, 1, :]
			pred_b_logits = logits[:, 2, :]
			pred_r_logits = logits[:, 3, :]

			# Pérdidas del Oyente
			loss_a = F.cross_entropy(pred_a_logits, op_a_token_ids_tensor)
			loss_op = F.cross_entropy(pred_op_logits, operator_token_ids_tensor)
			loss_b = F.cross_entropy(pred_b_logits, op_b_token_ids_tensor)
			loss_r = F.cross_entropy(pred_r_logits, result_token_ids_tensor)
			loss_listener = loss_a + loss_op + loss_b + loss_r

			# 4. Pérdida de Consistencia del Emisor (Speaker Consistency Loss)
			# Fuerza al Speaker a calcular R en la posición 4 de sus logits de salida
			loss_speaker_cons = F.cross_entropy(speaker_logits[:, 3, :], result_token_ids_tensor)

			# Pérdida Conjunta Total
			loss = loss_listener + 1.0 * loss_speaker_cons
			loss.backward()

			opt_speaker.step()
			opt_listener.step()

			epoch_losses.append(loss.item())

			# Evaluación de aciertos
			preds_a = torch.argmax(pred_a_logits, dim=-1)
			preds_op = torch.argmax(pred_op_logits, dim=-1)
			preds_b = torch.argmax(pred_b_logits, dim=-1)
			preds_r = torch.argmax(pred_r_logits, dim=-1)

			a_ok = (preds_a == op_a_token_ids_tensor).sum().item()
			op_ok = (preds_op == operator_token_ids_tensor).sum().item()
			b_ok = (preds_b == op_b_token_ids_tensor).sum().item()
			r_ok = (preds_r == result_token_ids_tensor).sum().item()

			# Consenso absoluto: los 4 elementos decodificados perfectamente
			correct_joint = (
				(
					(preds_a == op_a_token_ids_tensor)
					& (preds_op == operator_token_ids_tensor)
					& (preds_b == op_b_token_ids_tensor)
					& (preds_r == result_token_ids_tensor)
				)
				.sum()
				.item()
			)

			epoch_correct += correct_joint
			epoch_a_correct += a_ok
			epoch_op_correct += op_ok
			epoch_b_correct += b_ok
			epoch_r_correct += r_ok
			epoch_total += batch_size

			interactions[idx_speaker, idx_listener] += batch_size
			successes[idx_speaker, idx_listener] += correct_joint

			# Logging
			if step % 40 == 0:
				sample_a_name = breeder.get_operand_name(op_a_targets[0])
				sample_op_name = breeder.get_operator_name(operator_targets[0])
				sample_b_name = breeder.get_operand_name(op_b_targets[0])
				sample_r_name = breeder.get_operand_name(result_targets[0])

				pred_a_name = translator.decode([preds_a[0].item()])
				pred_op_name = translator.decode([preds_op[0].item()])
				pred_b_name = translator.decode([preds_b[0].item()])
				pred_r_name = translator.decode([preds_r[0].item()])

				print(
					f"  step {step:3d} | τ={tau:.3f} TF={tf_ratio:.0%} | loss={loss.item():.3f} | "
					f"target=({sample_a_name} {sample_op_name} {sample_b_name} = {sample_r_name}) → pred=({pred_a_name} {pred_op_name} {pred_b_name} = {pred_r_name})"
				)

			current_step += 1

		# Calcular fitness acumulado de cada agente
		for i in range(pop_size):
			sent_total = interactions[i, :].sum() + interactions[:, i].sum()
			sent_correct = successes[i, :].sum() + successes[:, i].sum()
			fitness[i] = sent_correct / (sent_total + 1e-10)

		avg_loss = np.mean(epoch_losses)
		acc_a = (epoch_a_correct / epoch_total) * 100
		acc_op = (epoch_op_correct / epoch_total) * 100
		acc_b = (epoch_b_correct / epoch_total) * 100
		acc_j = (epoch_correct / epoch_total) * 100

		print(
			f"Pérdida promedio: {avg_loss:.4f} | Operando A: {acc_a:.2f}% | Operador: {acc_op:.2f}% | Operando B: {acc_b:.2f}% | Consenso Aritmético: {acc_j:.2f}%"
		)
		print(f"Fitness de la Población: {[f'Agent_{i}: {f * 100:.2f}%' for i, f in enumerate(fitness)]}")

		if epoch < nursery_end:
			current_phase = "guarderia"
		elif epoch < transition_end:
			current_phase = "recreo"
		else:
			current_phase = "autonomia"

		svd_interval = config.get("svd_interval", 1)
		svd_phases = config.get("svd_phases", ["recreo", "autonomia"])

		worst_idx = -1
		parent_a_idx = -1
		parent_b_idx = -1

		if current_phase in svd_phases and epoch % svd_interval == 0:
			worst_idx = int(np.argmin(fitness))
			best_indices = np.argsort(fitness)[-2:]
			parent_a_idx = int(best_indices[1])
			parent_b_idx = int(best_indices[0])
			print(f"[Evolución] Reemplazando Agent_{worst_idx} con hijo SVD de Agent_{parent_a_idx} y Agent_{parent_b_idx}")
			svd_crossover(parent_a=population[parent_a_idx], parent_b=population[parent_b_idx], child=population[worst_idx], alpha=0.5, sigma=0.01)
			optimizers[worst_idx] = torch.optim.AdamW(filter(lambda p: p.requires_grad, population[worst_idx].parameters()), lr=lr)

		logger.log_epoch(
			epoch=epoch + 1,
			loss_avg=avg_loss,
			acc_concept=acc_a,
			acc_emotion=acc_op,
			acc_joint=acc_j,
			fitness=fitness.tolist(),
			worst_agent=worst_idx,
			parent_a=parent_a_idx,
			parent_b=parent_b_idx,
			acc_homeostasis=acc_b,
		)

		if epoch >= transition_end and acc_j >= 80.0:
			print(f"\n🏆 ¡HIT ALCANZADO! La población ha superado el 80% de entendimiento aritmético ({acc_j:.2f}%).")
			logger.log_event("promotion", {"grade": 1, "acc_joint": acc_j})
			break

	best_idx = np.argsort(fitness)[-1]
	checkpoint_save_path = os.path.join(exp_dir, "best_agent.pt")
	torch.save(population[best_idx].state_dict(), checkpoint_save_path)
	print(f"💾 Checkpoint del mejor agente guardado en {checkpoint_save_path}")

	logger.close()


if __name__ == "__main__":
	run_math_arena()
