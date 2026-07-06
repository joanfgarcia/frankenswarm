import argparse
import importlib.util
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.bitnet.data.generalization_breeder import CompositionalMathDatasetBreeder
from src.bitnet.model.modeling_bitnet import BitNet4LayerModel
from src.bitnet.operators.operators import OperatorRegistry
from src.bitnet.telemetry.telemetry import ExperimentLogger
from src.bitnet.translation.translator import SovereignTranslator


def svd_crossover(parent_a: nn.Module, parent_b: nn.Module, child: nn.Module, alpha: float = 0.5, sigma: float = 0.01):
	"""Aplica cruzamiento y perturbación SVD en el espacio de parámetros entrenables."""
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
	parser = argparse.ArgumentParser(description="Motor de Entrenamiento Genérico Frankenswarm")
	parser.add_argument("--config", type=str, default=None, help="Ruta al archivo de configuración JSON del experimento")
	args, unknown = parser.parse_known_args()

	base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	default_path = os.path.join(base_dir, "configs", "experiments", "default.json")

	if not os.path.exists(default_path):
		# Generar default básico si no existe
		default_config = {
			"experiment_id": "DEFAULT",
			"vocab_size": 8192,
			"hidden_dim": 128,
			"num_layers": 2,
			"pop_size": 4,
			"epochs": 50,
			"steps_per_epoch": 200,
			"batch_size": 32,
			"lr": 0.001,
			"weight_decay": 0.01,
			"tau_start": 1.0,
			"tau_min": 0.3,
			"nursery_end": 5,
			"transition_end": 20,
			"tf_min": 0.20,
			"use_logit_mask": True,
			"seed": 42,
			"split_ratio": 0.90,
			"operators": {"suma": {"enabled": True, "loss_weight": 1.0}, "resta": {"enabled": True, "loss_weight": 1.0}},
			"micro_vocab_words": ["cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez", "suma", "resta"],
		}
		return default_config

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


def load_experiment_hooks(experiment_id: str, base_dir: str):
	"""Busca y carga dinámicamente el script de hooks.py específico para el experimento."""
	hooks_path = os.path.join(base_dir, "configs", "experiments", experiment_id, "hooks.py")
	if os.path.exists(hooks_path):
		try:
			spec = importlib.util.spec_from_file_location("exp_hooks", hooks_path)
			hooks = importlib.util.module_from_spec(spec)
			spec.loader.exec_module(hooks)
			print(f"🔌 Hooks del experimento cargados desde {hooks_path}")
			return hooks
		except Exception as e:
			print(f"❌ Error al cargar hooks desde {hooks_path}: {e}")
	return None


def run_training():
	config = load_config()
	experiment_id = config["experiment_id"]
	print(f"=== 🧮 Iniciando Arena Aritmética Genérica — {experiment_id} ===")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"[Device]: {device}")

	seed = config.get("seed", 42)
	if seed is None:
		seed = 42
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

	# 1. Cargar Traductor y Hooks
	translator = SovereignTranslator()
	vocab_embeddings = translator.get_concept_embeddings()
	hooks = load_experiment_hooks(experiment_id, base_dir)

	# 2. Inicializar Breeder Composicional basado en operadores de Config
	operators_config = config.get("operators", {"suma": {"loss_weight": 1.0}, "resta": {"loss_weight": 1.2}})
	enabled_operators = list(operators_config.keys())
	split_ratio = config.get("split_ratio", 0.9)

	breeder = CompositionalMathDatasetBreeder(translator, split_ratio=split_ratio, seed=seed, enabled_operators=enabled_operators)
	print(f"📊 Partición Aritmética Dinámica: Train={len(breeder.train_equations)} | Test={len(breeder.test_equations)}")

	# 3. Inicializar Población
	pop_size = config["pop_size"]
	hidden_dim = config["hidden_dim"]
	num_layers = config["num_layers"]
	use_pos_embedding = config.get("use_pos_embedding", False)

	population = [
		BitNet4LayerModel(vocab_embeddings=vocab_embeddings, hidden_dim=hidden_dim, num_layers=num_layers, use_pos_embedding=use_pos_embedding).to(
			device
		)
		for _ in range(pop_size)
	]

	resume_checkpoint = config.get("resume_checkpoint")
	if resume_checkpoint:
		if not os.path.isabs(resume_checkpoint):
			resume_checkpoint = os.path.join(base_dir, resume_checkpoint)
		if os.path.exists(resume_checkpoint):
			print(f"💾 [HOTSTART] Cargando pesos desde checkpoint {resume_checkpoint}...")
			try:
				state_dict = torch.load(resume_checkpoint, map_location=device)
				for model in population:
					model.load_state_dict(state_dict)
				print("✅ Inicialización de población completada con éxito.")
			except Exception as e:
				print(f"⚠️ Error cargando checkpoint: {e}. Inicializando pesos aleatorios.")
		else:
			print(f"ℹ️ Checkpoint {resume_checkpoint} no encontrado. Inicializando con pesos aleatorios.")

	lr = config["lr"]
	wd = config.get("weight_decay", 0.01)
	print(f"🏋️ [Optimizer] L2 Regularization (Weight Decay): {wd}")
	optimizers = [torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd) for model in population]

	# 4. Logit Masking
	use_logit_mask = config["use_logit_mask"]
	logit_mask = None
	if use_logit_mask:
		micro_vocab_words = config["micro_vocab_words"]
		logit_mask = torch.zeros(8192, dtype=torch.bool, device=device)
		for word in micro_vocab_words:
			tids = translator.encode(word)
			if tids:
				logit_mask[tids[0]] = True
		print(f"🔒 Logit Masking activado: {len(micro_vocab_words)} palabras permitidas.")

	# 5. Inicializar Pesos de Pérdida Vectorizados
	max_op_idx = 0
	op_weights = {}
	for op_name in enabled_operators:
		info = OperatorRegistry.get_operator_info(op_name)
		if info:
			idx = info["idx"]
			max_op_idx = max(max_op_idx, idx)
			# Por defecto peso 1.0 si no se define en config
			weight = operators_config[op_name].get("loss_weight", 1.0)
			op_weights[idx] = weight

	# Construir el tensor de pesos mapeados
	weights_list = [1.0] * (max_op_idx + 1)
	for idx, w in op_weights.items():
		weights_list[idx] = w
	weights_tensor = torch.tensor(weights_list, dtype=torch.float32, device=device)

	print(f"⚖️ [Pérdidas] Multiplicadores por ID de operador: {weights_list}")

	# 6. Preparar bucle
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

	# Gancho de ciclo de vida: on_train_begin
	if hooks and hasattr(hooks, "on_train_begin"):
		hooks.on_train_begin(population, optimizers, config, device)

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

		# Gancho de ciclo de vida: on_epoch_begin
		if hooks and hasattr(hooks, "on_epoch_begin"):
			hooks.on_epoch_begin(epoch, population, optimizers, breeder, config)

		epoch_losses = []
		epoch_correct = 0
		epoch_a_correct = 0
		epoch_op_correct = 0
		epoch_b_correct = 0
		epoch_r_correct = 0
		epoch_total = 0

		interactions = np.zeros((pop_size, pop_size))
		successes = np.zeros((pop_size, pop_size))

		# Asegurar modo entrenamiento
		for model in population:
			model.train()

		# Determinar ecuaciones activas para este epoch (Soporta filtrado dinámico o curriculum base)
		current_eqs = None
		if hooks and hasattr(hooks, "get_train_equations"):
			current_eqs = hooks.get_train_equations(epoch, breeder, config)
		else:
			curriculum_mode = config.get("curriculum_mode", None)
			curriculum_stage1_epochs = config.get("curriculum_stage1_epochs", 25)
			if curriculum_mode == "stage_restas" and epoch < curriculum_stage1_epochs:
				# Stage 1: Solo restas con resultado positivo
				current_eqs = [eq for eq in breeder.train_equations if eq[1] == 1 and eq[3] > 0]
			else:
				current_eqs = breeder.train_equations

		for step in range(steps_per_epoch):
			tau = max(tau_min, tau_start * (1.0 - current_step / total_steps))

			# Lote aritmético
			op_a_targets, op_a_token_ids, operator_targets, operator_token_ids, op_b_targets, op_b_token_ids, result_targets, result_token_ids = (
				breeder.generate_batch(batch_size, mode="train", custom_eqs=current_eqs)
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

			# 1. Entrada del Hablante: [A, op, B, 0]
			speaker_input = torch.zeros((batch_size, 4), dtype=torch.long, device=device)
			speaker_input[:, 0] = op_a_token_ids_tensor
			speaker_input[:, 1] = operator_token_ids_tensor
			speaker_input[:, 2] = op_b_token_ids_tensor

			# Gancho de ciclo de vida: on_batch_start
			if hooks and hasattr(hooks, "on_batch_start"):
				speaker_input, teacher_input = hooks.on_batch_start(
					speaker_input,
					op_a_token_ids_tensor,
					operator_token_ids_tensor,
					op_b_token_ids_tensor,
					result_token_ids_tensor,
					epoch,
					step,
					config,
				)

			# Ejecución del Speaker
			speaker_logits = speaker(speaker_input, logit_mask=logit_mask)

			# Canal Diferenciable
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

			# 3. El Oyente decodifica
			logits = listener(message_input, logit_mask=logit_mask)
			pred_a_logits = logits[:, 0, :]
			pred_op_logits = logits[:, 1, :]
			pred_b_logits = logits[:, 2, :]
			pred_r_logits = logits[:, 3, :]

			# Obtener peso dinámico indexado vectorialmente por ID de operador
			operator_targets_tensor = torch.from_numpy(operator_targets).to(device)
			sample_weights = weights_tensor[operator_targets_tensor]

			# Pérdidas por muestra del Oyente
			loss_a_s = F.cross_entropy(pred_a_logits, op_a_token_ids_tensor, reduction="none")
			loss_op_s = F.cross_entropy(pred_op_logits, operator_token_ids_tensor, reduction="none")
			loss_b_s = F.cross_entropy(pred_b_logits, op_b_token_ids_tensor, reduction="none")
			loss_r_s = F.cross_entropy(pred_r_logits, result_token_ids_tensor, reduction="none")

			loss_listener_s = loss_a_s + loss_op_s + loss_b_s + loss_r_s
			loss_listener = (loss_listener_s * sample_weights).mean()

			# 4. Pérdida de Consistencia del Emisor por muestra
			loss_speaker_cons_s = F.cross_entropy(speaker_logits[:, 3, :], result_token_ids_tensor, reduction="none")
			loss_speaker_cons = (loss_speaker_cons_s * sample_weights).mean()

			# Pérdida Conjunta Total
			loss = loss_listener + 1.0 * loss_speaker_cons

			# Gancho de ciclo de vida: on_loss_calculated
			if hooks and hasattr(hooks, "on_loss_calculated"):
				loss = hooks.on_loss_calculated(loss, logits, speaker_logits, sample_weights, config)

			loss.backward()

			opt_speaker.step()
			opt_listener.step()

			epoch_losses.append(loss.item())

			# Aciertos de entrenamiento
			preds_a = torch.argmax(pred_a_logits, dim=-1)
			preds_op = torch.argmax(pred_op_logits, dim=-1)
			preds_b = torch.argmax(pred_b_logits, dim=-1)
			preds_r = torch.argmax(pred_r_logits, dim=-1)

			a_ok = (preds_a == op_a_token_ids_tensor).sum().item()
			op_ok = (preds_op == operator_token_ids_tensor).sum().item()
			b_ok = (preds_b == op_b_token_ids_tensor).sum().item()
			r_ok = (preds_r == result_token_ids_tensor).sum().item()

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

			current_step += 1

		# Calcular fitness acumulado
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
			f"[Train] Pérdida: {avg_loss:.4f} | Operando A: {acc_a:.2f}% | Operador: {acc_op:.2f}% | Operando B: {acc_b:.2f}% | Consenso: {acc_j:.2f}%"
		)

		# --- FASE DE VALIDACIÓN GENÉRICA (SISTEMÁTICA / TEST SPLIT) ---
		model_eval_correct = 0
		model_eval_total = 0

		# Diccionario dinámico para almacenar estadísticas por índice global de operador
		op_eval_stats = {}
		test_eqs = breeder.test_equations

		# Poner agentes en modo evaluación
		for model in population:
			model.eval()

		with torch.no_grad():
			# Evaluar todas las parejas posibles de agentes en todo el conjunto test
			for idx_spk in range(pop_size):
				for idx_lis in range(pop_size):
					if idx_spk == idx_lis:
						continue
					spk_model = population[idx_spk]
					lis_model = population[idx_lis]

					for a, op, b, r in test_eqs:
						a_token_id = breeder.operand_token_ids[a]
						op_token_id = breeder.operator_token_ids_map[op]
						b_token_id = breeder.operand_token_ids[b]
						r_token_id = breeder.operand_token_ids[r]

						# Input Speaker
						spk_in = torch.tensor([[a_token_id, op_token_id, b_token_id, 0]], dtype=torch.long, device=device)
						msg = spk_model.generate_message(spk_in, tau=0.1, hard=True, logit_mask=logit_mask)
						logits_lis = lis_model(msg, logit_mask=logit_mask)

						pred_a = torch.argmax(logits_lis[0, 0, :]).item()
						pred_op = torch.argmax(logits_lis[0, 1, :]).item()
						pred_b = torch.argmax(logits_lis[0, 2, :]).item()
						pred_r = torch.argmax(logits_lis[0, 3, :]).item()

						joint_ok = pred_a == a_token_id and pred_op == op_token_id and pred_b == b_token_id and pred_r == r_token_id

						# Acumular estadísticas dinámicamente
						if op not in op_eval_stats:
							op_eval_stats[op] = {"total": 0, "correct": 0}
						op_eval_stats[op]["total"] += 1
						if joint_ok:
							op_eval_stats[op]["correct"] += 1

						if joint_ok:
							model_eval_correct += 1
						model_eval_total += 1

		# Calcular porcentajes
		acc_joint_test = (model_eval_correct / model_eval_total) * 100 if model_eval_total > 0 else 0.0

		# Construir traza desglosada dinámicamente
		print_parts = [f"Global: {acc_joint_test:.2f}%"]
		logged_accs = {}
		for op_name in enabled_operators:
			info = OperatorRegistry.get_operator_info(op_name)
			if info:
				op_idx = info["idx"]
				stats = op_eval_stats.get(op_idx, {"total": 0, "correct": 0})
				acc = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0.0

				# Mapear nombres a emojis
				emoji = "➕" if op_name == "suma" else ("➖" if op_name == "resta" else "✖️")
				print_parts.append(f"{emoji} {op_name.capitalize()}: {acc:.2f}%")
				logged_accs[f"acc_{op_name}"] = acc

		print("🧪 [Test Generalización] " + " | ".join(print_parts))

		# Evolución
		current_phase = "guarderia" if epoch < nursery_end else ("recreo" if epoch < transition_end else "autonomia")
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

		# Registrar telemetría
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
			acc_homeostasis=acc_joint_test,
		)

		# Gancho de ciclo de vida: on_epoch_end
		if hooks and hasattr(hooks, "on_epoch_end"):
			hooks.on_epoch_end(epoch, avg_loss, logged_accs, population, config)

		# Detención temprana declarativa basada en config
		early_stopping = config.get("early_stopping", None)
		if early_stopping and epoch >= transition_end:
			target_op = early_stopping.get("operator")
			min_acc = early_stopping.get("min_acc", 75.0)
			target_acc = logged_accs.get(f"acc_{target_op}", 0.0)
			if target_acc >= min_acc:
				print(f"\n🏆 ¡HIT DE GENERALIZACIÓN ALCANZADO! {target_op} en test set: {target_acc:.2f}% (mínimo requerido: {min_acc:.2f}%).")
				logger.log_event("early_stop", {"operator": target_op, "acc": target_acc})
				break

	# Guardar checkpoint final
	best_idx = np.argsort(fitness)[-1]
	checkpoint_save_path = os.path.join(exp_dir, "best_agent.pt")
	torch.save(population[best_idx].state_dict(), checkpoint_save_path)
	print(f"💾 Checkpoint del mejor agente guardado en {checkpoint_save_path}")

	logger.close()


if __name__ == "__main__":
	run_training()
