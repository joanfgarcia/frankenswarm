import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.bitnet.dataset_breeder import ReferentialDatasetBreeder
from src.bitnet.modeling_bitnet import BitNet4LayerModel
from src.bitnet.telemetry import ExperimentLogger
from src.bitnet.translator import SovereignTranslator


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

			# Solo aplicamos SVD a tensores bidimensionales (matrices de pesos de proyección)
			if p_a.ndim == 2:
				# Interpolación lineal de las matrices de peso
				w_avg = alpha * p_a + (1.0 - alpha) * p_b
				try:
					# Descomposición SVD
					u, s, vh = torch.linalg.svd(w_avg, full_matrices=False)
					# Perturbación de valores singulares
					noise = torch.randn_like(s) * sigma
					s_perturbed = s + noise
					s_perturbed.clamp_(min=0.0)
					# Reconstrucción de la matriz cruzada
					w_child = u @ torch.diag(s_perturbed) @ vh
					param.copy_(w_child)
				except Exception:
					# Fallback en caso de error numérico en SVD
					param.copy_(w_avg)
			else:
				# Para vectores (bias, escalas), hacemos interpolación simple con ruido
				noise = torch.randn_like(p_a) * sigma
				param.copy_(alpha * p_a + (1.0 - alpha) * p_b + noise)


def load_config() -> dict:
	parser = argparse.ArgumentParser(description="Arena de Entrenamiento Frankenswarm")
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


def run_arena():
	config = load_config()
	experiment_id = config["experiment_id"]
	print(f"=== 🌋 Iniciando Arena de Comunicación Emergente PopuLoRA — {experiment_id} ===")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"[Device]: {device}")

	# Fijar semilla aleatoria si se especifica en la configuración
	seed = config.get("seed")
	if seed is not None:
		print(f"🌱 [SEED] Fijando semilla aleatoria: {seed}")
		np.random.seed(seed)
		torch.manual_seed(seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed_all(seed)

	# Crear directorios específicos para el experimento
	base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	exp_dir = os.path.join(base_dir, "storage", "experiments", experiment_id)
	os.makedirs(exp_dir, exist_ok=True)

	# Guardar copia de los parámetros reales utilizados
	config_copy_path = os.path.join(exp_dir, "config.json")
	with open(config_copy_path, "w", encoding="utf-8") as f:
		json.dump(config, f, ensure_ascii=False, indent=4)
	print(f"📄 Parámetros del experimento guardados en {config_copy_path}")

	# 1. Cargar el Traductor de Capa 1
	translator = SovereignTranslator()
	vocab_embeddings = translator.get_concept_embeddings()

	# 2. Inicializar Breeder
	breeder = ReferentialDatasetBreeder(translator)

	# 3. Inicializar Población de Alumnos (BitNet de 4 capas)
	pop_size = config["pop_size"]
	hidden_dim = config["hidden_dim"]
	num_layers = config["num_layers"]
	population = [
		BitNet4LayerModel(vocab_embeddings=vocab_embeddings, hidden_dim=hidden_dim, num_layers=num_layers).to(device) for _ in range(pop_size)
	]

	# Cargar checkpoint anterior para heredar la proto-sintaxis si se configura
	resume_checkpoint = config.get("resume_checkpoint")
	if resume_checkpoint:
		if not os.path.isabs(resume_checkpoint):
			resume_checkpoint = os.path.join(base_dir, resume_checkpoint)
		if os.path.exists(resume_checkpoint):
			print(f"💾 [HOTSTART] Cargando pesos desde checkpoint {resume_checkpoint}...")
			try:
				state_dict = torch.load(resume_checkpoint, map_location=device)
				for idx, model in enumerate(population):
					model.load_state_dict(state_dict)
				print("✅ Inicialización de población completada con éxito.")
			except Exception as e:
				print(f"⚠️ Error cargando checkpoint: {e}. Inicializando pesos aleatorios.")
		else:
			print(f"ℹ️ Checkpoint {resume_checkpoint} no encontrado. Inicializando con pesos aleatorios.")

	freeze_core = config.get("freeze_core", False)
	if freeze_core:
		print("🔒 [SOVEREIGN LAYER] Congelando Specialist Core (Capa 3). Solo se entrenan Traductores (Capas 2 y 4).")
		for model in population:
			for param in model.core_layers.parameters():
				param.requires_grad = False
			if getattr(model, "pos_embedding", None) is not None:
				model.pos_embedding.requires_grad = False

	# Optimizadores individuales
	lr = config["lr"]
	optimizers = [torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr) for model in population]

	# 4. Definir Micro-Vocabulario mediante logit mask si está configurado
	use_logit_mask = config["use_logit_mask"]
	logit_mask = None
	if use_logit_mask:
		micro_vocab_words = config["micro_vocab_words"]
		if not micro_vocab_words:
			micro_vocab_words = [
				"gato",
				"perro",
				"casa",
				"árbol",
				"agua",
				"fuego",
				"tierra",
				"aire",
				"sol",
				"luna",
				"peligro",
				"seguridad",
				"búnker",
				"agente",
				"código",
				"miedo",
				"alegría",
				"ira",
				"tristeza",
				"dolor",
				"hambre",
				"neutral",
				"urgencia",
			]
		micro_vocab_words = list(set(micro_vocab_words))
		logit_mask = torch.zeros(8192, dtype=torch.bool, device=device)
		for word in micro_vocab_words:
			tids = translator.encode(word)
			if tids:
				logit_mask[tids[0]] = True
		print(f"🔒 Logit Masking activado: {len(micro_vocab_words)} palabras permitidas.")

	# Telemetría de TrueSkill/Fitness
	fitness = np.zeros(pop_size)
	epochs = config["epochs"]
	steps_per_epoch = config["steps_per_epoch"]
	batch_size = config["batch_size"]

	# Parámetros de Gumbel-Softmax
	tau_start = config["tau_start"]
	tau_min = config["tau_min"]
	total_steps = epochs * steps_per_epoch

	# Teacher Forcing Schedule y Anclaje Semántico
	nursery_end = config["nursery_end"]
	transition_end = config["transition_end"]
	tf_min = config["tf_min"]

	# Configurar Logger para escribir en el subdirectorio del experimento
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
		epoch_concept_correct = 0
		epoch_emotion_correct = 0
		epoch_homeostasis_correct = 0
		epoch_total = 0

		# Matriz de interacción para evaluar fitness
		interactions = np.zeros((pop_size, pop_size))
		successes = np.zeros((pop_size, pop_size))

		for step in range(steps_per_epoch):
			# Decaimiento (annealing) de temperatura Gumbel
			tau = max(tau_min, tau_start * (1.0 - current_step / total_steps))

			# Generar lote de conceptos objetivos, estados emocionales y homeostasis
			concept_targets, concept_token_ids, emotion_targets, emotion_token_ids, homeostasis_targets, homeostasis_token_ids = (
				breeder.generate_batch(batch_size)
			)
			concept_token_ids_tensor = torch.from_numpy(concept_token_ids).long().to(device)
			emotion_token_ids_tensor = torch.from_numpy(emotion_token_ids).long().to(device)
			homeostasis_token_ids_tensor = torch.from_numpy(homeostasis_token_ids).long().to(device)

			# Seleccionar dos agentes distintos de la población
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

			# 1. El Hablante recibe el target conceptual, estado emocional y homeostático, y emite un mensaje de longitud 4
			# Construir entrada: [concept_token_id, emotion_token_id, homeostasis_token_id, 0]
			speaker_input = torch.zeros((batch_size, 4), dtype=torch.long, device=device)
			speaker_input[:, 0] = concept_token_ids_tensor
			speaker_input[:, 1] = emotion_token_ids_tensor
			speaker_input[:, 2] = homeostasis_token_ids_tensor

			# ── Scheduled Teacher Forcing ──
			use_teacher = torch.rand(batch_size, device=device) < tf_ratio

			if use_teacher.all():
				message_input = speaker_input
			elif use_teacher.any():
				speaker_message = speaker.generate_message(speaker_input, tau=tau, hard=True, logit_mask=logit_mask)
				teacher_onehot = F.one_hot(speaker_input, num_classes=8192).float()
				mask = use_teacher.view(-1, 1, 1).float()
				message_input = mask * teacher_onehot + (1 - mask) * speaker_message
			else:
				message_input = speaker.generate_message(speaker_input, tau=tau, hard=True, logit_mask=logit_mask)

			# 2. El Oyente recibe el mensaje y predice:
			# - El concepto en el paso 1 (logits[:, 1, :])
			# - La emoción en el paso 2 (logits[:, 2, :])
			# - La homeostasis en el paso 3 (logits[:, 3, :])
			logits = listener(message_input, logit_mask=logit_mask)  # (batch_size, 4, 8192)
			pred_concept_logits = logits[:, 1, :]  # (batch_size, 8192)
			pred_emotion_logits = logits[:, 2, :]  # (batch_size, 8192)
			pred_homeostasis_logits = logits[:, 3, :]  # (batch_size, 8192)

			# 3. Calcular Pérdida conjunta (Entropía cruzada conceptual + afectiva + homeostática)
			loss_concept = F.cross_entropy(pred_concept_logits, concept_token_ids_tensor)
			loss_emotion = F.cross_entropy(pred_emotion_logits, emotion_token_ids_tensor)
			loss_homeostasis = F.cross_entropy(pred_homeostasis_logits, homeostasis_token_ids_tensor)
			loss = loss_concept + 1.0 * loss_emotion + 0.5 * loss_homeostasis
			loss.backward()

			# Actualizar parámetros (Capa 3 y proyecciones de Capa 2/4)
			opt_speaker.step()
			opt_listener.step()

			# Registrar telemetría
			epoch_losses.append(loss.item())

			preds_concept = torch.argmax(pred_concept_logits, dim=-1)
			preds_emotion = torch.argmax(pred_emotion_logits, dim=-1)
			preds_homeostasis = torch.argmax(pred_homeostasis_logits, dim=-1)

			c_ok = (preds_concept == concept_token_ids_tensor).sum().item()
			e_ok = (preds_emotion == emotion_token_ids_tensor).sum().item()
			h_ok = (preds_homeostasis == homeostasis_token_ids_tensor).sum().item()

			# Entendimiento mutuo exitoso si se descodifican correctamente los TRES componentes
			correct_joint = (
				(
					(preds_concept == concept_token_ids_tensor)
					& (preds_emotion == emotion_token_ids_tensor)
					& (preds_homeostasis == homeostasis_token_ids_tensor)
				)
				.sum()
				.item()
			)

			epoch_correct += correct_joint
			epoch_concept_correct += c_ok
			epoch_emotion_correct += e_ok
			epoch_homeostasis_correct += h_ok
			epoch_total += batch_size

			interactions[idx_speaker, idx_listener] += batch_size
			successes[idx_speaker, idx_listener] += correct_joint

			# Extraer tokens del mensaje (para el primer elemento del lote)
			if message_input.ndim == 2:
				token_ids = message_input[0].tolist()
			else:
				token_ids = torch.argmax(message_input[0], dim=-1).tolist()
			message_tokens = [translator.decode([tid]) for tid in token_ids]

			sample_c = breeder.get_concept_name(concept_targets[0])
			sample_e = breeder.get_emotion_name(emotion_targets[0])
			sample_h = breeder.get_homeostasis_name(homeostasis_targets[0])
			pred_c_name = translator.decode([preds_concept[0].item()])
			pred_e_name = translator.decode([preds_emotion[0].item()])
			pred_h_name = translator.decode([preds_homeostasis[0].item()])

			logger.log_step(
				epoch=epoch + 1,
				step=step,
				loss=loss.item(),
				loss_concept=loss_concept.item(),
				loss_emotion=loss_emotion.item(),
				tau=tau,
				speaker_id=idx_speaker,
				listener_id=idx_listener,
				target_concept=sample_c,
				target_emotion=sample_e,
				pred_concept=pred_c_name,
				pred_emotion=pred_e_name,
				concept_correct=c_ok,
				emotion_correct=e_ok,
				joint_correct=correct_joint,
				batch_size=batch_size,
				message_tokens=message_tokens,
				tf_ratio=tf_ratio,
				loss_homeostasis=loss_homeostasis.item(),
				target_homeostasis=sample_h,
				pred_homeostasis=pred_h_name,
				homeostasis_correct=h_ok,
			)

			if step % 40 == 0:
				print(
					f"  step {step:3d} | τ={tau:.3f} TF={tf_ratio:.0%} | loss={loss.item():.3f} | "
					f"target=({sample_c},{sample_e},{sample_h}) → pred=({pred_c_name},{pred_e_name},{pred_h_name})"
				)

			current_step += 1

		# Calcular fitness acumulado de cada agente (accuracy promedio)
		for i in range(pop_size):
			sent_total = interactions[i, :].sum() + interactions[:, i].sum()
			sent_correct = successes[i, :].sum() + successes[:, i].sum()
			fitness[i] = sent_correct / (sent_total + 1e-10)

		avg_loss = np.mean(epoch_losses)
		acc_c = (epoch_concept_correct / epoch_total) * 100
		acc_e = (epoch_emotion_correct / epoch_total) * 100
		acc_h = (epoch_homeostasis_correct / epoch_total) * 100
		acc_j = (epoch_correct / epoch_total) * 100

		print(
			f"Pérdida promedio: {avg_loss:.4f} | Concepto: {acc_c:.2f}% | Emoción: {acc_e:.2f}% | Homeostasis: {acc_h:.2f}% | Entendimiento Mutuo (Accuracy): {acc_j:.2f}%"
		)
		print(f"Fitness de la Población: {[f'Agent_{i}: {f * 100:.2f}%' for i, f in enumerate(fitness)]}")

		# Determinar la fase actual del schedule para co-evolución
		if epoch < nursery_end:
			current_phase = "guarderia"
		elif epoch < transition_end:
			current_phase = "recreo"
		else:
			current_phase = "autonomia"

		svd_interval = config.get("svd_interval", 1)
		svd_phases = config.get("svd_phases", ["recreo", "autonomia"])

		# Co-evolución dinámica basada en config
		if current_phase in svd_phases and epoch % svd_interval == 0:
			worst_idx = int(np.argmin(fitness))
			best_indices = np.argsort(fitness)[-2:]  # Los dos mejores
			parent_a_idx = int(best_indices[1])
			parent_b_idx = int(best_indices[0])
			print(f"[Evolución] Reemplazando Agent_{worst_idx} (peor fitness) con hijo SVD de Agent_{parent_a_idx} y Agent_{parent_b_idx}")
			svd_crossover(parent_a=population[parent_a_idx], parent_b=population[parent_b_idx], child=population[worst_idx], alpha=0.5, sigma=0.01)
			# Reiniciar el optimizador del peor agente tras la mutación de pesos
			optimizers[worst_idx] = torch.optim.AdamW(filter(lambda p: p.requires_grad, population[worst_idx].parameters()), lr=lr)
			logger.log_event(
				"evolution",
				{
					"worst": worst_idx,
					"parent_a": parent_a_idx,
					"parent_b": parent_b_idx,
				},
			)
		else:
			worst_idx = -1
			parent_a_idx = -1
			parent_b_idx = -1

		logger.log_epoch(
			epoch=epoch + 1,
			loss_avg=avg_loss,
			acc_concept=acc_c,
			acc_emotion=acc_e,
			acc_joint=acc_j,
			fitness=fitness.tolist(),
			worst_agent=worst_idx,
			parent_a=parent_a_idx,
			parent_b=parent_b_idx,
			acc_homeostasis=acc_h,
		)

		# Promoción de grado académica
		if epoch >= transition_end and acc_j >= 80.0:
			print("\n🏆 ¡HIT ALCANZADO! La población ha superado el 80% de entendimiento mutuo.")
			print("PROMOTED TO GRADE 1: Lengua adquirida de forma emergente. Desbloqueando Aritmética.")
			logger.log_event("promotion", {"grade": 1, "acc_joint": acc_j})
			break

	# Guardar el mejor agente entrenado en el directorio del experimento
	best_idx = np.argsort(fitness)[-1]
	checkpoint_save_path = os.path.join(exp_dir, "best_agent.pt")
	torch.save(population[best_idx].state_dict(), checkpoint_save_path)
	print(f"💾 Checkpoint del mejor agente guardado en {checkpoint_save_path}")

	logger.close()


if __name__ == "__main__":
	run_arena()
