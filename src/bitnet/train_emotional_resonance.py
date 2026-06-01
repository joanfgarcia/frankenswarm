"""
EXP_033: Entrenamiento con Resonancia Emocional.

La emoción se inyecta DENTRO del bucle latente como vector que modula
la trayectoria del pensamiento. El mismo input con emociones distintas
produce trayectorias y respuestas distintas.

Condiciones experimentales:
    A: baseline (forward estándar, sin emoción)
    B: resonancia pura (clock_2, sin emoción)
    C: resonancia + fear_amplifier en loss (sin emoción interna)
    D: resonancia + emoción inyectada (additive/gated/first_only)
    E: resonancia + emoción + fear_amplifier

Origen: Joan Garcia — "la emoción es la que toma las decisiones"
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from src.bitnet.modeling_bitnet import BitNet4LayerModel
from src.bitnet.operators_logic import (
	CONCEPT_NAMES_EXT,
	EMOTION_NAMES,
	N_EMOTIONS,
	build_emotional_chains,
	get_bifurcation_pairs,
)
from src.bitnet.telemetry import ExperimentLogger
from src.bitnet.translator import SovereignTranslator


def svd_crossover(parent_a, parent_b, child, alpha=0.5, sigma=0.01):
	"""Crossover SVD heredado de train_loop.py."""
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


def build_chain_dataset_emotional(translator: SovereignTranslator, max_depth: int = 2):
	"""
	Construye dataset de cadenas emocionales para EXP_033.
	Cada cadena tiene un emotion_id que determina el destino correcto.
	"""
	emo_chains = build_emotional_chains(max_depth=max_depth)
	concept_tids = {}
	for name in CONCEPT_NAMES_EXT:
		tids = translator.encode(name)
		concept_tids[name] = tids[0] if tids else 0

	chains = []
	for ec in emo_chains:
		start_name = CONCEPT_NAMES_EXT[ec["start"]]
		end_name = CONCEPT_NAMES_EXT[ec["end"]]
		chains.append({
			"start": ec["start"],
			"start_tid": concept_tids[start_name],
			"end": ec["end"],
			"end_tid": concept_tids[end_name],
			"emotion_id": ec["emotion_id"],
			"fear": ec["fear"],
			"depth": ec["depth"],
			"path_names": ec["path_names"],
			"intermediate_tids": [concept_tids[CONCEPT_NAMES_EXT[ec["intermediate"]]]] if "intermediate" in ec else [],
		})

	return chains, concept_tids


def run_emotional_resonance_training():
	parser = argparse.ArgumentParser(description="EXP_033: Resonancia Emocional")
	parser.add_argument("--config", type=str, required=True, help="Path to experiment config JSON")
	args, _ = parser.parse_known_args()

	base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

	config_path = args.config if os.path.isabs(args.config) else os.path.join(base_dir, args.config)
	with open(config_path, encoding="utf-8") as f:
		config = json.load(f)

	experiment_id = config["experiment_id"]
	condition = config.get("condition", "D")  # A, B, C, D, E
	use_resonance = condition in ("B", "C", "D", "E")
	use_emotion = condition in ("D", "E")
	use_fear_loss = condition in ("C", "E")
	emotion_mode = config.get("emotion", {}).get("mode", "additive")

	print(f"=== 🧠 Resonancia Emocional — {experiment_id} ===")
	print(f"    Condición={condition} | resonancia={use_resonance} | "
		  f"emoción={use_emotion} ({emotion_mode}) | fear_loss={use_fear_loss}")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"[Device]: {device}")

	seed = config.get("seed", 42)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)

	# Directorio del experimento
	exp_dir = os.path.join(base_dir, "storage", "experiments", experiment_id)
	os.makedirs(exp_dir, exist_ok=True)
	with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf-8") as f:
		json.dump(config, f, ensure_ascii=False, indent=4)

	# 1. Dataset emocional
	translator = SovereignTranslator()
	vocab_embeddings = translator.get_concept_embeddings()

	chains, concept_tids = build_chain_dataset_emotional(translator, max_depth=config.get("max_chain_depth", 2))
	print(f"🔗 Cadenas emocionales: {len(chains)}")
	for emo_name in EMOTION_NAMES:
		emo_id = EMOTION_NAMES.index(emo_name)
		count = sum(1 for c in chains if c["emotion_id"] == emo_id)
		print(f"   {emo_name}: {count}")

	# Bifurcaciones para evaluación
	bifurcation_pairs = get_bifurcation_pairs()
	print(f"🔀 Pares de bifurcación: {len(bifurcation_pairs)}")

	op_implica_tid = translator.encode("implica")[0]

	# 2. Modelo
	model_cfg = config.get("model", {})
	pop_size = config.get("pop_size", 4)
	resonance_cfg = config.get("resonance", {})
	max_res_steps = resonance_cfg.get("max_resonance_steps", 5)
	emotion_cfg = config.get("emotion", {})

	population = [
		BitNet4LayerModel(
			vocab_embeddings=vocab_embeddings,
			hidden_dim=model_cfg.get("hidden_dim", 256),
			num_layers=model_cfg.get("num_layers", 3),
			use_pos_embedding=model_cfg.get("use_pos_embedding", True),
			max_resonance_steps=max_res_steps if use_resonance else 0,
			# EXP_033: Emotion embeddings
			n_emotions=N_EMOTIONS if use_emotion else 0,
			emotion_dim=emotion_cfg.get("dim", 64) if use_emotion else 0,
			emotion_mode=emotion_mode if use_emotion else "additive",
		).to(device)
		for _ in range(pop_size)
	]

	train_cfg = config.get("training", {})
	lr = train_cfg.get("lr", 3e-4)
	wd = train_cfg.get("weight_decay", 0.05)
	optimizers = [
		torch.optim.AdamW(filter(lambda p: p.requires_grad, m.parameters()), lr=lr, weight_decay=wd)
		for m in population
	]

	# Logit mask (vocabulario extendido para EXP_033)
	logit_mask = None
	if config.get("use_logit_mask", True):
		micro_vocab = config.get("micro_vocab_words", CONCEPT_NAMES_EXT + ["implica"])
		logit_mask = torch.zeros(vocab_embeddings.shape[0], dtype=torch.bool, device=device)
		for word in micro_vocab:
			tids = translator.encode(word)
			if tids:
				logit_mask[tids[0]] = True

	# 3. Training loop
	curriculum = config.get("curriculum", {})
	epochs = train_cfg.get("epochs", 200)
	steps_per_epoch = train_cfg.get("steps_per_epoch", 200)
	batch_size = train_cfg.get("batch_size", 32)
	tau_start = curriculum.get("tau_start", 1.0)
	tau_min = curriculum.get("tau_min", 0.3)
	nursery_end = curriculum.get("nursery_end", 30)
	transition_end = curriculum.get("transition_end", 80)
	tf_min = curriculum.get("tf_min", 0.1)
	grad_clip = train_cfg.get("grad_clip", 1.0)
	total_steps = epochs * steps_per_epoch

	n_steps = int(resonance_cfg.get("n_steps", 2))
	pos_mode = resonance_cfg.get("pos_mode", "clock")
	loss_mode = resonance_cfg.get("loss_mode", "weighted")
	intermediate_loss_weight = resonance_cfg.get("intermediate_loss_weight", 0.2)
	fear_amplifier = config.get("fear_amplifier", 3.0) if use_fear_loss else 0.0

	fitness = np.zeros(pop_size)
	logger = ExperimentLogger(
		experiment_id,
		{**config, "device": str(device)},
		log_path=os.path.join(exp_dir, "telemetry.jsonl"),
	)

	# Early stopping (lección EXP_032: sobreentrenamiento después de ep ~100-162)
	early_stopping_cfg = config.get("early_stopping", {})
	patience = early_stopping_cfg.get("patience", 30)
	best_acc_joint = 0.0
	epochs_without_improvement = 0

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
		epoch_correct, epoch_total = 0, 0
		epoch_bifurcation_correct, epoch_bifurcation_total = 0, 0
		interactions = np.zeros((pop_size, pop_size))
		successes = np.zeros((pop_size, pop_size))

		for model in population:
			model.train()

		for step in range(steps_per_epoch):
			tau = max(tau_min, tau_start * (1.0 - current_step / total_steps))

			batch_chains = [chains[np.random.randint(len(chains))] for _ in range(batch_size)]

			# Speaker/Listener
			idx_s = np.random.randint(0, pop_size)
			idx_l = np.random.randint(0, pop_size)
			while idx_l == idx_s and pop_size > 1:
				idx_l = np.random.randint(0, pop_size)

			speaker, listener = population[idx_s], population[idx_l]
			opt_s, opt_l = optimizers[idx_s], optimizers[idx_l]
			opt_s.zero_grad()
			opt_l.zero_grad()

			# Preparar input
			start_tids = torch.tensor([c["start_tid"] for c in batch_chains], dtype=torch.long, device=device)
			end_tids = torch.tensor([c["end_tid"] for c in batch_chains], dtype=torch.long, device=device)
			fears = torch.tensor(
				[1.0 + c["fear"] * fear_amplifier for c in batch_chains],
				dtype=torch.float32, device=device,
			)

			# Input: [start, implica, 0, 0]
			current_input = torch.zeros((batch_size, 4), dtype=torch.long, device=device)
			current_input[:, 0] = start_tids
			current_input[:, 1] = op_implica_tid

			# Emotion IDs (solo para condiciones D y E)
			emotion_ids = None
			if use_emotion:
				emotion_ids = torch.tensor(
					[c["emotion_id"] for c in batch_chains],
					dtype=torch.long, device=device,
				)

			# ═══════════════════════════════════════════
			# FORWARD
			# ═══════════════════════════════════════════
			if not use_resonance:
				# Condición A: forward estándar
				speaker_logits = speaker(current_input, logit_mask=logit_mask)
				total_loss = torch.tensor(0.0, device=device)
				meta = None

			elif loss_mode == "final":
				speaker_logits, meta = speaker.forward_resonance(
					current_input, n_steps=n_steps, pos_mode=pos_mode,
					logit_mask=logit_mask, emotion_ids=emotion_ids,
				)
				total_loss = torch.tensor(0.0, device=device)

			elif loss_mode in ("every", "weighted"):
				intermediate_targets_dict = dict.fromkeys(range(n_steps))
				speaker_logits, intermediate_logits = speaker.forward_resonance_training(
					current_input, n_steps=n_steps, pos_mode=pos_mode,
					intermediate_targets=intermediate_targets_dict,
					logit_mask=logit_mask, emotion_ids=emotion_ids,
				)
				total_loss = torch.tensor(0.0, device=device)

				weight = 1.0 if loss_mode == "every" else intermediate_loss_weight
				for step_idx, logits_mid in intermediate_logits:
					loss_mid = F.cross_entropy(logits_mid[:, 2, :], end_tids, reduction="none")
					total_loss = total_loss + (loss_mid * fears * weight).mean()

				meta = None

			# Listener
			speaker_msg = F.gumbel_softmax(speaker_logits, tau=tau, hard=False, dim=-1)

			if torch.rand(1).item() < tf_ratio:
				teacher = current_input.clone()
				teacher[:, 2] = end_tids
				msg_to_listener = F.one_hot(teacher, num_classes=vocab_embeddings.shape[0]).float()
			else:
				msg_to_listener = speaker_msg

			listener_logits = listener(msg_to_listener, logit_mask=logit_mask)

			# Loss final
			loss_final = F.cross_entropy(listener_logits[:, 2, :], end_tids, reduction="none")
			total_loss = total_loss + (loss_final * fears).mean()

			# Loss speaker
			loss_speaker = F.cross_entropy(speaker_logits[:, 2, :], end_tids, reduction="none")
			total_loss = total_loss + (loss_speaker * fears * 0.3).mean()

			# Backprop
			total_loss.backward()
			torch.nn.utils.clip_grad_norm_(speaker.parameters(), max_norm=grad_clip)
			torch.nn.utils.clip_grad_norm_(listener.parameters(), max_norm=grad_clip)
			opt_s.step()
			opt_l.step()

			epoch_losses.append(total_loss.item())

			# Métricas
			with torch.no_grad():
				pred_final = torch.argmax(listener_logits[:, 2, :], dim=-1)
				pred_speaker = torch.argmax(speaker_logits[:, 2, :], dim=-1)
				final_ok = (pred_final == end_tids).sum().item()

			epoch_correct += final_ok
			epoch_total += batch_size
			interactions[idx_s, idx_l] += batch_size
			successes[idx_s, idx_l] += final_ok

			# Print progress
			if step % 50 == 0:
				c = batch_chains[0]
				path = " → ".join(c["path_names"])
				emo_name = EMOTION_NAMES[c["emotion_id"]]
				pred_name = translator.decode([pred_final[0].item()])
				target_name = CONCEPT_NAMES_EXT[c["end"]]
				print(
					f"  step {step:3d} | τ={tau:.3f} TF={tf_ratio:.0%} | "
					f"loss={total_loss.item():.3f} | 😶 {emo_name} | 🔗 {path} | "
					f"pred={pred_name}({'✓' if pred_name == target_name else '✗'})"
				)

			current_step += 1

		# Fitness
		for i in range(pop_size):
			sent_total = interactions[i, :].sum() + interactions[:, i].sum()
			sent_correct = successes[i, :].sum() + successes[:, i].sum()
			fitness[i] = sent_correct / (sent_total + 1e-10)

		avg_loss = np.mean(epoch_losses)
		acc_joint = (epoch_correct / epoch_total) * 100

		# ═══════════════════════════════════════════
		# EVALUACIÓN DE BIFURCACIONES
		# ═══════════════════════════════════════════
		best_idx = int(np.argmax(fitness))
		best_model = population[best_idx]
		best_model.eval()

		bif_correct, bif_total = 0, 0
		with torch.no_grad():
			for pair in bifurcation_pairs:
				src_tid = concept_tids[pair["source"]]
				x_test = torch.zeros((2, 4), dtype=torch.long, device=device)
				x_test[:, 0] = src_tid
				x_test[:, 1] = op_implica_tid

				emo_a_id = EMOTION_NAMES.index(pair["emo_a"])
				emo_b_id = EMOTION_NAMES.index(pair["emo_b"])
				expected_a = concept_tids[pair["dest_a"]]
				expected_b = concept_tids[pair["dest_b"]]

				if use_resonance and use_emotion:
					test_emos = torch.tensor([emo_a_id, emo_b_id], dtype=torch.long, device=device)
					logits_test, _ = best_model.forward_resonance(
						x_test, n_steps=n_steps, pos_mode=pos_mode,
						logit_mask=logit_mask, emotion_ids=test_emos,
					)
				elif use_resonance:
					logits_test, _ = best_model.forward_resonance(
						x_test, n_steps=n_steps, pos_mode=pos_mode,
						logit_mask=logit_mask,
					)
				else:
					logits_test = best_model(x_test, logit_mask=logit_mask)

				preds = torch.argmax(logits_test[:, 2, :], dim=-1)
				if preds[0].item() == expected_a:
					bif_correct += 1
				if preds[1].item() == expected_b:
					bif_correct += 1
				bif_total += 2

		acc_bifurcation = (bif_correct / bif_total) * 100 if bif_total > 0 else 0.0

		print(f"Loss: {avg_loss:.4f} | Joint: {acc_joint:.2f}% | "
			  f"Bifurcación: {acc_bifurcation:.1f}% ({bif_correct}/{bif_total})")

		logger.log_epoch(
			epoch=epoch + 1,
			loss_avg=avg_loss,
			acc_concept=acc_joint,
			acc_emotion=acc_bifurcation,
			acc_joint=acc_joint,
			fitness=fitness.tolist(),
			worst_agent=int(np.argmin(fitness)),
			parent_a=-1,
			parent_b=-1,
			acc_homeostasis=acc_joint,
		)

		# Early stopping + guardar mejor
		if acc_joint > best_acc_joint:
			best_acc_joint = acc_joint
			epochs_without_improvement = 0
			save_path = os.path.join(exp_dir, "best_agent.pt")
			torch.save(best_model.state_dict(), save_path)
		else:
			epochs_without_improvement += 1

		# Reset contador al entrar en autonomía (el modelo necesita
		# adaptarse al nuevo régimen sin teacher forcing)
		if epoch == transition_end:
			epochs_without_improvement = 0
			best_acc_autonomy = acc_joint
			print(f"🦅 Entrando en autonomía — reset early stopping (best={best_acc_joint:.2f}%)")

		# Solo aplicar early stopping después de al menos 10 epochs en autonomía
		if epoch >= transition_end + 10 and epochs_without_improvement >= patience:
			print(f"⏹️ Early stopping: {patience} epochs sin mejora (best={best_acc_joint:.2f}%)")
			break

		# Evolución SVD
		svd_interval = config.get("svd_interval", 3)
		current_phase = "guarderia" if epoch < nursery_end else ("recreo" if epoch < transition_end else "autonomia")
		svd_phases = config.get("svd_phases", ["recreo", "autonomia"])

		if current_phase in svd_phases and epoch % svd_interval == 0:
			worst_idx = int(np.argmin(fitness))
			best = np.argsort(fitness)[-2:]
			svd_crossover(population[best[1]], population[best[0]], population[worst_idx])
			optimizers[worst_idx] = torch.optim.AdamW(
				filter(lambda p: p.requires_grad, population[worst_idx].parameters()),
				lr=lr, weight_decay=wd,
			)

	# Guardar resultado final
	best_idx = int(np.argmax(fitness))
	save_path = os.path.join(exp_dir, "best_agent.pt")
	torch.save(population[best_idx].state_dict(), save_path)
	print(f"💾 Mejor agente guardado en {save_path} (acc_joint={best_acc_joint:.2f}%)")

	logger.close()
	return best_acc_joint


if __name__ == "__main__":
	run_emotional_resonance_training()
