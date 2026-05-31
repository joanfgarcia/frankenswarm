"""
EXP_034: Entrenamiento con Glifos Ternarios — Vocabulario Vivo.

El modelo piensa en un metalenguaje universal. Cada token es un glifo
de 65 trits (primos semánticos de Wierzbicka). Los embeddings se COMPONEN
por radicales, no se buscan en tabla opaca.

Cambios vs EXP_033:
  - No usa SovereignTranslator ni fastembed (los glifos son el embedding)
  - No hay logit_mask (vocabulario ya es de 26 palabras)
  - Input: [start, start, 0, 0] (sin token "implica")
  - El modelo aprende relaciones causales emocionales en espacio universal

Origen: Joan Garcia — "estamos trabajando en metalenguaje"
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from src.bitnet.modeling_bitnet import BitNet4LayerModel
from src.bitnet.glyph_vocabulary import (
	WORD_NAMES, WORD_INDEX, N_WORDS, N_EMOTIONS, EMOTION_NAMES,
	EMOTION_INDEX, GLYPH_TABLE, CURRICULUM_PHASES,
	build_emotional_chains, get_bifurcation_pairs,
)
from src.bitnet.telemetry import ExperimentLogger


def svd_crossover(parent_a, parent_b, child, alpha=0.5, sigma=0.01):
	"""Crossover SVD para evolución poblacional."""
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


def run_glyph_training():
	parser = argparse.ArgumentParser(description="EXP_034: Glifos Ternarios")
	parser.add_argument("--config", type=str, required=True, help="Path to experiment config JSON")
	args, _ = parser.parse_known_args()

	base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	config_path = args.config if os.path.isabs(args.config) else os.path.join(base_dir, args.config)
	with open(config_path, encoding="utf-8") as f:
		config = json.load(f)

	experiment_id = config["experiment_id"]
	condition = config.get("condition", "D")
	use_resonance = condition in ("B", "C", "D", "E")
	use_emotion = condition in ("D", "E")
	use_fear_loss = condition in ("C", "E")
	emotion_mode = config.get("emotion", {}).get("mode", "first_only")

	print(f"═══ 🔤 Glifos Ternarios — {experiment_id} ═══")
	print(f"    Condición={condition} | resonancia={use_resonance} | "
	      f"emoción={use_emotion} ({emotion_mode}) | fear_loss={use_fear_loss}")
	print(f"    Vocabulario: {N_WORDS} palabras × 65 primos ternarios")

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

	# ═══════════════════════════════════════════
	# 1. Dataset
	# ═══════════════════════════════════════════
	chains = build_emotional_chains(max_depth=config.get("max_chain_depth", 2))
	print(f"🔗 Cadenas de entrenamiento: {len(chains)}")
	for emo_name in EMOTION_NAMES:
		emo_id = EMOTION_INDEX[emo_name]
		count = sum(1 for c in chains if c["emotion_id"] == emo_id)
		print(f"   {emo_name}: {count}")

	bifurcation_pairs = get_bifurcation_pairs()
	print(f"🔀 Pares de bifurcación: {len(bifurcation_pairs)}")

	# ═══════════════════════════════════════════
	# 2. Modelo con Glifos
	# ═══════════════════════════════════════════
	model_cfg = config.get("model", {})
	pop_size = config.get("pop_size", 4)
	resonance_cfg = config.get("resonance", {})
	max_res_steps = resonance_cfg.get("max_resonance_steps", 5)
	emotion_cfg = config.get("emotion", {})

	population = [
		BitNet4LayerModel(
			use_glyphs=True,  # ← EXP_034: Glifos ternarios
			hidden_dim=model_cfg.get("hidden_dim", 256),
			num_layers=model_cfg.get("num_layers", 3),
			use_pos_embedding=model_cfg.get("use_pos_embedding", True),
			max_resonance_steps=max_res_steps if use_resonance else 0,
			n_emotions=N_EMOTIONS if use_emotion else 0,
			emotion_dim=emotion_cfg.get("dim", 64) if use_emotion else 0,
			emotion_mode=emotion_mode if use_emotion else "additive",
		).to(device)
		for _ in range(pop_size)
	]

	n_params = sum(p.numel() for p in population[0].parameters() if p.requires_grad)
	n_prime = sum(p.numel() for n, p in population[0].named_parameters() if "prime" in n)
	print(f"📊 Params: {n_params:,} total | {n_prime:,} en primos semánticos")

	train_cfg = config.get("training", {})
	lr = train_cfg.get("lr", 3e-4)
	wd = train_cfg.get("weight_decay", 0.05)
	optimizers = [
		torch.optim.AdamW(filter(lambda p: p.requires_grad, m.parameters()), lr=lr, weight_decay=wd)
		for m in population
	]

	# ═══════════════════════════════════════════
	# 3. Training loop
	# ═══════════════════════════════════════════
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
		{**config, "device": str(device), "n_primes": 65, "n_words": N_WORDS, "mode": "glyphs"},
		log_path=os.path.join(exp_dir, "telemetry.jsonl"),
	)

	# Early stopping
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

			# Input: [start, start, 0, 0] — sin token "implica"
			# Los glifos no necesitan separadores, la relación causal es implícita
			start_ids = torch.tensor([c["start"] for c in batch_chains], dtype=torch.long, device=device)
			end_ids = torch.tensor([c["end"] for c in batch_chains], dtype=torch.long, device=device)
			fears = torch.tensor(
				[1.0 + c["fear"] * fear_amplifier for c in batch_chains],
				dtype=torch.float32, device=device,
			)

			current_input = torch.zeros((batch_size, 4), dtype=torch.long, device=device)
			current_input[:, 0] = start_ids
			current_input[:, 1] = start_ids  # repetición del concepto fuente

			# Emotion IDs
			emotion_ids = None
			if use_emotion:
				emotion_ids = torch.tensor(
					[c["emotion_id"] for c in batch_chains],
					dtype=torch.long, device=device,
				)

			# ═══ FORWARD ═══
			if not use_resonance:
				# Condición A: forward estándar
				speaker_logits = speaker(current_input)
				total_loss = torch.tensor(0.0, device=device)

			elif loss_mode == "final":
				speaker_logits, meta = speaker.forward_resonance(
					current_input, n_steps=n_steps, pos_mode=pos_mode,
					emotion_ids=emotion_ids,
				)
				total_loss = torch.tensor(0.0, device=device)

			elif loss_mode in ("every", "weighted"):
				intermediate_targets_dict = {i: None for i in range(n_steps)}
				speaker_logits, intermediate_logits = speaker.forward_resonance_training(
					current_input, n_steps=n_steps, pos_mode=pos_mode,
					intermediate_targets=intermediate_targets_dict,
					emotion_ids=emotion_ids,
				)
				total_loss = torch.tensor(0.0, device=device)

				weight = 1.0 if loss_mode == "every" else intermediate_loss_weight
				for step_idx, logits_mid in intermediate_logits:
					loss_mid = F.cross_entropy(logits_mid[:, 2, :], end_ids, reduction="none")
					total_loss = total_loss + (loss_mid * fears * weight).mean()

			# Listener (con Gumbel-Softmax)
			speaker_msg = F.gumbel_softmax(speaker_logits, tau=tau, hard=False, dim=-1)

			if torch.rand(1).item() < tf_ratio:
				# Teacher forcing: one-hot del ground truth
				teacher = current_input.clone()
				teacher[:, 2] = end_ids
				msg_to_listener = F.one_hot(teacher, num_classes=N_WORDS).float()
			else:
				msg_to_listener = speaker_msg

			listener_logits = listener(msg_to_listener)

			# Loss final (listener)
			loss_final = F.cross_entropy(listener_logits[:, 2, :], end_ids, reduction="none")
			total_loss = total_loss + (loss_final * fears).mean()

			# Loss speaker (auxiliar)
			loss_speaker = F.cross_entropy(speaker_logits[:, 2, :], end_ids, reduction="none")
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
				final_ok = (pred_final == end_ids).sum().item()

			epoch_correct += final_ok
			epoch_total += batch_size
			interactions[idx_s, idx_l] += batch_size
			successes[idx_s, idx_l] += final_ok

			# Print progress
			if step % 50 == 0:
				c = batch_chains[0]
				src_name = c["source_name"]
				dst_name = c["dest_name"]
				emo_name = c["emotion_name"]
				pred_name = WORD_NAMES[pred_final[0].item()]
				ok = "✓" if pred_name == dst_name else "✗"
				print(
					f"  step {step:3d} | τ={tau:.3f} TF={tf_ratio:.0%} | "
					f"loss={total_loss.item():.3f} | 😶 {emo_name} | "
					f"🔗 {src_name}→{dst_name} | pred={pred_name}({ok})"
				)

			current_step += 1

		# Fitness
		for i in range(pop_size):
			sent_total = interactions[i, :].sum() + interactions[:, i].sum()
			sent_correct = successes[i, :].sum() + successes[:, i].sum()
			fitness[i] = sent_correct / (sent_total + 1e-10)

		avg_loss = np.mean(epoch_losses)
		acc_joint = (epoch_correct / epoch_total) * 100

		# ═══ EVALUACIÓN DE BIFURCACIONES ═══
		best_idx = int(np.argmax(fitness))
		best_model = population[best_idx]
		best_model.eval()

		bif_correct, bif_total = 0, 0
		with torch.no_grad():
			for pair in bifurcation_pairs:
				src_idx = WORD_INDEX[pair["source"]]
				x_test = torch.zeros((2, 4), dtype=torch.long, device=device)
				x_test[:, 0] = src_idx
				x_test[:, 1] = src_idx

				emo_a_id = EMOTION_INDEX[pair["emo_a"]]
				emo_b_id = EMOTION_INDEX[pair["emo_b"]]
				expected_a = WORD_INDEX[pair["dest_a"]]
				expected_b = WORD_INDEX[pair["dest_b"]]

				if use_resonance and use_emotion:
					test_emos = torch.tensor([emo_a_id, emo_b_id], dtype=torch.long, device=device)
					logits_test, _ = best_model.forward_resonance(
						x_test, n_steps=n_steps, pos_mode=pos_mode,
						emotion_ids=test_emos,
					)
				elif use_resonance:
					logits_test, _ = best_model.forward_resonance(
						x_test, n_steps=n_steps, pos_mode=pos_mode,
					)
				else:
					logits_test = best_model(x_test)

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

		# Early stopping
		if acc_joint > best_acc_joint:
			best_acc_joint = acc_joint
			epochs_without_improvement = 0
			save_path = os.path.join(exp_dir, "best_agent.pt")
			torch.save(best_model.state_dict(), save_path)
		else:
			epochs_without_improvement += 1

		# Reset al entrar en autonomía
		if epoch == transition_end:
			epochs_without_improvement = 0
			print(f"🦅 Entrando en autonomía — reset early stopping (best={best_acc_joint:.2f}%)")

		# Solo early stopping después de 10 epochs en autonomía
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
	run_glyph_training()
