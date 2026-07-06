"""
EXP_036: Entrenamiento con Deep Think — Metacognición.

El modelo piensa en dos fases:
  Fase 1 (Pensar):   Bucle latente → resultado candidato
  Fase 2 (Verificar): Re-inyectar resultado → verificar convergencia

Tres señales de loss:
  1. loss_think:  ¿Fase 1 produce la respuesta correcta?
  2. loss_verify: ¿Fase 2 confirma la respuesta correcta?
  3. loss_convergence: ¿Convergencia alta cuando correcto, baja cuando incorrecto?

Hipótesis:
  H₁: La verificación mejora accuracy sobre EXP_034.
  H₂: Convergencia correlaciona con correctness (métrica de confianza).
  H₃: El modelo aprende a "dudar" de sus errores.

Origen: Joan Garcia — "el pensamiento en voz alta me ha servido para
focalizar y no desviarme" / "coger el resultado y volver a iterar
para validar si es correcto"
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from src.bitnet.vocab.glyph_vocabulary import (
	EMOTION_INDEX,
	N_EMOTIONS,
	N_WORDS,
	WORD_INDEX,
	WORD_NAMES,
	build_emotional_chains,
	get_bifurcation_pairs,
)
from src.bitnet.model.modeling_bitnet import BitNet4LayerModel
from src.bitnet.telemetry.telemetry import ExperimentLogger


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


def run_deep_think_training():
	parser = argparse.ArgumentParser(description="EXP_036: Deep Think")
	parser.add_argument("--config", type=str, required=True)
	args, _ = parser.parse_known_args()

	base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	config_path = args.config if os.path.isabs(args.config) else os.path.join(base_dir, args.config)
	with open(config_path, encoding="utf-8") as f:
		config = json.load(f)

	experiment_id = config["experiment_id"]
	emotion_mode = config.get("emotion", {}).get("mode", "first_only")

	print(f"═══ 🧠 Deep Think — {experiment_id} ═══")
	print("    Metacognición: Pensar + Verificar + Convergencia")
	print(f"    Vocabulario: {N_WORDS} palabras × 65 primos ternarios")

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

	# ═══ Dataset ═══
	chains = build_emotional_chains(max_depth=config.get("max_chain_depth", 2))
	bifurcation_pairs = get_bifurcation_pairs()
	print(f"🔗 Cadenas: {len(chains)} | Bifurcaciones: {len(bifurcation_pairs)}")

	# ═══ Modelo ═══
	model_cfg = config.get("model", {})
	pop_size = config.get("pop_size", 4)
	resonance_cfg = config.get("resonance", {})
	max_res_steps = resonance_cfg.get("max_resonance_steps", 5)
	emotion_cfg = config.get("emotion", {})
	deep_think_cfg = config.get("deep_think", {})

	n_think = resonance_cfg.get("n_think", 2)
	n_verify = resonance_cfg.get("n_verify", 2)
	pos_mode = resonance_cfg.get("pos_mode", "clock")

	population = [
		BitNet4LayerModel(
			use_glyphs=True,
			hidden_dim=model_cfg.get("hidden_dim", 256),
			num_layers=model_cfg.get("num_layers", 3),
			use_pos_embedding=model_cfg.get("use_pos_embedding", True),
			max_resonance_steps=max_res_steps,
			n_emotions=N_EMOTIONS,
			emotion_dim=emotion_cfg.get("dim", 64),
			emotion_mode=emotion_mode,
		).to(device)
		for _ in range(pop_size)
	]

	n_params = sum(p.numel() for p in population[0].parameters() if p.requires_grad)
	print(f"📊 Params: {n_params:,}")

	# ═══ Pre-entrenamiento (Piaget: primero saber, luego dudar) ═══
	pretrained_from = config.get("pretrained_from")
	if pretrained_from:
		pretrained_path = pretrained_from if os.path.isabs(pretrained_from) else os.path.join(base_dir, pretrained_from)
		if os.path.exists(pretrained_path):
			state_dict = torch.load(pretrained_path, map_location=device, weights_only=True)
			for i, model in enumerate(population):
				model.load_state_dict(state_dict)
			print(f"🧒→🧑 Pre-trained: {pretrained_path}")
			print("    Fase Piaget: el modelo ya sabe, ahora aprende a dudar")
		else:
			print(f"⚠️ pretrained_from no encontrado: {pretrained_path}")

	train_cfg = config.get("training", {})
	lr = train_cfg.get("lr", 3e-4)
	wd = train_cfg.get("weight_decay", 0.05)
	optimizers = [
		torch.optim.AdamW(filter(lambda p: p.requires_grad, m.parameters()), lr=lr, weight_decay=wd)
		for m in population
	]

	# Loss weights
	think_weight = deep_think_cfg.get("think_loss_weight", 1.0)
	verify_weight = deep_think_cfg.get("verify_loss_weight", 0.5)
	convergence_weight = deep_think_cfg.get("convergence_weight", 0.3)

	# ═══ Training loop ═══
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

	fitness = np.zeros(pop_size)
	logger = ExperimentLogger(
		experiment_id,
		{**config, "device": str(device), "mode": "deep_think"},
		log_path=os.path.join(exp_dir, "telemetry.jsonl"),
	)

	early_stopping_cfg = config.get("early_stopping", {})
	patience = early_stopping_cfg.get("patience", 30)
	best_acc_joint = 0.0
	epochs_without_improvement = 0
	current_step = 0

	# Métricas de metacognición

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
		epoch_conv_correct, epoch_conv_wrong = [], []
		interactions = np.zeros((pop_size, pop_size))
		successes = np.zeros((pop_size, pop_size))

		for model in population:
			model.train()

		for step in range(steps_per_epoch):
			tau = max(tau_min, tau_start * (1.0 - current_step / total_steps))

			batch_chains = [chains[np.random.randint(len(chains))] for _ in range(batch_size)]

			idx_s = np.random.randint(0, pop_size)
			idx_l = np.random.randint(0, pop_size)
			while idx_l == idx_s and pop_size > 1:
				idx_l = np.random.randint(0, pop_size)

			speaker, listener = population[idx_s], population[idx_l]
			opt_s, opt_l = optimizers[idx_s], optimizers[idx_l]
			opt_s.zero_grad()
			opt_l.zero_grad()

			start_ids = torch.tensor([c["start"] for c in batch_chains], dtype=torch.long, device=device)
			end_ids = torch.tensor([c["end"] for c in batch_chains], dtype=torch.long, device=device)

			current_input = torch.zeros((batch_size, 4), dtype=torch.long, device=device)
			current_input[:, 0] = start_ids
			current_input[:, 1] = start_ids

			emotion_ids = torch.tensor(
				[c["emotion_id"] for c in batch_chains],
				dtype=torch.long, device=device,
			)

			# ═══ DEEP THINK FORWARD ═══
			logits_think, logits_verify, convergence, meta = speaker.forward_deep_think_training(
				current_input, n_think=n_think, n_verify=n_verify,
				pos_mode=pos_mode, emotion_ids=emotion_ids,
			)

			# Loss 1: ¿Fase 1 produce la respuesta correcta?
			loss_think = F.cross_entropy(logits_think[:, 2, :], end_ids)

			# Loss 2: ¿Fase 2 confirma la respuesta correcta?
			loss_verify = F.cross_entropy(logits_verify[:, 2, :], end_ids)

			# Loss 3: Convergencia
			# Queremos convergencia alta cuando el modelo acierta
			# y convergencia baja cuando se equivoca (aprende a "dudar")
			with torch.no_grad():
				think_correct = (logits_think[:, 2, :].argmax(dim=-1) == end_ids).float()

			# Objetivo: convergence → 1 cuando correcto, convergence → 0 cuando incorrecto
			convergence_target = think_correct  # (batch,)
			loss_convergence = F.mse_loss(convergence, convergence_target)

			total_loss = (
				think_weight * loss_think +
				verify_weight * loss_verify +
				convergence_weight * loss_convergence
			)

			# Listener (Gumbel-Softmax del resultado verificado)
			speaker_msg = F.gumbel_softmax(logits_verify, tau=tau, hard=False, dim=-1)

			if torch.rand(1).item() < tf_ratio:
				teacher = current_input.clone()
				teacher[:, 2] = end_ids
				msg_to_listener = F.one_hot(teacher, num_classes=N_WORDS).float()
			else:
				msg_to_listener = speaker_msg

			listener_logits = listener(msg_to_listener)
			loss_listener = F.cross_entropy(listener_logits[:, 2, :], end_ids)
			total_loss = total_loss + loss_listener

			# Backprop
			total_loss.backward()
			torch.nn.utils.clip_grad_norm_(speaker.parameters(), max_norm=grad_clip)
			torch.nn.utils.clip_grad_norm_(listener.parameters(), max_norm=grad_clip)
			opt_s.step()
			opt_l.step()

			epoch_losses.append(total_loss.item())

			# Métricas
			with torch.no_grad():
				pred_verify = torch.argmax(logits_verify[:, 2, :], dim=-1)
				verify_ok = (pred_verify == end_ids).sum().item()

				# Convergencia por correcto/incorrecto
				for i in range(batch_size):
					if pred_verify[i] == end_ids[i]:
						epoch_conv_correct.append(convergence[i].item())
					else:
						epoch_conv_wrong.append(convergence[i].item())

			epoch_correct += verify_ok
			epoch_total += batch_size
			interactions[idx_s, idx_l] += batch_size
			successes[idx_s, idx_l] += verify_ok

			if step % 50 == 0:
				c = batch_chains[0]
				pred_name = WORD_NAMES[pred_verify[0].item()]
				ok = "✓" if pred_name == c["dest_name"] else "✗"
				avg_conv = convergence.mean().item()
				print(
					f"  step {step:3d} | τ={tau:.3f} TF={tf_ratio:.0%} | "
					f"loss={total_loss.item():.3f} | conv={avg_conv:.3f} | "
					f"😶 {c['emotion_name']} | 🔗 {c['source_name']}→{c['dest_name']} | "
					f"pred={pred_name}({ok})"
				)

			current_step += 1

		# Fitness
		for i in range(pop_size):
			sent_total = interactions[i, :].sum() + interactions[:, i].sum()
			sent_correct = successes[i, :].sum() + successes[:, i].sum()
			fitness[i] = sent_correct / (sent_total + 1e-10)

		avg_loss = np.mean(epoch_losses)
		acc_joint = (epoch_correct / epoch_total) * 100

		# Métricas de metacognición
		avg_conv_correct = np.mean(epoch_conv_correct) if epoch_conv_correct else 0.0
		avg_conv_wrong = np.mean(epoch_conv_wrong) if epoch_conv_wrong else 0.0
		conv_gap = avg_conv_correct - avg_conv_wrong

		# ═══ BIFURCACIONES ═══
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

				test_emos = torch.tensor([emo_a_id, emo_b_id], dtype=torch.long, device=device)
				logits_test, meta_test = best_model.forward_deep_think(
					x_test, n_think=n_think, n_verify=n_verify,
					pos_mode=pos_mode, emotion_ids=test_emos,
				)

				preds = torch.argmax(logits_test[:, 2, :], dim=-1)
				if preds[0].item() == expected_a:
					bif_correct += 1
				if preds[1].item() == expected_b:
					bif_correct += 1
				bif_total += 2

		acc_bifurcation = (bif_correct / bif_total) * 100 if bif_total > 0 else 0.0

		print(f"Loss: {avg_loss:.4f} | Joint: {acc_joint:.2f}% | "
		      f"Bif: {acc_bifurcation:.1f}% | "
		      f"Conv✓: {avg_conv_correct:.3f} Conv✗: {avg_conv_wrong:.3f} Gap: {conv_gap:.3f}")

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

		if epoch == transition_end:
			epochs_without_improvement = 0
			print("🦅 Entrando en autonomía — reset early stopping")

		if epoch >= transition_end + 10 and epochs_without_improvement >= patience:
			print(f"⏹️ Early stopping: {patience} epochs sin mejora (best={best_acc_joint:.2f}%)")
			break

		# SVD
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

	# Final
	best_idx = int(np.argmax(fitness))
	save_path = os.path.join(exp_dir, "best_agent.pt")
	torch.save(population[best_idx].state_dict(), save_path)
	print(f"💾 Mejor agente guardado en {save_path} (acc_joint={best_acc_joint:.2f}%)")

	logger.close()
	return best_acc_joint


if __name__ == "__main__":
	run_deep_think_training()
