"""
EXP_032: Entrenamiento con Resonancia Continua — Bucle Latente 4→2.

A diferencia de EXP_029 (que encadenaba pasos por vocabulario 8192-dim),
este script entrena con el bucle CERRADO en espacio latente (256-dim).

El modelo itera sus core_layers N veces sobre el mismo hidden state
y solo al final proyecta a logits para evaluar la respuesta.

Topología:
    input → [Capa1→2: una vez] → h(256)
                                   │
                 ┌─────────────────┤
                 │  core_layers    │ × n_steps
                 │  + norm + clock │
                 └─────────────────┘
                        │
                 [Capa4→5: solo al final] → logits(8192)

Soporta 3 modos de loss (Eje C del grid):
    - 'final':    Loss solo al último paso (Resonancia Aislada pura)
    - 'every':    Loss en cada paso del bucle (supervisión total)
    - 'weighted': 0.2 × intermedia + 1.0 × final

Uso:
    python -m src.bitnet.train_resonance --config configs/experiments/EXP_032_none_3_final.json
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


def build_chain_dataset(translator: SovereignTranslator, max_depth: int = 3):
	"""Construye dataset de cadenas causales. Heredado de train_loop.py."""
	graph = get_causal_graph()
	concept_tids = {}
	for name in CONCEPT_NAMES:
		tids = translator.encode(name)
		concept_tids[name] = tids[0] if tids else 0

	chains = []

	def dfs(start_name, current_name, path, depth, max_fear):
		if depth >= 2 and len(path) > 1:
			start_idx = CONCEPT_NAMES.index(start_name)
			end_idx = CONCEPT_NAMES.index(current_name)
			intermediates = [CONCEPT_NAMES.index(p) for p in path[1:-1]]
			chains.append({
				"start": start_idx,
				"start_tid": concept_tids[start_name],
				"end": end_idx,
				"end_tid": concept_tids[current_name],
				"intermediates": intermediates,
				"intermediate_tids": [concept_tids[CONCEPT_NAMES[i]] for i in intermediates],
				"depth": len(path) - 1,
				"fear": max_fear,
				"path_names": list(path),
			})
		if depth >= max_depth:
			return
		current_idx = CONCEPT_NAMES.index(current_name)
		for next_idx, fear in graph.get(current_idx, []):
			next_name = CONCEPT_NAMES[next_idx]
			if next_name not in path:
				dfs(start_name, next_name, path + [next_name], depth + 1, max(max_fear, fear))

	for start_name in CONCEPT_NAMES:
		start_idx = CONCEPT_NAMES.index(start_name)
		if start_idx in graph:
			for next_idx, fear in graph[start_idx]:
				next_name = CONCEPT_NAMES[next_idx]
				dfs(start_name, next_name, [start_name, next_name], 1, fear)

	return chains, concept_tids


def get_n_steps_for_epoch(epoch, config):
	"""Calcula n_steps según el modo de profundidad y la fase del curriculum."""
	depth_mode = config["resonance"]["n_steps"]
	if depth_mode == "ramp":
		ramp = config.get("depth_ramp", {})
		nursery_end = config["curriculum"]["nursery_end"]
		transition_end = config["curriculum"]["transition_end"]

		if epoch < nursery_end:
			return ramp.get("nursery_steps", 1)
		elif epoch < transition_end:
			# Interpolar entre transition_steps_start y transition_steps_end
			progress = (epoch - nursery_end) / (transition_end - nursery_end)
			start = ramp.get("transition_steps_start", 2)
			end = ramp.get("transition_steps_end", 3)
			return int(start + progress * (end - start))
		else:
			# Autonomía: subir de autonomy_start a autonomy_end progresivamente
			total_autonomy = config["training"]["epochs"] - transition_end
			progress = min(1.0, (epoch - transition_end) / max(1, total_autonomy))
			start = ramp.get("autonomy_steps_start", 3)
			end = ramp.get("autonomy_steps_end", 5)
			return int(start + progress * (end - start))
	else:
		return int(depth_mode)


def run_resonance_training():
	parser = argparse.ArgumentParser(description="EXP_032: Resonancia Continua")
	parser.add_argument("--config", type=str, required=True, help="Path to experiment config JSON")
	args, _ = parser.parse_known_args()

	base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

	config_path = args.config if os.path.isabs(args.config) else os.path.join(base_dir, args.config)
	with open(config_path, encoding="utf-8") as f:
		config = json.load(f)

	experiment_id = config["experiment_id"]
	print(f"=== 🔄 Resonancia Continua — {experiment_id} ===")
	print(f"    pos_mode={config['resonance']['pos_mode']} | "
		  f"depth={config['resonance']['n_steps']} | "
		  f"loss_mode={config['resonance']['loss_mode']}")

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

	# 1. Dataset
	translator = SovereignTranslator()
	vocab_embeddings = translator.get_concept_embeddings()

	chains, concept_tids = build_chain_dataset(translator, max_depth=config.get("max_chain_depth", 3))
	print(f"🔗 Cadenas: {len(chains)}")

	chains_2 = [c for c in chains if c["depth"] == 2]
	print(f"   2 pasos: {len(chains_2)} | 3+ pasos: {len([c for c in chains if c['depth'] >= 3])}")

	op_implica_tid = translator.encode("implica")[0]

	# 2. Población
	model_cfg = config.get("model", {})
	pop_size = config.get("pop_size", 4)
	resonance_cfg = config["resonance"]
	max_res_steps = resonance_cfg.get("max_resonance_steps", 5)

	population = [
		BitNet4LayerModel(
			vocab_embeddings=vocab_embeddings,
			hidden_dim=model_cfg.get("hidden_dim", 256),
			num_layers=model_cfg.get("num_layers", 3),
			use_pos_embedding=model_cfg.get("use_pos_embedding", True),
			max_resonance_steps=max_res_steps if resonance_cfg["pos_mode"] == "clock" else 0,
		).to(device)
		for _ in range(pop_size)
	]

	# Hotstart
	resume = config.get("resume_checkpoint")
	if resume:
		resume_path = resume if os.path.isabs(resume) else os.path.join(base_dir, resume)
		if os.path.exists(resume_path):
			print(f"💾 [HOTSTART] Cargando desde {resume_path}...")
			state_dict = torch.load(resume_path, map_location=device, weights_only=True)
			for model in population:
				model.load_state_dict(state_dict, strict=False)
			print("✅ Transferencia completada.")

	train_cfg = config["training"]
	lr = train_cfg.get("lr", 3e-4)
	wd = train_cfg.get("weight_decay", 0.05)
	optimizers = [
		torch.optim.AdamW(filter(lambda p: p.requires_grad, m.parameters()), lr=lr, weight_decay=wd)
		for m in population
	]

	# Logit mask
	logit_mask = None
	if config.get("use_logit_mask", True):
		micro_vocab = config.get("micro_vocab_words", CONCEPT_NAMES + ["implica"])
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

	pos_mode = resonance_cfg["pos_mode"]
	loss_mode = resonance_cfg["loss_mode"]
	intermediate_loss_weight = resonance_cfg.get("intermediate_loss_weight", 0.2)
	fear_amplifier = config.get("fear_amplifier", 3.0)
	watcher_interval = resonance_cfg.get("watcher_interval_epochs", 10)

	fitness = np.zeros(pop_size)
	logger = ExperimentLogger(
		experiment_id,
		{**config, "device": str(device)},
		log_path=os.path.join(exp_dir, "telemetry.jsonl"),
	)

	# Watcher log
	watcher_log_path = os.path.join(exp_dir, "watcher_log.jsonl")
	watcher_f = open(watcher_log_path, "w", encoding="utf-8")

	# Stability metrics log
	stability_log_path = os.path.join(exp_dir, "stability_metrics.jsonl")
	stability_f = open(stability_log_path, "w", encoding="utf-8")

	current_step = 0
	best_acc_joint = 0.0

	for epoch in range(epochs):
		if epoch < nursery_end:
			tf_ratio, phase = 1.0, "🍼 Guardería"
		elif epoch < transition_end:
			progress = (epoch - nursery_end) / (transition_end - nursery_end)
			tf_ratio, phase = 1.0 - progress * (1.0 - tf_min), "🎮 Recreo"
		else:
			tf_ratio, phase = tf_min, "🦅 Autonomía"

		n_steps = get_n_steps_for_epoch(epoch, config)

		print(f"\n--- Época {epoch + 1}/{epochs} [{phase}] TF={tf_ratio:.0%} n_steps={n_steps} ---")

		epoch_losses = []
		epoch_correct, epoch_total = 0, 0
		epoch_intermediate_correct = 0
		interactions = np.zeros((pop_size, pop_size))
		successes = np.zeros((pop_size, pop_size))

		for model in population:
			model.train()

		# Curriculum de cadenas
		active_chains = (chains_2 if chains_2 else chains) if epoch < nursery_end else chains

		collect_watcher = (epoch % watcher_interval == 0)

		for step in range(steps_per_epoch):
			tau = max(tau_min, tau_start * (1.0 - current_step / total_steps))

			batch_chains = [active_chains[np.random.randint(len(active_chains))] for _ in range(batch_size)]

			# Speaker/Listener pairing
			idx_s = np.random.randint(0, pop_size)
			idx_l = np.random.randint(0, pop_size)
			while idx_l == idx_s and pop_size > 1:
				idx_l = np.random.randint(0, pop_size)

			speaker, listener = population[idx_s], population[idx_l]
			opt_s, opt_l = optimizers[idx_s], optimizers[idx_l]
			opt_s.zero_grad()
			opt_l.zero_grad()

			# ═══════════════════════════════════════════
			# RESONANCIA CONTINUA — EL BUCLE LATENTE
			# ═══════════════════════════════════════════
			# En vez de encadenar pasos por vocabulario (EXP_029),
			# el speaker itera sus core_layers N veces en 256-dim.
			# El listener recibe el resultado final y decodifica.

			# Preparar input
			start_tids = torch.tensor([c["start_tid"] for c in batch_chains], dtype=torch.long, device=device)
			end_tids = torch.tensor([c["end_tid"] for c in batch_chains], dtype=torch.long, device=device)
			fears = torch.tensor([1.0 + c["fear"] * fear_amplifier for c in batch_chains], dtype=torch.float32, device=device)

			# Input: [start, implica, 0, 0]
			current_input = torch.zeros((batch_size, 4), dtype=torch.long, device=device)
			current_input[:, 0] = start_tids
			current_input[:, 1] = op_implica_tid

			# Intermediate targets (para loss_every y loss_weighted)
			intermediate_tids = []
			for c in batch_chains:
				if c["intermediate_tids"]:
					intermediate_tids.append(c["intermediate_tids"][0])
				else:
					intermediate_tids.append(c["end_tid"])
			torch.tensor(intermediate_tids, dtype=torch.long, device=device)

			# ═══════════════════════════════════════════
			# FORWARD: Speaker con resonancia
			# ═══════════════════════════════════════════

			if loss_mode == "final":
				# Resonancia Aislada pura: solo loss al final
				speaker_logits, meta = speaker.forward_resonance(
					current_input, n_steps=n_steps, pos_mode=pos_mode,
					collect_watcher=(collect_watcher and step == 0), logit_mask=logit_mask,
				)
				total_loss = torch.tensor(0.0, device=device)

			elif loss_mode == "every":
				# Supervisión en cada paso: cada paso debe producir tokens correctos
				intermediate_targets_dict = dict.fromkeys(range(n_steps))
				speaker_logits, intermediate_logits = speaker.forward_resonance_training(
					current_input, n_steps=n_steps, pos_mode=pos_mode,
					intermediate_targets=intermediate_targets_dict, logit_mask=logit_mask,
				)
				total_loss = torch.tensor(0.0, device=device)

				# Loss intermedio en cada paso
				for _step_idx, logits_mid in intermediate_logits:
					loss_mid = F.cross_entropy(logits_mid[:, 2, :], end_tids, reduction="none")
					total_loss = total_loss + (loss_mid * fears).mean()

				meta = None  # No tenemos metadata en training mode

			elif loss_mode == "weighted":
				# Supervisión ligera intermedia + peso fuerte final
				intermediate_targets_dict = dict.fromkeys(range(n_steps))
				speaker_logits, intermediate_logits = speaker.forward_resonance_training(
					current_input, n_steps=n_steps, pos_mode=pos_mode,
					intermediate_targets=intermediate_targets_dict, logit_mask=logit_mask,
				)
				total_loss = torch.tensor(0.0, device=device)

				for _step_idx, logits_mid in intermediate_logits:
					loss_mid = F.cross_entropy(logits_mid[:, 2, :], end_tids, reduction="none")
					total_loss = total_loss + (loss_mid * fears * intermediate_loss_weight).mean()

				meta = None

			# ═══════════════════════════════════════════
			# LISTENER: Decodifica el mensaje final
			# ═══════════════════════════════════════════

			# El speaker produce su mejor mensaje, lo pasamos al listener
			speaker_msg = F.gumbel_softmax(speaker_logits, tau=tau, hard=False, dim=-1)

			# Teacher forcing
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

			# Loss del speaker: debe producir el token final correcto
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
				speaker_ok = (pred_speaker == end_tids).sum().item()

			epoch_correct += final_ok
			epoch_intermediate_correct += speaker_ok
			epoch_total += batch_size
			interactions[idx_s, idx_l] += batch_size
			successes[idx_s, idx_l] += final_ok

			# Watcher log (primer step de cada epoch watcher)
			if collect_watcher and step == 0 and meta and meta.get("watcher_samples"):
				watcher_entry = {
					"epoch": epoch + 1,
					"n_steps": n_steps,
					"trajectory_norms": meta["trajectory_norms"],
					"cosine_convergence": meta["cosine_convergence"],
					"norm_ratio": meta["norm_ratio"],
				}
				watcher_f.write(json.dumps(watcher_entry, default=str) + "\n")
				watcher_f.flush()

			# Print progress
			if step % 50 == 0:
				c = batch_chains[0]
				path = " → ".join(c["path_names"])
				pred_name = translator.decode([pred_final[0].item()])
				target_name = CONCEPT_NAMES[c["end"]]
				print(
					f"  step {step:3d} | τ={tau:.3f} TF={tf_ratio:.0%} N={n_steps} | "
					f"loss={total_loss.item():.3f} | 🔗 {path} | "
					f"pred={pred_name}({'✓' if pred_name == target_name else '✗'})"
				)

			current_step += 1

		# Fitness
		for i in range(pop_size):
			sent_total = interactions[i, :].sum() + interactions[:, i].sum()
			sent_correct = successes[i, :].sum() + successes[:, i].sum()
			fitness[i] = sent_correct / (sent_total + 1e-10)

		avg_loss = np.mean(epoch_losses)
		acc_speaker = (epoch_intermediate_correct / epoch_total) * 100
		acc_joint = (epoch_correct / epoch_total) * 100

		print(f"Loss: {avg_loss:.4f} | Speaker: {acc_speaker:.2f}% | Listener (joint): {acc_joint:.2f}%")

		# Log stability metrics
		stability_entry = {
			"epoch": epoch + 1,
			"n_steps": n_steps,
			"loss": avg_loss,
			"acc_speaker": acc_speaker,
			"acc_joint": acc_joint,
			"phase": phase.split(" ")[1] if " " in phase else phase,
			"tf_ratio": tf_ratio,
		}
		stability_f.write(json.dumps(stability_entry, default=str) + "\n")
		stability_f.flush()

		logger.log_epoch(
			epoch=epoch + 1,
			loss_avg=avg_loss,
			acc_concept=acc_speaker,
			acc_emotion=0.0,
			acc_joint=acc_joint,
			fitness=fitness.tolist(),
			worst_agent=int(np.argmin(fitness)),
			parent_a=-1,
			parent_b=-1,
			acc_homeostasis=acc_joint,
		)

		# Guardar si es el mejor
		if acc_joint > best_acc_joint:
			best_acc_joint = acc_joint
			best_idx = int(np.argmax(fitness))
			save_path = os.path.join(exp_dir, "best_agent.pt")
			torch.save(population[best_idx].state_dict(), save_path)

		# Evolución SVD
		svd_interval = config.get("svd_interval", 3)
		current_phase = "guarderia" if epoch < nursery_end else ("recreo" if epoch < transition_end else "autonomia")
		svd_phases = config.get("svd_phases", ["recreo", "autonomia"])

		if current_phase in svd_phases and epoch % svd_interval == 0:
			worst_idx = int(np.argmin(fitness))
			best = np.argsort(fitness)[-2:]
			print(f"[Evolución] Reemplazando Agent_{worst_idx} con hijo SVD de Agent_{best[1]} y Agent_{best[0]}")
			svd_crossover(population[best[1]], population[best[0]], population[worst_idx])
			optimizers[worst_idx] = torch.optim.AdamW(
				filter(lambda p: p.requires_grad, population[worst_idx].parameters()),
				lr=lr, weight_decay=wd,
			)

	# Guardar el mejor agente final
	best_idx = int(np.argmax(fitness))
	save_path = os.path.join(exp_dir, "best_agent.pt")
	torch.save(population[best_idx].state_dict(), save_path)
	print(f"💾 Mejor agente guardado en {save_path} (acc_joint={best_acc_joint:.2f}%)")

	watcher_f.close()
	stability_f.close()
	logger.close()

	return best_acc_joint


if __name__ == "__main__":
	run_resonance_training()
