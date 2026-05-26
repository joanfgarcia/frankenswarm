"""
train_populora_micro.py — Phase 0 Arena with Scheduled Teacher Forcing

Exp 004b: Fixes the Gumbel-Softmax bootstrapping problem identified in Exp 003/004a.

Three training phases:
  Phase 1 — Nursery (teacher forcing 100%): Each agent learns to decode ground truth
  Phase 2 — Supervised Play (forcing decreases 100%→0%): Mix of teacher and speaker messages
  Phase 3 — Autonomy (forcing 0%): Full emergent communication with Gumbel-Softmax

The key insight: you can't put two agents that don't know how to talk
and expect them to agree. You educate each one first, then let them communicate.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.bitnet.modeling_bitnet import BitNet4LayerModel
from src.bitnet.telemetry import ExperimentLogger


# ── Micro-Vocabulary ──────────────────────────────────────────────────
CONCEPTS = [
	"gato", "perro", "casa", "árbol", "agua",
	"fuego", "tierra", "aire", "sol", "luna",
	"peligro", "seguridad", "búnker", "agente", "código",
]
EMOTIONS = ["miedo", "alegría", "ira", "tristeza", "dolor", "hambre"]
MICRO_VOCAB = CONCEPTS + EMOTIONS


class DualHeadAgent(nn.Module):
	"""
	Unified speaker+listener agent with:
	- Shared BitNet base model
	- Positional embeddings
	- Dual classification heads (concept: 15, emotion: 6)
	- Gumbel-Softmax message generation
	"""

	def __init__(self, base_model: BitNet4LayerModel, num_concepts: int, num_emotions: int):
		super().__init__()
		self.base = base_model
		self.hidden_dim = base_model.hidden_dim
		self.num_concepts = num_concepts
		self.num_emotions = num_emotions

		# Positional embeddings (shared between speaker and listener roles)
		self.pos_embedding = nn.Embedding(3, self.hidden_dim)

		# Listener heads
		self.concept_head = nn.Linear(self.hidden_dim, num_concepts)
		self.emotion_head = nn.Linear(self.hidden_dim, num_emotions)

	def _encode(self, x: torch.Tensor) -> torch.Tensor:
		"""Shared encoding: embed → inbound proj → positional → core → norm."""
		if x.ndim == 2:
			embeds = F.embedding(x, self.base.vocab_embeddings)
		else:
			embeds = torch.matmul(x, self.base.vocab_embeddings)

		h = self.base.inbound_proj(embeds)
		positions = torch.arange(3, device=h.device)
		h = h + self.pos_embedding(positions).unsqueeze(0)

		for layer in self.base.core_layers:
			h = layer(h)
		return self.base.norm(h)

	def speak(self, x: torch.Tensor, tau: float = 1.0, hard: bool = True) -> torch.Tensor:
		"""Generate a Gumbel-Softmax message from token input."""
		h = self._encode(x)
		concept_proj = self.base.outbound_proj(h)
		logits = torch.matmul(concept_proj, self.base.vocab_embeddings.T)
		return F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)

	def listen(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		"""Decode a message into concept and emotion predictions."""
		h = self._encode(x)
		concept_logits = self.concept_head(h[:, 1, :])
		emotion_logits = self.emotion_head(h[:, 2, :])
		return concept_logits, emotion_logits


def build_micro_embeddings() -> np.ndarray:
	from fastembed import TextEmbedding
	model = TextEmbedding()
	embeddings = list(model.embed(MICRO_VOCAB))
	return np.array([e if isinstance(e, np.ndarray) else np.array(e) for e in embeddings], dtype=np.float32)


def svd_crossover(parent_a: nn.Module, parent_b: nn.Module, child: nn.Module, alpha: float = 0.5, sigma: float = 0.01):
	with torch.no_grad():
		pa_params = dict(parent_a.named_parameters())
		pb_params = dict(parent_b.named_parameters())
		for name, param in child.named_parameters():
			if not param.requires_grad:
				continue
			p_a = pa_params[name].data
			p_b = pb_params[name].data
			if p_a.ndim == 2:
				w_avg = alpha * p_a + (1.0 - alpha) * p_b
				try:
					u, s, vh = torch.linalg.svd(w_avg, full_matrices=False)
					cond = s[0] / (s[-1] + 1e-10)
					if cond > 1e4:
						param.copy_(w_avg)
						continue
					noise = torch.randn_like(s) * sigma
					param.copy_(u @ torch.diag((s + noise).clamp_(min=0.0)) @ vh)
				except Exception:
					param.copy_(w_avg)
			else:
				param.copy_(alpha * p_a + (1.0 - alpha) * p_b + torch.randn_like(p_a) * sigma)


def run_arena():
	print("=== 🌋 Arena PopuLoRA — Scheduled Teacher Forcing (Exp. 004b) ===")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"[Device]: {device}")

	vocab_embeddings = build_micro_embeddings()
	num_concepts = len(CONCEPTS)
	num_emotions = len(EMOTIONS)
	vocab_size = len(MICRO_VOCAB)

	# ── Hyperparameters ──
	pop_size = 4
	epochs = 30
	steps_per_epoch = 200
	batch_size = 32
	lr = 1e-3
	tau_start = 1.0
	tau_min = 0.3  # Higher minimum τ — don't anneal too aggressively
	beta_emotion = 0.5
	total_steps = epochs * steps_per_epoch

	# ── Teacher Forcing Schedule ──
	# Phase 1 (epochs 1-5):   100% teacher forcing — "Nursery"
	# Phase 2 (epochs 6-20):  Linear decay 100% → 0% — "Supervised Play"
	# Phase 3 (epochs 21-30): 0% teacher forcing — "Autonomy"
	nursery_end = 5
	transition_end = 20

	params = {
		"experiment": "004b_scheduled_teacher_forcing",
		"vocab_size": vocab_size, "num_concepts": num_concepts, "num_emotions": num_emotions,
		"hidden_dim": 256, "num_layers": 4, "pop_size": pop_size,
		"epochs": epochs, "steps_per_epoch": steps_per_epoch, "batch_size": batch_size,
		"lr": lr, "tau_start": tau_start, "tau_min": tau_min, "beta_emotion": beta_emotion,
		"nursery_end": nursery_end, "transition_end": transition_end,
		"architecture": "dual_head_unified", "device": str(device),
	}
	logger = ExperimentLogger("004b", params)

	# ── Population ──
	agents = [
		DualHeadAgent(
			BitNet4LayerModel(vocab_embeddings=vocab_embeddings, hidden_dim=256, num_layers=4).to(device),
			num_concepts, num_emotions
		).to(device)
		for _ in range(pop_size)
	]
	optimizers = [torch.optim.AdamW(a.parameters(), lr=lr) for a in agents]

	fitness = np.zeros(pop_size)
	current_step = 0

	for epoch in range(epochs):
		# ── Compute teacher forcing ratio for this epoch ──
		if epoch < nursery_end:
			tf_ratio = 1.0  # Phase 1: full teacher forcing
			phase_name = "🍼 Guardería"
		elif epoch < transition_end:
			progress = (epoch - nursery_end) / (transition_end - nursery_end)
			tf_ratio = 1.0 - progress  # Phase 2: linear decay
			phase_name = "🎮 Recreo"
		else:
			tf_ratio = 0.0  # Phase 3: full autonomy
			phase_name = "🦅 Autonomía"

		print(f"\n--- Época {epoch + 1}/{epochs} [{phase_name}] TF={tf_ratio:.0%} ---")
		epoch_losses = []
		concept_correct = 0
		emotion_correct = 0
		joint_correct = 0
		epoch_total = 0

		interactions = np.zeros((pop_size, pop_size))
		successes = np.zeros((pop_size, pop_size))

		for step in range(steps_per_epoch):
			tau = max(tau_min, tau_start * (1.0 - current_step / total_steps))

			# Generate batch
			concept_indices = np.random.randint(0, num_concepts, size=(batch_size,))
			emotion_indices = np.random.randint(0, num_emotions, size=(batch_size,))
			concept_targets = torch.from_numpy(concept_indices).long().to(device)
			emotion_targets = torch.from_numpy(emotion_indices).long().to(device)

			concept_token_ids = concept_targets
			emotion_token_ids = torch.from_numpy(emotion_indices + num_concepts).long().to(device)

			# Select speaker/listener pair
			idx_s = np.random.randint(0, pop_size)
			idx_l = np.random.randint(0, pop_size)
			while idx_l == idx_s:
				idx_l = np.random.randint(0, pop_size)

			speaker = agents[idx_s]
			listener = agents[idx_l]
			opt_s = optimizers[idx_s]
			opt_l = optimizers[idx_l]
			opt_s.zero_grad()
			opt_l.zero_grad()

			# Speaker input
			speaker_input = torch.zeros((batch_size, 3), dtype=torch.long, device=device)
			speaker_input[:, 0] = concept_token_ids
			speaker_input[:, 1] = emotion_token_ids

			# ── Scheduled Teacher Forcing ──
			# Decide per-sample whether to use teacher forcing or speaker message
			use_teacher = torch.rand(batch_size, device=device) < tf_ratio

			if use_teacher.all():
				# Full teacher forcing — skip Gumbel entirely
				message_input = speaker_input  # discrete tokens, ndim=2
			elif use_teacher.any():
				# Mixed: some samples use teacher, some use speaker
				speaker_message = speaker.speak(speaker_input, tau=tau, hard=True)  # (B, 3, 21)
				# Teacher message as one-hot
				teacher_onehot = F.one_hot(speaker_input, num_classes=vocab_size).float()  # (B, 3, 21)
				# Mix per sample
				mask = use_teacher.view(-1, 1, 1).float()
				message_input = mask * teacher_onehot + (1 - mask) * speaker_message  # (B, 3, 21)
			else:
				# Full autonomy — speaker only
				message_input = speaker.speak(speaker_input, tau=tau, hard=True)

			# Listener decodes
			pred_concept_logits, pred_emotion_logits = listener.listen(message_input)

			# Loss
			loss_concept = F.cross_entropy(pred_concept_logits, concept_targets)
			loss_emotion = F.cross_entropy(pred_emotion_logits, emotion_targets)
			loss = loss_concept + beta_emotion * loss_emotion
			loss.backward()

			opt_s.step()
			opt_l.step()

			# Metrics
			epoch_losses.append(loss.item())
			preds_c = torch.argmax(pred_concept_logits, dim=-1)
			preds_e = torch.argmax(pred_emotion_logits, dim=-1)

			c_ok = (preds_c == concept_targets).sum().item()
			e_ok = (preds_e == emotion_targets).sum().item()
			j_ok = ((preds_c == concept_targets) & (preds_e == emotion_targets)).sum().item()

			concept_correct += c_ok
			emotion_correct += e_ok
			joint_correct += j_ok
			epoch_total += batch_size

			interactions[idx_s, idx_l] += batch_size
			successes[idx_s, idx_l] += j_ok
			current_step += 1

			# Telemetry
			sample_c = CONCEPTS[concept_indices[0]]
			sample_e = EMOTIONS[emotion_indices[0]]
			pred_c_name = CONCEPTS[preds_c[0].item()] if preds_c[0].item() < num_concepts else "?"
			pred_e_name = EMOTIONS[preds_e[0].item()] if preds_e[0].item() < num_emotions else "?"

			logger.log_step(
				epoch=epoch + 1, step=step, loss=loss.item(),
				loss_concept=loss_concept.item(), loss_emotion=loss_emotion.item(),
				tau=tau, speaker_id=idx_s, listener_id=idx_l,
				target_concept=sample_c, target_emotion=sample_e,
				pred_concept=pred_c_name, pred_emotion=pred_e_name,
				concept_correct=c_ok, emotion_correct=e_ok, joint_correct=j_ok,
				batch_size=batch_size,
			)

			if step % 40 == 0:
				print(f"  step {step:3d} | τ={tau:.3f} TF={tf_ratio:.0%} | loss={loss.item():.3f} "
					  f"(C:{loss_concept.item():.3f} E:{loss_emotion.item():.3f}) | "
					  f"target=({sample_c},{sample_e}) → pred=({pred_c_name},{pred_e_name})")

		# Epoch summary
		for i in range(pop_size):
			sent_total = interactions[i, :].sum() + interactions[:, i].sum()
			sent_correct = successes[i, :].sum() + successes[:, i].sum()
			fitness[i] = sent_correct / (sent_total + 1e-10)

		avg_loss = np.mean(epoch_losses)
		acc_c = concept_correct / epoch_total * 100
		acc_e = emotion_correct / epoch_total * 100
		acc_j = joint_correct / epoch_total * 100

		print(f"Loss: {avg_loss:.4f} | Concepto: {acc_c:.2f}% | Emoción: {acc_e:.2f}% | Conjunta: {acc_j:.2f}%")
		print(f"Fitness: {[f'A{i}:{f * 100:.1f}%' for i, f in enumerate(fitness)]}")

		logger.log_epoch(
			epoch=epoch + 1, loss_avg=avg_loss,
			acc_concept=acc_c, acc_emotion=acc_e, acc_joint=acc_j,
			fitness=fitness.tolist(), worst_agent=int(np.argmin(fitness)),
			parent_a=-1, parent_b=-1,
		)

		# Evolution (only in Phase 2+)
		if epoch >= nursery_end:
			worst_idx = int(np.argmin(fitness))
			best_indices = np.argsort(fitness)[-2:]
			print(f"[Evolución] Agent_{worst_idx} → hijo SVD de A{best_indices[1]} × A{best_indices[0]}")
			svd_crossover(agents[best_indices[1]], agents[best_indices[0]], agents[worst_idx])
			optimizers[worst_idx] = torch.optim.AdamW(agents[worst_idx].parameters(), lr=lr)

			logger.log_event("evolution", {
				"worst": worst_idx, "parent_a": int(best_indices[1]), "parent_b": int(best_indices[0]),
			})

		# Promotion check (only meaningful in Phase 3)
		if epoch >= transition_end and acc_j >= 80.0:
			print("\n🏆 ¡HIT! Entendimiento mutuo ≥ 80% en autonomía completa.")
			print("PROMOTED TO GRADE 1: Lengua emergente adquirida.")
			logger.log_event("promotion", {"grade": 1, "acc_joint": acc_j})
			break

	logger.close()
	print(f"\n[Telemetry] storage/telemetry/EXP_004b.jsonl")


if __name__ == "__main__":
	run_arena()
