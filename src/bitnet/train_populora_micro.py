"""
train_populora_micro.py — Phase 0 Dual-Head Micro-Vocabulary Arena

v3: Fixes structural issues from Exp. 001/002:
- Separate concept head (15 classes) and emotion head (6 classes)
- Learned positional embeddings (3 positions)
- β = 0.5 (raised from 0.3 — no longer competing for same logits)
- Full JSONL telemetry via ExperimentLogger
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
MICRO_VOCAB = CONCEPTS + EMOTIONS  # 21 tokens total


class DualHeadListener(nn.Module):
	"""
	Wraps a BitNet4LayerModel with:
	- Learned positional embeddings
	- Separate classification heads for concept (15) and emotion (6)
	"""

	def __init__(self, base_model: BitNet4LayerModel, num_concepts: int, num_emotions: int):
		super().__init__()
		self.base = base_model
		self.hidden_dim = base_model.hidden_dim

		# Positional embeddings: 3 positions (input, concept_output, emotion_output)
		self.pos_embedding = nn.Embedding(3, self.hidden_dim)

		# Separate output heads
		self.concept_head = nn.Linear(self.hidden_dim, num_concepts)
		self.emotion_head = nn.Linear(self.hidden_dim, num_emotions)

	def forward_with_heads(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		"""
		Forward pass through base model's core, then split into dual heads.
		x: (batch, seq=3, vocab_size=21) — Gumbel-Softmax message
		Returns: (concept_logits, emotion_logits)
		"""
		# Embed through Layer 1 (frozen vocab embeddings) + Layer 2 (inbound proj)
		if x.ndim == 2:
			embeds = F.embedding(x, self.base.vocab_embeddings)
		else:
			embeds = torch.matmul(x, self.base.vocab_embeddings)

		h = self.base.inbound_proj(embeds)  # (batch, 3, hidden_dim)

		# Add positional embeddings
		positions = torch.arange(3, device=h.device)
		h = h + self.pos_embedding(positions).unsqueeze(0)

		# Layer 3: Core transformer
		for layer in self.base.core_layers:
			h = layer(h)
		h = self.base.norm(h)

		# Dual heads: concept from position 1, emotion from position 2
		concept_logits = self.concept_head(h[:, 1, :])  # (batch, 15)
		emotion_logits = self.emotion_head(h[:, 2, :])  # (batch, 6)

		return concept_logits, emotion_logits


class DualHeadSpeaker(nn.Module):
	"""
	Wraps a BitNet4LayerModel with positional embeddings for the speaker role.
	Uses the base model's generate_message for Gumbel-Softmax emission.
	"""

	def __init__(self, base_model: BitNet4LayerModel):
		super().__init__()
		self.base = base_model
		self.pos_embedding = nn.Embedding(3, base_model.hidden_dim)

	def generate_message(self, x: torch.Tensor, tau: float = 1.0, hard: bool = True) -> torch.Tensor:
		"""Generate a differentiable message with positional awareness."""
		# Embed input tokens
		embeds = F.embedding(x, self.base.vocab_embeddings)
		h = self.base.inbound_proj(embeds)

		# Add positional embeddings
		positions = torch.arange(3, device=h.device)
		h = h + self.pos_embedding(positions).unsqueeze(0)

		# Core transformer
		for layer in self.base.core_layers:
			h = layer(h)
		h = self.base.norm(h)

		# Outbound to vocab logits
		concept_proj = self.base.outbound_proj(h)
		logits = torch.matmul(concept_proj, self.base.vocab_embeddings.T)

		# Gumbel-Softmax
		return F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)


def build_micro_embeddings() -> np.ndarray:
	"""Generate 384-dim embeddings for the 21-token micro-vocabulary."""
	from fastembed import TextEmbedding

	model = TextEmbedding()
	embeddings = list(model.embed(MICRO_VOCAB))
	return np.array([e if isinstance(e, np.ndarray) else np.array(e) for e in embeddings], dtype=np.float32)


def svd_crossover(parent_a: nn.Module, parent_b: nn.Module, child: nn.Module, alpha: float = 0.5, sigma: float = 0.01):
	"""SVD crossover with condition number monitoring."""
	with torch.no_grad():
		for name, param in child.named_parameters():
			if not param.requires_grad:
				continue
			p_a = dict(parent_a.named_parameters())[name].data
			p_b = dict(parent_b.named_parameters())[name].data

			if p_a.ndim == 2:
				w_avg = alpha * p_a + (1.0 - alpha) * p_b
				try:
					u, s, vh = torch.linalg.svd(w_avg, full_matrices=False)
					cond = s[0] / (s[-1] + 1e-10)
					if cond > 1e4:
						param.copy_(w_avg)
						continue
					noise = torch.randn_like(s) * sigma
					s_perturbed = (s + noise).clamp_(min=0.0)
					param.copy_(u @ torch.diag(s_perturbed) @ vh)
				except Exception:
					param.copy_(w_avg)
			else:
				noise = torch.randn_like(p_a) * sigma
				param.copy_(alpha * p_a + (1.0 - alpha) * p_b + noise)


def run_arena():
	print("=== 🌋 Arena PopuLoRA — Dual-Head Micro (Exp. 003) ===")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"[Device]: {device}")

	# 1. Build micro-embeddings
	vocab_embeddings = build_micro_embeddings()
	num_concepts = len(CONCEPTS)
	num_emotions = len(EMOTIONS)
	vocab_size = len(MICRO_VOCAB)
	print(f"[Vocab]: {vocab_size} tokens ({num_concepts}C + {num_emotions}E) | Dual-Head mode")

	# 2. Hyperparameters
	pop_size = 4
	epochs = 15
	steps_per_epoch = 200
	batch_size = 32
	lr = 1e-3
	tau_start = 1.0
	tau_min = 0.1
	beta_emotion = 0.5
	total_steps = epochs * steps_per_epoch

	# 3. Telemetry
	params = {
		"vocab_size": vocab_size,
		"num_concepts": num_concepts,
		"num_emotions": num_emotions,
		"hidden_dim": 256,
		"num_layers": 4,
		"pop_size": pop_size,
		"epochs": epochs,
		"steps_per_epoch": steps_per_epoch,
		"batch_size": batch_size,
		"lr": lr,
		"tau_start": tau_start,
		"tau_min": tau_min,
		"beta_emotion": beta_emotion,
		"architecture": "dual_head",
		"positional_encoding": True,
		"device": str(device),
	}
	logger = ExperimentLogger("003", params)

	# 4. Initialize population with dual-head wrappers
	base_models = [
		BitNet4LayerModel(vocab_embeddings=vocab_embeddings, hidden_dim=256, num_layers=4).to(device)
		for _ in range(pop_size)
	]
	speakers = [DualHeadSpeaker(m).to(device) for m in base_models]
	listeners = [DualHeadListener(m, num_concepts, num_emotions).to(device) for m in base_models]

	# Each agent's parameters = speaker params + listener params (shared base + own heads)
	optimizers = [
		torch.optim.AdamW(
			list(speakers[i].parameters()) + list(listeners[i].parameters()),
			lr=lr,
		)
		for i in range(pop_size)
	]

	fitness = np.zeros(pop_size)
	current_step = 0

	for epoch in range(epochs):
		print(f"\n--- Época {epoch + 1}/{epochs} ---")
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

			# Token IDs in micro-vocab for speaker input
			concept_token_ids = concept_targets  # [0..14]
			emotion_token_ids = torch.from_numpy(emotion_indices + num_concepts).long().to(device)  # [15..20]

			# Select speaker/listener pair
			idx_s = np.random.randint(0, pop_size)
			idx_l = np.random.randint(0, pop_size)
			while idx_l == idx_s:
				idx_l = np.random.randint(0, pop_size)

			opt_s = optimizers[idx_s]
			opt_l = optimizers[idx_l]
			opt_s.zero_grad()
			opt_l.zero_grad()

			# Speaker input: [concept_token, emotion_token, 0]
			speaker_input = torch.zeros((batch_size, 3), dtype=torch.long, device=device)
			speaker_input[:, 0] = concept_token_ids
			speaker_input[:, 1] = emotion_token_ids

			# Speaker generates message (Gumbel-Softmax)
			message = speakers[idx_s].generate_message(speaker_input, tau=tau, hard=True)

			# Listener predicts via dual heads
			pred_concept_logits, pred_emotion_logits = listeners[idx_l].forward_with_heads(message)

			# Loss: concept (15 classes) + β * emotion (6 classes)
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

			# Telemetry: log every step
			sample_c = CONCEPTS[concept_indices[0]]
			sample_e = EMOTIONS[emotion_indices[0]]
			pred_c_idx = preds_c[0].item()
			pred_e_idx = preds_e[0].item()
			pred_c_name = CONCEPTS[pred_c_idx] if pred_c_idx < num_concepts else f"?{pred_c_idx}"
			pred_e_name = EMOTIONS[pred_e_idx] if pred_e_idx < num_emotions else f"?{pred_e_idx}"

			logger.log_step(
				epoch=epoch + 1, step=step, loss=loss.item(),
				loss_concept=loss_concept.item(), loss_emotion=loss_emotion.item(),
				tau=tau, speaker_id=idx_s, listener_id=idx_l,
				target_concept=sample_c, target_emotion=sample_e,
				pred_concept=pred_c_name, pred_emotion=pred_e_name,
				concept_correct=c_ok, emotion_correct=e_ok, joint_correct=j_ok,
				batch_size=batch_size,
			)

			# Debug logging every 20 steps
			if step % 20 == 0:
				print(f"  step {step:3d} | τ={tau:.3f} | loss={loss.item():.3f} "
					  f"(C:{loss_concept.item():.3f} E:{loss_emotion.item():.3f}) | "
					  f"target=({sample_c},{sample_e}) → pred=({pred_c_name},{pred_e_name})")

		# Epoch aggregation
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

		# Evolution
		worst_idx = np.argmin(fitness)
		best_indices = np.argsort(fitness)[-2:]
		print(f"[Evolución] Agent_{worst_idx} → hijo SVD de A{best_indices[1]} × A{best_indices[0]}")

		logger.log_epoch(
			epoch=epoch + 1, loss_avg=avg_loss,
			acc_concept=acc_c, acc_emotion=acc_e, acc_joint=acc_j,
			fitness=fitness.tolist(), worst_agent=worst_idx,
			parent_a=int(best_indices[1]), parent_b=int(best_indices[0]),
		)

		svd_crossover(speakers[best_indices[1]], speakers[best_indices[0]], speakers[worst_idx])
		svd_crossover(listeners[best_indices[1]], listeners[best_indices[0]], listeners[worst_idx])
		optimizers[worst_idx] = torch.optim.AdamW(
			list(speakers[worst_idx].parameters()) + list(listeners[worst_idx].parameters()), lr=lr,
		)

		# Promotion check
		if acc_j >= 80.0:
			print("\n🏆 ¡HIT! Entendimiento mutuo ≥ 80%. PROMOTED TO GRADE 1.")
			logger.log_event("promotion", {"grade": 1, "acc_joint": acc_j})
			break

	logger.close()
	print(f"\n[Telemetry] Datos guardados en: storage/telemetry/EXP_003.jsonl")


if __name__ == "__main__":
	run_arena()
