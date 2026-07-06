"""
train_proto_syntax.py — Exp 005: Proto-Syntax Analysis Arena

Same architecture as 004b (Scheduled Teacher Forcing) but with:
1. Message token logging — records what the speaker actually emits
2. Checkpointing — saves agent state_dicts every N epochs
3. Extended autonomy — 40 epochs of TF=0% after the nursery/transition phases
4. Confusion matrix logging — tracks systematic error patterns

Goal: Understand WHAT the agents are saying, not just IF they're right.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.bitnet.model.modeling_bitnet import BitNet4LayerModel
from src.bitnet.telemetry.telemetry import ExperimentLogger

# ── Micro-Vocabulary ──────────────────────────────────────────────────
CONCEPTS = [
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
]
EMOTIONS = ["miedo", "alegría", "ira", "tristeza", "dolor", "hambre"]
MICRO_VOCAB = CONCEPTS + EMOTIONS


class DualHeadAgent(nn.Module):
	def __init__(self, base_model: BitNet4LayerModel, num_concepts: int, num_emotions: int):
		super().__init__()
		self.base = base_model
		self.hidden_dim = base_model.hidden_dim
		self.num_concepts = num_concepts
		self.num_emotions = num_emotions
		self.pos_embedding = nn.Embedding(3, self.hidden_dim)
		self.concept_head = nn.Linear(self.hidden_dim, num_concepts)
		self.emotion_head = nn.Linear(self.hidden_dim, num_emotions)

	def _encode(self, x: torch.Tensor) -> torch.Tensor:
		embeds = F.embedding(x, self.base.vocab_embeddings) if x.ndim == 2 else torch.matmul(x, self.base.vocab_embeddings)
		h = self.base.inbound_proj(embeds)
		positions = torch.arange(3, device=h.device)
		h = h + self.pos_embedding(positions).unsqueeze(0)
		for layer in self.base.core_layers:
			h = layer(h)
		return self.base.norm(h)

	def speak(self, x: torch.Tensor, tau: float = 1.0, hard: bool = True) -> torch.Tensor:
		h = self._encode(x)
		concept_proj = self.base.outbound_proj(h)
		logits = torch.matmul(concept_proj, self.base.vocab_embeddings.T)
		return F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)

	def listen(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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


def save_checkpoint(agents, epoch, experiment_id):
	"""Save agent state_dicts to storage/checkpoints/."""
	base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	ckpt_dir = os.path.join(base, "storage", "checkpoints", f"EXP_{experiment_id}")
	os.makedirs(ckpt_dir, exist_ok=True)
	for i, agent in enumerate(agents):
		path = os.path.join(ckpt_dir, f"agent_{i}_epoch_{epoch:03d}.pt")
		torch.save(agent.state_dict(), path)
	print(f"  [Checkpoint] Guardado en {ckpt_dir} (época {epoch})")


def run_arena():
	print("=== 🔬 Arena PopuLoRA — Proto-Syntax Analysis (Exp. 005) ===")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"[Device]: {device}")

	vocab_embeddings = build_micro_embeddings()
	num_concepts = len(CONCEPTS)
	num_emotions = len(EMOTIONS)
	vocab_size = len(MICRO_VOCAB)

	# ── Config ──
	pop_size = 4
	nursery_epochs = 5
	transition_epochs = 15
	autonomy_epochs = 40
	epochs = nursery_epochs + transition_epochs + autonomy_epochs  # 60 total
	steps_per_epoch = 200
	batch_size = 32
	lr = 1e-3
	tau_start = 1.0
	tau_min = 0.3
	beta_emotion = 0.5
	total_steps = epochs * steps_per_epoch
	checkpoint_every = 10

	params = {
		"experiment": "005_proto_syntax",
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
		"nursery_epochs": nursery_epochs,
		"transition_epochs": transition_epochs,
		"autonomy_epochs": autonomy_epochs,
		"message_logging": True,
		"checkpointing": True,
		"device": str(device),
		"frozen_embeddings": True,
		"embedding_source": "fastembed/all-MiniLM-L6-v2",
		"embedding_dim": 384,
	}
	logger = ExperimentLogger("005", params)

	# ── Population ──
	agents = [
		DualHeadAgent(BitNet4LayerModel(vocab_embeddings=vocab_embeddings, hidden_dim=256, num_layers=4).to(device), num_concepts, num_emotions).to(
			device
		)
		for _ in range(pop_size)
	]
	optimizers = [torch.optim.AdamW(a.parameters(), lr=lr) for a in agents]

	fitness = np.zeros(pop_size)
	current_step = 0

	# Confusion matrices for autonomy phase
	concept_confusion = np.zeros((num_concepts, num_concepts), dtype=np.int64)
	emotion_confusion = np.zeros((num_emotions, num_emotions), dtype=np.int64)

	for epoch in range(epochs):
		# ── Phase logic ──
		if epoch < nursery_epochs:
			tf_ratio = 1.0
			phase_name = "🍼 Guardería"
		elif epoch < nursery_epochs + transition_epochs:
			progress = (epoch - nursery_epochs) / transition_epochs
			tf_ratio = 1.0 - progress
			phase_name = "🎮 Recreo"
		else:
			tf_ratio = 0.0
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

			concept_indices = np.random.randint(0, num_concepts, size=(batch_size,))
			emotion_indices = np.random.randint(0, num_emotions, size=(batch_size,))
			concept_targets = torch.from_numpy(concept_indices).long().to(device)
			emotion_targets = torch.from_numpy(emotion_indices).long().to(device)

			concept_token_ids = concept_targets
			emotion_token_ids = torch.from_numpy(emotion_indices + num_concepts).long().to(device)

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

			speaker_input = torch.zeros((batch_size, 3), dtype=torch.long, device=device)
			speaker_input[:, 0] = concept_token_ids
			speaker_input[:, 1] = emotion_token_ids

			# ── Scheduled Teacher Forcing ──
			use_teacher = torch.rand(batch_size, device=device) < tf_ratio
			speaker_message = None

			if use_teacher.all():
				message_input = speaker_input
			elif use_teacher.any():
				speaker_message = speaker.speak(speaker_input, tau=tau, hard=True)
				teacher_onehot = F.one_hot(speaker_input, num_classes=vocab_size).float()
				mask = use_teacher.view(-1, 1, 1).float()
				message_input = mask * teacher_onehot + (1 - mask) * speaker_message
			else:
				speaker_message = speaker.speak(speaker_input, tau=tau, hard=True)
				message_input = speaker_message

			pred_concept_logits, pred_emotion_logits = listener.listen(message_input)

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

			# ── Confusion matrix (autonomy only) ──
			if tf_ratio == 0.0:
				for i in range(batch_size):
					concept_confusion[concept_indices[i], preds_c[i].item()] += 1
					emotion_confusion[emotion_indices[i], preds_e[i].item()] += 1

			# ── Extract message tokens for logging ──
			msg_tokens_log = None
			if speaker_message is not None and tf_ratio < 1.0:
				msg_argmax = torch.argmax(speaker_message[0], dim=-1)  # first sample
				msg_tokens_log = [MICRO_VOCAB[t.item()] for t in msg_argmax]

			sample_c = CONCEPTS[concept_indices[0]]
			sample_e = EMOTIONS[emotion_indices[0]]
			pred_c_name = CONCEPTS[preds_c[0].item()] if preds_c[0].item() < num_concepts else "?"
			pred_e_name = EMOTIONS[preds_e[0].item()] if preds_e[0].item() < num_emotions else "?"

			logger.log_step(
				epoch=epoch + 1,
				step=step,
				loss=loss.item(),
				loss_concept=loss_concept.item(),
				loss_emotion=loss_emotion.item(),
				tau=tau,
				speaker_id=idx_s,
				listener_id=idx_l,
				target_concept=sample_c,
				target_emotion=sample_e,
				pred_concept=pred_c_name,
				pred_emotion=pred_e_name,
				concept_correct=c_ok,
				emotion_correct=e_ok,
				joint_correct=j_ok,
				batch_size=batch_size,
				message_tokens=msg_tokens_log,
				tf_ratio=tf_ratio,
			)

			if step % 40 == 0:
				msg_str = f" msg={msg_tokens_log}" if msg_tokens_log else ""
				print(
					f"  step {step:3d} | τ={tau:.3f} TF={tf_ratio:.0%} | loss={loss.item():.3f} | "
					f"({sample_c},{sample_e})→({pred_c_name},{pred_e_name}){msg_str}"
				)

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
			epoch=epoch + 1,
			loss_avg=avg_loss,
			acc_concept=acc_c,
			acc_emotion=acc_e,
			acc_joint=acc_j,
			fitness=fitness.tolist(),
			worst_agent=int(np.argmin(fitness)),
			parent_a=-1,
			parent_b=-1,
		)

		# Evolution (Phase 2+)
		if epoch >= nursery_epochs:
			worst_idx = int(np.argmin(fitness))
			best_indices = np.argsort(fitness)[-2:]
			print(f"[Evolución] Agent_{worst_idx} → hijo SVD de A{best_indices[1]} × A{best_indices[0]}")
			svd_crossover(agents[best_indices[1]], agents[best_indices[0]], agents[worst_idx])
			optimizers[worst_idx] = torch.optim.AdamW(agents[worst_idx].parameters(), lr=lr)

		# Checkpointing
		if (epoch + 1) % checkpoint_every == 0:
			save_checkpoint(agents, epoch + 1, "005")

	# ── Final checkpoint ──
	save_checkpoint(agents, epochs, "005")

	# ── Log confusion matrices ──
	logger.log_event(
		"confusion_concept",
		{
			"labels": CONCEPTS,
			"matrix": concept_confusion.tolist(),
		},
	)
	logger.log_event(
		"confusion_emotion",
		{
			"labels": EMOTIONS,
			"matrix": emotion_confusion.tolist(),
		},
	)

	logger.close()

	# ── Print confusion analysis ──
	print("\n" + "=" * 60)
	print("📊 ANÁLISIS DE CONFUSIÓN (épocas de autonomía)")
	print("=" * 60)

	print("\n🎯 Top-5 confusiones conceptuales:")
	np.fill_diagonal(concept_confusion, 0)
	flat_idx = np.argsort(concept_confusion.ravel())[::-1][:5]
	for idx in flat_idx:
		r, c = divmod(idx, num_concepts)
		count = concept_confusion[r, c]
		if count > 0:
			print(f"  {CONCEPTS[r]:12s} → {CONCEPTS[c]:12s}  ({count} veces)")

	print("\n😢 Top-5 confusiones emocionales:")
	np.fill_diagonal(emotion_confusion, 0)
	flat_idx = np.argsort(emotion_confusion.ravel())[::-1][:5]
	for idx in flat_idx:
		r, c = divmod(idx, num_emotions)
		count = emotion_confusion[r, c]
		if count > 0:
			print(f"  {EMOTIONS[r]:12s} → {EMOTIONS[c]:12s}  ({count} veces)")

	print("\n[Telemetry] storage/telemetry/EXP_005.jsonl")
	print("[Checkpoints] storage/checkpoints/EXP_005/")


if __name__ == "__main__":
	run_arena()
