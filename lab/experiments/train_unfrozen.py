"""
train_unfrozen.py — Exp 005b: Unfrozen Embeddings Arena

Same as Exp 005 but with vocab_embeddings converted from frozen buffer
to trainable Parameter. This answers Sonnet's question:
"How much of the proto-language structure is genuinely emergent
vs inherited from fastembed geometry?"

If confusions (fuego↔sol) resolve → they were fastembed artifacts.
If confusions persist → they are structural limitations of the model.
If the positional grammar [C,C,E] changes → it was also fastembed-dependent.
If it persists → it's genuinely emergent.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.bitnet.modeling_bitnet import BitNet4LayerModel
from src.bitnet.telemetry import ExperimentLogger

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


def unfreeze_embeddings(model: BitNet4LayerModel):
	"""Convert vocab_embeddings from buffer to trainable Parameter."""
	embed_data = model.vocab_embeddings.data.clone()
	del model.vocab_embeddings
	model.vocab_embeddings = nn.Parameter(embed_data, requires_grad=True)


class DualHeadAgent(nn.Module):
	def __init__(self, base_model: BitNet4LayerModel, num_concepts: int, num_emotions: int):
		super().__init__()
		self.base = base_model
		self.hidden_dim = base_model.hidden_dim
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

	def speak(self, x, tau=1.0, hard=True):
		h = self._encode(x)
		concept_proj = self.base.outbound_proj(h)
		logits = torch.matmul(concept_proj, self.base.vocab_embeddings.T)
		return F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)

	def listen(self, x):
		h = self._encode(x)
		return self.concept_head(h[:, 1, :]), self.emotion_head(h[:, 2, :])


def build_micro_embeddings():
	from fastembed import TextEmbedding

	model = TextEmbedding()
	embeddings = list(model.embed(MICRO_VOCAB))
	return np.array([e if isinstance(e, np.ndarray) else np.array(e) for e in embeddings], dtype=np.float32)


def svd_crossover(parent_a, parent_b, child, alpha=0.5, sigma=0.01):
	with torch.no_grad():
		pa_params = dict(parent_a.named_parameters())
		pb_params = dict(parent_b.named_parameters())
		for name, param in child.named_parameters():
			if not param.requires_grad:
				continue
			p_a = pa_params[name].data
			p_b = pb_params[name].data
			if p_a.ndim == 2:
				w_avg = alpha * p_a + (1 - alpha) * p_b
				try:
					u, s, vh = torch.linalg.svd(w_avg, full_matrices=False)
					if s[0] / (s[-1] + 1e-10) > 1e4:
						param.copy_(w_avg)
						continue
					param.copy_(u @ torch.diag((s + torch.randn_like(s) * sigma).clamp_(min=0.0)) @ vh)
				except Exception:
					param.copy_(w_avg)
			else:
				param.copy_(alpha * p_a + (1 - alpha) * p_b + torch.randn_like(p_a) * sigma)


def save_checkpoint(agents, epoch, experiment_id):
	base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	ckpt_dir = os.path.join(base, "storage", "checkpoints", f"EXP_{experiment_id}")
	os.makedirs(ckpt_dir, exist_ok=True)
	for i, agent in enumerate(agents):
		torch.save(agent.state_dict(), os.path.join(ckpt_dir, f"agent_{i}_epoch_{epoch:03d}.pt"))
	# Also save the embeddings separately for comparison
	torch.save(agents[0].base.vocab_embeddings.data.cpu(), os.path.join(ckpt_dir, f"embeddings_epoch_{epoch:03d}.pt"))
	print(f"  [Checkpoint] Guardado (época {epoch})")


def run_arena():
	print("=== 🔓 Arena PopuLoRA — Unfrozen Embeddings (Exp. 005b) ===")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	vocab_embeddings = build_micro_embeddings()
	num_concepts, num_emotions = len(CONCEPTS), len(EMOTIONS)
	vocab_size = len(MICRO_VOCAB)

	# Save original embeddings for later comparison
	original_embeddings = torch.from_numpy(vocab_embeddings.copy()).float()

	pop_size = 4
	nursery_epochs, transition_epochs, autonomy_epochs = 5, 15, 40
	epochs = nursery_epochs + transition_epochs + autonomy_epochs
	steps_per_epoch, batch_size, lr = 200, 32, 1e-3
	tau_start, tau_min, beta_emotion = 1.0, 0.3, 0.5
	total_steps = epochs * steps_per_epoch

	params = {
		"experiment": "005b_unfrozen_embeddings",
		"frozen_embeddings": False,
		"vocab_size": vocab_size,
		"hidden_dim": 256,
		"num_layers": 4,
		"pop_size": pop_size,
		"epochs": epochs,
		"steps_per_epoch": steps_per_epoch,
		"batch_size": batch_size,
		"lr": lr,
		"tau_min": tau_min,
		"beta_emotion": beta_emotion,
		"nursery_epochs": nursery_epochs,
		"transition_epochs": transition_epochs,
	}
	logger = ExperimentLogger("005b", params)

	# Build agents with UNFROZEN embeddings
	agents = []
	for _ in range(pop_size):
		base = BitNet4LayerModel(vocab_embeddings=vocab_embeddings, hidden_dim=256, num_layers=4).to(device)
		unfreeze_embeddings(base)  # ← THE KEY CHANGE
		agent = DualHeadAgent(base, num_concepts, num_emotions).to(device)
		agents.append(agent)

	optimizers = [torch.optim.AdamW(a.parameters(), lr=lr) for a in agents]
	fitness = np.zeros(pop_size)
	current_step = 0

	concept_confusion = np.zeros((num_concepts, num_concepts), dtype=np.int64)
	emotion_confusion = np.zeros((num_emotions, num_emotions), dtype=np.int64)

	for epoch in range(epochs):
		if epoch < nursery_epochs:
			tf_ratio, phase_name = 1.0, "🍼 Guardería"
		elif epoch < nursery_epochs + transition_epochs:
			tf_ratio = 1.0 - (epoch - nursery_epochs) / transition_epochs
			phase_name = "🎮 Recreo"
		else:
			tf_ratio, phase_name = 0.0, "🦅 Autonomía"

		print(f"\n--- Época {epoch + 1}/{epochs} [{phase_name}] TF={tf_ratio:.0%} ---")
		epoch_losses, concept_correct, emotion_correct, joint_correct, epoch_total = [], 0, 0, 0, 0
		interactions = np.zeros((pop_size, pop_size))
		successes = np.zeros((pop_size, pop_size))

		for step in range(steps_per_epoch):
			tau = max(tau_min, tau_start * (1.0 - current_step / total_steps))
			concept_indices = np.random.randint(0, num_concepts, size=(batch_size,))
			emotion_indices = np.random.randint(0, num_emotions, size=(batch_size,))
			concept_targets = torch.from_numpy(concept_indices).long().to(device)
			emotion_targets = torch.from_numpy(emotion_indices).long().to(device)
			emotion_token_ids = torch.from_numpy(emotion_indices + num_concepts).long().to(device)

			idx_s, idx_l = np.random.randint(0, pop_size), np.random.randint(0, pop_size)
			while idx_l == idx_s:
				idx_l = np.random.randint(0, pop_size)

			speaker, listener = agents[idx_s], agents[idx_l]
			optimizers[idx_s].zero_grad()
			optimizers[idx_l].zero_grad()

			speaker_input = torch.zeros((batch_size, 3), dtype=torch.long, device=device)
			speaker_input[:, 0] = concept_targets
			speaker_input[:, 1] = emotion_token_ids

			use_teacher = torch.rand(batch_size, device=device) < tf_ratio
			speaker_message = None
			if use_teacher.all():
				message_input = speaker_input
			elif use_teacher.any():
				speaker_message = speaker.speak(speaker_input, tau=tau)
				teacher_onehot = F.one_hot(speaker_input, num_classes=vocab_size).float()
				mask = use_teacher.view(-1, 1, 1).float()
				message_input = mask * teacher_onehot + (1 - mask) * speaker_message
			else:
				speaker_message = speaker.speak(speaker_input, tau=tau)
				message_input = speaker_message

			pred_c, pred_e = listener.listen(message_input)
			loss = F.cross_entropy(pred_c, concept_targets) + beta_emotion * F.cross_entropy(pred_e, emotion_targets)
			loss_c = F.cross_entropy(pred_c, concept_targets).item()
			loss_e = F.cross_entropy(pred_e, emotion_targets).item()
			loss.backward()
			optimizers[idx_s].step()
			optimizers[idx_l].step()

			preds_c = torch.argmax(pred_c, dim=-1)
			preds_e = torch.argmax(pred_e, dim=-1)
			c_ok = (preds_c == concept_targets).sum().item()
			e_ok = (preds_e == emotion_targets).sum().item()
			j_ok = ((preds_c == concept_targets) & (preds_e == emotion_targets)).sum().item()
			concept_correct += c_ok
			emotion_correct += e_ok
			joint_correct += j_ok
			epoch_total += batch_size
			epoch_losses.append(loss.item())
			interactions[idx_s, idx_l] += batch_size
			successes[idx_s, idx_l] += j_ok
			current_step += 1

			if tf_ratio == 0.0:
				for i in range(batch_size):
					concept_confusion[concept_indices[i], preds_c[i].item()] += 1
					emotion_confusion[emotion_indices[i], preds_e[i].item()] += 1

			msg_tokens_log = None
			if speaker_message is not None and tf_ratio < 1.0:
				msg_argmax = torch.argmax(speaker_message[0], dim=-1)
				msg_tokens_log = [MICRO_VOCAB[t.item()] for t in msg_argmax]

			sample_c, sample_e = CONCEPTS[concept_indices[0]], EMOTIONS[emotion_indices[0]]
			pred_c_name = CONCEPTS[preds_c[0].item()] if preds_c[0].item() < num_concepts else "?"
			pred_e_name = EMOTIONS[preds_e[0].item()] if preds_e[0].item() < num_emotions else "?"

			logger.log_step(
				epoch=epoch + 1,
				step=step,
				loss=loss.item(),
				loss_concept=loss_c,
				loss_emotion=loss_e,
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

		for i in range(pop_size):
			sent = interactions[i, :].sum() + interactions[:, i].sum()
			fitness[i] = (successes[i, :].sum() + successes[:, i].sum()) / (sent + 1e-10)

		avg_loss = np.mean(epoch_losses)
		acc_c = concept_correct / epoch_total * 100
		acc_e = emotion_correct / epoch_total * 100
		acc_j = joint_correct / epoch_total * 100
		print(f"Loss: {avg_loss:.4f} | Concepto: {acc_c:.2f}% | Emoción: {acc_e:.2f}% | Conjunta: {acc_j:.2f}%")

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

		if epoch >= nursery_epochs:
			worst = int(np.argmin(fitness))
			best = np.argsort(fitness)[-2:]
			svd_crossover(agents[best[1]], agents[best[0]], agents[worst])
			optimizers[worst] = torch.optim.AdamW(agents[worst].parameters(), lr=lr)

		if (epoch + 1) % 10 == 0:
			save_checkpoint(agents, epoch + 1, "005b")

	save_checkpoint(agents, epochs, "005b")

	# ── Embedding drift analysis ──
	print("\n" + "=" * 60)
	print("📐 EMBEDDING DRIFT ANALYSIS")
	print("   ¿Cuánto se han movido los embeddings de fastembed?")
	print("=" * 60)

	final_embeddings = agents[0].base.vocab_embeddings.data.cpu()
	drift = torch.norm(final_embeddings - original_embeddings, dim=1)
	for i, token in enumerate(MICRO_VOCAB):
		print(f"  {token:12s} drift={drift[i]:.4f}")

	# cosine similarity between fuego and sol before/after
	def cos_sim(a, b):
		return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

	print(f"\n  fuego↔sol  ANTES: {cos_sim(original_embeddings[5], original_embeddings[8]):.4f}")
	print(f"  fuego↔sol  DESPUÉS: {cos_sim(final_embeddings[5], final_embeddings[8]):.4f}")
	print(f"  agua↔gato  ANTES: {cos_sim(original_embeddings[4], original_embeddings[0]):.4f}")
	print(f"  agua↔gato  DESPUÉS: {cos_sim(final_embeddings[4], final_embeddings[0]):.4f}")

	# Confusion analysis
	print("\n🎯 Top-5 confusiones conceptuales:")
	np.fill_diagonal(concept_confusion, 0)
	for idx in np.argsort(concept_confusion.ravel())[::-1][:5]:
		r, c = divmod(idx, num_concepts)
		if concept_confusion[r, c] > 0:
			print(f"  {CONCEPTS[r]:12s} → {CONCEPTS[c]:12s}  ({concept_confusion[r, c]} veces)")

	print("\n😢 Top-5 confusiones emocionales:")
	np.fill_diagonal(emotion_confusion, 0)
	for idx in np.argsort(emotion_confusion.ravel())[::-1][:5]:
		r, c = divmod(idx, num_emotions)
		if emotion_confusion[r, c] > 0:
			print(f"  {EMOTIONS[r]:12s} → {EMOTIONS[c]:12s}  ({emotion_confusion[r, c]} veces)")

	logger.log_event(
		"embedding_drift",
		{
			"tokens": MICRO_VOCAB,
			"drift": drift.tolist(),
			"fuego_sol_before": cos_sim(original_embeddings[5], original_embeddings[8]),
			"fuego_sol_after": cos_sim(final_embeddings[5], final_embeddings[8]),
		},
	)
	logger.log_event("confusion_concept", {"labels": CONCEPTS, "matrix": concept_confusion.tolist()})
	logger.log_event("confusion_emotion", {"labels": EMOTIONS, "matrix": emotion_confusion.tolist()})
	logger.close()
	print("\n[Telemetry] storage/telemetry/EXP_005b.jsonl")


if __name__ == "__main__":
	run_arena()
