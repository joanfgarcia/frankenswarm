"""
train_diagnostic.py — Exp 004a: Teacher Forcing Diagnostic

Purpose: Isolate whether the listener architecture can learn AT ALL
by bypassing the Gumbel-Softmax communication channel entirely.

Instead of: speaker(target) → Gumbel-Softmax → message → listener(message) → prediction
We do:      target → one-hot encoding → listener(one_hot) → prediction

If the listener learns → the problem is the speaker/channel (Gumbel-Softmax)
If the listener doesn't learn → the problem is the listener architecture itself
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.bitnet.modeling_bitnet import BitNet4LayerModel
from src.bitnet.telemetry import ExperimentLogger

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


class DiagnosticListener(nn.Module):
	"""Listener with dual heads and positional encoding — same as Exp 003."""

	def __init__(self, base_model: BitNet4LayerModel, num_concepts: int, num_emotions: int):
		super().__init__()
		self.base = base_model
		self.hidden_dim = base_model.hidden_dim
		self.pos_embedding = nn.Embedding(3, self.hidden_dim)
		self.concept_head = nn.Linear(self.hidden_dim, num_concepts)
		self.emotion_head = nn.Linear(self.hidden_dim, num_emotions)

	def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		embeds = F.embedding(x, self.base.vocab_embeddings) if x.ndim == 2 else torch.matmul(x, self.base.vocab_embeddings)

		h = self.base.inbound_proj(embeds)
		positions = torch.arange(3, device=h.device)
		h = h + self.pos_embedding(positions).unsqueeze(0)

		for layer in self.base.core_layers:
			h = layer(h)
		h = self.base.norm(h)

		concept_logits = self.concept_head(h[:, 1, :])
		emotion_logits = self.emotion_head(h[:, 2, :])
		return concept_logits, emotion_logits


def build_micro_embeddings() -> np.ndarray:
	from fastembed import TextEmbedding

	model = TextEmbedding()
	embeddings = list(model.embed(MICRO_VOCAB))
	return np.array([e if isinstance(e, np.ndarray) else np.array(e) for e in embeddings], dtype=np.float32)


def run_diagnostic():
	print("=== 🔬 Exp 004a: Teacher Forcing Diagnostic ===")
	print("=== Bypassing Gumbel-Softmax — feeding ground truth to listener ===")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"[Device]: {device}")

	vocab_embeddings = build_micro_embeddings()
	num_concepts = len(CONCEPTS)
	num_emotions = len(EMOTIONS)
	vocab_size = len(MICRO_VOCAB)

	# Single listener — no population, no evolution, no speaker
	base_model = BitNet4LayerModel(vocab_embeddings=vocab_embeddings, hidden_dim=256, num_layers=4).to(device)
	listener = DiagnosticListener(base_model, num_concepts, num_emotions).to(device)
	optimizer = torch.optim.AdamW(listener.parameters(), lr=1e-3)

	epochs = 20
	steps_per_epoch = 200
	batch_size = 32
	beta_emotion = 0.5

	params = {
		"experiment": "004a_teacher_forcing",
		"vocab_size": vocab_size,
		"num_concepts": num_concepts,
		"num_emotions": num_emotions,
		"hidden_dim": 256,
		"num_layers": 4,
		"epochs": epochs,
		"steps_per_epoch": steps_per_epoch,
		"batch_size": batch_size,
		"lr": 1e-3,
		"beta_emotion": beta_emotion,
		"gumbel_softmax": False,
		"teacher_forcing": True,
		"device": str(device),
	}
	logger = ExperimentLogger("004a", params)

	for epoch in range(epochs):
		print(f"\n--- Época {epoch + 1}/{epochs} ---")
		epoch_losses = []
		concept_correct = 0
		emotion_correct = 0
		joint_correct = 0
		epoch_total = 0

		for step in range(steps_per_epoch):
			concept_indices = np.random.randint(0, num_concepts, size=(batch_size,))
			emotion_indices = np.random.randint(0, num_emotions, size=(batch_size,))

			concept_targets = torch.from_numpy(concept_indices).long().to(device)
			emotion_targets = torch.from_numpy(emotion_indices).long().to(device)

			# ── TEACHER FORCING: build input directly from ground truth ──
			# Position 0: concept token ID, Position 1: emotion token ID, Position 2: padding (0)
			teacher_input = torch.zeros((batch_size, 3), dtype=torch.long, device=device)
			teacher_input[:, 0] = concept_targets  # concept in [0..14]
			teacher_input[:, 1] = emotion_targets + num_concepts  # emotion in [15..20]
			# Position 2 stays 0 (padding)

			optimizer.zero_grad()

			concept_logits, emotion_logits = listener(teacher_input)

			loss_concept = F.cross_entropy(concept_logits, concept_targets)
			loss_emotion = F.cross_entropy(emotion_logits, emotion_targets)
			loss = loss_concept + beta_emotion * loss_emotion
			loss.backward()
			optimizer.step()

			epoch_losses.append(loss.item())
			preds_c = torch.argmax(concept_logits, dim=-1)
			preds_e = torch.argmax(emotion_logits, dim=-1)

			c_ok = (preds_c == concept_targets).sum().item()
			e_ok = (preds_e == emotion_targets).sum().item()
			j_ok = ((preds_c == concept_targets) & (preds_e == emotion_targets)).sum().item()

			concept_correct += c_ok
			emotion_correct += e_ok
			joint_correct += j_ok
			epoch_total += batch_size

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
				tau=0.0,
				speaker_id=-1,
				listener_id=0,
				target_concept=sample_c,
				target_emotion=sample_e,
				pred_concept=pred_c_name,
				pred_emotion=pred_e_name,
				concept_correct=c_ok,
				emotion_correct=e_ok,
				joint_correct=j_ok,
				batch_size=batch_size,
			)

			if step % 40 == 0:
				print(
					f"  step {step:3d} | loss={loss.item():.3f} "
					f"(C:{loss_concept.item():.3f} E:{loss_emotion.item():.3f}) | "
					f"target=({sample_c},{sample_e}) → pred=({pred_c_name},{pred_e_name})"
				)

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
			fitness=[0.0],
			worst_agent=-1,
			parent_a=-1,
			parent_b=-1,
		)

		if acc_j >= 95.0:
			print("\n🔬 DIAGNÓSTICO POSITIVO: El listener PUEDE aprender. El problema es el canal.")
			logger.log_event("diagnostic_positive", {"acc_joint": acc_j})
			break

	logger.close()

	# Final verdict
	print("\n" + "=" * 60)
	if acc_j > 20:
		print("✅ VEREDICTO: El listener aprende con teacher forcing.")
		print("   → El cuello de botella es el canal Gumbel-Softmax / speaker.")
	else:
		print("❌ VEREDICTO: El listener NO aprende ni con teacher forcing.")
		print("   → El problema es la arquitectura del listener / BitNet.")
	print("=" * 60)
	print("\n[Telemetry] storage/telemetry/EXP_004a.jsonl")


if __name__ == "__main__":
	run_diagnostic()
