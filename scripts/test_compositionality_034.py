"""
EXP_034 — Test de Composicionalidad Zero-Shot.

La pregunta: ¿puede el modelo entender una palabra NUEVA que nunca ha
visto durante el entrenamiento, solo porque su glifo (vector de trits)
comparte primos con palabras conocidas?

Si sí → los primos semánticos son realmente los "átomos del significado"
y el modelo ha aprendido a componer, no a memorizar.

Uso:
    python scripts/test_compositionality_034.py --model storage/experiments/TASTING_034_D_first_only/best_agent.pt
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.bitnet.glyph_vocabulary import (
	EMOTION_INDEX,
	GLYPH_TABLE,
	N_EMOTIONS,
	WORD_NAMES,
	_make_glyph,
)
from src.bitnet.modeling_bitnet import BitNet4LayerModel

# ═══════════════════════════════════════════════════════════════════
# PALABRAS NUEVAS (nunca vistas en entrenamiento)
# Definidas solo por sus trits — el modelo debe deducir su significado
# ═══════════════════════════════════════════════════════════════════

NEW_WORDS = {
	# "trampa": un artefacto para cazar — debería comportarse como caza/piedra
	"trampa": _make_glyph(
		algo=1, cosa=1, hacer=1, malo=1, pequeño=1,
		tocar=1, ver=-1,  # oculta
	),

	# "fruta": comida que viene de árbol — debería comportarse como comida
	"fruta": _make_glyph(
		algo=1, bueno=1, querer=1, vivir=1, tocar=1,
		cuerpo=1, arriba=1, pequeño=1, ver=1,  # como comida + árbol
	),

	# "manada": grupo grande de depredadores — debería ser peligro
	"manada": _make_glyph(
		gente=1, alguien=1, malo=1, grande=1, mucho=1,
		mover=1, ver=1, morir=1,  # como depredador + grupo
	),

	# "lago": como río pero fijo — debería dar agua
	"lago": _make_glyph(
		algo=1, agua_prima=1, grande=1, ver=1, bueno=1,
		vivir=1, abajo=1, mover=-1,  # como río pero sin mover
	),

	# "luna": como sol pero nocturna — debería asociarse a noche
	"luna": _make_glyph(
		algo=1, grande=1, ver=1, arriba=1, lejos=1,
		luz=1, oscuro=1, frío=1,  # como sol pero fría y nocturna
	),
}


def cosine_similarity_to_vocabulary(new_glyph, glyph_table, word_names):
	"""Muestra las palabras más similares a un glifo nuevo."""
	new_f = new_glyph.astype(float)
	norm_new = np.linalg.norm(new_f)
	if norm_new == 0:
		return []

	sims = []
	for i in range(len(word_names)):
		existing = glyph_table[i].astype(float)
		norm_ex = np.linalg.norm(existing)
		if norm_ex > 0:
			cos = np.dot(new_f, existing) / (norm_new * norm_ex)
			sims.append((cos, word_names[i]))

	sims.sort(reverse=True)
	return sims[:5]


def test_zero_shot_inference(model, device, new_words, n_steps=2):
	"""
	Testea si el modelo puede hacer inferencia con palabras nuevas.

	Para cada palabra nueva:
	1. Añadimos temporalmente su glifo a la tabla
	2. La usamos como input
	3. Vemos qué predice el modelo (debería predecir palabras conocidas coherentes)
	"""
	model.eval()
	print(f"\n{'═'*70}")
	print("  TEST DE COMPOSICIONALIDAD ZERO-SHOT")
	print(f"{'═'*70}")

	for new_name, new_glyph in new_words.items():
		print(f"\n  ── {new_name} ──")

		# Similitud con vocabulario existente
		sims = cosine_similarity_to_vocabulary(new_glyph, GLYPH_TABLE, WORD_NAMES)
		print("  Vecinos (coseno de trits):")
		for cos, name in sims:
			print(f"    {name:<15} cos={cos:.3f}")

		# Crear tabla extendida temporalmente
		extended_table = np.vstack([GLYPH_TABLE, new_glyph.reshape(1, -1)])
		new_idx = len(WORD_NAMES)  # índice temporal

		# Construir modelo temporal con tabla extendida
		# En vez de eso, computamos el embedding directamente
		with torch.no_grad():
			# Embedding de la palabra nueva
			glyph_tensor = torch.from_numpy(new_glyph).float().to(device)
			new_embed = glyph_tensor @ model.glyph_embedding.prime_embeddings  # (hidden_dim,)

			# Input: repetir el embedding como si fuera un token
			# Construimos manualmente el hidden state
			h = new_embed.unsqueeze(0).unsqueeze(0).expand(1, 4, -1)  # (1, 4, hidden_dim)

			# Pasar por core layers
			for layer in model.core_layers:
				h = layer(h)
			h = model.norm(h)

			# Decodificar
			logits = model.glyph_embedding.decode_logits(h)  # (1, 4, vocab_size)
			probs = F.softmax(logits[0, 0, :], dim=-1)

			top5 = torch.topk(probs, 5)
			print("  Predicción sin emoción (top 5):")
			for prob, idx in zip(top5.values, top5.indices):
				print(f"    {WORD_NAMES[idx.item()]:<15} p={prob.item():.3f}")

			# Con emociones
			if model.emotion_embeddings is not None:
				print("  Con emociones:")
				for emo_name in ["miedo", "alegría", "hambre"]:
					emo_id = EMOTION_INDEX[emo_name]
					emo_vec = model.emotion_embeddings(
						torch.tensor([emo_id], device=device)
					)
					emo_proj = model.emotion_proj(emo_vec)  # (1, hidden_dim)

					# Inyectar emoción (first_only: solo al inicio)
					h_emo = new_embed.unsqueeze(0).unsqueeze(0).expand(1, 4, -1).clone()
					h_emo[:, 0, :] = h_emo[:, 0, :] + emo_proj.squeeze(0)

					for layer in model.core_layers:
						h_emo = layer(h_emo)
					h_emo = model.norm(h_emo)

					logits_emo = model.glyph_embedding.decode_logits(h_emo)
					pred_idx = logits_emo[0, 0, :].argmax().item()
					pred_prob = F.softmax(logits_emo[0, 0, :], dim=-1)[pred_idx].item()
					print(f"    + {emo_name:<10} → {WORD_NAMES[pred_idx]:<15} p={pred_prob:.3f}")


def main():
	parser = argparse.ArgumentParser(description="EXP_034: Test de composicionalidad")
	parser.add_argument("--model", type=str, required=True, help="Path to best_agent.pt")
	args = parser.parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Cargar modelo
	model = BitNet4LayerModel(
		use_glyphs=True,
		hidden_dim=256,
		num_layers=3,
		use_pos_embedding=True,
		max_resonance_steps=5,
		n_emotions=N_EMOTIONS,
		emotion_dim=64,
		emotion_mode="first_only",
	).to(device)

	state_dict = torch.load(args.model, map_location=device, weights_only=True)
	model.load_state_dict(state_dict)
	print(f"✅ Modelo cargado: {args.model}")

	test_zero_shot_inference(model, device, NEW_WORDS, n_steps=2)

	print(f"\n{'═'*70}")
	print("  FIN DEL TEST DE COMPOSICIONALIDAD")
	print(f"{'═'*70}")


if __name__ == "__main__":
	main()
