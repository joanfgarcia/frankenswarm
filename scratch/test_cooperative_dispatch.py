import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np

# Añadir src al path
base_dir = "/home/joan/Documents/IA/frankenswarm"
sys.path.append(os.path.join(base_dir, "src"))

from bitnet.modeling_bitnet import BitNet4LayerModel
from bitnet.glyph_vocabulary import N_EMOTIONS, EMOTION_INDEX

def run_cooperative_dispatch_demo():
	print("═══ 📡 Demo: Flujo Cooperativo Gritar/Escuchar con Despacho a Bit ═══")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"[Device]: {device}")

	# Paths a los checkpoints
	path_a = "storage/checkpoints/sovereign_school/model_milestone_2_years.pt"
	path_b = "storage/checkpoints/sovereign_school/model_milestone_3_years.pt"

	# 1. Cargar vocabulario de 15k
	expanded_glyphs_path = os.path.join(base_dir, "configs", "expanded_glyphs.json")
	with open(expanded_glyphs_path, encoding="utf-8") as f:
		vocab_data = json.load(f)
		words = vocab_data["words"]
		glyphs = np.array(vocab_data["glyphs"], dtype=np.float32)
	word_to_idx = {w: i for i, w in enumerate(words)}

	# 2. Inicializar Model A (Tronco Local / Nico y Sofy, dim 256)
	print("Inicializando Modelos de Acción (Nico y Sofy - 256-dim)...")
	model_a = BitNet4LayerModel(
		use_glyphs=True,
		glyph_table=glyphs,
		hidden_dim=256,
		num_layers=6,
		use_pos_embedding=True,
		n_emotions=N_EMOTIONS,
		emotion_dim=64,
		emotion_mode="first_only",
	).to(device)
	sd_a = torch.load(path_a, map_location=device, weights_only=True)
	model_a.load_state_dict(sd_a, strict=False)

	# 3. Inicializar Model B (Bit / Experto Cognitivo, dim 384)
	print("Inicializando Modelo Cognitivo (Bit - 384-dim)...")
	model_b = BitNet4LayerModel(
		use_glyphs=True,
		glyph_table=glyphs,
		hidden_dim=384,
		num_layers=6,
		use_pos_embedding=True,
		n_emotions=N_EMOTIONS,
		emotion_dim=64,
		emotion_mode="first_only",
	).to(device)
	sd_b = torch.load(path_b, map_location=device, weights_only=True)
	model_b.load_state_dict(sd_b, strict=False)

	model_a.eval()
	model_b.eval()

	# ══════════════════════════════════════════════════════════════════
	# PASO 1: Nico (Monito A) tiene hambre extrema en el Bosque y "grita"
	# ══════════════════════════════════════════════════════════════════
	print("\n--- PASO 1: Nico siente hambre en el Bosque y emite un grito ---")
	# Input Nico: [bosque, comida, hambre, padding]
	nico_words = ["bosque", "comida", "hambre", "agua"]
	nico_ids = [word_to_idx[w] for w in nico_words]
	x_nico = torch.tensor([nico_ids], dtype=torch.long, device=device)
	emo_nico = torch.tensor([EMOTION_INDEX["hambre"]], dtype=torch.long, device=device)

	with torch.no_grad():
		logits_nico = model_a(x_nico)
		# Nico decodifica su hidden state final para ver qué palabra conceptual "grita"
		shout_idx = logits_nico.argmax(dim=-1)[0, 2].item() # Usamos token en pos 2
		shout_word = words[shout_idx]
		print(f" Nico grita al entorno la señal semántica: '{shout_word}' (token idx: {shout_idx})")

	# ══════════════════════════════════════════════════════════════════
	# PASO 2: Sofy (Monito B) escucha la señal en la Cueva
	# ══════════════════════════════════════════════════════════════════
	print("\n--- PASO 2: Sofy recibe el grito de Nico y lo procesa de forma híbrida ---")
	# Input Sofy: [cueva, received_location="bosque", received_concept="hambre", padding]
	# Mapeamos la percepción de Sofy incluyendo el grito de Nico
	sofy_words = ["cueva", "bosque", shout_word if shout_word in word_to_idx else "peligro", "yo_palabra"]
	sofy_ids = [word_to_idx[w] for w in sofy_words]
	x_sofy = torch.tensor([sofy_ids], dtype=torch.long, device=device)
	emo_sofy = torch.tensor([EMOTION_INDEX["miedo"]], dtype=torch.long, device=device) # Miedo / Alerta al oír un grito

	print(f" Entrada de Sofy al escuchar a Nico: {sofy_words} -> IDs: {sofy_ids}")

	# 5. Ejecutar forward de Sofy con despacho a Bit para toma de decisiones
	with torch.no_grad():
		# 5.1 Early layers en Sofy (Model A, Capas 0-2)
		h = model_a._embed_input(x_sofy)
		
		# Inyección de emoción (Miedo)
		if model_a.emotion_embeddings is not None:
			emo_emb_a = model_a.emotion_embeddings(emo_sofy)
			h = h + model_a.emotion_proj(emo_emb_a).unsqueeze(1)

		for i in range(3):
			h = model_a.core_layers[i](h)
		h = model_a.norm(h)
		print(f" Hidden state en Sofy (Model A capa 3): {h.shape}")

		# 5.2 DESPACHO: Proyectar al vocabulario común de 15k y re-embeber en Bit (Model B, 384-dim)
		logits_sofy_a = model_a._decode_hidden(h)
		probs_sofy_a = F.softmax(logits_sofy_a / 1.0, dim=-1)

		word_embeds_b = model_b.glyph_embedding.get_word_embeddings()
		h_b = torch.matmul(probs_sofy_a, word_embeds_b) # (1, 4, 384)
		print(f" Despachado a Bit (Model B, 384-dim): {h_b.shape}")

		# 5.3 COMPUTACIÓN DEL EXPERTO: Bit procesa en capas 3, 4
		for i in range(3, 5):
			h_b = model_b.core_layers[i](h_b)
		h_b = model_b.norm(h_b)

		# 5.4 RETORNO: Decodificar en Bit a vocabulario y re-embeber en Sofy (Model A, 256-dim)
		logits_sofy_b = model_b._decode_hidden(h_b)
		probs_sofy_b = F.softmax(logits_sofy_b / 1.0, dim=-1)

		# Qué concepto decide Bit que debe ser la prioridad del plan
		plan_idx = logits_sofy_b.argmax(dim=-1)[0, 2].item()
		print(f" 🧠 Bit (Modelo Cognitivo) decide conceptualmente: '{words[plan_idx]}' (plan idx: {plan_idx})")

		word_embeds_a = model_a.glyph_embedding.get_word_embeddings()
		h_a_back = torch.matmul(probs_sofy_b, word_embeds_a) # (1, 4, 256)
		print(f" Retornado a Sofy (Model A, 256-dim): {h_a_back.shape}")

		# 5.5 Capas finales en Sofy (Model A, Capas 3-5)
		for i in range(3, 6):
			h_a_back = model_a.core_layers[i](h_a_back)
		h_a_back = model_a.norm(h_a_back)

		# 5.6 Cabeza de Acción de Sofy toma la decisión física final
		h_action = h_a_back[:, 2, :] # Extraemos token pos 2
		action_logits = model_a.action_head(h_action)
		action_idx = action_logits.argmax(dim=-1).item()
		
		action_names = ["comer", "beber", "dormir", "mover", "ver", "piedra"]
		action_selected = action_names[action_idx]
		print(f" Sofy ejecuta la acción física final: '{action_selected}'")

	print("\n✅ Flujo cooperativo híbrido completado con éxito.")

if __name__ == "__main__":
	run_cooperative_dispatch_demo()
