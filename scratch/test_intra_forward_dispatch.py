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
from bitnet.glyph_vocabulary import N_EMOTIONS

def run_dispatch_test():
	print("═══ 🔬 Test: Despacho de Tensores Intra-Forward (Tensors as API) ═══")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"[Device]: {device}")

	# Paths a los checkpoints
	path_a = "storage/checkpoints/sovereign_school/model_milestone_2_years.pt"
	path_b = "storage/checkpoints/sovereign_school/model_milestone_3_years.pt"

	# 1. Cargar vocabulario para verificar compatibilidad de dimensiones
	expanded_glyphs_path = os.path.join(base_dir, "configs", "expanded_glyphs.json")
	with open(expanded_glyphs_path, encoding="utf-8") as f:
		vocab_data = json.load(f)
		words = vocab_data["words"]
		glyphs = np.array(vocab_data["glyphs"], dtype=np.float32)

	print(f"Vocab size: {len(words)}, Glyphs shape: {glyphs.shape}")

	# 2. Inicializar Model A (hidden_dim=256, layers=6)
	print("Inicializando Modelo A (256-dim)...")
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
	print("Modelo A cargado correctamente.")

	# 3. Inicializar Model B (hidden_dim=384, layers=6)
	print("Inicializando Modelo B (384-dim)...")
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
	print("Modelo B cargado correctamente.")

	# Poner ambos en modo eval
	model_a.eval()
	model_b.eval()

	# 4. Preparar entrada de prueba
	# Frase simple: "tengo hambre cueva"
	input_words = ["yo_palabra", "hambre", "cueva", "dormir"]
	input_ids = []
	for w in input_words:
		if w in words:
			input_ids.append(words.index(w))
		else:
			input_ids.append(words.index("<unk>"))

	x = torch.tensor([input_ids], dtype=torch.long, device=device) # (1, 4)
	emotion_id = torch.tensor([words.index("yo_palabra")], dtype=torch.long, device=device) # emoción dummy

	print(f"Palabras de entrada: {input_words} -> IDs: {input_ids}")

	# 5. Ejecutar Forward Pass Híbrido con Despacho de Tensores
	with torch.no_grad():
		# --- TRONCO LOCAL: Modelo A (0 -> Capa 3) ---
		# 5.1 Embeddings
		h = model_a._embed_input(x)
		
		# 5.2 Capas tempranas (capas 0, 1, 2)
		for i in range(3):
			h = model_a.core_layers[i](h)
		h = model_a.norm(h)
		print(f"Estado en A (capa 3): {h.shape}")

		# 5.3 Proyectar al vocabulario base de 15,005 palabras
		logits_a = model_a._decode_hidden(h) # (batch, seq, 15005)
		probs_a = F.softmax(logits_a / 1.0, dim=-1)
		print(f"Distribución semántica probs_a: {probs_a.shape}")

		# 5.4 Re-embeber en el espacio oculto del Modelo B (384-dim)
		word_embeds_b = model_b.glyph_embedding.get_word_embeddings()
		h_b = torch.matmul(probs_a, word_embeds_b) # (batch, seq, 384)
		print(f"Re-embebido en B (384-dim): {h_b.shape}")

		# --- COGNICIÓN DISTRIBUIDA: Modelo B (capas 3, 4) ---
		for i in range(3, 5):
			h_b = model_b.core_layers[i](h_b)
		h_b = model_b.norm(h_b)
		print(f"Estado en B (capa 5): {h_b.shape}")

		# 5.5 Decodificar a logits de B y obtener distribución
		logits_b = model_b._decode_hidden(h_b)
		probs_b = F.softmax(logits_b / 1.0, dim=-1)
		print(f"Distribución semántica probs_b: {probs_b.shape}")

		# 5.6 Re-embeber en el espacio oculto del Modelo A (256-dim)
		word_embeds_a = model_a.glyph_embedding.get_word_embeddings()
		h_a_back = torch.matmul(probs_b, word_embeds_a) # (batch, seq, 256)
		print(f"Re-embebido de vuelta en A (256-dim): {h_a_back.shape}")

		# --- TRONCO LOCAL: Modelo A (Capa 3 -> Fin) ---
		for i in range(3, 6):
			h_a_back = model_a.core_layers[i](h_a_back)
		h_a_back = model_a.norm(h_a_back)

		# Decodificar logits finales
		final_logits = model_a._decode_hidden(h_a_back)
		print(f"Logits finales: {final_logits.shape}")

		# Ver qué palabras se predicen
		predicted_ids = final_logits.argmax(dim=-1)[0].tolist()
		predicted_words = [words[idx] for idx in predicted_ids]
		print(f"Palabras predichas por el modelo híbrido: {predicted_words}")

		# Comparar con el Modelo A nativo puro
		logits_a_pure = model_a(x)
		pure_ids = logits_a_pure.argmax(dim=-1)[0].tolist()
		pure_words = [words[idx] for idx in pure_ids]
		print(f"Palabras predichas por el Modelo A puro: {pure_words}")

	print("✅ Test de despacho de tensores completado exitosamente.")

if __name__ == "__main__":
	run_dispatch_test()
