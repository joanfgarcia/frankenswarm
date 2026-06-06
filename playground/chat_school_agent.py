import argparse
import json
import os
import re

import numpy as np
import torch
import torch.nn.functional as F

from src.bitnet.dictionary_tool import SovereignDictionary
from src.bitnet.modeling_bitnet import BitNet4LayerModel


def tokenize(text: str, word_to_idx: dict) -> list[int]:
	words = re.findall(r'[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_<>\-]+', text.lower())
	return [word_to_idx.get(w, 1) for w in words]

def sample_next_token(logits, temperature=0.7, repetition_penalty=1.2, generated_tokens=None, word_to_idx=None):
	# Apply repetition penalty
	if generated_tokens and word_to_idx:
		for token in set(generated_tokens):
			logits[token] -= repetition_penalty
			
	# Filter special tokens from being generated if possible
	logits[word_to_idx.get("<pad>", 0)] = -1e9
	logits[word_to_idx.get("<unk>", 1)] = -1e9
	
	if temperature <= 0.0:
		return logits.argmax(dim=-1).item()
		
	logits = logits / max(temperature, 1e-5)
	probs = F.softmax(logits, dim=-1)
	next_token = torch.multinomial(probs, num_samples=1).item()
	return next_token

def run_chat(args):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	base_dir = "/home/joan/Documents/IA/frankenswarm"
	expanded_glyphs_path = os.path.join(base_dir, "configs", "expanded_glyphs.json")
	model_path = os.path.join(base_dir, "storage", "checkpoints", "sovereign_school", "model_final.pt")
	state_path = os.path.join(base_dir, "storage", "checkpoints", "sovereign_school", "school_state.json")
	
	if not os.path.exists(model_path):
		model_path = os.path.join(base_dir, "storage", "checkpoints", "sovereign_school", "model_current.pt")
		if not os.path.exists(model_path):
			print(f"❌ Error: No se encontró el modelo en {model_path}. Por favor entrena primero el modelo.")
			return
		
	# 1. Cargar vocabulario y glifos
	with open(expanded_glyphs_path, encoding="utf-8") as f:
		vocab_data = json.load(f)
		words = vocab_data["words"]
		glyphs = np.array(vocab_data["glyphs"], dtype=np.float32)
		
	word_to_idx = {w: i for i, w in enumerate(words)}
	idx_to_word = dict(enumerate(words))
	
	# 2. Inicializar Diccionario Soberano para limpiar la entrada del usuario
	print("📖 Inicializando Diccionario Soberano...")
	dictionary = SovereignDictionary(expanded_glyphs_path)
	
	# 3. Cargar arquitectura dinámica del estado
	hidden_dim = 256
	num_layers = 6
	if os.path.exists(state_path):
		try:
			with open(state_path, encoding="utf-8") as sf:
				state_data = json.load(sf)
				hidden_dim = state_data.get("hidden_dim", 256)
				num_layers = state_data.get("num_layers", 6)
				print(f"📖 Configuración de arquitectura leída: hidden_dim={hidden_dim}, capas={num_layers}")
		except Exception as e:
			print(f"⚠️ Error al leer school_state.json: {e}")
			
	# 4. Inicializar modelo
	print(f"🧠 Cargando modelo Causal BitNet ({hidden_dim} dim, {num_layers} capas)...")
	model = BitNet4LayerModel(
		use_glyphs=True,
		glyph_table=glyphs,
		hidden_dim=hidden_dim,
		num_layers=num_layers,
		use_pos_embedding=True,
		is_causal=True,
		max_seq_len=128
	).to(device)
	model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
	model.eval()
	
	print("\n" + "═"*60)
	print("💬 AGENTE SOBERANO FORMADO EN EL COLEGIO (NIÑO DE 8 AÑOS)")
	print("   El modelo razona a través de 65 primos semánticos composicionales.")
	print("   Tu entrada se mapeará al vocabulario base para asegurar comprensión.")
	print("   Escribe 'salir' para terminar el chat.")
	print("═"*60 + "\n")
	
	
	while True:
		try:
			user_input = input("Tú > ").strip()
			if user_input.lower() in ["salir", "exit", "quit"]:
				print("\n¡Adiós!")
				break
			if not user_input:
				continue
				
			# Limpiar y mapear palabras de la entrada al vocabulario base
			raw_words = re.findall(r'[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_]+', user_input.lower())
			mapped_words = [dictionary.map_to_base_word(w) for w in raw_words]
			cleaned_input = " ".join(mapped_words)
			
			if cleaned_input:
				print(f"🎒 [Mapeado]: {cleaned_input}")
			else:
				print("🎒 [Mapeado]: (vacío)")

				continue
				
			# Mapear e inyectar el turno directamente (sin acumular historial, ya que el modelo fue entrenado con frases independientes)
			context = [word_to_idx.get(w, 1) for w in mapped_words]
			
			# Limitar el contexto a max_seq_len (128)
			if len(context) > 128:
				context = context[-128:]
				
			# Generar la predicción directa de un único token (Argmax)
			with torch.no_grad():
				input_len = len(context)
				padded_input = list(context)
				padded_input = padded_input + [0] * (128 - len(padded_input)) if len(padded_input) < 128 else padded_input[-128:]
					
				x = torch.tensor([padded_input], dtype=torch.long, device=device)
				logits = model(x)
				
				# El modelo predice el siguiente token inmediato al final del input
				last_token_idx = input_len - 1
				next_token = logits[0, last_token_idx].argmax(dim=-1).item()
				
			response_text = idx_to_word.get(next_token, "?")
			print(f"Yo > {response_text}\n")

			
		except KeyboardInterrupt:
			print("\n¡Adiós!")
			break
		except Exception as e:
			print(f"⚠️ Error: {e}")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--temperature", type=float, default=0.6, help="Temperatura de muestreo (default: 0.6)")
	parser.add_argument("--penalty", type=float, default=1.5, help="Penalización por repetición (default: 1.5)")
	args = parser.parse_args()
	run_chat(args)
