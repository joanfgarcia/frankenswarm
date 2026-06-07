import os
import json
import re
import torch
import torch.nn.functional as F
import numpy as np
import sys

sys.path.append("/home/joan/Documents/IA/frankenswarm")

from src.bitnet.modeling_bitnet import BitNet4LayerModel
from src.bitnet.dictionary_tool import SovereignDictionary

def main():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	base_dir = "/home/joan/Documents/IA/frankenswarm"
	expanded_glyphs_path = os.path.join(base_dir, "configs", "expanded_glyphs.json")
	model_path = os.path.join(base_dir, "storage", "checkpoints", "sovereign_school", "model_current.pt")
	state_path = os.path.join(base_dir, "storage", "checkpoints", "sovereign_school", "school_state.json")
	
	with open(expanded_glyphs_path, "r", encoding="utf-8") as f:
		vocab_data = json.load(f)
		words = vocab_data["words"]
		glyphs = np.array(vocab_data["glyphs"], dtype=np.float32)
		
	word_to_idx = {w: i for i, w in enumerate(words)}
	idx_to_word = {i: w for i, w in enumerate(words)}
	
	# Load architecture configuration
	hidden_dim = 256
	num_layers = 6
	if os.path.exists(state_path):
		with open(state_path, "r", encoding="utf-8") as sf:
			state_data = json.load(sf)
			hidden_dim = state_data.get("hidden_dim", 256)
			num_layers = state_data.get("num_layers", 6)
			print(f"Loaded architecture: hidden_dim={hidden_dim}, layers={num_layers}")

	dictionary = SovereignDictionary(expanded_glyphs_path)
	
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
	
	test_dialogue = [
		"el gato duerme",
		"el perro corre",
		"cuánto es tres más tres? es",
		"cuánto es seis menos cuatro? es",
		"el agua del río corre hacia el",
		"los árboles dan oxígeno y",
		"el corazón bombea sangre al",
		"cómo te llamas",
		"de dónde eres",
		"la capital de España es",
		"la tierra gira alrededor del",
		"los mapas muestran los ríos y",
		"si equis más dos es cinco entonces equis es",
		"toda causa produce un",
		"el laberinto es una biblioteca de espejos",
		"el Aleph es un punto que contiene todo el",
		"qué es el búnker"
	]
	
	print("\n--- Testing Natural Stop Behavior ---")
	for input_text in test_dialogue:
		raw_words = re.findall(r'[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_]+', input_text.lower())
		mapped_words = [dictionary.map_to_base_word(w) for w in raw_words]
		cleaned_input = " ".join(mapped_words)
		
		context = [word_to_idx.get(w, 1) for w in mapped_words]
		if len(context) > 128:
			context = context[-128:]
			
		generated = []
		tokens_with_pad = []
		with torch.no_grad():
			for _ in range(15):
				input_len = len(context)
				padded_input = list(context)
				if len(padded_input) < 128:
					padded_input = padded_input + [0] * (128 - len(padded_input))
				else:
					padded_input = padded_input[-128:]
					
				x = torch.tensor([padded_input], dtype=torch.long, device=device)
				logits = model(x)
				
				last_token_idx = input_len - 1
				next_token = logits[0, last_token_idx].argmax(dim=-1).item()
				
				tokens_with_pad.append(next_token)
				if next_token <= 1 or idx_to_word.get(next_token) in ["<pad>", "<unk>", "[eos]"]:
					break
				generated.append(next_token)
				context.append(next_token)
				
		response_words = [idx_to_word.get(t, "?") for t in generated]
		response_text = " ".join(response_words)
		
		print(f"Pregunta: \"{input_text}\"")
		print(f"Mapeado:  \"{cleaned_input}\"")
		print(f"Respuesta: \"{response_text}\"")
		print(f"Tokens generados (con parada): {[idx_to_word.get(t, '?') for t in tokens_with_pad]}")
		print()

if __name__ == '__main__':
	main()
