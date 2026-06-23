import os
import sys
import json
import re
import argparse
import numpy as np
import torch

# Añadir src y base_dir al path
base_dir = "/home/joan/Documents/IA/frankenswarm"
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, "src"))

from bitnet.modeling_bitnet import BitNet4LayerModel
from bitnet.dictionary_tool import SovereignDictionary

def tokenize(text: str, word_to_idx: dict) -> list[int]:
	words = re.findall(r'[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_<>\-]+', text.lower())
	return [word_to_idx.get(w, word_to_idx.get("<unk>", 1)) for w in words]

def run_milestone_chat():
	parser = argparse.ArgumentParser(description="Interactive Chat with Sovereign School Milestones")
	parser.add_argument("--age", type=int, default=3, choices=[2, 3, 4, 5, 6, 7, 8], help="Age milestone checkpoint to load")
	parser.add_argument("--max_tokens", type=int, default=15, help="Maximum tokens to generate")
	parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature")
	parser.add_argument("--penalty", type=float, default=2.0, help="Repetition penalty value")
	args = parser.parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	expanded_glyphs_path = os.path.join(base_dir, "configs", "expanded_glyphs.json")
	
	checkpoint_name = f"model_milestone_{args.age}_years.pt"
	checkpoint_path = os.path.join(base_dir, "storage", "checkpoints", "sovereign_school", checkpoint_name)

	if not os.path.exists(checkpoint_path):
		print(f"❌ Checkpoint no encontrado en {checkpoint_path}")
		return

	# Configurar dimensiones según la edad (get_stage_config)
	age_dims = {
		2: 256,
		3: 384,
		4: 512,
		5: 640,
		6: 768,
		7: 896,
		8: 1024
	}
	hidden_dim = age_dims[args.age]
	num_layers = 6

	print(f"🤖 Cargando Modelo de la Escuela Soberana (Edad: {args.age} años)...")
	print(f"   [Dimensiones]: hidden_dim={hidden_dim}, capas={num_layers}")
	print(f"   [Device]: {device}")

	# Cargar vocabulario base
	with open(expanded_glyphs_path, encoding="utf-8") as f:
		vocab_data = json.load(f)
		words = vocab_data["words"]
		glyphs = np.array(vocab_data["glyphs"], dtype=np.float32)

	word_to_idx = {w: i for i, w in enumerate(words)}
	idx_to_word = {i: w for w, i in word_to_idx.items()}

	# Inicializar modelo
	model = BitNet4LayerModel(
		use_glyphs=True,
		glyph_table=glyphs,
		hidden_dim=hidden_dim,
		num_layers=num_layers,
		use_pos_embedding=True
	).to(device)

	state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
	model.load_state_dict(state_dict, strict=False)
	model.eval()

	print("\n" + "═"*60)
	print(f"💬 CHAT INTERACTIVO CON BIT ({args.age} AÑOS)")
	print("   Escribe 'salir' para finalizar.")
	print("═"*60 + "\n")

	while True:
		try:
			user_input = input("Tú > ").strip()
			if not user_input:
				continue
			if user_input.lower() in ["salir", "exit", "quit"]:
				break

			# Preprocesar entrada
			input_tokens = tokenize("tú: " + user_input, word_to_idx)
			
			response_tokens = []
			current_input = torch.tensor([input_tokens], dtype=torch.long, device=device)

			with torch.no_grad():
				for _ in range(args.max_tokens):
					logits = model(current_input) # (1, seq_len, vocab_size)
					next_token_logits = logits[0, -1, :].clone()

					# Filtrar tokens de control y relleno de la generación
					next_token_logits[word_to_idx.get("<pad>", 0)] = -1e9
					next_token_logits[word_to_idx.get("<unk>", 1)] = -1e9
					next_token_logits[word_to_idx.get("yo", 0)] = -1e9
					next_token_logits[word_to_idx.get("tú", 0)] = -1e9

					# Penalización por repetición
					for prev_tok in set(response_tokens):
						next_token_logits[prev_tok] -= args.penalty

					# Muestreo estocástico con temperatura
					probs = torch.softmax(next_token_logits / args.temp, dim=-1)
					next_token = torch.multinomial(probs, num_samples=1).item()

					if next_token == word_to_idx.get("<pad>", 0) or next_token == word_to_idx.get("noche", 0):
						break

					response_tokens.append(next_token)

					# Concatenar para autoregresión
					next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
					current_input = torch.cat([current_input, next_token_tensor], dim=1)

			response_words = [idx_to_word.get(t, "?") for t in response_tokens]
			response_text = " ".join([w for w in response_words if w not in ["<pad>", "<unk>", "yo", "tú"]])
			print(f"Bit > {response_text}\n")

		except KeyboardInterrupt:
			break
		except Exception as e:
			print(f"⚠️ Error: {e}")

if __name__ == "__main__":
	run_milestone_chat()
