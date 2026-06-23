import os
import sys
import json
import re
import numpy as np
import torch

base_dir = "/home/joan/Documents/IA/frankenswarm"
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, "src"))

from bitnet.modeling_bitnet import BitNet4LayerModel

def tokenize(text: str, word_to_idx: dict) -> list[int]:
	words = re.findall(r'[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_<>\-]+', text.lower())
	return [word_to_idx.get(w, word_to_idx.get("<unk>", 1)) for w in words]

def verify_model():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	expanded_glyphs_path = os.path.join(base_dir, "configs", "expanded_glyphs.json")
	checkpoint_path = os.path.join(base_dir, "storage", "checkpoints", "sovereign_school", "model_milestone_4_years.pt")

	if not os.path.exists(checkpoint_path):
		print(f"❌ Checkpoint not found at {checkpoint_path}")
		sys.exit(1)

	# Dimensions for age 4
	hidden_dim = 512
	num_layers = 6

	print(f"Loading milestone model (Age: 4 years)...")
	print(f"Dimension: {hidden_dim}, Layers: {num_layers}, Device: {device}")

	with open(expanded_glyphs_path, encoding="utf-8") as f:
		vocab_data = json.load(f)
		words = vocab_data["words"]
		glyphs = np.array(vocab_data["glyphs"], dtype=np.float32)

	word_to_idx = {w: i for i, w in enumerate(words)}
	idx_to_word = {i: w for w, i in word_to_idx.items()}

	model = BitNet4LayerModel(
		use_glyphs=True,
		glyph_table=glyphs,
		hidden_dim=hidden_dim,
		num_layers=num_layers,
		use_pos_embedding=True
	).to(device)

	state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
	model.load_state_dict(state_dict, strict=True)
	model.eval()
	print("✅ Checkpoint loaded with strict=True. Dimensions match perfectly!")

	prompts = [
		"hola",
		"cómo estás",
		"quién eres",
		"el sol brilla",
		"si toco el fuego",
		"tengo frío",
		"quiero jugar"
	]

	print("\nGenerating programmatic answers:")
	for prompt in prompts:
		input_tokens = tokenize("tú: " + prompt, word_to_idx)
		current_input = torch.tensor([input_tokens], dtype=torch.long, device=device)
		response_tokens = []

		with torch.no_grad():
			for _ in range(15):
				logits = model(current_input)
				next_token_logits = logits[0, -1, :].clone()

				# Filter control tokens
				next_token_logits[word_to_idx.get("<pad>", 0)] = -1e9
				next_token_logits[word_to_idx.get("<unk>", 1)] = -1e9
				next_token_logits[word_to_idx.get("yo", 0)] = -1e9
				next_token_logits[word_to_idx.get("tú", 0)] = -1e9

				# Repetition penalty
				for prev_tok in set(response_tokens):
					next_token_logits[prev_tok] -= 2.0

				# Sample
				probs = torch.softmax(next_token_logits / 0.7, dim=-1)
				next_token = torch.multinomial(probs, num_samples=1).item()

				if next_token == word_to_idx.get("<pad>", 0) or next_token == word_to_idx.get("noche", 0):
					break

				response_tokens.append(next_token)
				next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
				current_input = torch.cat([current_input, next_token_tensor], dim=1)

		response_words = [idx_to_word.get(t, "?") for t in response_tokens]
		response_text = " ".join([w for w in response_words if w not in ["<pad>", "<unk>", "yo", "tú"]])
		print(f"Q: '{prompt}' -> A: '{response_text}'")

if __name__ == '__main__':
	verify_model()
