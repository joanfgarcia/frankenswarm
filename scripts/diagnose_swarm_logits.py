import os
import json
import re
import numpy as np
import torch
import torch.nn.functional as F

from src.bitnet.modeling_bitnet import BitNet4LayerModel


def get_entropy(probs):
	return -torch.sum(probs * torch.log(probs + 1e-9)).item()


def tokenize_words(text, word_to_idx):
	w_list = re.findall(r'[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_<>\-]+', text.lower())
	return [word_to_idx.get(w, 1) for w in w_list]


def main():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	base_dir = "/home/joan/Documents/IA/frankenswarm"
	expanded_glyphs_path = os.path.join(base_dir, "configs", "expanded_glyphs.json")
	model_path = os.path.join(base_dir, "storage", "checkpoints", "EXP_072", "model_final.pt")

	if not os.path.exists(model_path):
		print(f"❌ Error: No se encontró el modelo en {model_path}")
		return

	# Cargar vocabulario
	with open(expanded_glyphs_path, encoding="utf-8") as f:
		vocab_data = json.load(f)
		words = vocab_data["words"]
		glyphs = np.array(vocab_data["glyphs"], dtype=np.float32)

	word_to_idx = {w: i for i, w in enumerate(words)}
	idx_to_word = dict(enumerate(words))

	model = BitNet4LayerModel(
		use_glyphs=True,
		glyph_table=glyphs,
		hidden_dim=512,
		num_layers=6,
		use_pos_embedding=True,
		is_causal=True,
		max_seq_len=64
	).to(device)
	model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
	model.eval()

	# Nico inicia: "yo tengo perro yo"
	ctx = tokenize_words("yo tengo perro", word_to_idx) + [word_to_idx.get("yo")]
	
	input_ids = list(ctx)
	print("\nDiagnóstico de Logits paso a paso para 'yo tengo perro yo':")
	print("-" * 60)

	for step in range(5):
		curr_input = list(input_ids)[-64:]
		input_len = len(curr_input)
		if len(curr_input) < 64:
			curr_input = curr_input + [0] * (64 - len(curr_input))

		x = torch.tensor([curr_input], dtype=torch.long, device=device)
		with torch.no_grad():
			logits = model(x)

		last_token_idx = input_len - 1
		next_token_logits = logits[0, last_token_idx].clone()

		# Aplicar rep penalty
		for token in set(input_ids):
			next_token_logits[token] -= 1.2

		next_token_logits[0] = -1e9
		next_token_logits[1] = -1e9

		probs = F.softmax(next_token_logits, dim=-1)
		entropy = get_entropy(probs)

		# Top 5
		top_5_vals, top_5_indices = torch.topk(probs, 5)
		top_5_list = []
		for val, idx in zip(top_5_vals, top_5_indices):
			top_5_list.append(f"'{idx_to_word[idx.item()]}': {val.item():.4f}")

		selected_idx = top_5_indices[0].item()
		selected_prob = top_5_vals[0].item()
		selected_word = idx_to_word[selected_idx]

		print(f"Step {step+1:02d}: Selected='{selected_word}' | Prob={selected_prob:.4f} | Entropy={entropy:.4f}")
		print(f"   Alternatives: {', '.join(top_5_list)}")

		input_ids.append(selected_idx)


if __name__ == "__main__":
	main()
