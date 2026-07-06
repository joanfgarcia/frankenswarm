import json
import os
import torch
import torch.nn.functional as F
import numpy as np

def run_audit():
	print("═══ 🔬 Projected Word Embeddings Auditor ═══")
	
	base_dir = "/home/joan/Documents/IA/frankenswarm"
	expanded_glyphs_path = os.path.join(base_dir, "configs", "expanded_glyphs.json")
	state_path = os.path.join(base_dir, "storage", "checkpoints", "sovereign_school", "school_state.json")
	checkpoint_path = os.path.join(base_dir, "storage", "checkpoints", "sovereign_school", "model_current.pt")
	
	if not os.path.exists(checkpoint_path):
		print(f"❌ Checkpoint not found at: {checkpoint_path}")
		return
		
	# 1. Load school state
	with open(state_path, encoding="utf-8") as f:
		state = json.load(f)
		hidden_dim = state.get("hidden_dim", 128)
		num_layers = state.get("num_layers", 6)
		current_epoch = state.get("current_epoch", 1)
		
	print(f"📖 School State: current_epoch={current_epoch}, hidden_dim={hidden_dim}, layers={num_layers}")
	
	# 2. Load glyph table
	with open(expanded_glyphs_path, encoding="utf-8") as f:
		vocab_data = json.load(f)
		words = vocab_data["words"]
		glyphs = np.array(vocab_data["glyphs"], dtype=np.float32)
		
	word_to_idx = {w: i for i, w in enumerate(words)}
	idx_to_word = dict(enumerate(words))
	vocab_size = len(words)
	print(f"Loaded vocabulary: {vocab_size} words.")
	
	# 3. Instantiate model and load checkpoint
	from src.bitnet.model.modeling_bitnet import BitNet4LayerModel
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")
	
	model = BitNet4LayerModel(
		use_glyphs=True,
		glyph_table=glyphs,
		hidden_dim=hidden_dim,
		num_layers=num_layers,
		use_pos_embedding=True,
		is_causal=True,
		max_seq_len=128,
	).to(device)
	
	model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
	model.eval()
	print("🧠 Checkpoint loaded successfully.")
	
	# 4. Extract projected word embeddings
	with torch.no_grad():
		word_embeds = model.glyph_embedding.get_word_embeddings()  # (vocab_size, hidden_dim)
		word_norms = F.normalize(word_embeds, dim=-1)  # (vocab_size, hidden_dim)
		
		# Compute some stats
		print("\n🔍 Calculating overall embedding alignment statistics...")
		
		# Sample a subset of pairs to avoid OOM for N=15005 on large matrices
		n_samples = min(10000, vocab_size)
		sample_indices = torch.randperm(vocab_size)[:n_samples].to(device)
		sample_norms = word_norms[sample_indices]
		
		# Similarity matrix for samples: (n_samples, n_samples)
		sim_matrix = torch.matmul(sample_norms, sample_norms.T)
		
		# Mask out identity diagonal
		mask = ~torch.eye(n_samples, dtype=torch.bool, device=device)
		off_diag_sims = sim_matrix[mask]
		
		mean_sim = off_diag_sims.mean().item()
		std_sim = off_diag_sims.std().item()
		max_sim = off_diag_sims.max().item()
		min_sim = off_diag_sims.min().item()
		
		print(f"  - Mean Cosine Similarity: {mean_sim:.4f}")
		print(f"  - Std Dev: {std_sim:.4f}")
		print(f"  - Max Similarity: {max_sim:.4f}")
		print(f"  - Min Similarity: {min_sim:.4f}")
		
		# Check specific OOV words
		target_oovs = ["asteroide", "población"]
		target_actives = ["yo", "sentir", "madre", "decir", "calor", "frío", "gato", "perro"]
		
		print("\n🔍 Checking alignment of OOV words with active training words:")
		for oov in target_oovs:
			if oov not in word_to_idx:
				continue
			oov_idx = word_to_idx[oov]
			oov_norm = word_norms[oov_idx] # (hidden_dim,)
			
			print(f"\n  • Word: '{oov}'")
			similarities = []
			for active in target_actives:
				if active not in word_to_idx:
					continue
				act_idx = word_to_idx[active]
				act_norm = word_norms[act_idx]
				cos_sim = torch.dot(oov_norm, act_norm).item()
				similarities.append((active, cos_sim))
				
			# Sort by similarity
			similarities.sort(key=lambda x: x[1], reverse=True)
			for word, sim in similarities:
				print(f"    - CosSim with '{word}': {sim:.4f}")
				
		# Find overall highest similarities in the vocabulary (> 0.85)
		print("\n🔍 Scanning for high-similarity collisions (Cosine Similarity > 0.85) in vocab...")
		collisions = []
		block_size = 1000
		for i in range(0, vocab_size, block_size):
			end_i = min(i + block_size, vocab_size)
			block_norms = word_norms[i:end_i]
			
			# Compute similarity of this block against all words
			block_sims = torch.matmul(block_norms, word_norms.T)
			
			# Find matches > 0.85
			rows, cols = torch.where(block_sims > 0.85)
			for r, c in zip(rows.tolist(), cols.tolist()):
				global_r = i + r
				if global_r < c: # Only keep unique pairs and ignore diagonal
					collisions.append((words[global_r], words[c], block_sims[r, c].item()))
					
		print(f"  ✓ Found {len(collisions)} pairs with similarity > 0.85.")
		if collisions:
			print("\n  Top 10 highest-similarity word pairs:")
			# Sort by similarity descending
			collisions.sort(key=lambda x: x[2], reverse=True)
			for w1, w2, sim in collisions[:10]:
				print(f"    - '{w1}' ↔ '{w2}' | CosSim: {sim:.4f}")

if __name__ == "__main__":
	run_audit()
