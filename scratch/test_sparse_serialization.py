import os
import json
import torch
import torch.nn.functional as F  # noqa: N812
import numpy as np

def run_sparse_validation():
	base_dir = "/home/joan/Documents/IA/frankenswarm"
	glyphs_path = os.path.join(base_dir, "configs", "expanded_glyphs.json")
	
	if not os.path.exists(glyphs_path):
		print("❌ Glifos no encontrados.")
		return
		
	print("📡 Cargando vocabulario y glifos para validación de compresión...")
	with open(glyphs_path, "r", encoding="utf-8") as f:
		vocab_data = json.load(f)
		glyphs = np.array(vocab_data["glyphs"], dtype=np.float32)
		
	vocab_size = len(vocab_data["words"])
	glyph_dim = glyphs.shape[1]
	print(f"Vocabulario: {vocab_size} palabras | Dimensión del glifo: {glyph_dim}")
	
	# Convertir glifos a tensor de embedding (Mock de W_E_b de 384-dim)
	# Para simular la matriz de embedding del modelo B, creamos una proyección lineal de glifos a 384-dim
	torch.manual_seed(42)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	W_E_glyph = torch.tensor(glyphs, dtype=torch.float32, device=device) # [15005, 65]
	proj_B = torch.randn(glyph_dim, 384, device=device) / np.sqrt(glyph_dim) # [65, 384]
	W_E_b = W_E_glyph @ proj_B # [15005, 384]
	
	# Generar logits de salida del Modelo A (mock)
	batch_size = 1
	seq_len = 4
	logits_a = torch.randn(batch_size, seq_len, vocab_size, device=device) * 2.0
	
	# 1. Proyección Densa Completa (Tensors as API original)
	probs_a_dense = F.softmax(logits_a / 1.0, dim=-1) # [1, 4, 15005]
	h_b_dense = probs_a_dense @ W_E_b # [1, 4, 384]
	
	# 2. Proyección Dispersa Top-K (Propuesta de Lumo)
	k_values = [10, 50, 100, 500]
	print("\n📊 Resultados de Validación Top-K:")
	print(f"{'K':<6} | {'Bytes Payload':<15} | {'Reducción %':<12} | {'Similitud Coseno':<18} | {'Distancia L2':<12}")
	print("-" * 70)
	
	# Tamaño densa: B * S * V * 4 bytes + 12 bytes header
	dense_size = batch_size * seq_len * vocab_size * 4 + 12
	
	for k in k_values:
		# Obtener top-k logits y sus índices
		values, indices = torch.topk(logits_a, k=k, dim=-1) # [1, 4, k]
		
		# Aplicar softmax localmente sobre los top-k para que la distribución sume 1.0
		probs_topk = F.softmax(values, dim=-1) # [1, 4, k]
		
		# Re-embebido disperso eficiente: probs_topk @ W_E_b[indices]
		# W_E_b[indices] tiene dimensiones [B, S, k, d_b]
		# Multiplicamos elemento a elemento y sumamos en la dimensión k
		selected_embeddings = W_E_b[indices] # [B, S, k, 384]
		h_b_sparse = (probs_topk.unsqueeze(-1) * selected_embeddings).sum(dim=-2) # [B, S, 384]
		
		# Calcular similitud coseno entre h_b_dense y h_b_sparse
		cos_sim = F.cosine_similarity(h_b_dense.view(-1, 384), h_b_sparse.view(-1, 384), dim=-1).mean().item()
		l2_dist = torch.norm(h_b_dense - h_b_sparse, p=2).item()
		
		# Tamaño de payload disperso:
		# Cada token disperso necesita: 1 índice (int32 = 4 bytes) + 1 valor (float32 = 4 bytes) = 8 bytes
		# Total: B * S * k * 8 bytes + 12 bytes header
		sparse_size = batch_size * seq_len * k * 8 + 12
		reduction = (1.0 - (sparse_size / dense_size)) * 100
		
		print(f"{k:<6} | {sparse_size:<15,} | {reduction:.2f}% | {cos_sim:.6f} | {l2_dist:.6f}")

if __name__ == "__main__":
	run_sparse_validation()
