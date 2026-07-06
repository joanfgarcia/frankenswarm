import torch
import torch.nn as nn

from src.bitnet.model.modeling_bitnet import BitLinear, BitNetTransformerBlock, RMSNorm


class BitNetCausalLM(nn.Module):
	"""
	Modelo de Lenguaje Causal de 1.58 bits (BitNet) diseñado para conversación.
	Reutiliza bloques de atención causal de BitNet y soporta la inyección temporal h_prev
	para retención de memoria conversacional de corto y medio plazo.
	"""

	def __init__(self, vocab_size: int = 10000, hidden_dim: int = 256, num_layers: int = 6, num_heads: int = 4, max_seq_len: int = 256):
		super().__init__()
		self.vocab_size = vocab_size
		self.hidden_dim = hidden_dim
		self.num_layers = num_layers
		self.max_seq_len = max_seq_len

		# Capa 1: Embeddings de Tokens aprendibles estándar
		self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
		
		# Capa 2: Embeddings Posicionales aprendibles estándar
		self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)

		# Capa 3: Specialist Core (Ternary Transformer Blocks causales)
		self.blocks = nn.ModuleList([
			BitNetTransformerBlock(dim=hidden_dim, num_heads=num_heads, mlp_ratio=4, is_causal=True)
			for _ in range(num_layers)
		])
		self.norm = RMSNorm(hidden_dim)

		# Capa 4: Cabeza LM Causal de 1.58 bits
		self.lm_head = BitLinear(hidden_dim, vocab_size, bias=False)

	def forward(self, input_ids: torch.Tensor, h_prev: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
		"""
		Paso forward del modelado de lenguaje causal.
		Soporta la inyección de h_prev en el primer token del input para transferir memoria latente.
		"""
		batch_size, seq_len = input_ids.shape
		assert seq_len <= self.max_seq_len, f"La longitud de secuencia {seq_len} excede el máximo {self.max_seq_len}"

		# Lookup de embeddings
		h = self.token_embedding(input_ids)  # (B, seq_len, hidden_dim)

		# Sumar embeddings posicionales
		pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
		h = h + self.pos_embedding(pos_ids)

		# Inyectar la memoria del turno anterior (h_prev) si está disponible
		if h_prev is not None:
			# Sumamos el vector h_prev al primer token del turno actual.
			# Esto introduce el contexto latente acumulado al inicio de la frase,
			# permitiendo que la atención causal lo propague a todo el nuevo texto.
			h[:, 0, :] = h[:, 0, :] + 0.5 * h_prev

		# Pasar por el Core de Transformers ternarios
		for block in self.blocks:
			h = block(h)
		h = self.norm(h)

		# Extraer el hidden state del último token para el siguiente paso de resonancia
		# Representa el resumen comprimido de todo el contexto conversacional actual
		next_h_prev = h[:, -1, :]

		# Proyectar a logits sobre el vocabulario
		logits = self.lm_head(h)  # (B, seq_len, vocab_size)

		return logits, next_h_prev
