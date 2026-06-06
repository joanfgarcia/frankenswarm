import torch

from src.bitnet.glyph_vocabulary import N_EMOTIONS
from src.bitnet.modeling_bitnet import BitNet4LayerModel


class RecurrentBitNetModel(BitNet4LayerModel):
	"""
	POC of Temporal Resonance: Subclass of BitNet4LayerModel that allows
	feeding back the final hidden state of the previous tick (Layer 4)
	into the input stage of the current tick (Layer 2).
	"""
	def __init__(self, *args, recurrent_decay=0.6, **kwargs):
		super().__init__(*args, **kwargs)
		self.recurrent_decay = recurrent_decay

	def forward_recurrent(
		self,
		x: torch.Tensor,
		h_prev: torch.Tensor = None,
		n_steps: int = 3,
		pos_mode: str = "clock",
		emotion_ids: torch.Tensor = None,
	) -> tuple[torch.Tensor, torch.Tensor]:
		"""
		Fórmula:
			h = Capa1_2(x) + decaimiento * h_prev
			for step in range(n_steps):
				h = Capa3(h)
			logits = Capa4_5(h)
		"""
		# ── Entrada: Capas 1→2 (una sola vez) ──
		h = self._embed_input(x)

		# ── Acoplamiento del estado latente anterior (Resonancia Temporal) ──
		if h_prev is not None:
			# Sumamos el estado anterior atenuado al estado de entrada actual
			h = h + self.recurrent_decay * h_prev

		# Preparación del vector emocional
		emo_vec = None
		if emotion_ids is not None and self.emotion_embeddings is not None:
			emo_emb = self.emotion_embeddings(emotion_ids)
			emo_vec = self.emotion_proj(emo_emb).unsqueeze(1)

		# Bucle Latente (Resonancia de Profundidad)
		for step in range(n_steps):
			if pos_mode == "clock" and getattr(self, "resonance_clock", None) is not None:
				h = h + self.resonance_clock[:, step, :].unsqueeze(1)

			if emo_vec is not None and (self.emotion_mode == "additive" or self.emotion_mode == "first_only" and step == 0):
				h = h + emo_vec

			for layer in self.core_layers:
				h = layer(h)
			h = self.norm(h)

		# ── Salida: Capas 4→5 ──
		logits = self._decode_hidden(h)

		# Devolvemos los logits para la acción y el nuevo estado latente para el siguiente tick
		return logits, h.detach()

def run_poc():
	print("=" * 60)
	print("🧪 POC: RESONANCIA TEMPORAL EN BITNET 🧪")
	print("=" * 60)

	# Configuración de prueba
	hidden_dim = 128
	num_layers = 3
	max_resonance_steps = 5
	max_seq_len = 6 # Telemetría de tamaño 6 (loc, what, body, backpack, sig_loc, sig_what)

	print("1. Inicializando RecurrentBitNetModel...")
	model = RecurrentBitNetModel(
		use_glyphs=True,
		hidden_dim=hidden_dim,
		num_layers=num_layers,
		use_pos_embedding=True,
		max_resonance_steps=max_resonance_steps,
		n_emotions=N_EMOTIONS,
		emotion_dim=32,
		emotion_mode="first_only",
		max_seq_len=max_seq_len,
		recurrent_decay=0.5
	)

	# Simular una secuencia de 5 ticks de observaciones discretas dentro del vocabulario
	# Cada observación representa la telemetría del monito en ese tick
	print("\n2. Generando secuencia de inputs (5 Ticks)...")
	torch.manual_seed(1234)
	sequence_inputs = [torch.randint(0, model.vocab_size, (1, max_seq_len)) for _ in range(5)]
	emotion_ids = torch.tensor([0]) # emoción basal

	# Variables para rastrear el estado persistente y estabilidad
	h_state = None
	
	print("\n3. Ejecutando simulación con Resonancia Temporal:")
	print("-" * 60)
	print(f"{'Tick':<6} | {'Input (Tokens)':<25} | {'Norma Hidden':<12} | {'Sim. Coseno con anterior':<25}")
	print("-" * 60)

	for tick, x in enumerate(sequence_inputs):
		# Guardamos el hidden anterior para calcular la similitud
		h_before = h_state
		
		# Ejecución del paso recurrente
		logits, h_state = model.forward_recurrent(
			x,
			h_prev=h_state,
			n_steps=2,
			pos_mode="clock",
			emotion_ids=emotion_ids
		)
		
		# Calcular similitud de coseno si existe un estado anterior
		if h_before is not None:
			cos_sim = torch.nn.functional.cosine_similarity(h_state, h_before, dim=-1).mean().item()
			sim_str = f"{cos_sim:.4f}"
		else:
			sim_str = "N/A (Primer Tick)"
			
		norm_str = f"{h_state.norm().item():.4f}"
		tokens_str = str(x.tolist()[0])
		
		print(f"T={tick}    | {tokens_str:<25} | {norm_str:<12} | {sim_str:<25}")

	print("-" * 60)
	print("\n4. Representación del Flujo de Datos:")
	print("   [Tick T] ──> Capa 1 (Sensorial) ──┐")
	print("                                      ├──> Capa 2 (Entrada) <── [Decaimiento 0.5] ──┐")
	print("   [Pensamiento]                      │                                             │")
	print("      └─> 2× [Capa 3 (Core)] ─────────┴──> Capa 4 (Salida) ─────────────────────────┴─> (Siguiente Tick)")
	print("                                                │")
	print("                                                └──> Capa 5 (Acción/Voz)")
	print("=" * 60)
	print("Prueba completada con éxito.")

if __name__ == "__main__":
	run_poc()
