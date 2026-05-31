import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightQuantSTE(torch.autograd.Function):
	"""
	Cuantización Ternaria {-1, 0, 1} utilizando Straight-Through Estimator (STE).
	Mantiene los gradientes continuos en el paso backward.
	"""

	@staticmethod
	def forward(ctx, weight):
		scale = weight.abs().mean().clamp(min=1e-5)
		quant = (weight / scale).round().clamp(-1, 1) * scale
		return quant

	@staticmethod
	def backward(ctx, grad_output):
		return grad_output


class ActivationQuantSTE(torch.autograd.Function):
	"""
	Cuantización simétrica de 8 bits [-128, 127] utilizando Straight-Through Estimator (STE).
	Mantiene los gradientes continuos en el paso backward.
	"""

	@staticmethod
	def forward(ctx, x):
		scale = x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
		quant = (x * 127 / scale).round().clamp(-128, 127) * scale / 127
		return quant

	@staticmethod
	def backward(ctx, grad_output):
		return grad_output


class BitLinear(nn.Module):
	"""
	Capa lineal de BitNet b1.58.
	Aplica WeightQuantSTE a los pesos y ActivationQuantSTE a las activaciones.
	"""

	def __init__(self, in_features: int, out_features: int, bias: bool = True):
		super().__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
		if bias:
			self.bias = nn.Parameter(torch.Tensor(out_features))
		else:
			self.register_parameter("bias", None)
		self.reset_parameters()

	def reset_parameters(self):
		nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
		if self.bias is not None:
			fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
			bound = 1 / np.sqrt(fan_in)
			nn.init.uniform_(self.bias, -bound, bound)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		quant_w = WeightQuantSTE.apply(self.weight)
		quant_x = ActivationQuantSTE.apply(x)
		return F.linear(quant_x, quant_w, self.bias)


class RMSNorm(nn.Module):
	"""Root Mean Square Layer Normalization."""

	def __init__(self, dim: int, eps: float = 1e-6):
		super().__init__()
		self.eps = eps
		self.weight = nn.Parameter(torch.ones(dim))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		variance = x.pow(2).mean(-1, keepdim=True)
		return x * torch.rsqrt(variance + self.eps) * self.weight


class BitNetAttention(nn.Module):
	"""Mecanismo de atención multi-cabezal utilizando BitLinear."""

	def __init__(self, dim: int, num_heads: int = 4):
		super().__init__()
		self.dim = dim
		self.num_heads = num_heads
		self.head_dim = dim // num_heads

		self.q_proj = BitLinear(dim, dim, bias=False)
		self.k_proj = BitLinear(dim, dim, bias=False)
		self.v_proj = BitLinear(dim, dim, bias=False)
		self.out_proj = BitLinear(dim, dim, bias=False)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		batch_size, seq_len, _ = x.shape
		q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
		k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
		v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

		scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
		attn = F.softmax(scores, dim=-1)
		context = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
		return self.out_proj(context)


class BitNetMLP(nn.Module):
	"""Feed-Forward Network utilizando BitLinear y GELU."""

	def __init__(self, dim: int, hidden_dim: int):
		super().__init__()
		self.up_proj = BitLinear(dim, hidden_dim, bias=False)
		self.down_proj = BitLinear(hidden_dim, dim, bias=False)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.down_proj(F.gelu(self.up_proj(x)))


class BitNetTransformerBlock(nn.Module):
	"""Bloque transformer de BitNet con RMSNorm."""

	def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: int = 4):
		super().__init__()
		self.attn_norm = RMSNorm(dim)
		self.attn = BitNetAttention(dim, num_heads)
		self.mlp_norm = RMSNorm(dim)
		self.mlp = BitNetMLP(dim, dim * mlp_ratio)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = x + self.attn(self.attn_norm(x))
		x = x + self.mlp(self.mlp_norm(x))
		return x


class BitNet4LayerModel(nn.Module):
	"""
	Modelo de 4 capas acoplado al vocabulario conceptual discreto.
	Implementa la ruta diferenciable para Gumbel-Softmax en el juego referencial.
	"""

	def __init__(self, vocab_embeddings: np.ndarray, hidden_dim: int = 256, num_layers: int = 4, use_pos_embedding: bool = False, max_resonance_steps: int = 0, n_emotions: int = 0, emotion_dim: int = 0, emotion_mode: str = "additive"):
		super().__init__()
		self.vocab_size, self.vocab_dim = vocab_embeddings.shape
		self.hidden_dim = hidden_dim
		self.use_pos_embedding = use_pos_embedding
		self.max_resonance_steps = max_resonance_steps
		self.emotion_mode = emotion_mode  # 'additive', 'gated', 'first_only'

		# Registrar los embeddings del vocabulario conceptual como un buffer no entrenable (Capa 1 fija)
		self.register_buffer("vocab_embeddings", torch.from_numpy(vocab_embeddings).float())

		# Capa 2: Inbound Translator (Proyección del embedding de 384-dim al espacio oculto del Core de 256-dim)
		self.inbound_proj = nn.Linear(self.vocab_dim, hidden_dim, bias=False)

		# Capa de Posición: Embeddings Posicionales Aprendibles (Longitud máxima 4)
		if self.use_pos_embedding:
			self.pos_embedding = nn.Parameter(torch.randn(1, 4, hidden_dim) * 0.02)
		else:
			self.register_parameter("pos_embedding", None)

		# Capa 3: Specialist Core (Ternary Transformer)
		self.core_layers = nn.ModuleList([BitNetTransformerBlock(dim=hidden_dim, num_heads=4, mlp_ratio=4) for _ in range(num_layers)])
		self.norm = RMSNorm(hidden_dim)

		# Capa 4: Outbound Translator (Proyección del espacio oculto de 256-dim al espacio conceptual de 384-dim)
		self.outbound_proj = nn.Linear(hidden_dim, self.vocab_dim, bias=False)

		# Resonancia Continua: Reloj posicional para el bucle latente (EXP_032)
		# Solo se inicializa si max_resonance_steps > 0
		if max_resonance_steps > 0:
			self.resonance_clock = nn.Parameter(torch.randn(1, max_resonance_steps, hidden_dim) * 0.02)
		else:
			self.register_parameter("resonance_clock", None)

		# ── EXP_033: Resonancia Emocional ──
		# Embedding de emociones → espacio oculto (la emoción como brújula)
		if n_emotions > 0 and emotion_dim > 0:
			self.emotion_embeddings = nn.Embedding(n_emotions, emotion_dim)
			self.emotion_proj = nn.Linear(emotion_dim, hidden_dim, bias=False)
			if emotion_mode == "gated":
				self.emotion_gate = nn.Linear(hidden_dim, hidden_dim, bias=False)
		else:
			self.emotion_embeddings = None
			self.emotion_proj = None

	def forward(self, x: torch.Tensor, logit_mask: torch.Tensor = None) -> torch.Tensor:
		"""
		Paso forward.
		x puede ser:
		- Un tensor de enteros de tamaño (batch_size, seq_len) conteniendo Token IDs discretos.
		- Un tensor float de tamaño (batch_size, seq_len, vocab_size) conteniendo vectores one-hot relajados (Gumbel-Softmax).
		"""
		seq_len = x.shape[1]
		# Capa 1 a Capa 2: Proyección al espacio oculto
		if x.ndim == 2:
			# Ruta discreta convencional (Token IDs)
			# Indexación directa sobre los embeddings fijos
			embeds = F.embedding(x, self.vocab_embeddings)  # (batch_size, seq_len, 384)
		else:
			# Ruta continua diferenciable (Gumbel-Softmax)
			# x es (batch_size, seq_len, 8192)
			embeds = torch.matmul(x, self.vocab_embeddings)  # (batch_size, seq_len, 384)

		h = self.inbound_proj(embeds)  # (batch_size, seq_len, 256)

		# Sumar embeddings posicionales si están habilitados o presentes
		if getattr(self, "pos_embedding", None) is not None:
			h = h + self.pos_embedding[:, :seq_len, :]

		# Capa 3: Specialist Core
		for layer in self.core_layers:
			h = layer(h)
		h = self.norm(h)

		# Capa 4: Outbound Translator
		# Proyectar el espacio oculto al espacio conceptual de Capa 1
		concept_proj = self.outbound_proj(h)  # (batch_size, seq_len, 384)

		# Mapear a logits multiplicando por la transpuesta de los embeddings del vocabulario fijos
		# (batch_size, seq_len, 384) x (384, 8192) -> (batch_size, seq_len, 8192)
		logits = torch.matmul(concept_proj, self.vocab_embeddings.T)
		if logit_mask is not None:
			logits = logits.masked_fill(~logit_mask, -1e9)
		return logits

	def generate_message(self, x: torch.Tensor, tau: float = 1.0, hard: bool = True, logit_mask: torch.Tensor = None) -> torch.Tensor:
		"""
		Genera un mensaje utilizando Gumbel-Softmax para mantener la diferenciabilidad del canal.
		Devuelve un tensor de vectores one-hot relajados.
		"""
		logits = self.forward(x, logit_mask=logit_mask)
		# Aplicamos Gumbel-Softmax sobre la dimensión del vocabulario
		message = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)
		return message

	def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
		key = prefix + "pos_embedding"
		if key in state_dict and state_dict[key] is not None and getattr(self, "pos_embedding", None) is None:
			param_shape = state_dict[key].shape
			self.pos_embedding = nn.Parameter(torch.zeros(param_shape, device=state_dict[key].device))
			self.use_pos_embedding = True
		super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

	# ── Resonancia Continua (EXP_032) ─────────────────────────────────────────────

	def _embed_input(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Capas 1→2: Proyecta input (token IDs o Gumbel-Softmax) al espacio oculto.
		Factorizado para reutilizar en forward() y forward_resonance().
		"""
		if x.ndim == 2:
			embeds = F.embedding(x, self.vocab_embeddings)
		else:
			embeds = torch.matmul(x, self.vocab_embeddings)
		h = self.inbound_proj(embeds)
		if getattr(self, "pos_embedding", None) is not None:
			seq_len = x.shape[1]
			h = h + self.pos_embedding[:, :seq_len, :]
		return h

	def _decode_hidden(self, h: torch.Tensor, logit_mask: torch.Tensor = None) -> torch.Tensor:
		"""
		Capas 4→5: Proyecta hidden state a logits sobre vocabulario.
		"""
		concept_proj = self.outbound_proj(h)
		logits = torch.matmul(concept_proj, self.vocab_embeddings.T)
		if logit_mask is not None:
			logits = logits.masked_fill(~logit_mask, -1e9)
		return logits

	def _sample_watcher(self, h: torch.Tensor) -> dict:
		"""
		Osciloscopio: espía qué 'piensa' el modelo sin afectar al bucle.
		Proyecta el hidden state a tokens mediante Capas 4→5 dentro de no_grad.
		No altera pesos ni hidden state. Pura lectura.
		"""
		concept_proj = self.outbound_proj(h)
		logits = torch.matmul(concept_proj, self.vocab_embeddings.T)
		tokens = logits.argmax(dim=-1)
		probs = F.softmax(logits, dim=-1)
		entropy = -(probs * probs.log().clamp(min=-100)).sum(dim=-1)
		return {
			"tokens": tokens,
			"top_logit": logits.max(dim=-1).values,
			"entropy": entropy,
		}

	def forward_resonance(
		self,
		x: torch.Tensor,
		n_steps: int = 3,
		pos_mode: str = "none",
		collect_watcher: bool = False,
		logit_mask: torch.Tensor = None,
		emotion_ids: torch.Tensor = None,
	) -> tuple[torch.Tensor, dict]:
		"""
		Bucle latente cerrado: el hidden state itera N veces por el core
		SIN salir a vocabulario (8192-dim). Solo al final se proyecta a logits.

		Topología:
			input → [Capa1→2: una sola vez] → h(256)
			                                     │
			                   ┌─────────────────┤
			                   │  core_layers    │
			                   │  + norm         │ × n_steps
			                   │  + clock (opt)  │
			                   └─────────────────┘
			                         │
			                   [Capa4→5: solo al final] → logits(8192)

		Args:
			x: Input tensor (token IDs o Gumbel-Softmax).
			n_steps: Número de iteraciones del bucle latente.
			pos_mode: 'none' | 'entry' | 'clock'
				- 'none':  Sin pos. embedding en el bucle.
				- 'entry': Pos. embedding de secuencia sumados una sola vez (ya hecho en _embed_input).
				- 'clock': Resonance clock re-sumado en cada iteración del bucle.
			collect_watcher: Si True, muestrea tokens en cada paso (sin gradiente).
			logit_mask: Máscara de logits para el paso final.

		Returns:
			(logits, metadata) donde metadata contiene métricas de estabilidad y watcher.
		"""
		# ── Entrada: Capas 1→2 (una sola vez) ──
		h = self._embed_input(x)
		h0_norm = h.norm(dim=-1).mean().item()

		# ── EXP_033: Preparar vector emocional ──
		emo_vec = None
		if emotion_ids is not None and self.emotion_embeddings is not None:
			# emotion_ids: (batch,) → emo_vec: (batch, 1, hidden_dim)
			emo_emb = self.emotion_embeddings(emotion_ids)  # (batch, emotion_dim)
			emo_vec = self.emotion_proj(emo_emb).unsqueeze(1)  # (batch, 1, hidden_dim)

		# ── Bucle Latente: Core itera N veces ──
		trajectory_norms = [h0_norm]
		cosine_convergence = []
		watcher_samples = []

		for step in range(n_steps):
			h_prev = h.detach()  # Para métricas (no afecta al gradiente del bucle)

			# Clock: re-sumar embedding de resonancia en cada paso
			if pos_mode == "clock" and getattr(self, "resonance_clock", None) is not None:
				h = h + self.resonance_clock[:, step, :].unsqueeze(1)  # (1, 1, 256) broadcast

			# ── EXP_033: Inyección emocional en el bucle ──
			if emo_vec is not None:
				if self.emotion_mode == "additive":
					h = h + emo_vec  # La emoción desplaza cada paso
				elif self.emotion_mode == "gated":
					gate = torch.sigmoid(self.emotion_gate(emo_vec.squeeze(1))).unsqueeze(1)
					h = h * gate + emo_vec  # La emoción filtra y desplaza
				elif self.emotion_mode == "first_only":
					if step == 0:
						h = h + emo_vec  # Impulso emocional solo al inicio

			# Paso por el core completo
			for layer in self.core_layers:
				h = layer(h)
			h = self.norm(h)

			# Métricas de estabilidad
			step_norm = h.norm(dim=-1).mean().item()
			trajectory_norms.append(step_norm)

			# Convergencia: coseno entre estado actual y anterior
			with torch.no_grad():
				cos_sim = F.cosine_similarity(h, h_prev, dim=-1).mean().item()
				cosine_convergence.append(cos_sim)

			# Watcher: muestreo asíncrono sin romper el bucle
			if collect_watcher:
				with torch.no_grad():
					watcher_samples.append(self._sample_watcher(h))

		# ── Salida: Capas 4→5 (solo al final) ──
		logits = self._decode_hidden(h, logit_mask=logit_mask)

		metadata = {
			"trajectory_norms": trajectory_norms,
			"cosine_convergence": cosine_convergence,
			"norm_ratio": trajectory_norms[-1] / (h0_norm + 1e-8),
			"watcher_samples": watcher_samples,
			"final_hidden": h.detach(),
		}
		return logits, metadata

	def forward_resonance_training(
		self,
		x: torch.Tensor,
		n_steps: int = 3,
		pos_mode: str = "none",
		intermediate_targets: dict[int, torch.Tensor] | None = None,
		logit_mask: torch.Tensor = None,
		emotion_ids: torch.Tensor = None,
	) -> tuple[torch.Tensor, list[tuple[int, torch.Tensor]]]:
		"""
		Variante de forward_resonance para entrenamiento con BPTT.
		Permite loss intermedio en pasos configurables.

		Args:
			x: Input tensor.
			n_steps: Iteraciones del bucle latente.
			pos_mode: 'none' | 'entry' | 'clock'.
			intermediate_targets: Dict {step_index: target_tensor}.
				Si se proporciona, se computan logits intermedios en esos pasos
				para calcular loss parcial. None = solo loss al final.
			logit_mask: Máscara de logits.

		Returns:
			(final_logits, intermediate_logits) donde intermediate_logits es
			lista de (step, logits) para los pasos con supervisión.
		"""
		h = self._embed_input(x)

		# ── EXP_033: Preparar vector emocional ──
		emo_vec = None
		if emotion_ids is not None and self.emotion_embeddings is not None:
			emo_emb = self.emotion_embeddings(emotion_ids)
			emo_vec = self.emotion_proj(emo_emb).unsqueeze(1)

		intermediate_logits = []
		target_steps = set(intermediate_targets.keys()) if intermediate_targets else set()

		for step in range(n_steps):
			if pos_mode == "clock" and getattr(self, "resonance_clock", None) is not None:
				h = h + self.resonance_clock[:, step, :].unsqueeze(1)

			# ── EXP_033: Inyección emocional ──
			if emo_vec is not None:
				if self.emotion_mode == "additive":
					h = h + emo_vec
				elif self.emotion_mode == "gated":
					gate = torch.sigmoid(self.emotion_gate(emo_vec.squeeze(1))).unsqueeze(1)
					h = h * gate + emo_vec
				elif self.emotion_mode == "first_only":
					if step == 0:
						h = h + emo_vec

			for layer in self.core_layers:
				h = layer(h)
			h = self.norm(h)

			# Loss intermedio: computar logits en este paso si se requiere supervisión
			if step in target_steps:
				logits_mid = self._decode_hidden(h, logit_mask=logit_mask)
				intermediate_logits.append((step, logits_mid))

		# Logits finales
		final_logits = self._decode_hidden(h, logit_mask=logit_mask)
		return final_logits, intermediate_logits
