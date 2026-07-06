import numpy as np
import torch

# EXP_034: Glifos ternarios composicionales
# Importación lazy para backward-compatibility
try:
	from src.bitnet.vocab.glyph_vocabulary import GLYPH_TABLE, N_PRIMES, GlyphEmbedding
except ImportError:
	GlyphEmbedding = None
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

	def __init__(self, dim: int, num_heads: int = 4, is_causal: bool = False):
		super().__init__()
		self.dim = dim
		self.num_heads = num_heads
		self.head_dim = dim // num_heads
		self.is_causal = is_causal

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
		if self.is_causal and seq_len > 1:
			mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
			fill_value = -65000.0 if scores.dtype == torch.float16 else -1e9
			scores = scores.masked_fill(mask, fill_value)
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

	def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: int = 4, is_causal: bool = False):
		super().__init__()
		self.attn_norm = RMSNorm(dim)
		self.attn = BitNetAttention(dim, num_heads, is_causal=is_causal)
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

	def __init__(self, vocab_embeddings: np.ndarray = None, hidden_dim: int = 256, num_layers: int = 4, use_pos_embedding: bool = False, max_resonance_steps: int = 0, n_emotions: int = 0, emotion_dim: int = 0, emotion_mode: str = "additive", use_glyphs: bool = False, glyph_table: np.ndarray = None, action_head_width: int = None, is_causal: bool = False, max_seq_len: int = 64):
		super().__init__()
		self.hidden_dim = hidden_dim
		self.use_pos_embedding = use_pos_embedding
		self.max_resonance_steps = max_resonance_steps
		self.emotion_mode = emotion_mode  # 'additive', 'gated', 'first_only'
		self.use_glyphs = use_glyphs
		self.is_causal = is_causal

		# ── EXP_034: Modo Glifos Ternarios ──
		if use_glyphs:
			assert GlyphEmbedding is not None, "glyph_vocabulary.py not found"
			self.glyph_embedding = GlyphEmbedding(
				hidden_dim=hidden_dim,
				glyph_table=glyph_table,
			)
			self.vocab_size = self.glyph_embedding.vocab_size
			self.vocab_dim = hidden_dim  # Los glifos producen hidden_dim directamente
			# No necesitamos inbound_proj ni outbound_proj ni vocab_embeddings
			self.register_buffer("vocab_embeddings", None)
			self.inbound_proj = None
			self.outbound_proj = None
		else:
			# ── Modo clasico: fastembed lookup ──
			assert vocab_embeddings is not None, "vocab_embeddings required when use_glyphs=False"
			self.vocab_size, self.vocab_dim = vocab_embeddings.shape
			self.glyph_embedding = None
			# Registrar los embeddings del vocabulario conceptual como un buffer no entrenable (Capa 1 fija)
			self.register_buffer("vocab_embeddings", torch.from_numpy(vocab_embeddings).float())
			# Capa 2: Inbound Translator (Proyección del embedding de 384-dim al espacio oculto del Core de 256-dim)
			self.inbound_proj = nn.Linear(self.vocab_dim, hidden_dim, bias=False)
			# Capa 4: Outbound Translator (Proyección del espacio oculto de 256-dim al espacio conceptual de 384-dim)
			self.outbound_proj = nn.Linear(hidden_dim, self.vocab_dim, bias=False)

		# Capa de Posición: Embeddings Posicionales Aprendibles
		if self.use_pos_embedding:
			self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.02)
		else:
			self.register_parameter("pos_embedding", None)

		# Capa 3: Specialist Core (Ternary Transformer)
		self.core_layers = nn.ModuleList([BitNetTransformerBlock(dim=hidden_dim, num_heads=4, mlp_ratio=4, is_causal=is_causal) for _ in range(num_layers)])
		self.norm = RMSNorm(hidden_dim)

		# Resonancia Continua: Reloj posicional para el bucle latente (EXP_032)
		if max_resonance_steps > 0:
			self.resonance_clock = nn.Parameter(torch.randn(1, max_resonance_steps, hidden_dim) * 0.02)
		else:
			self.register_parameter("resonance_clock", None)

		# ── EXP_033: Resonancia Emocional ──
		if n_emotions > 0 and emotion_dim > 0:
			self.emotion_embeddings = nn.Embedding(n_emotions, emotion_dim)
			self.emotion_proj = nn.Linear(emotion_dim, hidden_dim, bias=False)
			if emotion_mode == "gated":
				self.emotion_gate = nn.Linear(hidden_dim, hidden_dim, bias=False)
		else:
			self.emotion_embeddings = None
			self.emotion_proj = None

		# ── EXP_039: Cabeza de Acción (puente saber→actuar) ──
		_ahw = action_head_width if action_head_width is not None else hidden_dim // 2
		self.action_head = nn.Sequential(
			nn.Linear(hidden_dim, _ahw),
			nn.GELU(),
			nn.Linear(_ahw, 6),  # 6 acciones: comer, beber, dormir, mover, ver, piedra
		)

		# ── EXP_050: Cabeza de Valor para Actor-Critic/PPO ──
		self.value_head = nn.Sequential(
			nn.Linear(hidden_dim, _ahw),
			nn.GELU(),
			nn.Linear(_ahw, 1),
		)

		# ── EXP_071: Cabeza de Proyección de Glifos (Definición -> 65 Primos) ──
		self.glyph_projection_head = nn.Sequential(
			nn.Linear(hidden_dim, 128),
			nn.GELU(),
			nn.Linear(128, 65),
			nn.Tanh()
		)

	def register_new_word(self, word_name: str, glyph_vector: torch.Tensor):
		"""
		Registra dinámicamente una nueva palabra en el modelo.
		Añade el vector de trits (65,) a self.glyph_embedding.glyph_table.
		"""
		assert self.use_glyphs, "La expansión de vocabulario dinámico solo es compatible con el modo Glifos (use_glyphs=True)"
		device = self.glyph_embedding.glyph_table.device
		glyph_vector = glyph_vector.to(device).float()
		new_table = torch.cat([self.glyph_embedding.glyph_table, glyph_vector.unsqueeze(0)], dim=0)
		del self.glyph_embedding.glyph_table
		self.glyph_embedding.register_buffer("glyph_table", new_table)
		self.vocab_size = self.glyph_embedding.vocab_size

	def forward(self, x: torch.Tensor, logit_mask: torch.Tensor = None, tau: float = None) -> torch.Tensor:
		"""
		Paso forward estándar (sin resonancia).
		x puede ser:
		- Tensor de enteros (batch_size, seq_len): Token IDs discretos
		- Tensor float (batch_size, seq_len, vocab_size): Gumbel-Softmax
		Soporta modo clásico (fastembed) y modo glifo (EXP_034).
		"""
		h = self._embed_input(x)

		# Capa 3: Specialist Core
		for layer in self.core_layers:
			h = layer(h)
		h = self.norm(h)

		# Capa 4→5: Decode a logits
		logits = self._decode_hidden(h, logit_mask=logit_mask)
		return logits

	def generate_message(self, x: torch.Tensor, tau: float = 1.0, hard: bool = True, logit_mask: torch.Tensor = None, mode: str = "gumbel") -> torch.Tensor:
		"""
		Genera un mensaje discreto/relajado manteniendo la diferenciabilidad del canal.
		Soporta:
		  - "gumbel": Gumbel-Softmax (con temperatura tau).
		  - "ste": Straight-Through Estimator (argmax + gradiente de softmax).
		"""
		logits = self.forward(x, logit_mask=logit_mask)
		if mode == "ste":
			# Softmax probability for backward pass
			y_soft = F.softmax(logits / tau, dim=-1)
			# Argmax for forward pass
			indices = logits.argmax(dim=-1)
			y_hard = F.one_hot(indices, num_classes=logits.size(-1)).float()
			# Straight-Through trick
			message = y_hard - y_soft.detach() + y_soft
		else:
			# Default: Gumbel-Softmax
			message = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)
		return message

	def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
		key = prefix + "pos_embedding"
		if key in state_dict and state_dict[key] is not None:
			param_shape = state_dict[key].shape
			if getattr(self, "pos_embedding", None) is None or self.pos_embedding.shape != param_shape:
				self.pos_embedding = nn.Parameter(torch.zeros(param_shape, device=state_dict[key].device))
				self.use_pos_embedding = True
		super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

	# ── Resonancia Continua (EXP_032) ─────────────────────────────────────────────

	def _embed_input(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Capas 1→2: Proyecta input (token IDs o Gumbel-Softmax) al espacio oculto.
		Soporta modo clásico (fastembed) y modo glifo (EXP_034).
		"""
		if self.use_glyphs:
			# EXP_034: Glifos → composición de primos → hidden_dim directamente
			if x.ndim == 2:
				h = self.glyph_embedding(x)  # (batch, seq, hidden_dim)
			else:
				# Gumbel path: soft_tokens @ word_embeddings
				word_embeds = self.glyph_embedding.get_word_embeddings()
				h = torch.matmul(x, word_embeds)  # (batch, seq, hidden_dim)
		else:
			# Modo clásico: fastembed lookup + inbound projection
			embeds = F.embedding(x, self.vocab_embeddings) if x.ndim == 2 else torch.matmul(x, self.vocab_embeddings)
			h = self.inbound_proj(embeds)

		if getattr(self, "pos_embedding", None) is not None:
			seq_len = x.shape[1]
			pos = self.pos_embedding[:, :seq_len, :]
			if pos.shape[1] < seq_len:
				# Extra tokens (e.g., communication channel) get no positional encoding
				pad = torch.zeros(1, seq_len - pos.shape[1], self.hidden_dim, device=pos.device)
				pos = torch.cat([pos, pad], dim=1)
			h = h + pos
		return h

	def _decode_hidden(self, h: torch.Tensor, logit_mask: torch.Tensor = None) -> torch.Tensor:
		"""
		Capas 4→5: Proyecta hidden state a logits sobre vocabulario.
		Soporta modo clásico (fastembed) y modo glifo (EXP_034).
		"""
		if self.use_glyphs:
			# EXP_034: Cosine similarity con word embeddings composicionales
			logits = self.glyph_embedding.decode_logits(h)
		else:
			# Modo clásico: outbound projection + similarity
			concept_proj = self.outbound_proj(h)
			logits = torch.matmul(concept_proj, self.vocab_embeddings.T)

		if logit_mask is not None:
			logits = logits.masked_fill(~logit_mask, -1e9)
		return logits

	def _sample_watcher(self, h: torch.Tensor) -> dict:
		"""
		Osciloscopio: espía qué 'piensa' el modelo sin afectar al bucle.
		Proyecta el hidden state a tokens. Pura lectura, sin gradiente.
		"""
		logits = self._decode_hidden(h)  # Reutiliza el path correcto (glyph o classic)
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
		h_prev: torch.Tensor = None,
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
			h_prev: Estado latente de la iteración temporal anterior (Resonancia Temporal).

		Returns:
			(logits, metadata) donde metadata contiene métricas de estabilidad y watcher.
		"""
		# ── Entrada: Capas 1→2 (una sola vez) ──
		h = self._embed_input(x)
		if h_prev is not None:
			h = h + 0.5 * h_prev
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
				elif self.emotion_mode == "first_only" and step == 0:
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
			"hidden": h,  # EXP_039: non-detached for action_head gradient
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
		h_prev: torch.Tensor = None,
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
			h_prev: Estado latente de la iteración temporal anterior (Resonancia Temporal).

		Returns:
			(final_logits, intermediate_logits) donde intermediate_logits es
			lista de (step, logits) para los pasos con supervisión.
		"""
		h = self._embed_input(x)
		if h_prev is not None:
			h = h + 0.5 * h_prev

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
				elif self.emotion_mode == "first_only" and step == 0:
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

	# ── Metacognición (EXP_036) ───────────────────────────────────────────────

	def forward_deep_think(
		self,
		x: torch.Tensor,
		n_think: int = 3,
		n_verify: int = 2,
		pos_mode: str = "clock",
		logit_mask: torch.Tensor = None,
		emotion_ids: torch.Tensor = None,
		max_rethink: int = 1,
		convergence_threshold: float = 0.95,
	) -> tuple[torch.Tensor, dict]:
		"""
		Pensamiento en dos fases: Pensar + Verificar.

		Fase 1 (Pensar): Bucle latente normal (resonancia).
		Fase 2 (Verificar): Decodificar resultado, re-inyectarlo como input,
		                     y ejecutar un segundo bucle. Si el pensamiento
		                     converge (cos(h1, h2) > threshold), el modelo
		                     está "seguro". Si diverge, re-piensa.

		Analogía humana:
		  Fase 1: Piensas la respuesta mentalmente.
		  Fase 2: La dices en voz alta y te escuchas.
		  Si suena bien: la confirmas.
		  Si suena raro: "espera, déjame repensarlo".

		Origen: Joan Garcia — "el pensamiento en voz alta me ha servido
		para focalizar y no desviarme"

		Args:
		    x: Input tensor (token IDs o Gumbel-Softmax).
		    n_think: Steps del bucle de pensamiento (Fase 1).
		    n_verify: Steps del bucle de verificación (Fase 2).
		    pos_mode: Modo posicional ('none', 'entry', 'clock').
		    logit_mask: Máscara de logits.
		    emotion_ids: IDs de emoción para modular el pensamiento.
		    max_rethink: Máximo de iteraciones adicionales si no converge.
		    convergence_threshold: Umbral coseno para considerar convergente.

		Returns:
		    (logits, metadata) donde metadata incluye confianza y trazas.
		"""
		rethink_trace = []

		# ═══ FASE 1: PENSAR (silencio) ═══
		logits_think, meta_think = self.forward_resonance(
			x, n_steps=n_think, pos_mode=pos_mode,
			logit_mask=logit_mask, emotion_ids=emotion_ids,
		)
		h_think = meta_think["final_hidden"]  # (batch, seq, hidden_dim)

		current_logits = logits_think
		current_hidden = h_think

		for rethink_step in range(1 + max_rethink):
			# ═══ FASE 2: VERIFICAR (en voz alta) ═══
			# Decodificar a tokens discretos
			with torch.no_grad():
				result_tokens = current_logits.argmax(dim=-1)  # (batch, seq)

			# Re-inyectar como nuevo input
			logits_verify, meta_verify = self.forward_resonance(
				result_tokens, n_steps=n_verify, pos_mode=pos_mode,
				logit_mask=logit_mask, emotion_ids=emotion_ids,
			)
			h_verify = meta_verify["final_hidden"]

			# Medir convergencia: ¿piensa lo mismo?
			with torch.no_grad():
				# Coseno por posición de secuencia, promedio sobre batch
				convergence = F.cosine_similarity(
					current_hidden, h_verify, dim=-1
				).mean(dim=-1)  # (batch,)
				avg_convergence = convergence.mean().item()

			rethink_trace.append({
				"step": rethink_step,
				"convergence": avg_convergence,
				"converged": avg_convergence >= convergence_threshold,
				"tokens_think": result_tokens[:1].tolist() if result_tokens.shape[0] > 0 else [],
				"tokens_verify": logits_verify.argmax(dim=-1)[:1].tolist() if logits_verify.shape[0] > 0 else [],
			})

			if avg_convergence >= convergence_threshold:
				# Convergente → confiado
				break

			# No convergente → el resultado de Fase 2 se convierte
			# en el nuevo "pensamiento" a verificar
			current_logits = logits_verify
			current_hidden = h_verify

		# Output final: el último resultado verificado
		final_logits = logits_verify
		final_convergence = avg_convergence

		metadata = {
			"phase1_hidden": h_think,
			"phase2_hidden": h_verify,
			"hidden": h_verify,  # EXP_039: para action_head
			"convergence": final_convergence,
			"converged": final_convergence >= convergence_threshold,
			"n_rethinks": len(rethink_trace),
			"rethink_trace": rethink_trace,
			"meta_think": meta_think,
			"meta_verify": meta_verify,
		}

		return final_logits, metadata

	def forward_deep_think_training(
		self,
		x: torch.Tensor,
		n_think: int = 3,
		n_verify: int = 2,
		pos_mode: str = "clock",
		logit_mask: torch.Tensor = None,
		emotion_ids: torch.Tensor = None,
		tau: float = 0.5,
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
		"""
		Variante de forward_deep_think para entrenamiento con BPTT.

		Mantiene el grafo computacional entre Fase 1 y Fase 2 para
		que los gradientes fluyan de la verificación al pensamiento.

		Returns:
		    (logits_think, logits_verify, convergence, metadata)
		    - logits_think: (batch, seq, vocab) — resultado de Fase 1
		    - logits_verify: (batch, seq, vocab) — resultado de Fase 2
		    - convergence: (batch,) — coseno entre hidden states
		    - metadata: dict con trazas
		"""
		# ═══ FASE 1: PENSAR ═══
		# Usamos forward_resonance_training para mantener gradientes
		intermediate_targets_think = dict.fromkeys(range(n_think))
		logits_think, intermediates_think = self.forward_resonance_training(
			x, n_steps=n_think, pos_mode=pos_mode,
			intermediate_targets=intermediate_targets_think,
			logit_mask=logit_mask, emotion_ids=emotion_ids,
		)

		# Capturar hidden state final de Fase 1 (con gradiente)
		h_think = self._embed_input(x)
		emo_vec = None
		if emotion_ids is not None and self.emotion_embeddings is not None:
			emo_emb = self.emotion_embeddings(emotion_ids)
			emo_vec = self.emotion_proj(emo_emb).unsqueeze(1)
		for step in range(n_think):
			if pos_mode == "clock" and getattr(self, "resonance_clock", None) is not None:
				h_think = h_think + self.resonance_clock[:, step, :].unsqueeze(1)
			if emo_vec is not None and (self.emotion_mode == "additive" or self.emotion_mode == "first_only" and step == 0):
				h_think = h_think + emo_vec
			for layer in self.core_layers:
				h_think = layer(h_think)
			h_think = self.norm(h_think)

		# ═══ PUENTE: Decodificar a tokens (Gumbel-Softmax diferenciable) ═══
		logits_bridge = self._decode_hidden(h_think, logit_mask=logit_mask)
		soft_tokens = F.gumbel_softmax(logits_bridge, tau=tau, hard=False, dim=-1)

		# ═══ FASE 2: VERIFICAR (re-inyección diferenciable) ═══
		h_verify = self._embed_input(soft_tokens)
		if emo_vec is not None and (self.emotion_mode == "additive" or self.emotion_mode == "first_only"):
			h_verify = h_verify + emo_vec
		for step in range(n_verify):
			if pos_mode == "clock" and getattr(self, "resonance_clock", None) is not None:
				# Usar slots diferentes del clock para verificación
				clock_idx = min(n_think + step, self.resonance_clock.shape[1] - 1)
				h_verify = h_verify + self.resonance_clock[:, clock_idx, :].unsqueeze(1)
			for layer in self.core_layers:
				h_verify = layer(h_verify)
			h_verify = self.norm(h_verify)

		logits_verify = self._decode_hidden(h_verify, logit_mask=logit_mask)

		# ═══ CONVERGENCIA (diferenciable) ═══
		convergence = F.cosine_similarity(
			h_think, h_verify, dim=-1
		).mean(dim=-1)  # (batch,)

		metadata = {
			"intermediates_think": intermediates_think,
			"h_think": h_think,
			"h_verify": h_verify,
		}

		return logits_think, logits_verify, convergence, metadata

