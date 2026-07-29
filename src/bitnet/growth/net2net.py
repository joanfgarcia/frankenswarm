"""
Net2Net: Crecimiento neural preservando conocimiento.

Implementa Net2WiderNet (más neuronas) para el action head.
El modelo crece cuando lo necesita sin perder lo aprendido.

Ref: Chen, Goodfellow, Shlens — "Net2Net: Accelerating Learning via Knowledge Transfer" (2015)
Aplicación: Neurogenésis artificial guiada por convergencia.

Origen: Joan Garcia — "Net2Net en caliente?" — 2026-05-31
"""

import numpy as np
import torch
import torch.nn as nn


def net2wider_linear(layer_in: nn.Linear, layer_out: nn.Linear, new_width: int, noise_std: float = 0.01):
	"""
	Net2WiderNet: Ampliar una capa hidden preservando la función.

	layer_in:	Linear(in_features, old_width)  → será Linear(in_features, new_width)
	layer_out:	Linear(old_width, out_features)  → será Linear(new_width, out_features)
	new_width:	nuevo ancho (> old_width)

	Mecanismo:
		1. Las neuronas originales se mantienen intactas.
		2. Las neuronas nuevas copian neuronas existentes (random).
		3. Los pesos de salida de las copias se dividen por n_copies.
		4. Se añade ruido pequeño para romper simetría.

	Returns:
		(new_layer_in, new_layer_out) con los mismos comportamientos funcionales.
	"""
	old_width = layer_in.out_features
	assert new_width > old_width, f"new_width ({new_width}) must be > old_width ({old_width})"

	in_features = layer_in.in_features
	out_features = layer_out.out_features
	n_new = new_width - old_width

	# ═══ Seleccionar qué neuronas copiar ═══
	# Elegir aleatoriamente qué neuronas existentes clonar
	clone_indices = torch.randint(0, old_width, (n_new,))

	# ═══ Nueva capa de entrada (in → new_width) ═══
	new_in = nn.Linear(in_features, new_width, bias=layer_in.bias is not None)
	with torch.no_grad():
		# Copiar pesos originales
		new_in.weight[:old_width] = layer_in.weight.clone()
		if layer_in.bias is not None:
			new_in.bias[:old_width] = layer_in.bias.clone()

		# Copiar neuronas clonadas + ruido
		for i, src_idx in enumerate(clone_indices):
			new_in.weight[old_width + i] = layer_in.weight[src_idx].clone()
			new_in.weight[old_width + i] += torch.randn_like(new_in.weight[old_width + i]) * noise_std * 0.1
			if layer_in.bias is not None:
				new_in.bias[old_width + i] = layer_in.bias[src_idx].clone()

	# ═══ Nueva capa de salida (new_width → out) ═══
	new_out = nn.Linear(new_width, out_features, bias=layer_out.bias is not None)
	with torch.no_grad():
		# Copiar pesos originales
		new_out.weight[:, :old_width] = layer_out.weight.clone()
		if layer_out.bias is not None:
			new_out.bias[:] = layer_out.bias.clone()

		# Para cada neurona clonada: dividir pesos de salida
		# Contar cuántas copias tiene cada neurona original
		copy_count = torch.ones(old_width)
		for src_idx in clone_indices:
			copy_count[src_idx] += 1

		# Ajustar pesos originales por n_copies
		for j in range(old_width):
			if copy_count[j] > 1:
				new_out.weight[:, j] /= copy_count[j]

		# Pesos de las copias = peso original / n_copies
		for i, src_idx in enumerate(clone_indices):
			new_out.weight[:, old_width + i] = layer_out.weight[:, src_idx].clone() / copy_count[src_idx]
			new_out.weight[:, old_width + i] += torch.randn_like(new_out.weight[:, old_width + i]) * noise_std

	return new_in, new_out


def grow_action_head(model: nn.Module, growth_factor: float = 1.5, noise_std: float = 0.01) -> dict:
	"""
	Hacer crecer el action_head del modelo via Net2WiderNet.

	action_head = Sequential(
		Linear(hidden_dim, width),     # layer 0
		GELU(),                         # layer 1
		Linear(width, 6),              # layer 2
	)

	Crece la capa hidden (layer 0 out / layer 2 in) por growth_factor.

	Returns:
		dict con info del crecimiento
	"""
	head = model.action_head
	layer_in = head[0]  # Linear(hidden_dim, old_width)
	layer_out = head[2]  # Linear(old_width, 6)

	old_width = layer_in.out_features
	new_width = int(old_width * growth_factor)

	# Hacer crecer
	new_layer_in, new_layer_out = net2wider_linear(layer_in, layer_out, new_width, noise_std=noise_std)

	# Mover a mismo device
	device = next(model.parameters()).device
	new_layer_in = new_layer_in.to(device)
	new_layer_out = new_layer_out.to(device)

	# Reemplazar en el modelo
	model.action_head[0] = new_layer_in
	model.action_head[2] = new_layer_out

	old_params = old_width * (layer_in.in_features + 1) + 6 * (old_width + 1)
	new_params = new_width * (layer_in.in_features + 1) + 6 * (new_width + 1)

	info = {
		"old_width": old_width,
		"new_width": new_width,
		"old_params": old_params,
		"new_params": new_params,
		"growth_factor": growth_factor,
		"neurons_added": new_width - old_width,
	}

	return info


def get_head_aligned_mapping(old_dim, new_dim, num_heads=4):
	old_head_dim = old_dim // num_heads
	new_head_dim = new_dim // num_heads
	n_new_per_head = new_head_dim - old_head_dim

	# Generate random clone indices within each head
	head_clone_indices = torch.randint(0, old_head_dim, (n_new_per_head,))

	# Calculate copy count for one head
	head_copy_count = torch.ones(old_head_dim)
	for src_idx in head_clone_indices:
		head_copy_count[src_idx] += 1

	# Build global mapping g, copy_count, and is_clone mask
	g = []
	copy_count = []
	is_clone = []
	for h in range(num_heads):
		head_start = h * old_head_dim
		for k in range(new_head_dim):
			if k < old_head_dim:
				src_idx = head_start + k
				is_clone.append(False)
			else:
				src_idx = head_start + head_clone_indices[k - old_head_dim].item()
				is_clone.append(True)
			g.append(src_idx)

		for k in range(old_head_dim):
			copy_count.append(head_copy_count[k].item())

	return (
		torch.tensor(g, dtype=torch.long),
		torch.tensor(copy_count, dtype=torch.float),
		torch.tensor(is_clone, dtype=torch.bool),
	)


def get_random_mapping(old_dim, new_dim):
	n_new = new_dim - old_dim
	clone_indices = torch.randint(0, old_dim, (n_new,))
	copy_count = torch.ones(old_dim)
	for src_idx in clone_indices:
		copy_count[src_idx] += 1

	g = list(range(old_dim)) + clone_indices.tolist()
	is_clone = [False] * old_dim + [True] * n_new

	return (
		torch.tensor(g, dtype=torch.long),
		torch.tensor(copy_count, dtype=torch.float),
		torch.tensor(is_clone, dtype=torch.bool),
	)


def net2wider_model(
	old_model: nn.Module,
	new_hidden_dim: int,
	noise_std: float = 0.01,
	old_optimizer: torch.optim.Optimizer = None,
	new_optimizer: torch.optim.Optimizer = None
) -> nn.Module:
	"""
	Net2WiderNet: Expande la dimensión oculta (hidden_dim) de todo el modelo BitNet4LayerModel.
	Transfiere y adapta todos los pesos y bias de forma funcionalmente equivalente (preserva la salida).
	"""
	from src.bitnet.model.modeling_bitnet import BitNet4LayerModel

	old_hidden_dim = old_model.hidden_dim
	assert new_hidden_dim > old_hidden_dim, f"new_hidden_dim ({new_hidden_dim}) must be > old_hidden_dim ({old_hidden_dim})"

	# 1. Recrear el nuevo modelo con la misma estructura pero con new_hidden_dim
	use_pos_embedding = old_model.use_pos_embedding
	max_resonance_steps = old_model.resonance_clock.shape[1] if getattr(old_model, "resonance_clock", None) is not None else 0
	n_emotions = old_model.emotion_embeddings.num_embeddings if old_model.emotion_embeddings is not None else 0
	emotion_dim = old_model.emotion_embeddings.embedding_dim if old_model.emotion_embeddings is not None else 0
	emotion_mode = old_model.emotion_mode
	use_glyphs = old_model.use_glyphs

	device = next(old_model.parameters()).device

	if use_glyphs:
		glyph_table = old_model.glyph_embedding.glyph_table.cpu().float().numpy()
		vocab_embeddings = None
	else:
		glyph_table = None
		vocab_embeddings = old_model.vocab_embeddings.cpu().numpy()

	# Configuración de action_head y value_head
	old_ahw = old_model.action_head[0].out_features
	new_ahw = new_hidden_dim // 2

	new_model = BitNet4LayerModel(
		vocab_embeddings=vocab_embeddings,
		hidden_dim=new_hidden_dim,
		num_layers=len(old_model.core_layers),
		use_pos_embedding=use_pos_embedding,
		max_resonance_steps=max_resonance_steps,
		n_emotions=n_emotions,
		emotion_dim=emotion_dim,
		emotion_mode=emotion_mode,
		use_glyphs=use_glyphs,
		glyph_table=glyph_table,
		action_head_width=new_ahw,
		is_causal=old_model.is_causal,
		max_seq_len=old_model.pos_embedding.shape[1] if getattr(old_model, "pos_embedding", None) is not None else 64,
	).to(device)

	# 2. Generar mapeos
	num_heads = 4  # hardcoded en BitNetTransformerBlock
	g, copy_count, is_clone = get_head_aligned_mapping(old_hidden_dim, new_hidden_dim, num_heads=num_heads)
	g = g.to(device)
	copy_count = copy_count.to(device)
	is_clone = is_clone.to(device)

	g_ahw, ahw_copy_count, ahw_is_clone = get_random_mapping(old_ahw, new_ahw)
	g_ahw = g_ahw.to(device)
	ahw_copy_count = ahw_copy_count.to(device)
	ahw_is_clone = ahw_is_clone.to(device)

	def transfer_state(old_p, new_p, map_fn, scale_fn=None):
		if old_optimizer is None or new_optimizer is None:
			return
		if old_p not in old_optimizer.state:
			return
		old_state = old_optimizer.state[old_p]
		new_state = {}
		if "step" in old_state:
			step_val = old_state["step"]
			new_state["step"] = step_val.clone() if isinstance(step_val, torch.Tensor) else step_val
		if "exp_avg" in old_state:
			m = old_state["exp_avg"]
			mapped_m = map_fn(m)
			if scale_fn is not None:
				mapped_m = scale_fn(mapped_m, 1)
			new_state["exp_avg"] = mapped_m.clone()
		if "exp_avg_sq" in old_state:
			v = old_state["exp_avg_sq"]
			mapped_v = map_fn(v)
			if scale_fn is not None:
				mapped_v = scale_fn(mapped_v, 2)
			new_state["exp_avg_sq"] = mapped_v.clone()
		new_optimizer.state[new_p] = new_state

	with torch.no_grad():
		# --- A. Glyph Embedding ---
		if use_glyphs:
			old_pe = old_model.glyph_embedding.prime_embeddings.data
			new_pe = old_pe[:, g]
			noise = torch.randn_like(new_pe) * noise_std
			new_pe = new_pe + noise * is_clone.unsqueeze(0)
			new_model.glyph_embedding.prime_embeddings.copy_(new_pe)

			transfer_state(
				old_model.glyph_embedding.prime_embeddings,
				new_model.glyph_embedding.prime_embeddings,
				lambda x: x[:, g],
				lambda x, power: x / (copy_count[g].unsqueeze(0) ** power)
			)
		else:
			old_w = old_model.inbound_proj.weight.data
			new_w = old_w[g, :]
			noise = torch.randn_like(new_w) * noise_std
			new_w = new_w + noise * is_clone.unsqueeze(1)
			new_model.inbound_proj.weight.copy_(new_w)

			transfer_state(
				old_model.inbound_proj.weight,
				new_model.inbound_proj.weight,
				lambda x: x[g, :],
				lambda x, power: x / (copy_count[g].unsqueeze(1) ** power)
			)

			if old_model.inbound_proj.bias is not None:
				new_model.inbound_proj.bias.copy_(old_model.inbound_proj.bias.data[g])
				transfer_state(
					old_model.inbound_proj.bias,
					new_model.inbound_proj.bias,
					lambda x: x[g],
					lambda x, power: x / (copy_count[g] ** power)
				)

		# --- B. Pos Embedding & Resonance Clock ---
		if use_pos_embedding and getattr(old_model, "pos_embedding", None) is not None:
			old_pos = old_model.pos_embedding.data
			new_pos = old_pos[:, :, g]
			noise = torch.randn_like(new_pos) * noise_std
			new_pos = new_pos + noise * is_clone.view(1, 1, -1)
			new_model.pos_embedding.copy_(new_pos)

			transfer_state(
				old_model.pos_embedding,
				new_model.pos_embedding,
				lambda x: x[:, :, g],
				lambda x, power: x / (copy_count[g].view(1, 1, -1) ** power)
			)

		if getattr(old_model, "resonance_clock", None) is not None:
			old_clock = old_model.resonance_clock.data
			new_clock = old_clock[:, :, g]
			noise = torch.randn_like(new_clock) * noise_std
			new_clock = new_clock + noise * is_clone.view(1, 1, -1)
			new_model.resonance_clock.copy_(new_clock)

			transfer_state(
				old_model.resonance_clock,
				new_model.resonance_clock,
				lambda x: x[:, :, g],
				lambda x, power: x / (copy_count[g].view(1, 1, -1) ** power)
			)

		# --- C. Emotion Proj ---
		if old_model.emotion_proj is not None:
			old_w = old_model.emotion_proj.weight.data
			new_w = old_w[g, :]
			noise = torch.randn_like(new_w) * noise_std
			new_w = new_w + noise * is_clone.unsqueeze(1)
			new_model.emotion_proj.weight.copy_(new_w)

			transfer_state(
				old_model.emotion_proj.weight,
				new_model.emotion_proj.weight,
				lambda x: x[g, :],
				lambda x, power: x / (copy_count[g].unsqueeze(1) ** power)
			)

			if old_model.emotion_proj.bias is not None:
				new_model.emotion_proj.bias.copy_(old_model.emotion_proj.bias.data[g])
				transfer_state(
					old_model.emotion_proj.bias,
					new_model.emotion_proj.bias,
					lambda x: x[g],
					lambda x, power: x / (copy_count[g] ** power)
				)

			if emotion_mode == "gated":
				old_w = old_model.emotion_gate.weight.data
				new_w = old_w[g][:, g] / copy_count[g].unsqueeze(0)
				noise = torch.randn_like(new_w) * noise_std
				clone_mask = is_clone.unsqueeze(0) | is_clone.unsqueeze(1)
				new_w = new_w + noise * clone_mask
				new_model.emotion_gate.weight.copy_(new_w)

				transfer_state(
					old_model.emotion_gate.weight,
					new_model.emotion_gate.weight,
					lambda x: x[g][:, g],
					lambda x, power: x / ((copy_count[g].unsqueeze(0) * copy_count[g].unsqueeze(1)) ** power)
				)

				if old_model.emotion_gate.bias is not None:
					new_model.emotion_gate.bias.copy_(old_model.emotion_gate.bias.data[g])
					transfer_state(
						old_model.emotion_gate.bias,
						new_model.emotion_gate.bias,
						lambda x: x[g],
						lambda x, power: x / (copy_count[g] ** power)
					)

			new_model.emotion_embeddings.weight.copy_(old_model.emotion_embeddings.weight.data)
			transfer_state(
				old_model.emotion_embeddings.weight,
				new_model.emotion_embeddings.weight,
				lambda x: x,
				None
			)

		# --- D. Core Layers (Transformer Blocks) ---
		for l_idx, (old_block, new_block) in enumerate(zip(old_model.core_layers, new_model.core_layers, strict=False)):
			new_block.attn_norm.weight.copy_(old_block.attn_norm.weight.data[g])
			transfer_state(
				old_block.attn_norm.weight,
				new_block.attn_norm.weight,
				lambda x: x[g],
				lambda x, power: x / (copy_count[g] ** power)
			)
			if getattr(old_block.attn_norm, "bias", None) is not None:
				new_block.attn_norm.bias.copy_(old_block.attn_norm.bias.data[g])
				transfer_state(
					old_block.attn_norm.bias,
					new_block.attn_norm.bias,
					lambda x: x[g],
					lambda x, power: x / (copy_count[g] ** power)
				)

			# Attention: q_proj, k_proj, v_proj
			old_head_dim = old_hidden_dim // num_heads
			new_head_dim = new_hidden_dim // num_heads
			scale_factor = np.sqrt(new_head_dim / old_head_dim)

			for proj_name in ["q_proj", "k_proj", "v_proj"]:
				old_proj = getattr(old_block.attn, proj_name)
				new_proj = getattr(new_block.attn, proj_name)
				old_w = old_proj.weight.data
				new_w = old_w[g][:, g] / copy_count[g].unsqueeze(0)

				if proj_name == "q_proj":
					scale_vector = scale_factor / torch.sqrt(copy_count[g])
					new_w = new_w * scale_vector.unsqueeze(1)
					scale_w = (1.0 / copy_count[g].unsqueeze(0)) * (scale_factor / torch.sqrt(copy_count[g].unsqueeze(1)))
				elif proj_name == "k_proj":
					scale_vector = 1.0 / torch.sqrt(copy_count[g])
					new_w = new_w * scale_vector.unsqueeze(1)
					scale_w = (1.0 / copy_count[g].unsqueeze(0)) * (1.0 / torch.sqrt(copy_count[g].unsqueeze(1)))
				elif proj_name == "v_proj":
					scale_w = 1.0 / copy_count[g].unsqueeze(0)

				noise = torch.randn_like(new_w) * noise_std
				clone_mask = is_clone.unsqueeze(0) | is_clone.unsqueeze(1)
				new_w = new_w + noise * clone_mask
				new_proj.weight.copy_(new_w)

				transfer_state(
					old_proj.weight,
					new_proj.weight,
					lambda x: x[g][:, g],
					lambda x, power, s_w=scale_w: x * (s_w ** power)
				)

				if old_proj.bias is not None:
					new_proj.bias.copy_(old_proj.bias.data[g])
					transfer_state(
						old_proj.bias,
						new_proj.bias,
						lambda x: x[g],
						lambda x, power: x / (copy_count[g] ** power)
					)

			# out_proj
			old_proj = old_block.attn.out_proj
			new_proj = new_block.attn.out_proj
			old_w = old_proj.weight.data
			new_w = old_w[g][:, g] / copy_count[g].unsqueeze(0)
			noise = torch.randn_like(new_w) * noise_std
			clone_mask = is_clone.unsqueeze(0) | is_clone.unsqueeze(1)
			new_w = new_w + noise * clone_mask
			new_proj.weight.copy_(new_w)

			transfer_state(
				old_proj.weight,
				new_proj.weight,
				lambda x: x[g][:, g],
				lambda x, power: x / (copy_count[g].unsqueeze(0) ** power)
			)

			if old_proj.bias is not None:
				new_proj.bias.copy_(old_proj.bias.data[g])
				transfer_state(
					old_proj.bias,
					new_proj.bias,
					lambda x: x[g],
					lambda x, power: x / (copy_count[g] ** power)
				)

			# mlp_norm
			new_block.mlp_norm.weight.copy_(old_block.mlp_norm.weight.data[g])
			transfer_state(
				old_block.mlp_norm.weight,
				new_block.mlp_norm.weight,
				lambda x: x[g],
				lambda x, power: x / (copy_count[g] ** power)
			)
			if getattr(old_block.mlp_norm, "bias", None) is not None:
				new_block.mlp_norm.bias.copy_(old_block.mlp_norm.bias.data[g])
				transfer_state(
					old_block.mlp_norm.bias,
					new_block.mlp_norm.bias,
					lambda x: x[g],
					lambda x, power: x / (copy_count[g] ** power)
				)

			# mlp up_proj
			old_up = old_block.mlp.up_proj
			new_up = new_block.mlp.up_proj
			old_mlp_dim = old_up.out_features
			new_mlp_dim = new_up.out_features

			g_mlp, mlp_copy_count, mlp_is_clone = get_random_mapping(old_mlp_dim, new_mlp_dim)
			g_mlp = g_mlp.to(device)
			mlp_copy_count = mlp_copy_count.to(device)
			mlp_is_clone = mlp_is_clone.to(device)

			old_w = old_up.weight.data
			new_w = old_w[g_mlp][:, g] / copy_count[g].unsqueeze(0)
			noise = torch.randn_like(new_w) * noise_std
			clone_mask = mlp_is_clone.unsqueeze(1) | is_clone.unsqueeze(0)
			new_w = new_w + noise * clone_mask
			new_up.weight.copy_(new_w)

			transfer_state(
				old_up.weight,
				new_up.weight,
				lambda x, g_m=g_mlp: x[g_m][:, g],
				lambda x, power: x / (copy_count[g].unsqueeze(0) ** power)
			)

			if old_up.bias is not None:
				new_up.bias.copy_(old_up.bias.data[g_mlp])
				transfer_state(
					old_up.bias,
					new_up.bias,
					lambda x, g_m=g_mlp: x[g_m],
					lambda x, power, m_cc=mlp_copy_count, g_m=g_mlp: x / (m_cc[g_m] ** power)
				)

			# mlp down_proj
			old_down = old_block.mlp.down_proj
			new_down = new_block.mlp.down_proj
			old_w = old_down.weight.data
			new_w = old_w[g][:, g_mlp] / mlp_copy_count[g_mlp].unsqueeze(0)
			noise = torch.randn_like(new_w) * noise_std
			clone_mask = is_clone.unsqueeze(1) | mlp_is_clone.unsqueeze(0)
			new_w = new_w + noise * clone_mask
			new_down.weight.copy_(new_w)

			transfer_state(
				old_down.weight,
				new_down.weight,
				lambda x, g_m=g_mlp: x[g][:, g_m],
				lambda x, power, m_cc=mlp_copy_count, g_m=g_mlp: x / (m_cc[g_m].unsqueeze(0) ** power)
			)

			if old_down.bias is not None:
				new_down.bias.copy_(old_down.bias.data[g])
				transfer_state(
					old_down.bias,
					new_down.bias,
					lambda x: x[g],
					lambda x, power: x / (copy_count[g] ** power)
				)

		# --- E. Norm ---
		new_model.norm.weight.copy_(old_model.norm.weight.data[g])
		transfer_state(
			old_model.norm.weight,
			new_model.norm.weight,
			lambda x: x[g],
			lambda x, power: x / (copy_count[g] ** power)
		)
		if getattr(old_model.norm, "bias", None) is not None:
			new_model.norm.bias.copy_(old_model.norm.bias.data[g])
			transfer_state(
				old_model.norm.bias,
				new_model.norm.bias,
				lambda x: x[g],
				lambda x, power: x / (copy_count[g] ** power)
			)

		# --- F. Outbound Proj ---
		if not use_glyphs:
			old_w = old_model.outbound_proj.weight.data
			new_w = old_w[:, g] / copy_count[g].unsqueeze(0)
			noise = torch.randn_like(new_w) * noise_std
			new_w = new_w + noise * is_clone.unsqueeze(0)
			new_model.outbound_proj.weight.copy_(new_w)

			transfer_state(
				old_model.outbound_proj.weight,
				new_model.outbound_proj.weight,
				lambda x: x[:, g],
				lambda x, power: x / (copy_count[g].unsqueeze(0) ** power)
			)

			if old_model.outbound_proj.bias is not None:
				new_model.outbound_proj.bias.copy_(old_model.outbound_proj.bias.data)
				transfer_state(
					old_model.outbound_proj.bias,
					new_model.outbound_proj.bias,
					lambda x: x,
					None
				)

		# --- G. Action Head ---
		old_w = old_model.action_head[0].weight.data
		new_w = old_w[g_ahw][:, g] / copy_count[g].unsqueeze(0)
		noise = torch.randn_like(new_w) * noise_std
		clone_mask = ahw_is_clone.unsqueeze(1) | is_clone.unsqueeze(0)
		new_w = new_w + noise * clone_mask
		new_model.action_head[0].weight.copy_(new_w)
		new_model.action_head[0].bias.copy_(old_model.action_head[0].bias.data[g_ahw])

		transfer_state(
			old_model.action_head[0].weight,
			new_model.action_head[0].weight,
			lambda x: x[g_ahw][:, g],
			lambda x, power: x / (copy_count[g].unsqueeze(0) ** power)
		)
		transfer_state(
			old_model.action_head[0].bias,
			new_model.action_head[0].bias,
			lambda x: x[g_ahw],
			lambda x, power: x / (ahw_copy_count[g_ahw] ** power)
		)

		old_w = old_model.action_head[2].weight.data
		new_w = old_w[:, g_ahw] / ahw_copy_count[g_ahw].unsqueeze(0)
		noise = torch.randn_like(new_w) * noise_std
		new_w = new_w + noise * ahw_is_clone.unsqueeze(0)
		new_model.action_head[2].weight.copy_(new_w)
		new_model.action_head[2].bias.copy_(old_model.action_head[2].bias.data)

		transfer_state(
			old_model.action_head[2].weight,
			new_model.action_head[2].weight,
			lambda x: x[:, g_ahw],
			lambda x, power: x / (ahw_copy_count[g_ahw].unsqueeze(0) ** power)
		)
		transfer_state(
			old_model.action_head[2].bias,
			new_model.action_head[2].bias,
			lambda x: x,
			None
		)

		# --- H. Value Head ---
		old_w = old_model.value_head[0].weight.data
		new_w = old_w[g_ahw][:, g] / copy_count[g].unsqueeze(0)
		noise = torch.randn_like(new_w) * noise_std
		clone_mask = ahw_is_clone.unsqueeze(1) | is_clone.unsqueeze(0)
		new_model.value_head[0].weight.copy_(new_w)
		new_model.value_head[0].bias.copy_(old_model.value_head[0].bias.data[g_ahw])

		transfer_state(
			old_model.value_head[0].weight,
			new_model.value_head[0].weight,
			lambda x: x[g_ahw][:, g],
			lambda x, power: x / (copy_count[g].unsqueeze(0) ** power)
		)
		transfer_state(
			old_model.value_head[0].bias,
			new_model.value_head[0].bias,
			lambda x: x[g_ahw],
			lambda x, power: x / (ahw_copy_count[g_ahw] ** power)
		)

		old_w = old_model.value_head[2].weight.data
		new_w = old_w[:, g_ahw] / ahw_copy_count[g_ahw].unsqueeze(0)
		noise = torch.randn_like(new_w) * noise_std
		new_w = new_w + noise * ahw_is_clone.unsqueeze(0)
		new_model.value_head[2].weight.copy_(new_w)
		new_model.value_head[2].bias.copy_(old_model.value_head[2].bias.data)

		transfer_state(
			old_model.value_head[2].weight,
			new_model.value_head[2].weight,
			lambda x: x[:, g_ahw],
			lambda x, power: x / (ahw_copy_count[g_ahw].unsqueeze(0) ** power)
		)
		transfer_state(
			old_model.value_head[2].bias,
			new_model.value_head[2].bias,
			lambda x: x,
			None
		)

		# --- I. Glyph Projection Head ---
		old_w = old_model.glyph_projection_head[0].weight.data
		new_w = old_w[:, g] / copy_count[g].unsqueeze(0)
		noise = torch.randn_like(new_w) * noise_std
		new_w = new_w + noise * is_clone.unsqueeze(0)
		new_model.glyph_projection_head[0].weight.copy_(new_w)
		new_model.glyph_projection_head[0].bias.copy_(old_model.glyph_projection_head[0].bias.data)

		transfer_state(
			old_model.glyph_projection_head[0].weight,
			new_model.glyph_projection_head[0].weight,
			lambda x: x[:, g],
			lambda x, power: x / (copy_count[g].unsqueeze(0) ** power)
		)
		transfer_state(
			old_model.glyph_projection_head[0].bias,
			new_model.glyph_projection_head[0].bias,
			lambda x: x,
			None
		)

		new_model.glyph_projection_head[2].weight.copy_(old_model.glyph_projection_head[2].weight.data)
		new_model.glyph_projection_head[2].bias.copy_(old_model.glyph_projection_head[2].bias.data)

		transfer_state(
			old_model.glyph_projection_head[2].weight,
			new_model.glyph_projection_head[2].weight,
			lambda x: x,
			None
		)
		transfer_state(
			old_model.glyph_projection_head[2].bias,
			new_model.glyph_projection_head[2].bias,
			lambda x: x,
			None
		)

	if new_optimizer is not None:
		new_optimizer.param_groups[0]['params'] = list(new_model.parameters())

	return new_model


def verify_net2net_preservation(model: nn.Module, test_input: torch.Tensor, pre_output: torch.Tensor, tolerance: float = 0.01) -> bool:
	"""
	Verificar que Net2Net preservó la función.
	El output después del crecimiento debe ser ≈ al de antes.
	"""
	model.eval()
	with torch.no_grad():
		post_output = model.action_head(test_input)
		diff = (pre_output - post_output).abs().max().item()
	return diff < tolerance, diff


# ── Test ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
	print("═══ Net2Net — Test de preservación ═══\n")

	# Simular action head
	hidden_dim = 256
	head = nn.Sequential(
		nn.Linear(hidden_dim, 128),
		nn.GELU(),
		nn.Linear(128, 6),
	)

	# Input de prueba
	x = torch.randn(4, hidden_dim)

	# Output antes de crecer
	with torch.no_grad():
		y_before = head(x).clone()

	print(f"Antes:  width=128, params={sum(p.numel() for p in head.parameters()):,}")
	print(f"Output: {y_before[0].tolist()}")

	# Crecer
	layer_in = head[0]
	layer_out = head[2]
	new_in, new_out = net2wider_linear(layer_in, layer_out, 192)
	head[0] = new_in
	head[2] = new_out

	# Output después de crecer
	with torch.no_grad():
		y_after = head(x).clone()

	print(f"\nDespués: width=192, params={sum(p.numel() for p in head.parameters()):,}")
	print(f"Output: {y_after[0].tolist()}")

	# Verificar
	diff = (y_before - y_after).abs().max().item()
	print(f"\nMax diff: {diff:.6f}")
	print(f"Preservado: {'✅ SÍ' if diff < 0.01 else '❌ NO'}")

	# Segundo crecimiento
	layer_in = head[0]
	layer_out = head[2]
	new_in, new_out = net2wider_linear(layer_in, layer_out, 256)
	head[0] = new_in
	head[2] = new_out

	with torch.no_grad():
		y_after2 = head(x).clone()

	diff2 = (y_before - y_after2).abs().max().item()
	print(f"\nDoble crecimiento: width=256, params={sum(p.numel() for p in head.parameters()):,}")
	print(f"Max diff desde original: {diff2:.6f}")
	print(f"Preservado: {'✅ SÍ' if diff2 < 0.05 else '❌ NO'}")

	# --- Test de Modelo Completo ---
	print("\n═══ Net2Net — Test de Modelo Completo (BitNet4LayerModel) ═══\n")
	try:
		import numpy as np

		from src.bitnet.model.modeling_bitnet import BitNet4LayerModel

		# Inicializar modelo de 256 dim
		vocab_embeddings = np.random.randn(26, 384)
		model_256 = BitNet4LayerModel(
			vocab_embeddings=vocab_embeddings,
			hidden_dim=256,
			num_layers=2,
			use_pos_embedding=True,
			max_resonance_steps=5,
			n_emotions=6,
			emotion_dim=32,
			emotion_mode="gated",
			use_glyphs=False,
		)

		model_256.eval()
		x_input = torch.randint(0, 26, (2, 10))
		emotion_ids = torch.randint(0, 6, (2,))

		with torch.no_grad():
			# Probar logits y outputs de resonancia antes del crecimiento
			logits_before, meta_before = model_256.forward_resonance(x_input, n_steps=3, pos_mode="clock", emotion_ids=emotion_ids)
			# Action head output
			action_before = model_256.action_head(meta_before["final_hidden"])

		print("Creciendo modelo 256 -> 512...")
		# Deshabilitar ruido temporalmente para test de equivalencia estricta
		model_512 = net2wider_model(model_256, 512, noise_std=0.0)
		model_512.eval()

		with torch.no_grad():
			logits_after, meta_after = model_512.forward_resonance(x_input, n_steps=3, pos_mode="clock", emotion_ids=emotion_ids)
			action_after = model_512.action_head(meta_after["final_hidden"])

		diff_logits = (logits_before - logits_after).abs().max().item()
		diff_action = (action_before - action_after).abs().max().item()

		print(f"Max diff en logits de resonancia: {diff_logits:.6f}")
		print(f"Max diff en action head: {diff_action:.6f}")
		if diff_logits < 1e-4 and diff_action < 1e-4:
			print("✅ Equivalencia funcional estricta superada!")
		else:
			print("❌ Error en equivalencia funcional.")

	except Exception as e:
		print(f"❌ Error en test de modelo completo: {e}")
		import traceback

		traceback.print_exc()
