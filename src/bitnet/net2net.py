"""
Net2Net: Crecimiento neural preservando conocimiento.

Implementa Net2WiderNet (más neuronas) para el action head.
El modelo crece cuando lo necesita sin perder lo aprendido.

Ref: Chen, Goodfellow, Shlens — "Net2Net: Accelerating Learning via Knowledge Transfer" (2015)
Aplicación: Neurogenésis artificial guiada por convergencia.

Origen: Joan Garcia — "Net2Net en caliente?" — 2026-05-31
"""


import torch
import torch.nn as nn


def net2wider_linear(layer_in: nn.Linear, layer_out: nn.Linear, new_width: int, noise_std: float = 0.01):
	"""
	Net2WiderNet: Ampliar una capa hidden preservando la función.

	layer_in:   Linear(in_features, old_width)  → será Linear(in_features, new_width)
	layer_out:  Linear(old_width, out_features)  → será Linear(new_width, out_features)
	new_width:  nuevo ancho (> old_width)

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
	    Linear(hidden_dim, width),     ← layer 0
	    GELU(),                         ← layer 1
	    Linear(width, 6),              ← layer 2
	)

	Crece la capa hidden (layer 0 out / layer 2 in) por growth_factor.

	Returns:
	    dict con info del crecimiento
	"""
	head = model.action_head
	layer_in = head[0]   # Linear(hidden_dim, old_width)
	layer_out = head[2]  # Linear(old_width, 6)

	old_width = layer_in.out_features
	new_width = int(old_width * growth_factor)

	# Hacer crecer
	new_layer_in, new_layer_out = net2wider_linear(
		layer_in, layer_out, new_width, noise_std=noise_std
	)

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
