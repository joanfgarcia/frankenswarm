import copy

import torch
import torch.nn as nn

from src.bitnet.growth.net2net import net2wider_linear


def project_model_to_width(parent_model, target_width, device):
	"""
	Crea una copia de parent_model y alinea quirúrgicamente sus cabezales de acción
	y valor exactamente hasta target_width.
	"""
	model_copy = copy.deepcopy(parent_model).to(device)
	
	# Obtener ancho actual
	curr_w = model_copy.action_head[0].out_features
	if curr_w == target_width:
		return model_copy
		
	# Si la red es mayor que el target, realizamos un recorte simple (surgery)
	if curr_w > target_width:
		with torch.no_grad():
			# Recortar action head
			old_in = model_copy.action_head[0]
			old_out = model_copy.action_head[2]
			new_in = nn.Linear(old_in.in_features, target_width).to(device)
			new_out = nn.Linear(target_width, old_out.out_features).to(device)
			new_in.weight.copy_(old_in.weight[:target_width])
			new_in.bias.copy_(old_in.bias[:target_width])
			new_out.weight.copy_(old_out.weight[:, :target_width])
			new_out.bias.copy_(old_out.bias)
			model_copy.action_head[0] = new_in
			model_copy.action_head[2] = new_out
			
			# Recortar value head
			old_val_in = model_copy.value_head[0]
			old_val_out = model_copy.value_head[2]
			new_val_in = nn.Linear(old_val_in.in_features, target_width).to(device)
			new_val_out = nn.Linear(target_width, old_val_out.out_features).to(device)
			new_val_in.weight.copy_(old_val_in.weight[:target_width])
			new_val_in.bias.copy_(old_val_in.bias[:target_width])
			new_val_out.weight.copy_(old_val_out.weight[:, :target_width])
			new_val_out.bias.copy_(old_val_out.bias)
			model_copy.value_head[0] = new_val_in
			model_copy.value_head[2] = new_val_out
		return model_copy

	# Si la red es menor, la hacemos crecer mediante Net2WiderNet
	new_in, new_out = net2wider_linear(
		model_copy.action_head[0], model_copy.action_head[2], target_width, noise_std=0.01
	)
	model_copy.action_head[0] = new_in.to(device)
	model_copy.action_head[2] = new_out.to(device)

	new_val_in, new_val_out = net2wider_linear(
		model_copy.value_head[0], model_copy.value_head[2], target_width, noise_std=0.01
	)
	model_copy.value_head[0] = new_val_in.to(device)
	model_copy.value_head[2] = new_val_out.to(device)

	return model_copy


def align_tensor(src_tensor, target_shape, default_tensor=None):
	"""
	Alinea src_tensor al target_shape cortándolo o rellenándolo.
	Usa default_tensor como base si se provee.
	"""
	if src_tensor.shape == target_shape:
		return src_tensor
	out = default_tensor.clone() if default_tensor is not None else torch.zeros(target_shape, device=src_tensor.device, dtype=src_tensor.dtype)
	slices_src = []
	slices_out = []
	for dim_src, dim_out in zip(src_tensor.shape, target_shape, strict=False):
		m = min(dim_src, dim_out)
		slices_src.append(slice(0, m))
		slices_out.append(slice(0, m))
	out[tuple(slices_out)] = src_tensor[tuple(slices_src)]
	return out


def svd_crossover(parent_a, parent_b, child, alpha=0.5, sigma=0.01):
	"""
	Realiza el cruzamiento SVD de pesos entre parent_a y parent_b y escribe en child.
	Asume que parent_a, parent_b y child tienen la misma arquitectura y dimensiones.
	Soporta alineación de tensores en caso de discrepancias de tamaño por carga de pesos parciales.
	"""
	with torch.no_grad():
		for name, param in child.named_parameters():
			if not param.requires_grad:
				continue
			p_a = parent_a.state_dict()[name]
			p_b = parent_b.state_dict()[name]
			
			aligned_a = align_tensor(p_a, param.shape, default_tensor=param)
			aligned_b = align_tensor(p_b, param.shape, default_tensor=param)
			
			if aligned_a.ndim == 2:
				w_avg = alpha * aligned_a + (1.0 - alpha) * aligned_b
				try:
					u, s, vh = torch.linalg.svd(w_avg, full_matrices=False)
					s_perturbed = (s + torch.randn_like(s) * sigma).clamp_(min=0.0)
					param.copy_(u @ torch.diag(s_perturbed) @ vh)
				except Exception:
					param.copy_(w_avg)
			else:
				param.copy_(alpha * aligned_a + (1.0 - alpha) * aligned_b + torch.randn_like(aligned_a) * sigma)


def recombine_parents(parent_model_a, parent_model_b, child_model, target_width, device):
	"""
	Alinea parent_model_a y parent_model_b a target_width,
	y luego cruza sus pesos mediante SVD crossover hacia child_model.
	"""
	projected_a = project_model_to_width(parent_model_a, target_width, device)
	projected_b = project_model_to_width(parent_model_b, target_width, device)
	svd_crossover(projected_a, projected_b, child_model, alpha=0.5, sigma=0.01)
