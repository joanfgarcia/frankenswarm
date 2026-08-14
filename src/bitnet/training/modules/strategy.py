from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

WEIGHT_DECAY = 0.05

# Parámetros que en `wd_mode="no_embed"` quedan exentos de weight decay: las
# tablas de embedding (entrada/salida), las posicionales y los primos del glifo.
#
# Por qué existe el modo: con decay uniforme, cada step aplica decay a TODAS las
# columnas de `inbound_proj` (el gradiente es denso aunque solo unas pocas filas
# reciban señal), así que el embedding de una palabra rara se encoge hacia cero
# entre sus actualizaciones infrecuentes. El brazo glifo no sufre eso — sus 65
# primos se actualizan en cada batch. Con decay uniforme los dos brazos NO
# reciben el mismo trato de regularización, y la penalización cae justo sobre lo
# que miden OOD y gen_valid: la generalización a lo poco visto.
NO_DECAY_SUFFIXES = ("inbound_proj.weight", "outbound_proj.weight", "pos_embedding", "prime_embeddings")


def build_param_groups(model, weight_decay: float = WEIGHT_DECAY, wd_mode: str = "uniform"):
	"""Agrupa parámetros para AdamW según el trato de weight decay.

	`wd_mode="uniform"` (histórico) devuelve un grupo único: es el trato con el
	que corrieron las réplicas DL-006 y se mantiene por defecto para no cambiar
	un instrumento a mitad de comparativa (D3). `"no_embed"` exime las tablas de
	embedding, que es el trato simétrico entre brazos.
	"""
	if wd_mode == "uniform":
		return [{"params": list(model.parameters()), "weight_decay": weight_decay}]

	decay, no_decay = [], []
	for name, param in model.named_parameters():
		if not param.requires_grad:
			continue
		(no_decay if name.endswith(NO_DECAY_SUFFIXES) else decay).append(param)
	return [
		{"params": decay, "weight_decay": weight_decay},
		{"params": no_decay, "weight_decay": 0.0},
	]


@runtime_checkable
class TrainingStrategy(Protocol):
	"""Protocolo IOC para estrategias de entrenamiento.

	Cada estrategia encapsula la lógica de forward, backward, y optimización.
	El orchestrator delega en la estrategia sin conocer los detalles de
	mixed precision, 8-bit optimizer, o torch.compile.
	"""

	def create_optimizer(self, model, lr_scale: float) -> object:
		"""Crea y devuelve un optimizador configurado para el modelo."""
		...

	def training_step(self, model, batch_x, vocab_size: int, tau: float, device) -> float:
		"""Ejecuta un step de entrenamiento (forward + backward + step).
		Devuelve el loss escalar.
		"""
		...

	def get_amp_context(self, device):
		"""Devuelve el contexto de autocast para mixed precision."""
		...

	@property
	def name(self) -> str:
		"""Nombre descriptivo de la estrategia."""
		...


@dataclass
class FP32Strategy:
	"""Estrategia FP32 puro (sin mixed precision)."""

	wd_mode: str = "uniform"

	@property
	def name(self) -> str:
		return "fp32"

	def create_optimizer(self, model, lr_scale: float):
		import torch
		return torch.optim.AdamW(build_param_groups(model, wd_mode=self.wd_mode), lr=4e-4 * lr_scale, weight_decay=WEIGHT_DECAY)

	def training_step(self, model, batch_x, vocab_size: int, tau: float, device):
		import torch
		import torch.nn.functional as F

		model.train()
		inputs = batch_x[:, :-1].to(device)
		targets = batch_x[:, 1:].to(device)

		logits = model(inputs, tau=tau)
		loss_elementwise = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1), reduction="none")
		loss_elementwise = loss_elementwise.reshape(targets.shape)
		is_zero = (targets == 0).to(torch.int32)
		cumsum_zero = torch.cumsum(is_zero, dim=-1)
		mask = (cumsum_zero <= 1).to(logits.dtype)
		loss = (loss_elementwise * mask).sum() / mask.sum()

		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
		return loss.item()

	def get_amp_context(self, device):
		from contextlib import nullcontext
		return nullcontext()


@dataclass
class BF16Strategy:
	"""Estrategia BF16 autocast (pesos maestros FP32, activaciones BF16)."""

	wd_mode: str = "uniform"

	@property
	def name(self) -> str:
		return "bf16"

	def create_optimizer(self, model, lr_scale: float):
		import torch
		return torch.optim.AdamW(build_param_groups(model, wd_mode=self.wd_mode), lr=4e-4 * lr_scale, weight_decay=WEIGHT_DECAY)

	def training_step(self, model, batch_x, vocab_size: int, tau: float, device):
		import torch
		import torch.nn.functional as F

		model.train()
		inputs = batch_x[:, :-1].to(device)
		targets = batch_x[:, 1:].to(device)

		with self.get_amp_context(device):
			logits = model(inputs, tau=tau)
			loss_elementwise = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1), reduction="none")
			loss_elementwise = loss_elementwise.reshape(targets.shape)
			is_zero = (targets == 0).to(torch.int32)
			cumsum_zero = torch.cumsum(is_zero, dim=-1)
			mask = (cumsum_zero <= 1).to(logits.dtype)
			loss = (loss_elementwise * mask).sum() / mask.sum()

		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
		return loss.item()

	def get_amp_context(self, device):
		import torch
		if device.type == "cuda" and torch.cuda.is_bf16_supported():
			return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
		from contextlib import nullcontext
		return nullcontext()


@dataclass
class Opt8BitStrategy:
	"""Estrategia con estados de optimizador 8-bit (bitsandbytes).
	Ahorra ~75% VRAM para estados AdamW (m, v).
	"""

	wd_mode: str = "uniform"

	@property
	def name(self) -> str:
		return "opt8bit"

	def create_optimizer(self, model, lr_scale: float):
		import torch
		groups = build_param_groups(model, wd_mode=self.wd_mode)
		try:
			import bitsandbytes as bnb
			return bnb.optim.AdamW8bit(groups, lr=4e-4 * lr_scale, weight_decay=WEIGHT_DECAY, min_8bit_size=4096)
		except ImportError:
			print("⚠️ [OPT8BIT] bitsandbytes no instalado. Fallback a AdamW FP32.")
			return torch.optim.AdamW(groups, lr=4e-4 * lr_scale, weight_decay=WEIGHT_DECAY)

	def training_step(self, model, batch_x, vocab_size: int, tau: float, device):
		import torch
		import torch.nn.functional as F

		model.train()
		inputs = batch_x[:, :-1].to(device)
		targets = batch_x[:, 1:].to(device)

		logits = model(inputs, tau=tau)
		loss_elementwise = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1), reduction="none")
		loss_elementwise = loss_elementwise.reshape(targets.shape)
		is_zero = (targets == 0).to(torch.int32)
		cumsum_zero = torch.cumsum(is_zero, dim=-1)
		mask = (cumsum_zero <= 1).to(logits.dtype)
		loss = (loss_elementwise * mask).sum() / mask.sum()

		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
		return loss.item()

	def get_amp_context(self, device):
		from contextlib import nullcontext
		return nullcontext()


@dataclass
class BF16Opt8BitStrategy:
	"""Estrategia combinada: BF16 autocast + 8-bit optimizer states."""

	wd_mode: str = "uniform"

	@property
	def name(self) -> str:
		return "bf16+opt8bit"

	def create_optimizer(self, model, lr_scale: float):
		import torch
		groups = build_param_groups(model, wd_mode=self.wd_mode)
		try:
			import bitsandbytes as bnb
			return bnb.optim.AdamW8bit(groups, lr=4e-4 * lr_scale, weight_decay=WEIGHT_DECAY, min_8bit_size=4096)
		except ImportError:
			print("⚠️ [OPT8BIT] bitsandbytes no instalado. Fallback a AdamW FP32.")
			return torch.optim.AdamW(groups, lr=4e-4 * lr_scale, weight_decay=WEIGHT_DECAY)

	def training_step(self, model, batch_x, vocab_size: int, tau: float, device):
		import torch
		import torch.nn.functional as F

		model.train()
		inputs = batch_x[:, :-1].to(device)
		targets = batch_x[:, 1:].to(device)

		with self.get_amp_context(device):
			logits = model(inputs, tau=tau)
			loss_elementwise = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1), reduction="none")
			loss_elementwise = loss_elementwise.reshape(targets.shape)
			is_zero = (targets == 0).to(torch.int32)
			cumsum_zero = torch.cumsum(is_zero, dim=-1)
			mask = (cumsum_zero <= 1).to(logits.dtype)
			loss = (loss_elementwise * mask).sum() / mask.sum()

		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
		return loss.item()

	def get_amp_context(self, device):
		import torch
		if device.type == "cuda" and torch.cuda.is_bf16_supported():
			return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
		from contextlib import nullcontext
		return nullcontext()


def select_strategy(amp: str, opt8bit: str, wd_mode: str = "uniform") -> TrainingStrategy:
	"""Factory para seleccionar la estrategia de entrenamiento según flags.

	`amp` debe llegar ya resuelto ("bf16"/"off"): "auto" no se interpreta aquí.
	"""
	if amp == "bf16" and opt8bit == "on":
		return BF16Opt8BitStrategy(wd_mode=wd_mode)
	elif amp == "bf16":
		return BF16Strategy(wd_mode=wd_mode)
	elif opt8bit == "on":
		return Opt8BitStrategy(wd_mode=wd_mode)
	else:
		return FP32Strategy(wd_mode=wd_mode)
