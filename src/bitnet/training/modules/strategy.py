from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


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

	@property
	def name(self) -> str:
		return "fp32"

	def create_optimizer(self, model, lr_scale: float):
		import torch
		return torch.optim.AdamW(model.parameters(), lr=4e-4 * lr_scale, weight_decay=0.05)

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

	@property
	def name(self) -> str:
		return "bf16"

	def create_optimizer(self, model, lr_scale: float):
		import torch
		return torch.optim.AdamW(model.parameters(), lr=4e-4 * lr_scale, weight_decay=0.05)

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

	@property
	def name(self) -> str:
		return "opt8bit"

	def create_optimizer(self, model, lr_scale: float):
		import torch
		try:
			import bitsandbytes as bnb
			return bnb.optim.AdamW8bit(model.parameters(), lr=4e-4 * lr_scale, weight_decay=0.05, min_8bit_size=4096)
		except ImportError:
			print("⚠️ [OPT8BIT] bitsandbytes no instalado. Fallback a AdamW FP32.")
			return torch.optim.AdamW(model.parameters(), lr=4e-4 * lr_scale, weight_decay=0.05)

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

	@property
	def name(self) -> str:
		return "bf16+opt8bit"

	def create_optimizer(self, model, lr_scale: float):
		import torch
		try:
			import bitsandbytes as bnb
			return bnb.optim.AdamW8bit(model.parameters(), lr=4e-4 * lr_scale, weight_decay=0.05, min_8bit_size=4096)
		except ImportError:
			print("⚠️ [OPT8BIT] bitsandbytes no instalado. Fallback a AdamW FP32.")
			return torch.optim.AdamW(model.parameters(), lr=4e-4 * lr_scale, weight_decay=0.05)

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


def select_strategy(amp: str, opt8bit: str) -> TrainingStrategy:
	"""Factory para seleccionar la estrategia de entrenamiento según flags."""
	if amp == "bf16" and opt8bit == "on":
		return BF16Opt8BitStrategy()
	elif amp == "bf16":
		return BF16Strategy()
	elif opt8bit == "on":
		return Opt8BitStrategy()
	else:
		return FP32Strategy()
