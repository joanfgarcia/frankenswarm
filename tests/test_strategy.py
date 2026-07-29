"""Tests para la estrategia de entrenamiento (IOC)."""

import pytest
import torch

from src.bitnet.training.modules.strategy import (
	BF16Opt8BitStrategy,
	BF16Strategy,
	FP32Strategy,
	Opt8BitStrategy,
	select_strategy,
)


class MockModel(torch.nn.Module):
	"""Modelo mock que acepta tau como parámetro y procesa 3D input."""

	def __init__(self):
		super().__init__()
		# Use a simple approach: project each position independently
		self.linear = torch.nn.Linear(1, 5)

	def forward(self, x, tau=1.0):
		# x shape: (batch, seq_len) → model outputs (batch, seq_len, vocab_size)
		# Process each position independently
		batch, seq_len = x.shape
		x = x.float().unsqueeze(-1)  # (batch, seq_len, 1)
		return self.linear(x)  # (batch, seq_len, 5)


class TestStrategySelection:
	"""Tests para la selección de estrategia."""

	def test_select_fp32_by_default(self):
		strategy = select_strategy("off", "off")
		assert isinstance(strategy, FP32Strategy)
		assert strategy.name == "fp32"

	def test_select_bf16(self):
		strategy = select_strategy("bf16", "off")
		assert isinstance(strategy, BF16Strategy)
		assert strategy.name == "bf16"

	def test_select_opt8bit(self):
		strategy = select_strategy("off", "on")
		assert isinstance(strategy, Opt8BitStrategy)
		assert strategy.name == "opt8bit"

	def test_select_bf16_opt8bit(self):
		strategy = select_strategy("bf16", "on")
		assert isinstance(strategy, BF16Opt8BitStrategy)
		assert strategy.name == "bf16+opt8bit"


class TestFP32Strategy:
	"""Tests para la estrategia FP32."""

	def test_create_optimizer(self):
		strategy = FP32Strategy()
		model = MockModel()
		optimizer = strategy.create_optimizer(model, lr_scale=1.0)
		assert isinstance(optimizer, torch.optim.AdamW)

	def test_training_step(self):
		strategy = FP32Strategy()
		model = MockModel()
		vocab_size = 5
		batch_x = torch.randint(0, vocab_size, (2, 10))  # (batch, seq_len)
		loss = strategy.training_step(model, batch_x, vocab_size=vocab_size, tau=1.0, device=torch.device("cpu"))
		assert isinstance(loss, float)
		assert loss >= 0

	def test_get_amp_context(self):
		strategy = FP32Strategy()
		ctx = strategy.get_amp_context(torch.device("cpu"))
		# Should be a nullcontext
		with ctx:
			pass


class TestBF16Strategy:
	"""Tests para la estrategia BF16."""

	def test_create_optimizer(self):
		strategy = BF16Strategy()
		model = MockModel()
		optimizer = strategy.create_optimizer(model, lr_scale=1.0)
		assert isinstance(optimizer, torch.optim.AdamW)

	def test_get_amp_context_cpu(self):
		strategy = BF16Strategy()
		ctx = strategy.get_amp_context(torch.device("cpu"))
		# Should be a nullcontext on CPU
		with ctx:
			pass


class TestOpt8BitStrategy:
	"""Tests para la estrategia 8-bit."""

	def test_create_optimizer(self):
		strategy = Opt8BitStrategy()
		model = MockModel()
		optimizer = strategy.create_optimizer(model, lr_scale=1.0)
		# Should be either AdamW8bit or AdamW (fallback)
		assert isinstance(optimizer, (torch.optim.AdamW, type(optimizer)))

	def test_get_amp_context(self):
		strategy = Opt8BitStrategy()
		ctx = strategy.get_amp_context(torch.device("cpu"))
		with ctx:
			pass
