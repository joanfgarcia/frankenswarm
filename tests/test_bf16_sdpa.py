"""
Gates de validación de RFC-BITNET-VRAM-001 (Estrategia B, fases 1-2).

F2 — SDPA: la atención fusionada debe reproducir la implementación manual
     (softmax(QK^T/√d)V con máscara causal) dentro de tolerancia FP32.
F1 — BF16: bajo autocast el modelo debe entrenar con gradientes vivos en
     BitLinear (salud del STE) y RMSNorm debe mantener su contrato numérico.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.bitnet.model.modeling_bitnet import BitLinear, BitNet4LayerModel, BitNetAttention, RMSNorm


def manual_attention_reference(attn: BitNetAttention, x: torch.Tensor) -> torch.Tensor:
	"""La implementación previa a SDPA, conservada como oráculo del gate F2."""
	batch_size, seq_len, _ = x.shape
	q = attn.q_proj(x).view(batch_size, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)
	k = attn.k_proj(x).view(batch_size, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)
	v = attn.v_proj(x).view(batch_size, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)

	scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(attn.head_dim)
	if attn.is_causal and seq_len > 1:
		mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
		scores = scores.masked_fill(mask, -1e9)
	weights = F.softmax(scores, dim=-1)
	context = torch.matmul(weights, v).transpose(1, 2).contiguous().view(batch_size, seq_len, attn.dim)
	return attn.out_proj(context)


@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("seq_len", [1, 7, 32])
def test_sdpa_matches_manual_attention(is_causal, seq_len):
	torch.manual_seed(770)
	attn = BitNetAttention(dim=64, num_heads=4, is_causal=is_causal).eval()
	x = torch.randn(3, seq_len, 64)
	with torch.no_grad():
		expected = manual_attention_reference(attn, x)
		actual = attn(x)
	assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-4), \
		f"SDPA diverge de la atención manual (max abs diff: {(actual - expected).abs().max().item():.2e})"


def test_sdpa_preserves_gradient_flow():
	torch.manual_seed(770)
	attn = BitNetAttention(dim=64, num_heads=4, is_causal=True)
	x = torch.randn(2, 16, 64, requires_grad=True)
	attn(x).sum().backward()
	assert x.grad is not None and x.grad.abs().sum() > 0
	assert attn.q_proj.weight.grad is not None and attn.q_proj.weight.grad.abs().sum() > 0


def test_rmsnorm_matches_previous_formula_in_fp32():
	torch.manual_seed(7)
	norm = RMSNorm(48)
	with torch.no_grad():
		norm.weight.mul_(1.7)
	x = torch.randn(5, 9, 48)
	expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm.eps) * norm.weight
	assert torch.allclose(norm(x), expected, atol=1e-6)


def test_rmsnorm_computes_in_fp32_and_returns_input_dtype():
	norm = RMSNorm(32)
	x = torch.randn(4, 8, 32, dtype=torch.bfloat16)
	out = norm(x)
	assert out.dtype == torch.bfloat16
	# El resultado debe coincidir con la referencia FP32 dentro del error de BF16
	ref = norm(x.float())
	assert torch.allclose(out.float(), ref, atol=1e-2)


def _tiny_model():
	torch.manual_seed(65)
	vocab = np.random.randn(50, 24).astype(np.float32)
	return BitNet4LayerModel(
		vocab_embeddings=vocab,
		hidden_dim=32,
		num_layers=2,
		use_pos_embedding=True,
		is_causal=True,
		max_seq_len=16,
	)


def _autocast_device():
	return "cuda" if torch.cuda.is_available() else "cpu"


def test_bf16_autocast_forward_backward_keeps_ste_alive():
	"""Gate F1 (salud del STE, RFC §4.8.1): bajo autocast BF16 los gradientes
	de BitLinear deben seguir siendo no nulos y finitos."""
	device = _autocast_device()
	model = _tiny_model().to(device)
	x = torch.randint(0, 50, (4, 12), device=device)
	targets = torch.randint(0, 50, (4, 12), device=device)

	with torch.autocast(device_type=device, dtype=torch.bfloat16):
		logits = model(x)
		loss = F.cross_entropy(logits.reshape(-1, 50), targets.reshape(-1))
	loss.backward()

	bitlinear_grads = [
		m.weight.grad for m in model.modules() if isinstance(m, BitLinear)
	]
	assert len(bitlinear_grads) > 0
	for grad in bitlinear_grads:
		assert grad is not None, "BitLinear sin gradiente bajo autocast (STE roto)"
		assert torch.isfinite(grad).all(), "Gradiente no finito bajo autocast"
	total = sum(g.abs().sum().item() for g in bitlinear_grads)
	assert total > 0, "Todos los gradientes de BitLinear son cero (STE roto)"

	# Los pesos maestros y sus gradientes permanecen en FP32 (doctrina AMP:
	# el checkpoint no cambia de formato).
	for p in model.parameters():
		assert p.dtype == torch.float32
		if p.grad is not None:
			assert p.grad.dtype == torch.float32


def test_bf16_autocast_loss_close_to_fp32():
	"""El forward BF16 debe producir una loss cercana a la FP32 (mismo modelo, mismo batch)."""
	device = _autocast_device()
	model = _tiny_model().to(device).eval()
	x = torch.randint(0, 50, (4, 12), device=device)
	targets = torch.randint(0, 50, (4, 12), device=device)

	with torch.no_grad():
		logits_fp32 = model(x)
		loss_fp32 = F.cross_entropy(logits_fp32.reshape(-1, 50), targets.reshape(-1)).item()
		with torch.autocast(device_type=device, dtype=torch.bfloat16):
			logits_bf16 = model(x)
			loss_bf16 = F.cross_entropy(logits_bf16.reshape(-1, 50), targets.reshape(-1)).item()

	assert abs(loss_bf16 - loss_fp32) / max(abs(loss_fp32), 1e-8) < 0.05, \
		f"Loss BF16 ({loss_bf16:.4f}) se aleja >5% de FP32 ({loss_fp32:.4f})"
