#!/usr/bin/env python
"""
Tier 1 del benchmark BF16/SDPA (Aleth_Core/plans/benchmark_bf16_school.md §2).

Micro-benchmark del step de entrenamiento a dims grandes (896, 1024) con datos
sintéticos. NO lee ni escribe ningún checkpoint: modelo desde init aleatoria.
Responde dos preguntas que el A/B from-scratch (dims pequeñas) no puede:
  1. ¿Cuánto acelera/ahorra cada pieza (SDPA, BF16, torch.compile)?
  2. ¿Cabe dim=1024 en los 8 GB de la RTX, y con qué margen?

Uso (con la GPU libre — descargar antes el LLM del kernel):
	PYTHONPATH=. .venv/bin/python scripts/bench_school_step.py
	PYTHONPATH=. .venv/bin/python scripts/bench_school_step.py --dims 896 --steps 8

Un OOM no aborta la matriz: se registra como resultado (también es dato).
Resultados: tabla en consola + storage/benchmarks/tier1_step_bench.json
"""

import argparse
import gc
import json
import os
import statistics
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import sys

if BASE_DIR not in sys.path:
	sys.path.insert(0, BASE_DIR)

from src.bitnet.model.modeling_bitnet import BitNet4LayerModel, BitNetAttention

CONFIGS = ["fp32_manual", "fp32_sdpa", "bf16_sdpa", "bf16_sdpa_compile"]


def manual_attention_forward(self, x):
	"""La atención pre-SDPA (oráculo del test de equivalencia): línea base pre-D1."""
	batch_size, seq_len, _ = x.shape
	q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
	k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
	v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
	scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
	if self.is_causal and seq_len > 1:
		mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
		scores = scores.masked_fill(mask, -1e9)
	attn = F.softmax(scores, dim=-1)
	context = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
	return self.out_proj(context)


def load_glyphs():
	path = os.path.join(BASE_DIR, "configs", "expanded_glyphs.json")
	with open(path, encoding="utf-8") as f:
		data = json.load(f)
	return np.array(data["glyphs"], dtype=np.float32)


def run_config(name, dim, glyphs, device, steps, warmup, batch_size, seq_len):
	result = {"config": name, "dim": dim, "batch_size": batch_size, "seq_len": seq_len}
	orig_forward = BitNetAttention.forward
	if name == "fp32_manual":
		BitNetAttention.forward = manual_attention_forward
	model = None
	optimizer = None
	try:
		torch.manual_seed(770)
		model = BitNet4LayerModel(
			use_glyphs=True,
			glyph_table=glyphs,
			hidden_dim=dim,
			num_layers=6,
			use_pos_embedding=True,
			is_causal=True,
			max_seq_len=seq_len,
		).to(device)
		vocab_size = model.vocab_size
		optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4 * 128.0 / dim, weight_decay=0.05)
		fwd_model = torch.compile(model, fullgraph=False) if name.endswith("_compile") else model
		use_amp = name.startswith("bf16")

		torch.manual_seed(65)
		x = torch.randint(2, vocab_size, (batch_size, seq_len), device=device)
		inputs, targets = x[:, :-1], x[:, 1:]

		torch.cuda.reset_peak_memory_stats()
		times = []
		for step in range(warmup + steps):
			torch.cuda.synchronize()
			t0 = time.perf_counter()
			optimizer.zero_grad()
			ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
			with ctx:
				logits = fwd_model(inputs)
				loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			optimizer.step()
			torch.cuda.synchronize()
			dt = time.perf_counter() - t0
			if step == 0 and name.endswith("_compile"):
				result["first_step_s"] = round(dt, 2)  # coste de compilación en frío
			if step >= warmup:
				times.append(dt)

		ste_grad = model.core_layers[0].attn.q_proj.weight.grad
		result.update({
			"ms_per_step": round(statistics.median(times) * 1000, 1),
			"vram_peak_mb": round(torch.cuda.max_memory_allocated() / 2**20),
			"ste_grad_norm": round(ste_grad.norm().item(), 6) if ste_grad is not None else None,
			"last_loss": round(loss.item(), 4),
		})
	except torch.cuda.OutOfMemoryError:
		result["oom"] = True
	except RuntimeError as err:
		if "out of memory" in str(err).lower():
			result["oom"] = True
		else:
			result["error"] = str(err)[:300]
	finally:
		BitNetAttention.forward = orig_forward
		del model, optimizer
		gc.collect()
		torch.cuda.empty_cache()
	return result


def main():
	parser = argparse.ArgumentParser(description="Tier 1: micro-benchmark del step de la Escuela")
	parser.add_argument("--dims", type=str, default="896,1024")
	parser.add_argument("--steps", type=int, default=12, help="Steps medidos por config (mediana)")
	parser.add_argument("--warmup", type=int, default=3)
	parser.add_argument("--batch_size", type=int, default=64)
	parser.add_argument("--seq_len", type=int, default=128)
	parser.add_argument("--configs", type=str, default=",".join(CONFIGS))
	args = parser.parse_args()

	assert torch.cuda.is_available(), "Tier 1 mide la GPU: sin CUDA no tiene sentido"
	device = torch.device("cuda")
	free_mb = torch.cuda.mem_get_info()[0] / 2**20
	print(f"GPU: {torch.cuda.get_device_name(0)} | libre: {free_mb:,.0f} MB | torch {torch.__version__}")
	if free_mb < 7000:
		print("⚠️ GPU no exclusiva (¿LLM del kernel cargado?). Los picos de VRAM saldrán contaminados.")

	glyphs = load_glyphs()
	results = []
	for dim in [int(d) for d in args.dims.split(",")]:
		for name in args.configs.split(","):
			print(f"▶ dim={dim} · {name} ...", flush=True)
			r = run_config(name, dim, glyphs, device, args.steps, args.warmup, args.batch_size, args.seq_len)
			results.append(r)
			if r.get("oom"):
				print("   ✗ CUDA OOM (registrado como resultado)")
			elif r.get("error"):
				print(f"   ✗ error: {r['error']}")
			else:
				extra = f" | 1er step: {r['first_step_s']}s" if "first_step_s" in r else ""
				print(f"   ✓ {r['ms_per_step']} ms/step | VRAM pico: {r['vram_peak_mb']:,} MB | ∇STE: {r['ste_grad_norm']}{extra}")

	out_dir = os.path.join(BASE_DIR, "storage", "benchmarks")
	os.makedirs(out_dir, exist_ok=True)
	out_path = os.path.join(out_dir, "tier1_step_bench.json")
	payload = {
		"gpu": torch.cuda.get_device_name(0),
		"torch": torch.__version__,
		"steps_measured": args.steps,
		"results": results,
	}
	with open(out_path, "w", encoding="utf-8") as f:
		json.dump(payload, f, indent=2)
	print(f"\n💾 Resultados: {out_path}")

	print(f"\n{'config':<20}{'dim':>6}{'ms/step':>10}{'VRAM MB':>10}{'∇STE':>12}")
	for r in results:
		if r.get("oom"):
			print(f"{r['config']:<20}{r['dim']:>6}{'OOM':>10}{'—':>10}{'—':>12}")
		elif r.get("error"):
			print(f"{r['config']:<20}{r['dim']:>6}{'ERROR':>10}{'—':>10}{'—':>12}")
		else:
			print(f"{r['config']:<20}{r['dim']:>6}{r['ms_per_step']:>10}{r['vram_peak_mb']:>10,}{r['ste_grad_norm']:>12}")


if __name__ == "__main__":
	main()
