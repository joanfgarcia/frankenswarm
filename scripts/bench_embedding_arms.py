#!/usr/bin/env python
"""Banco de coste de los brazos de embedding (DL-006): glyph vs standard.

Mide el step de entrenamiento (forward + backward + optimizer) de los dos brazos
con el MISMO vocabulario, dim y batch, y reporta:
  - ms/step (mediana de N pasos, tras warmup)
  - pico de VRAM
  - tamaño serializado del checkpoint

Por qué existe: la cifra "el glifo decodifica 4.7× más barato" del informe DL-006
se midió (15 vs 70 s/época) contra un brazo estándar que materializaba una tabla
identidad de V×V y multiplicaba por ella — trabajo aritmético gratuito ajeno a la
arquitectura. Este banco separa el coste de la ARQUITECTURA del coste de la
IMPLEMENTACIÓN, y es el que produce la cifra publicable.

Uso (GPU libre — descargar antes el LLM del kernel):
	PYTHONPATH=. .venv/bin/python scripts/bench_embedding_arms.py --label after

Resultados: tabla en consola + storage/benchmarks/embedding_arms_bench.json
(acumulativo por label, para comparar antes/después).
"""

import argparse
import gc
import io
import json
import os
import statistics
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
	sys.path.insert(0, BASE_DIR)

from src.bitnet.model.modeling_bitnet import BitNet4LayerModel


def load_glyphs():
	path = os.path.join(BASE_DIR, "configs", "expanded_glyphs.json")
	with open(path, encoding="utf-8") as f:
		data = json.load(f)
	return np.array(data["glyphs"], dtype=np.float32)


def build_model(arm, glyphs, vocab_size, dim, num_layers, seq_len, device):
	torch.manual_seed(770)
	if arm == "glyph":
		return BitNet4LayerModel(
			use_glyphs=True,
			glyph_table=glyphs[:vocab_size],
			hidden_dim=dim,
			num_layers=num_layers,
			use_pos_embedding=True,
			is_causal=True,
			max_seq_len=seq_len,
		).to(device)
	return BitNet4LayerModel(
		use_glyphs=False,
		vocab_embeddings=np.eye(vocab_size, dtype=np.float32),
		hidden_dim=dim,
		num_layers=num_layers,
		use_pos_embedding=True,
		is_causal=True,
		max_seq_len=seq_len,
	).to(device)


def checkpoint_bytes(model):
	buf = io.BytesIO()
	torch.save(model.state_dict(), buf)
	return buf.getbuffer().nbytes


def run_arm(arm, glyphs, vocab_size, dim, num_layers, seq_len, batch_size, steps, warmup, device, amp):
	model = build_model(arm, glyphs, vocab_size, dim, num_layers, seq_len, device)
	optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4 * 128.0 / dim, weight_decay=0.05)

	torch.manual_seed(65)
	x = torch.randint(2, vocab_size, (batch_size, seq_len), device=device)
	inputs, targets = x[:, :-1], x[:, 1:]

	on_cuda = device.type == "cuda"
	if on_cuda:
		torch.cuda.reset_peak_memory_stats()

	times = []
	for step in range(warmup + steps):
		if on_cuda:
			torch.cuda.synchronize()
		t0 = time.perf_counter()
		optimizer.zero_grad()
		ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if (amp and on_cuda) else nullcontext()
		with ctx:
			logits = model(inputs)
			loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), targets.reshape(-1))
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
		optimizer.step()
		if on_cuda:
			torch.cuda.synchronize()
		if step >= warmup:
			times.append((time.perf_counter() - t0) * 1000.0)

	result = {
		"arm": arm,
		"dim": dim,
		"vocab_size": model.vocab_size,
		"params": sum(p.numel() for p in model.parameters()),
		"ms_per_step": round(statistics.median(times), 2),
		"ms_stdev": round(statistics.stdev(times), 2) if len(times) > 1 else 0.0,
		"peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1) if on_cuda else None,
		"checkpoint_mb": round(checkpoint_bytes(model) / 1024**2, 1),
		"final_loss": round(loss.item(), 4),
	}
	del model, optimizer, logits, loss
	gc.collect()
	if on_cuda:
		torch.cuda.empty_cache()
	return result


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--label", type=str, required=True, help="Etiqueta de la medición (p. ej. before/after)")
	parser.add_argument("--dim", type=int, default=128)
	parser.add_argument("--num_layers", type=int, default=6)
	parser.add_argument("--vocab", type=int, default=None, help="Default: todo el vocabulario de expanded_glyphs.json")
	parser.add_argument("--batch_size", type=int, default=64)
	parser.add_argument("--seq_len", type=int, default=128)
	parser.add_argument("--steps", type=int, default=12)
	parser.add_argument("--warmup", type=int, default=3)
	parser.add_argument("--amp", action="store_true", default=True)
	parser.add_argument("--no_amp", dest="amp", action="store_false")
	parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
	args = parser.parse_args()

	device = torch.device(args.device)
	glyphs = load_glyphs()
	vocab_size = args.vocab or glyphs.shape[0]

	print(f"═══ Banco de brazos de embedding · label={args.label} ═══")
	print(f"device={device} · dim={args.dim} · vocab={vocab_size} · batch={args.batch_size} · seq={args.seq_len} · amp={args.amp}")

	results = []
	for arm in ["glyph", "standard"]:
		print(f"\n▶ Brazo {arm}...")
		r = run_arm(
			arm, glyphs, vocab_size, args.dim, args.num_layers, args.seq_len,
			args.batch_size, args.steps, args.warmup, device, args.amp,
		)
		results.append(r)
		vram = f"{r['peak_vram_mb']} MB" if r["peak_vram_mb"] is not None else "n/a"
		print(f"  {r['ms_per_step']} ms/step (σ {r['ms_stdev']}) · params {r['params']:,} · VRAM {vram} · ckpt {r['checkpoint_mb']} MB")

	glyph, std = results[0], results[1]
	ratio = std["ms_per_step"] / glyph["ms_per_step"] if glyph["ms_per_step"] else float("nan")
	print(f"\n📊 standard/glyph = {ratio:.2f}× por step · ckpt {std['checkpoint_mb']}/{glyph['checkpoint_mb']} MB")

	out_dir = os.path.join(BASE_DIR, "storage", "benchmarks")
	os.makedirs(out_dir, exist_ok=True)
	out_path = os.path.join(out_dir, "embedding_arms_bench.json")
	payload = {}
	if os.path.exists(out_path):
		with open(out_path, encoding="utf-8") as f:
			payload = json.load(f)
	payload[args.label] = {
		"device": str(device),
		"dim": args.dim,
		"vocab_size": vocab_size,
		"batch_size": args.batch_size,
		"seq_len": args.seq_len,
		"amp": args.amp,
		"steps": args.steps,
		"results": results,
		"standard_over_glyph": round(ratio, 3),
	}
	with open(out_path, "w", encoding="utf-8") as f:
		json.dump(payload, f, indent=2)
	print(f"💾 {out_path}")


if __name__ == "__main__":
	main()
