"""Benchmark script para comparar rendimiento con --opt8bit on vs off.

Este script ejecuta un entrenamiento breve y mide:
- Tiempo por época
- VRAM pico
- Tasa de convergencia
"""

import json
import os
import time

import torch

from src.bitnet.model.modeling_bitnet import BitNet4LayerModel
from src.bitnet.training.modules.strategy import select_strategy


def create_mock_data(batch_size=64, seq_len=128, vocab_size=1000):
	"""Crea datos de entrenamiento mock."""
	return torch.randint(0, vocab_size, (batch_size, seq_len))


def benchmark_strategy(strategy_name, n_epochs=5, batch_size=64, seq_len=128, vocab_size=1000):
	"""Ejecuta un benchmark con la estrategia dada."""
	print(f"\n{'='*60}")
	print(f"Benchmarking: {strategy_name}")
	print(f"{'='*60}")

	# Seleccionar estrategia
	if strategy_name == "fp32":
		strategy = select_strategy("off", "off")
	elif strategy_name == "bf16":
		strategy = select_strategy("bf16", "off")
	elif strategy_name == "opt8bit":
		strategy = select_strategy("off", "on")
	elif strategy_name == "bf16+opt8bit":
		strategy = select_strategy("bf16", "on")
	else:
		raise ValueError(f"Estrategia desconocida: {strategy_name}")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}")

	# Crear glyphs mock (necesario para BitNet4LayerModel)
	import numpy as np
	vocab_size = 1000
	n_primes = 65  # Número de primos semánticos
	mock_glyphs = np.random.randn(vocab_size, n_primes).astype(np.float32)

	# Crear modelo
	model = BitNet4LayerModel(
		use_glyphs=True,
		glyph_table=mock_glyphs,
		hidden_dim=128,
		num_layers=2,
		use_pos_embedding=True,
		is_causal=True,
		max_seq_len=seq_len,
	).to(device)

	# Crear optimizador
	lr_scale = 128.0 / model.hidden_dim
	optimizer = strategy.create_optimizer(model, lr_scale)

	# Crear datos
	x_train = create_mock_data(batch_size, seq_len, vocab_size)

	# Medir VRAM pico
	if device.type == "cuda":
		torch.cuda.reset_peak_memory_stats()
		vram_before = torch.cuda.max_memory_allocated() / 2**20

	# Benchmark
	times = []
	losses = []
	for epoch in range(n_epochs):
		start_time = time.time()
		loss = strategy.training_step(model, x_train, vocab_size, tau=1.0, device=device)
		elapsed = time.time() - start_time
		times.append(elapsed)
		losses.append(loss)
		print(f"  Época {epoch+1}: loss={loss:.4f}, tiempo={elapsed:.3f}s")

	# Resultados
	avg_time = sum(times) / len(times)
	total_time = sum(times)
	if device.type == "cuda":
		vram_peak = torch.cuda.max_memory_allocated() / 2**20
	else:
		vram_peak = 0

	print(f"\nResultados:")
	print(f"  Tiempo promedio por época: {avg_time:.3f}s")
	print(f"  Tiempo total: {total_time:.3f}s")
	print(f"  VRAM pico: {vram_peak:.0f} MB")
	print(f"  Loss final: {losses[-1]:.4f}")

	return {
		"strategy": strategy_name,
		"avg_time": avg_time,
		"total_time": total_time,
		"vram_peak": vram_peak,
		"final_loss": losses[-1],
		"losses": losses,
	}


def main():
	"""Ejecuta benchmarks para todas las estrategias."""
	print("═══ Benchmark de Estrategias de Entrenamiento ═══")
	print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

	results = []
	for strategy in ["fp32", "opt8bit", "bf16", "bf16+opt8bit"]:
		try:
			result = benchmark_strategy(strategy, n_epochs=5)
			results.append(result)
		except Exception as e:
			print(f"Error con {strategy}: {e}")

	# Guardar resultados
	output_path = "benchmark_results.json"
	with open(output_path, "w") as f:
		json.dump(results, f, indent=2)
	print(f"\n💾 Resultados guardados en {output_path}")

	# Resumen
	print("\n═══ Resumen ═══")
	for r in results:
		print(f"{r['strategy']:15} | tiempo={r['avg_time']:.3f}s | VRAM={r['vram_peak']:.0f}MB | loss={r['final_loss']:.4f}")


if __name__ == "__main__":
	main()
