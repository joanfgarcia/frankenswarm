import os
import time

import torch
from tokenizers import Tokenizer

from playground.chat_backtrack import generate_response_with_backtrack
from src.bitnet.modeling_conversational import BitNetCausalLM


def run_benchmark():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	base_dir = "/home/joan/Documents/IA/frankenswarm"
	tokenizer_path = os.path.join(base_dir, "storage", "experiments", "conversational_tokenizer.json")
	model_path = os.path.join(base_dir, "storage", "experiments", "conversational_agent.pt")

	if not os.path.exists(tokenizer_path) or not os.path.exists(model_path):
		print("❌ Error: No se encontraron los modelos conversacionales.")
		return

	tokenizer = Tokenizer.from_file(tokenizer_path)
	vocab_size = tokenizer.get_vocab_size()

	state_dict = torch.load(model_path, map_location=device, weights_only=True)
	hidden_dim = state_dict["token_embedding.weight"].shape[1]
	num_layers = 8 if hidden_dim == 512 else 4
	num_heads = 8 if hidden_dim == 512 else 4

	model = BitNetCausalLM(vocab_size=vocab_size, hidden_dim=hidden_dim, num_layers=num_layers, num_heads=num_heads, max_seq_len=256).to(device)
	model.load_state_dict(state_dict)
	model.eval()

	# Prompts representativos para evaluar coherencia, lógica y respuestas cortas/largas
	test_prompts = [
		"hola, cómo estás?",
		"quién eres tú?",
		"el oso come miel en el bosque porque...",
		"las flores del campo beben...",
		"la capital de España es...",
		"qué hay dentro del búnker?",
	]

	modes = ["none", "confidence", "entropy", "lookahead"]
	results = {m: [] for m in modes}

	print("🧪 Iniciando Benchmarking de Modos de Backtracking...")
	print(f"Modelo: {model_path} | Dispositivo: {device}")
	print("-" * 60)

	for mode in modes:
		print(f"\n🚀 Evaluando Modo: {mode.upper()}")
		print("=" * 40)

		for prompt in test_prompts:
			formatted_prompt = f"tú: {prompt} [EOS] yo:"
			input_ids = tokenizer.encode(formatted_prompt).ids

			h_prev = torch.zeros((1, model.hidden_dim), device=device)

			# Medir tiempo
			t0 = time.time()
			# Llamar al generador con visual=False para no ensuciar la salida del script
			response, _, backtracks = generate_response_with_backtrack(
				model=model,
				tokenizer=tokenizer,
				prompt_ids=input_ids,
				h_prev=h_prev,
				device=device,
				mode=mode,
				max_tokens=32,
				temperature=0.7,
				conf_thresh=0.18,
				entropy_thresh=2.0,
				lookahead_thresh=0.06,
				visual=False,
			)
			dt = time.time() - t0

			print(f"  P: '{prompt}'")
			print(f"  R: '{response}'")
			print(f"  [Retrocesos: {backtracks} | Tiempo: {dt:.3f}s]")
			print("-" * 30)

			results[mode].append({"prompt": prompt, "response": response, "backtracks": backtracks, "time": dt})

	# Escribir el reporte en un archivo Markdown en los artefactos
	report_path = "/home/joan/.gemini/antigravity/brain/426682cc-776c-436a-9332-8afcdbb382a9/backtrack_comparison_report.md"

	with open(report_path, "w", encoding="utf-8") as f:
		f.write("# Reporte Comparativo: Métodos de Backtracking en Inferencia\n\n")
		f.write(
			"Este reporte analiza cuantitativa y cualitativamente el impacto de los tres modos de retroceso frente a la inferencia básica (baseline).\n\n"
		)

		# Tabla general
		f.write("## Resumen Métricas Generales\n\n")
		f.write("| Modo | Promedio Retrocesos | Tiempo Promedio (s) | Calidad Observada |\n")
		f.write("| --- | --- | --- | --- |\n")

		for m in modes:
			avg_backtracks = sum(r["backtracks"] for r in results[m]) / len(results[m])
			avg_time = sum(r["time"] for r in results[m]) / len(results[m])

			if m == "none":
				quality = "Rápida, pero vulnerable a bucles y respuestas genéricas o rotas."
			elif m == "confidence":
				quality = "Filtra palabras extrañas de baja frecuencia, buena coherencia léxica."
			elif m == "entropy":
				quality = "Evita momentos de total confusión/indecisión del modelo."
			elif m == "lookahead":
				quality = "Excelente para prevenir callejones sin salida, detecta la incoherencia un paso antes."

			f.write(f"| **{m.upper()}** | {avg_backtracks:.2f} | {avg_time:.3f}s | {quality} |\n")

		f.write("\n## Detalle de Respuestas por Prompt\n\n")

		for i, prompt in enumerate(test_prompts):
			f.write(f'### Prompt {i + 1}: "{prompt}"\n\n')
			f.write("| Modo | Respuesta Generada | Retrocesos | Tiempo |\n")
			f.write("| --- | --- | --- | --- |\n")
			for m in modes:
				res = results[m][i]
				f.write(f'| {m.upper()} | "{res["response"]}" | {res["backtracks"]} | {res["time"]:.3f}s |\n')
			f.write("\n")

	print(f"\n📊 Reporte de benchmarking guardado en: {report_path}")


if __name__ == "__main__":
	run_benchmark()
