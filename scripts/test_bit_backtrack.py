import json
import os
import re
import time

import numpy as np
import torch
import torch.nn.functional as F

from src.bitnet.modeling_bitnet import BitNet4LayerModel


def get_entropy(probs):
	"""Calcula la entropía de Shannon de la distribución de probabilidad."""
	return -torch.sum(probs * torch.log(probs + 1e-9)).item()


def sample_next_token_with_blacklist(logits, temperature=0.7, top_p=0.9, top_k=50, blacklist=None):
	"""Muestrea un token aplicando una lista negra de tokens prohibidos para este paso."""
	logits_clone = logits.clone()
	if blacklist:
		for token_id in blacklist:
			logits_clone[token_id] = -float("Inf")

	if temperature <= 0.0:
		pred_id = logits_clone.argmax(dim=-1).item()
		probs = F.softmax(logits_clone, dim=-1)
		return pred_id, probs[pred_id].item(), probs

	logits_clone = logits_clone / max(temperature, 1e-5)

	if top_k > 0:
		v, _ = torch.topk(logits_clone, min(top_k, logits_clone.size(-1)))
		logits_clone[logits_clone < v[-1]] = -float("Inf")

	if top_p < 1.0:
		sorted_logits, sorted_indices = torch.sort(logits_clone, descending=True)
		cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
		sorted_indices_to_remove = cumulative_probs > top_p
		sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
		sorted_indices_to_remove[0] = 0
		indices_to_remove = sorted_indices[sorted_indices_to_remove]
		logits_clone[indices_to_remove] = -float("Inf")

	probs = F.softmax(logits_clone, dim=-1)
	next_token = torch.multinomial(probs, num_samples=1).item()
	return next_token, probs[next_token].item(), probs


def tokenize(text: str, word_to_idx: dict) -> list[int]:
	words = re.findall(r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_<>\-]+", text.lower())
	return [word_to_idx.get(w, word_to_idx.get("<unk>", 0)) for w in words]


def generate_with_backtrack(
	model,
	word_to_idx,
	idx_to_word,
	prompt,
	mode="none",
	max_tokens=15,
	temperature=0.7,
	top_p=0.9,
	top_k=40,
	conf_thresh=0.15,
	lookahead_thresh=0.05,
	entropy_thresh=2.2,
	device="cpu",
):
	input_tokens = tokenize(prompt, word_to_idx)
	current_ids = list(input_tokens)
	generated = []

	blacklist_by_depth = {}
	depth = 0
	max_backtracks = 20
	backtrack_count = 0

	pad_idx = word_to_idx.get("<pad>", 0)
	unk_idx = word_to_idx.get("<unk>", 0)

	while len(generated) < max_tokens:
		# Ejecutar el paso forward
		input_tensor = torch.tensor([current_ids], dtype=torch.long, device=device)
		with torch.no_grad():
			logits = model(input_tensor)

		# Logits del último token
		last_logits = logits[0, -1, :].clone()
		last_logits[pad_idx] = -1e9
		last_logits[unk_idx] = -1e9

		curr_blacklist = blacklist_by_depth.get(depth, set())

		# Muestrear
		next_token, prob, probs = sample_next_token_with_blacklist(
			last_logits, temperature=temperature, top_p=top_p, top_k=top_k, blacklist=curr_blacklist
		)

		token_str = idx_to_word.get(next_token, "?")
		entropy = get_entropy(probs)

		trigger_backtrack = False

		if mode == "confidence" and prob < conf_thresh or mode == "entropy" and entropy > entropy_thresh:
			trigger_backtrack = True
		elif mode == "lookahead" and len(generated) < max_tokens - 1:
			# Lookahead
			temp_ids = current_ids + [next_token]
			temp_tensor = torch.tensor([temp_ids], dtype=torch.long, device=device)
			with torch.no_grad():
				temp_logits = model(temp_tensor)

			next_probs = F.softmax(temp_logits[0, -1, :], dim=-1)
			max_next_prob = next_probs.max().item()

			if max_next_prob < lookahead_thresh:
				trigger_backtrack = True

		if trigger_backtrack and backtrack_count < max_backtracks:
			backtrack_count += 1
			if depth not in blacklist_by_depth:
				blacklist_by_depth[depth] = set()
			blacklist_by_depth[depth].add(next_token)

			if len(blacklist_by_depth[depth]) >= 5 and depth > 0:
				blacklist_by_depth[depth] = set()
				depth -= 1
				if generated:
					generated.pop()
					current_ids.pop()
			continue

		# Confirmar
		generated.append(next_token)
		current_ids.append(next_token)
		depth += 1
		blacklist_by_depth[depth] = set()

		if token_str in ["<pad>", "<eos>", "yo:", "tú:"]:
			break

	response_words = [idx_to_word.get(t, "?") for t in generated]
	response_text = " ".join([w for w in response_words if w not in ["<pad>", "<unk>"]])
	return response_text, backtrack_count


def main():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	base_dir = "/home/joan/Documents/IA/frankenswarm"
	expanded_glyphs_path = os.path.join(base_dir, "configs", "expanded_glyphs.json")
	checkpoint_path = os.path.join(base_dir, "storage", "checkpoints", "EXP_071", "model_final.pt")

	if not os.path.exists(checkpoint_path):
		print(f"❌ Error: No se encontró el checkpoint en {checkpoint_path}")
		return

	# Cargar vocabulario base
	with open(expanded_glyphs_path, encoding="utf-8") as f:
		vocab_data = json.load(f)
		words = vocab_data["words"]
		glyphs = np.array(vocab_data["glyphs"], dtype=np.float32)

	word_to_idx = {w: i for i, w in enumerate(words)}
	idx_to_word = {i: w for w, i in word_to_idx.items()}

	# Inicializar modelo y cargar pesos
	print("🤖 Cargando modelo BitNet...")
	model = BitNet4LayerModel(use_glyphs=True, glyph_table=glyphs, hidden_dim=256, num_layers=4, use_pos_embedding=True).to(device)
	model.load_state_dict(torch.load(checkpoint_path, map_location=device))
	model.eval()

	test_cases = [
		{"prompt": "tú: hola cómo estás yo:", "label": "hola_como_estas"},
		{"prompt": "tú: el oso come yo:", "label": "oso_come"},
		{"prompt": "tú: las flores beben yo:", "label": "flores_beben"},
	]

	configs = [
		{"mode": "none", "params": {}, "desc": "Baseline (Sin Backtracking)"},
		{"mode": "confidence", "params": {"conf_thresh": 0.15}, "desc": "Confidence (thresh=0.15)"},
		{"mode": "confidence", "params": {"conf_thresh": 0.30}, "desc": "Confidence (thresh=0.30)"},
		{"mode": "entropy", "params": {"entropy_thresh": 2.0}, "desc": "Entropy (thresh=2.0)"},
		{"mode": "entropy", "params": {"entropy_thresh": 1.2}, "desc": "Entropy (thresh=1.2)"},
		{"mode": "lookahead", "params": {"lookahead_thresh": 0.05}, "desc": "Lookahead (thresh=0.05)"},
		{"mode": "lookahead", "params": {"lookahead_thresh": 0.15}, "desc": "Lookahead (thresh=0.15)"},
	]

	results = []

	print("\n🧪 Ejecutando matriz de pruebas sobre el chatbot Bit (modelo ternario EXP_071)...")
	print("-" * 80)

	for config in configs:
		mode = config["mode"]
		params = config["params"]
		desc = config["desc"]

		print(f"\nEvaluating: {desc}...")

		for case in test_cases:
			t0 = time.time()
			response, backtracks = generate_with_backtrack(
				model=model,
				word_to_idx=word_to_idx,
				idx_to_word=idx_to_word,
				prompt=case["prompt"],
				mode=mode,
				max_tokens=6,
				temperature=0.7,
				device=device,
				**params,
			)
			dt = time.time() - t0

			print(f"  [{case['label']}] -> R: '{response}' (Backtracks: {backtracks}, Time: {dt:.3f}s)")

			results.append({"desc": desc, "label": case["label"], "response": response, "backtracks": backtracks, "time": dt})

	# Escribir los resultados en una tabla de markdown en los artefactos
	report_path = "/home/joan/.gemini/antigravity/brain/426682cc-776c-436a-9332-8afcdbb382a9/backtrack_comparison_report.md"

	with open(report_path, "w", encoding="utf-8") as f:
		f.write("# Reporte Comparativo: Modos de Backtracking en Bit Chatbot (EXP_071)\n\n")
		f.write(
			"Este reporte compara cuantitativa y cualitativamente los diferentes modos de backtracking y sus umbrales sobre nuestro chatbot Bit (BitNet 4-layer con embeddings de glifos ternarios).\n\n"
		)

		f.write("## Tabla de Resultados (Bit Chatbot)\n\n")
		f.write("| Configuración | Caso de Prueba | Respuesta Generada | Retrocesos | Tiempo (s) |\n")
		f.write("| --- | --- | --- | --- | --- |\n")

		for r in results:
			f.write(f"| {r['desc']} | {r['label']} | `{r['response']}` | {r['backtracks']} | {r['time']:.3f} |\n")

	print(f"\n📊 Reporte de benchmarking guardado en: {report_path}")


if __name__ == "__main__":
	main()
