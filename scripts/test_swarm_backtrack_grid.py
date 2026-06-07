import json
import os
import re
import time

import numpy as np
import torch
import torch.nn.functional as F

from src.bitnet.modeling_bitnet import BitNet4LayerModel


def get_entropy(probs):
	"""Calcula la entropía de Shannon."""
	return -torch.sum(probs * torch.log(probs + 1e-9)).item()


def sample_next_token_with_blacklist(logits, temperature=0.7, top_p=0.9, top_k=40, blacklist=None):
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


def tokenize_words(text, word_to_idx):
	w_list = re.findall(r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_<>\-]+", text.lower())
	return [word_to_idx.get(w, 1) for w in w_list]


def generate_with_backtrack(
	model,
	context_tokens,
	word_to_idx,
	idx_to_word,
	mode="none",
	max_tokens=12,
	temperature=0.7,
	top_p=0.9,
	top_k=40,
	conf_thresh=0.10,
	lookahead_thresh=0.05,
	entropy_thresh=1.8,
	penalty_val=1.2,
	device="cpu",
):
	input_ids = list(context_tokens)
	generated = []

	blacklist_by_depth = {}
	depth = 0
	max_backtracks = 20
	backtrack_count = 0

	yo_tok = word_to_idx.get("yo")
	tu_tok = word_to_idx.get("tú")

	while len(generated) < max_tokens:
		# Formatear el input con padding a 64
		curr_input = list(input_ids)[-64:]
		input_len = len(curr_input)
		if len(curr_input) < 64:
			curr_input = curr_input + [0] * (64 - len(curr_input))

		x = torch.tensor([curr_input], dtype=torch.long, device=device)
		with torch.no_grad():
			logits = model(x)

		last_token_idx = input_len - 1
		next_token_logits = logits[0, last_token_idx].clone()

		# Aplicar penalización por repetición
		for token in set(input_ids):
			next_token_logits[token] -= penalty_val

		# Evitar pad/unk de salida si queremos palabras legibles
		next_token_logits[0] = -1e9
		next_token_logits[1] = -1e9

		curr_blacklist = blacklist_by_depth.get(depth, set())

		# Muestrear
		next_token, prob, probs = sample_next_token_with_blacklist(
			next_token_logits, temperature=temperature, top_p=top_p, top_k=top_k, blacklist=curr_blacklist
		)

		idx_to_word.get(next_token, "?")
		entropy = get_entropy(probs)

		trigger_backtrack = False

		if mode == "confidence" and prob < conf_thresh or mode == "entropy" and entropy > entropy_thresh:
			trigger_backtrack = True
		elif mode == "lookahead" and len(generated) < max_tokens - 1:
			# Evaluar paso siguiente temporalmente
			temp_ids = input_ids + [next_token]
			temp_curr = list(temp_ids)[-64:]
			temp_len = len(temp_curr)
			if len(temp_curr) < 64:
				temp_curr = temp_curr + [0] * (64 - len(temp_curr))

			temp_x = torch.tensor([temp_curr], dtype=torch.long, device=device)
			with torch.no_grad():
				temp_logits = model(temp_x)

			next_probs = F.softmax(temp_logits[0, temp_len - 1, :], dim=-1)
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
					input_ids.pop()
			continue

		# Confirmar
		generated.append(next_token)
		input_ids.append(next_token)
		depth += 1
		blacklist_by_depth[depth] = set()

		# Si genera fin de turno o pad, parar
		if next_token in (yo_tok, tu_tok):
			break

	response_text = " ".join([idx_to_word.get(t, "?") for t in generated if t not in [yo_tok, tu_tok]])
	return response_text, backtrack_count


def main():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	base_dir = "/home/joan/Documents/IA/frankenswarm"
	expanded_glyphs_path = os.path.join(base_dir, "configs", "expanded_glyphs.json")
	model_path = os.path.join(base_dir, "storage", "checkpoints", "EXP_072", "model_final.pt")

	if not os.path.exists(model_path):
		print(f"❌ Error: No se encontró el modelo en {model_path}")
		return

	# Cargar vocabulario
	with open(expanded_glyphs_path, encoding="utf-8") as f:
		vocab_data = json.load(f)
		words = vocab_data["words"]
		glyphs = np.array(vocab_data["glyphs"], dtype=np.float32)

	word_to_idx = {w: i for i, w in enumerate(words)}
	idx_to_word = dict(enumerate(words))

	# Inicializar modelo
	model = BitNet4LayerModel(
		use_glyphs=True, glyph_table=glyphs, hidden_dim=512, num_layers=6, use_pos_embedding=True, is_causal=True, max_seq_len=64
	).to(device)
	model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
	model.eval()

	# Construir contextos de prueba basados en las identidades
	# Caso 1: Nico comienza presentándose
	# "yo tengo perro yo"
	ctx_1 = tokenize_words("yo tengo perro", word_to_idx) + [word_to_idx.get("yo")]

	# Caso 2: Nico ha saludado, Sofi ha contestado y Nico vuelve a hablar
	# "yo tengo perro yo hola tú hola yo"
	ctx_2 = (
		tokenize_words("yo tengo perro", word_to_idx)
		+ [word_to_idx.get("yo")]
		+ tokenize_words("hola", word_to_idx)
		+ [word_to_idx.get("tú")]
		+ tokenize_words("hola", word_to_idx)
		+ [word_to_idx.get("yo")]
	)

	test_cases = [
		{"ctx": ctx_1, "label": "Nico inicia", "prompt_text": "yo tengo perro yo"},
		{"ctx": ctx_2, "label": "Saludo cruzado", "prompt_text": "yo tengo perro yo hola tú hola yo"},
	]

	configs = [
		{"mode": "none", "params": {}, "desc": "Baseline (Sin Backtracking)"},
		{"mode": "confidence", "params": {"conf_thresh": 0.02}, "desc": "Confidence (thresh=0.02)"},
		{"mode": "confidence", "params": {"conf_thresh": 0.04}, "desc": "Confidence (thresh=0.04)"},
		{"mode": "entropy", "params": {"entropy_thresh": 5.5}, "desc": "Entropy (thresh=5.5)"},
		{"mode": "entropy", "params": {"entropy_thresh": 6.0}, "desc": "Entropy (thresh=6.0)"},
		{"mode": "lookahead", "params": {"lookahead_thresh": 0.01}, "desc": "Lookahead (thresh=0.01)"},
		{"mode": "lookahead", "params": {"lookahead_thresh": 0.02}, "desc": "Lookahead (thresh=0.02)"},
	]

	results = []

	print("\n🧪 Iniciando Benchmarking de Backtracking sobre el Swarm Chatbot Nico (EXP_072)...")
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
				context_tokens=case["ctx"],
				word_to_idx=word_to_idx,
				idx_to_word=idx_to_word,
				mode=mode,
				max_tokens=10,
				temperature=0.7,
				device=device,
				**params,
			)
			dt = time.time() - t0

			print(f"  [{case['label']}] -> R: '{response}' (Backtracks: {backtracks}, Time: {dt:.3f}s)")

			results.append({"desc": desc, "label": case["label"], "response": response, "backtracks": backtracks, "time": dt})

	# Escribir los resultados en el reporte de markdown
	report_path = "/home/joan/.gemini/antigravity/brain/426682cc-776c-436a-9332-8afcdbb382a9/backtrack_comparison_report.md"

	with open(report_path, "w", encoding="utf-8") as f:
		f.write("# Reporte Comparativo: Backtracking en Swarm Chatbot Nico (EXP_072)\n\n")
		f.write(
			"Este reporte compara cuantitativa y cualitativamente los diferentes modos de backtracking y sus umbrales sobre nuestro modelo de enjambre Nico (BitNet 6-layer, 19.3M params, glifos de 65 trits).\n\n"
		)

		f.write("## Tabla de Resultados (Swarm Nico)\n\n")
		f.write("| Configuración | Caso de Prueba | Respuesta Generada | Retrocesos | Tiempo (s) |\n")
		f.write("| --- | --- | --- | --- | --- |\n")

		for r in results:
			f.write(f"| {r['desc']} | {r['label']} | `{r['response']}` | {r['backtracks']} | {r['time']:.3f} |\n")

	print(f"\n📊 Reporte de benchmarking guardado en: {report_path}")


if __name__ == "__main__":
	main()
