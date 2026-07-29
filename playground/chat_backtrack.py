import argparse
import os
import sys

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from src.bitnet.model.modeling_conversational import BitNetCausalLM


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


def generate_response_with_backtrack(
	model,
	tokenizer,
	prompt_ids,
	h_prev,
	device,
	mode="none",
	max_tokens=64,
	temperature=0.7,
	top_p=0.9,
	top_k=50,
	conf_thresh=0.15,
	lookahead_thresh=0.05,
	entropy_thresh=2.2,
	visual=True,
):
	"""
	Genera una respuesta token a token implementando backtracking dinámico.
	Soporta modos: 'none', 'confidence', 'entropy', 'lookahead'.
	"""
	eos_token_id = tokenizer.token_to_id("[EOS]")
	tokenizer.token_to_id("tú")
	tokenizer.token_to_id("yo")

	current_ids = list(prompt_ids)
	generated = []

	# Pila para deshacer (histórico de decisiones por cada token generado)
	# Cada elemento de la pila guarda:
	# (token_ids_state, token_id_elegido, probs, blacklist_actual)
	blacklist_by_depth = {}  # depth -> set of blacklisted token_ids

	depth = 0
	max_backtracks = 20
	backtrack_count = 0

	sys.stdout.write("Yo > ")
	sys.stdout.flush()

	while len(generated) < max_tokens:
		# 1. Ejecutar el paso forward
		input_tensor = torch.tensor([current_ids], dtype=torch.long, device=device)
		with torch.no_grad():
			logits, next_h = model(input_tensor, h_prev=h_prev)

		# Obtener logits del último token
		last_logits = logits[0, -1, :]

		# Obtener blacklist para la profundidad actual
		curr_blacklist = blacklist_by_depth.get(depth, set())

		# 2. Muestrear propuesta de token
		next_token, prob, probs = sample_next_token_with_blacklist(
			last_logits, temperature=temperature, top_p=top_p, top_k=top_k, blacklist=curr_blacklist
		)

		token_str = tokenizer.decode([next_token]).strip()
		entropy = get_entropy(probs)

		# 3. Evaluar criterios de backtracking
		trigger_backtrack = False

		if mode == "confidence" and prob < conf_thresh or mode == "entropy" and entropy > entropy_thresh:
			trigger_backtrack = True

		elif mode == "lookahead" and len(generated) < max_tokens - 1:
			# Lookahead: Evaluamos el siguiente token hipotético t_{i+1}
			temp_ids = current_ids + [next_token]
			temp_tensor = torch.tensor([temp_ids], dtype=torch.long, device=device)
			with torch.no_grad():
				temp_logits, _ = model(temp_tensor, h_prev=h_prev)

			# Probabilidades del siguiente paso
			next_probs = F.softmax(temp_logits[0, -1, :], dim=-1)
			max_next_prob = next_probs.max().item()

			if max_next_prob < lookahead_thresh:
				trigger_backtrack = True

		# 4. Procesar backtracking o avanzar
		if trigger_backtrack and backtrack_count < max_backtracks:
			backtrack_count += 1
			if visual:
				# Imprimir en consola en color rojo tachado el token que rechazamos
				sys.stdout.write(f"\033[91m~~{token_str}~~\033[0m ")
				sys.stdout.flush()

			# Añadir a la blacklist de esta profundidad
			if depth not in blacklist_by_depth:
				blacklist_by_depth[depth] = set()
			blacklist_by_depth[depth].add(next_token)

			# Si la blacklist de esta profundidad agota opciones viables, retrocedemos aún más (profundidad anterior)
			if len(blacklist_by_depth[depth]) >= 5 and depth > 0:
				# Resetear blacklist de la profundidad actual al retroceder
				blacklist_by_depth[depth] = set()
				depth -= 1
				# Eliminar el último token confirmado de las listas
				if generated:
					last_confirmed = generated.pop()
					current_ids.pop()
					if visual:
						sys.stdout.write(f"\033[93m[<- pop {tokenizer.decode([last_confirmed]).strip()}]\033[0m ")
						sys.stdout.flush()
			continue  # Reintentar el paso actual con la blacklist actualizada

		# Confirmar token generado
		generated.append(next_token)
		current_ids.append(next_token)

		if visual:
			# Imprimir el token confirmado en verde
			sys.stdout.write(f"\033[92m{token_str}\033[0m ")
			sys.stdout.flush()

		if next_token == eos_token_id or token_str in ["yo:", "tú:", "me:", "you:"]:
			break

		# Avanzar profundidad
		depth += 1
		# Limpiar blacklist de profundidades futuras al avanzar con éxito
		blacklist_by_depth[depth] = set()

	sys.stdout.write("\n")
	sys.stdout.flush()

	# Ejecutar el modelo final sobre la secuencia definitiva para obtener la memoria h_prev correcta
	final_tensor = torch.tensor([current_ids], dtype=torch.long, device=device)
	with torch.no_grad():
		_, final_h = model(final_tensor, h_prev=h_prev)

	# Limpiar tags de diálogo
	response_text = tokenizer.decode(generated).strip()
	response_text = response_text.replace("[EOS]", "").strip()

	return response_text, final_h, backtrack_count


def main():
	parser = argparse.ArgumentParser(description="Chat interactivo BitNet con backtracking")
	parser.add_argument("--mode", type=str, default="confidence", choices=["none", "confidence", "entropy", "lookahead"], help="Modo de backtracking")
	parser.add_argument("--temperature", type=float, default=0.7, help="Temperatura")
	parser.add_argument("--conf_thresh", type=float, default=0.18, help="Umbral de confianza")
	parser.add_argument("--entropy_thresh", type=float, default=2.0, help="Umbral de entropía")
	parser.add_argument("--lookahead_thresh", type=float, default=0.06, help="Umbral de lookahead")
	args = parser.parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	base_dir = "/home/joan/Documents/IA/frankenswarm"
	tokenizer_path = os.path.join(base_dir, "storage", "experiments", "conversational_tokenizer.json")
	model_path = os.path.join(base_dir, "storage", "experiments", "conversational_agent.pt")

	if not os.path.exists(tokenizer_path) or not os.path.exists(model_path):
		print("❌ Error: No se encontraron los modelos conversacionales en storage/experiments/")
		return

	print("📖 Cargando tokenizador...")
	tokenizer = Tokenizer.from_file(tokenizer_path)
	vocab_size = tokenizer.get_vocab_size()

	print("🧠 Cargando modelo BitNet...")
	state_dict = torch.load(model_path, map_location=device, weights_only=True)
	hidden_dim = state_dict["token_embedding.weight"].shape[1]
	num_layers = 8 if hidden_dim == 512 else 4
	num_heads = 8 if hidden_dim == 512 else 4

	model = BitNetCausalLM(vocab_size=vocab_size, hidden_dim=hidden_dim, num_layers=num_layers, num_heads=num_heads, max_seq_len=256).to(device)
	model.load_state_dict(state_dict)
	model.eval()

	print("\n" + "═" * 60)
	print("💬 PLAYGROUND CON INFERENCIA DE RETROCESO (BACKTRACKING)")
	print(f"   Modo activo: \033[93m{args.mode.upper()}\033[0m")
	print(f"   Configuración: Temp={args.temperature}")
	if args.mode == "confidence":
		print(f"   Umbral Confianza: P < {args.conf_thresh} -> Backtrack")
	elif args.mode == "entropy":
		print(f"   Umbral Entropía: H > {args.entropy_thresh} -> Backtrack")
	elif args.mode == "lookahead":
		print(f"   Umbral Lookahead: P_next < {args.lookahead_thresh} -> Backtrack")
	print("   Los tokens rechazados se mostrarán como \033[91m~~token~~\033[0m")
	print("═" * 60 + "\n")

	h_prev = torch.zeros((1, model.hidden_dim), device=device)

	while True:
		try:
			user_input = input("Tú > ")
			if user_input.strip().lower() in ["salir", "exit"]:
				break
			if not user_input.strip():
				continue

			prompt = f"tú: {user_input.strip()} [EOS] yo:"
			input_ids = tokenizer.encode(prompt).ids

			_, next_h, backtracks = generate_response_with_backtrack(
				model=model,
				tokenizer=tokenizer,
				prompt_ids=input_ids,
				h_prev=h_prev,
				device=device,
				mode=args.mode,
				temperature=args.temperature,
				conf_thresh=args.conf_thresh,
				entropy_thresh=args.entropy_thresh,
				lookahead_thresh=args.lookahead_thresh,
				visual=True,
			)
			h_prev = next_h
			print(f"\033[90m[Estadística: {backtracks} retrocesos realizados en este turno]\033[0m\n")

		except KeyboardInterrupt:
			print("\n¡Adiós!")
			break


if __name__ == "__main__":
	main()
