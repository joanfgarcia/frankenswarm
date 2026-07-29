import argparse
import os

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from src.bitnet.model.modeling_conversational import BitNetCausalLM


def sample_next_token(logits, temperature=0.7, top_p=0.9, top_k=50):
	if temperature <= 0.0:
		return logits.argmax(dim=-1).item()
		
	logits = logits / max(temperature, 1e-5)
	
	if top_k > 0:
		# Keep only the top k tokens
		v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
		logits[logits < v[-1]] = -float("Inf")
		
	if top_p < 1.0:
		sorted_logits, sorted_indices = torch.sort(logits, descending=True)
		cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
		
		# Remove tokens with cumulative probability above top_p (nucleus)
		sorted_indices_to_remove = cumulative_probs > top_p
		# Shift the indices to the right to keep the first token above top_p
		sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
		sorted_indices_to_remove[0] = 0
		
		indices_to_remove = sorted_indices[sorted_indices_to_remove]
		logits[indices_to_remove] = -float("Inf")
		
	probs = F.softmax(logits, dim=-1)
	next_token = torch.multinomial(probs, num_samples=1).item()
	return next_token

def run_chat_sandbox(args):
	lang = args.lang
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if lang == "en":
		tokenizer_path = "storage/experiments/conversational_tokenizer_en.json"
		model_path = "storage/experiments/conversational_agent_en.pt"
		user_prefix = "you"
		assistant_prefix = "me"
		header = "💬 SOVEREIGN CONVERSATIONAL AGENT (10-YEAR-OLD CHILD - ENGLISH)"
		exit_hint = "   Type 'exit' or 'quit' to terminate the chat."
		exit_words = ["exit", "quit", "salir"]
	else:
		tokenizer_path = "storage/experiments/conversational_tokenizer.json"
		model_path = "storage/experiments/conversational_agent.pt"
		user_prefix = "tú"
		assistant_prefix = "yo"
		header = "💬 AGENTE CONVERSACIONAL SOBERANO (NIÑO DE 10 AÑOS - ESPAÑOL)"
		exit_hint = "   Escribe 'salir' para terminar el chat."
		exit_words = ["salir"]
	
	if not os.path.exists(tokenizer_path) or not os.path.exists(model_path):
		print(f"Error: El tokenizador o el modelo entrenado no existen para '{lang}'. Ejecuta primero train_tokenizer.py --lang {lang} y train_conversational.py --lang {lang}")
		return
		
	# 1. Cargar tokenizador y modelo
	tokenizer = Tokenizer.from_file(tokenizer_path)
	vocab_size = tokenizer.get_vocab_size()
	
	state_dict = torch.load(model_path, map_location=device, weights_only=True)
	hidden_dim = state_dict["token_embedding.weight"].shape[1]
	if hidden_dim == 512:
		num_layers = 8
		num_heads = 8
	else:
		num_layers = 4
		num_heads = 4
		
	model = BitNetCausalLM(
		vocab_size=vocab_size,
		hidden_dim=hidden_dim,
		num_layers=num_layers,
		num_heads=num_heads,
		max_seq_len=256
	).to(device)
	
	model.load_state_dict(state_dict)
	model.eval()
	
	tokenizer.token_to_id("[PAD]")
	eos_token_id = tokenizer.token_to_id("[EOS]")
	
	print("\n" + "═"*60)
	print(header)
	print(exit_hint)
	print(f"   Parámetros: Temp={args.temperature}, Top-p={args.top_p}, Top-k={args.top_k}")
	print("   La memoria temporal h_prev se propaga entre turnos.")
	print("═"*60 + "\n")
	
	# Inicializar memoria latente conversacional
	h_prev = torch.zeros((1, model.hidden_dim), device=device)
	
	while True:
		try:
			input_prompt = "You: " if lang == "en" else "Tú: "
			user_input = input(input_prompt)
			if user_input.strip().lower() in exit_words:
				print("\n¡Goodbye!" if lang == "en" else "\n¡Adiós!")
				break
				
			if not user_input.strip():
				continue
				
			# Formatear entrada del usuario recreando el estilo de entrenamiento
			prompt = f"{user_prefix}: {user_input.strip()} [EOS] {assistant_prefix}:"
			input_ids = tokenizer.encode(prompt).ids
			current_ids = list(input_ids)
			
			# Decodificación autoregresiva token a token
			generated = []
			turn_h_prev = h_prev
			
			with torch.no_grad():
				for _ in range(64):  # Máximo 64 tokens de respuesta
					input_tensor = torch.tensor([current_ids], dtype=torch.long, device=device)
					logits, next_h = model(input_tensor, h_prev=turn_h_prev)
					
					# Obtener predicción del último token mediante muestreo estructurado
					next_token = sample_next_token(
						logits[0, -1, :],
						temperature=args.temperature,
						top_p=args.top_p,
						top_k=args.top_k
					)
					
					if next_token == eos_token_id:
						break
						
					# Evitar loops o tokens especiales de interlocutor si salen en medio
					token_str = tokenizer.decode([next_token])
					if token_str in [f"{user_prefix}:", f"{assistant_prefix}:"]:
						break
						
					generated.append(next_token)
					current_ids.append(next_token)
				
				# Guardar la memoria latente al final del turno
				h_prev = next_h
			
			response_text = tokenizer.decode(generated).strip()
			out_prefix = "Me: " if lang == "en" else "Yo: "
			print(f"{out_prefix}{response_text}\n")
			
		except (KeyboardInterrupt, EOFError):
			print("\n¡Goodbye!" if lang == "en" else "\n¡Adiós!")
			break

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--lang", type=str, default="es", choices=["es", "en"], help="Idioma del chat (es/en)")
	parser.add_argument("--temperature", type=float, default=0.7, help="Temperatura de muestreo (0.0 = greedy)")
	parser.add_argument("--top-p", type=float, default=0.9, help="Umbral de muestreo Nucleus (Top-p)")
	parser.add_argument("--top-k", type=int, default=50, help="Filtro de muestreo Top-k")
	args = parser.parse_args()
	run_chat_sandbox(args)
