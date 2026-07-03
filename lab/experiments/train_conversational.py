import glob
import json
import os
import random

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from src.bitnet.modeling_conversational import BitNetCausalLM


def load_dialogue_dataset(tokenizer_path, lang="es"):
	tokenizer = Tokenizer.from_file(tokenizer_path)
	
	# Cargar todos los diálogos
	dialogues = []
	files = glob.glob("configs/tiny_dialogues*.json")
	for fpath in files:
		if lang == "en":
			if "_en" not in fpath:
				continue
		else:
			if "_en" in fpath:
				continue
		with open(fpath, encoding="utf-8") as f:
			data = json.load(f)
			# Cada diálogo es una lista de turnos (strings)
			dialogues.extend(data)
			
	return dialogues, tokenizer

def train_conversational_model(lang="es", pretrained=False):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if lang == "en":
		tokenizer_path = "storage/experiments/conversational_tokenizer_en.json"
		model_save_path = "storage/experiments/conversational_agent_en.pt"
		pretrained_path = "storage/experiments/pretrained_base_en.pt"
	else:
		tokenizer_path = "storage/experiments/conversational_tokenizer.json"
		model_save_path = "storage/experiments/conversational_agent.pt"
		pretrained_path = "storage/experiments/pretrained_base_es.pt"
	
	if not os.path.exists(tokenizer_path):
		print(f"Error: El tokenizador no existe. Ejecuta primero train_tokenizer.py --lang {lang}")
		return

	dialogues, tokenizer = load_dialogue_dataset(tokenizer_path, lang=lang)
	vocab_size = tokenizer.get_vocab_size()
	print(f"Dataset cargado: {len(dialogues)} diálogos. Vocabulario BPE: {vocab_size} tokens.")

	# Configurar dimensiones según si es pre-entrenado o no
	if pretrained:
		hidden_dim = 512
		num_layers = 8
		num_heads = 8
		lr = 5e-5
		epochs = 10
	else:
		hidden_dim = 256
		num_layers = 4
		num_heads = 4
		lr = 2e-4
		epochs = 15

	# Instanciar modelo causal BitNet
	model = BitNetCausalLM(
		vocab_size=vocab_size,
		hidden_dim=hidden_dim,
		num_layers=num_layers,
		num_heads=num_heads,
		max_seq_len=256
	).to(device)

	if pretrained:
		if os.path.exists(pretrained_path):
			model.load_state_dict(torch.load(pretrained_path, map_location=device, weights_only=True))
			print(f"Loaded pretrained base weights from {pretrained_path}")
		else:
			print(f"Warning: Pretrained base weights {pretrained_path} not found. Training from scratch.")

	optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
	pad_token_id = tokenizer.token_to_id("[PAD]")
	
	# Usamos precisión mixta para el fine-tuning para ahorrar memoria
	scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
	batch_size = 16
	
	print(f"Iniciando entrenamiento del Agente Conversacional en {device}...")
	
	for epoch in range(1, epochs + 1):
		# Mezclar diálogos en cada época
		random.shuffle(dialogues)
		
		# --- FASE WAKE: SFT Clásico (Autoregresivo sobre secuencia completa) ---
		model.train()
		wake_loss_sum = 0.0
		wake_steps = 0
		
		for i in range(0, len(dialogues), batch_size):
			batch_dialogues = dialogues[i:i+batch_size]
			if not batch_dialogues:
				continue
				
			# Formatear cada diálogo uniendo los turnos con un espacio o un token especial
			batch_inputs = []
			for dialogue in batch_dialogues:
				full_text = " [EOS] ".join(dialogue) + " [EOS]"
				encoded = tokenizer.encode(full_text).ids
				# Limitar a max_seq_len
				if len(encoded) > 256:
					encoded = encoded[:256]
				batch_inputs.append(encoded)
				
			# Rellenar con [PAD] para unificar longitudes
			max_len = max(len(x) for x in batch_inputs)
			padded_inputs = [x + [pad_token_id] * (max_len - len(x)) for x in batch_inputs]
			
			input_tensor = torch.tensor(padded_inputs, dtype=torch.long, device=device)
			
			optimizer.zero_grad()
			
			# En el paso Wake de SFT, no pasamos h_prev (aprende la gramática básica autoregresiva)
			with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
				logits, _ = model(input_tensor)
				
				# Calcular pérdida Causal: predecir el siguiente token
				shift_logits = logits[..., :-1, :].contiguous()
				shift_labels = input_tensor[..., 1:].contiguous()
				
				loss = F.cross_entropy(
					shift_logits.view(-1, vocab_size),
					shift_labels.view(-1),
					ignore_index=pad_token_id
				)
			
			if scaler is not None:
				scaler.scale(loss).backward()
				scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
				scaler.step(optimizer)
				scaler.update()
			else:
				loss.backward()
				torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
				optimizer.step()
			
			wake_loss_sum += loss.item()
			wake_steps += 1
			
		avg_wake_loss = wake_loss_sum / max(1, wake_steps)
		
		# --- FASE SLEEP: Consolidación Secuencial BPTT (Turno a Turno con h_prev) ---
		sleep_loss_sum = 0.0
		sleep_steps = 0
		
		for i in range(0, len(dialogues), batch_size):
			batch_dialogues = dialogues[i:i+batch_size]
			if not batch_dialogues:
				continue
				
			optimizer.zero_grad()
			loss_batch = 0.0
			
			# Inicializar h_prev a cero para el lote
			h_prev = torch.zeros((len(batch_dialogues), model.hidden_dim), device=device)
			
			# Encontrar la longitud máxima de turnos en este lote
			max_turns = max(len(d) for d in batch_dialogues)
			
			# Procesar turno a turno (BPTT a través de h_prev)
			for turn_idx in range(max_turns):
				turn_inputs = []
				for dialogue in batch_dialogues:
					encoded = tokenizer.encode(dialogue[turn_idx]).ids if turn_idx < len(dialogue) else [pad_token_id]
					turn_inputs.append(encoded)
					
				# Rellenar turnos
				max_turn_len = max(len(x) for x in turn_inputs)
				padded_turns = [x + [pad_token_id] * (max_turn_len - len(x)) for x in turn_inputs]
				
				turn_tensor = torch.tensor(padded_turns, dtype=torch.long, device=device)
				
				# Paso forward pasando el h_prev acumulado
				with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
					logits, next_h_prev = model(turn_tensor, h_prev=h_prev)
					
					# Propagar h_prev al siguiente turno (conectando el grafo de gradientes)
					h_prev = next_h_prev
					
					# Pérdida de predicción sobre el turno actual
					shift_logits = logits[..., :-1, :].contiguous()
					shift_labels = turn_tensor[..., 1:].contiguous()
					
					loss_turn = F.cross_entropy(
						shift_logits.view(-1, vocab_size),
						shift_labels.view(-1),
						ignore_index=pad_token_id
					)
				
				loss_batch += loss_turn
				
			loss_batch = loss_batch / max_turns
			
			if scaler is not None:
				scaler.scale(loss_batch).backward()
				scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
				scaler.step(optimizer)
				scaler.update()
			else:
				loss_batch.backward()
				torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
				optimizer.step()
			
			sleep_loss_sum += loss_batch.item()
			sleep_steps += 1
			
		avg_sleep_loss = sleep_loss_sum / max(1, sleep_steps)
		
		print(f"Época {epoch:02d}/{epochs} | Loss Wake (SFT): {avg_wake_loss:.4f} | Loss Sleep (BPTT): {avg_sleep_loss:.4f}")
		
	# Guardar modelo final
	os.makedirs("storage/experiments", exist_ok=True)
	torch.save(model.state_dict(), model_save_path)
	print(f"Modelo entrenado y guardado en {model_save_path}")

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--lang", type=str, default="es", choices=["es", "en"], help="Idioma del modelo a entrenar")
	parser.add_argument("--pretrained", action="store_true", help="Cargar pesos del base pre-entrenado y usar dimensiones escaladas")
	args = parser.parse_args()
	train_conversational_model(args.lang, args.pretrained)
