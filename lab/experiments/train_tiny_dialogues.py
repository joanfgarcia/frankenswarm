import json
import os
import re

import numpy as np
import torch
import torch.nn.functional as F

from src.bitnet.model.modeling_bitnet import BitNet4LayerModel

# Definición de definiciones para entrenar el glyph_projection_head (Fase 2)
KNOWN_DEFINITIONS = {
	"agua": "líquido frío para beber y vivir",
	"comida": "algo bueno que come el cuerpo",
	"fuego": "cosa caliente con luz y peligro",
	"árbol": "planta grande en la tierra",
	"perro": "animal que es amigo del hombre",
	"gato": "animal pequeño que bebe leche",
	"sol": "objeto grande arriba con mucha luz",
	"noche": "tiempo oscuro con luna y frío",
	"bosque": "lugar grande con muchos árboles",
	"casa": "lugar seguro para vivir",
	"seguro": "bueno sin peligro",
	"grupo": "mucha gente junta",
	"piedra": "objeto duro que no vive",
	"luna": "objeto con luz en la noche",
	"río": "agua que se mueve",
	"beber": "tomar agua para vivir",
	"comer": "tomar comida para vivir",
	"dormir": "descansar el cuerpo en la cama",
	"miedo": "sentimiento malo por peligro",
	"alegría": "sentimiento bueno en el corazón"
}

def tokenize(text: str, word_to_idx: dict) -> list[int]:
	words = re.findall(r'[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_<>\-]+', text.lower())
	return [word_to_idx.get(w, word_to_idx.get("<unk>", 1)) for w in words]

def generate_response(model, prompt_tokens, word_to_idx, idx_to_word, device, max_gen_len=15, temperature=0.7, penalty_val=1.2):
	model.eval()
	input_ids = list(prompt_tokens)
	generated = []
	for _ in range(max_gen_len):
		# Pad/truncate input to max_seq_len (64)
		curr_input = input_ids[-64:]
		input_len = len(curr_input)
		if len(curr_input) < 64:
			curr_input = curr_input + [0] * (64 - len(curr_input))
		x = torch.tensor([curr_input], dtype=torch.long, device=device)
		with torch.no_grad():
			logits = model(x) # (1, 64, vocab_size)
		# Encontrar la posición del último token del prompt
		last_token_idx = input_len - 1
		next_token_logits = logits[0, last_token_idx].clone()
		
		# Aplicar penalización por repetición
		for token in set(input_ids):
			next_token_logits[token] -= penalty_val
			
		# Muestreo
		probs = F.softmax(next_token_logits / temperature, dim=-1)
		next_token = torch.multinomial(probs, num_samples=1).item()
		
		if next_token == 0: # <pad>
			break
		input_ids.append(next_token)
		generated.append(next_token)
		
	return [idx_to_word[t] for t in generated]

def run_training():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print("═══ 🧒 Entrenamiento de Causal BitNet en Diálogos (EXP_072) ═══")
	print(f"[Device]: {device}")
	
	base_dir = "/home/joan/Documents/IA/frankenswarm"
	expanded_glyphs_path = os.path.join(base_dir, "configs", "expanded_glyphs.json")
	os.path.join(base_dir, "configs", "tiny_dialogues.json")
	
	with open(expanded_glyphs_path, encoding="utf-8") as f:
		vocab_data = json.load(f)
		words = vocab_data["words"]
		glyphs = np.array(vocab_data["glyphs"], dtype=np.float32)
		
	word_to_idx = {w: i for i, w in enumerate(words)}
	idx_to_word = dict(enumerate(words))
	vocab_size = len(words)
	print(f"Vocabulario cargado: {vocab_size} palabras.")
	
	# Cargar diálogos generados
	dialogue_list = []
	config_dir = os.path.join(base_dir, "configs")
	if os.path.exists(config_dir):
		for filename in os.listdir(config_dir):
			if filename.startswith("tiny_dialogues") and filename.endswith(".json"):
				filepath = os.path.join(config_dir, filename)
				try:
					with open(filepath, encoding="utf-8") as f:
						part_list = json.load(f)
						dialogue_list.extend(part_list)
						print(f"Loaded {len(part_list)} dialogues from {filename}")
				except Exception as e:
					print(f"Error loading {filename}: {e}")
					
	if not dialogue_list:
		print("❌ No se encontraron diálogos para entrenar.")
		return
	print(f"Total dialogues loaded for training: {len(dialogue_list)}")
	
	# Inicializar modelo BitNet grande (hidden_dim=512, num_layers=6)
	model = BitNet4LayerModel(
		use_glyphs=True,
		glyph_table=glyphs,
		hidden_dim=512,
		num_layers=6,
		use_pos_embedding=True,
		is_causal=True,
		max_seq_len=64
	).to(device)
	
	# Contar parámetros del modelo
	n_params = sum(p.numel() for p in model.parameters())
	print(f"Model scaled to hidden_dim=512, num_layers=6. Total parameters: {n_params:,}")
	
	# ═══════════════════════════════════════════════════════════════════
	# FASE 1: Entrenamiento Causal de Diálogos
	# ═══════════════════════════════════════════════════════════════════
	print("\n--- Fase 1: Entrenamiento Causal de Diálogos ---")
	
	# Tokenizar y estructurar diálogos
	tokenized_dialogues = []
	for dialogue in dialogue_list:
		dialogue_tokens = []
		for turn in dialogue:
			# Cada turno tiene el formato "yo: frase" o "tú: frase"
			# Lo convertimos a la secuencia: ["yo", "frase", "tú", "frase"]
			# Quitar puntuación extra y limpiar
			match = re.match(r'^(yo|tú)\s*:\s*(.*)$', turn, re.IGNORECASE)
			if match:
				speaker = match.group(1).lower()
				content = match.group(2)
				turn_words = [speaker] + re.findall(r'[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_<>\-]+', content.lower())
				turn_tokens = [word_to_idx.get(w, 1) for w in turn_words]
				dialogue_tokens.extend(turn_tokens)
				
		if len(dialogue_tokens) >= 2:
			# Pad a seq_len = 64
			dialogue_tokens = dialogue_tokens + [0] * (64 - len(dialogue_tokens)) if len(dialogue_tokens) < 64 else dialogue_tokens[:64]
			tokenized_dialogues.append(dialogue_tokens)
			
	x_train = torch.tensor(tokenized_dialogues, dtype=torch.long, device=device)
	print(f"Processed {x_train.size(0)} valid training dialogue sequences of length 64.")
	
	optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
	
	model.train()
	epochs_fase1 = 150
	batch_size = 64
	
	for epoch in range(epochs_fase1):
		epoch_loss = 0.0
		permutation = torch.randperm(x_train.size(0))
		
		for i in range(0, x_train.size(0), batch_size):
			indices = permutation[i:i+batch_size]
			batch_x = x_train[indices] # (batch, seq_len)
			
			optimizer.zero_grad()
			
			# Causal language modeling shift
			inputs = batch_x[:, :-1]
			targets = batch_x[:, 1:]
			
			logits = model(inputs) # (batch, seq_len - 1, vocab_size)
			
			# Cross entropy ignorando <pad> (index 0)
			loss = F.cross_entropy(
				logits.reshape(-1, vocab_size), 
				targets.reshape(-1), 
				ignore_index=0
			)
			loss.backward()
			optimizer.step()
			
			epoch_loss += loss.item() * batch_x.size(0)
			
		epoch_loss /= x_train.size(0)
		if (epoch + 1) % 10 == 0 or epoch == 0:
			print(f"  Época {epoch+1:3d}/{epochs_fase1} | Causal Dialogue Loss: {epoch_loss:.4f}")
			
	# ═══════════════════════════════════════════════════════════════════
	# FASE 2: Alineación de la Cabeza de Proyección
	# ═══════════════════════════════════════════════════════════════════
	print("\n--- Fase 2: Entrenamiento de Proyección de Glifos ---")
	
	def_inputs = []
	def_targets = []
	
	for word_name, def_text in KNOWN_DEFINITIONS.items():
		if word_name in word_to_idx:
			tokens = tokenize(def_text, word_to_idx)
			# Pad a seq_len_def = 8
			seq_len_def = 8
			tokens = tokens + [0] * (seq_len_def - len(tokens)) if len(tokens) < seq_len_def else tokens[:seq_len_def]
				
			def_inputs.append(tokens)
			def_targets.append(model.glyph_embedding.glyph_table[word_to_idx[word_name]].cpu().numpy())
			
	x_def = torch.tensor(def_inputs, dtype=torch.long, device=device)
	y_def = torch.tensor(np.array(def_targets), dtype=torch.float, device=device) # (N_def, 65)
	
	epochs_fase2 = 30
	proj_optimizer = torch.optim.AdamW(
		list(model.glyph_projection_head.parameters()) + list(model.core_layers.parameters()), 
		lr=1e-3
	)
	
	for epoch in range(epochs_fase2):
		proj_optimizer.zero_grad()
		
		# Obtener hidden state del core
		h = model._embed_input(x_def)
		for layer in model.core_layers:
			h = layer(h)
		h = model.norm(h)
		
		h_mean = h.mean(dim=1) # (N_def, hidden_dim)
		pred_glyphs = model.glyph_projection_head(h_mean) # (N_def, 65)
		
		loss = F.mse_loss(pred_glyphs, y_def)
		loss.backward()
		proj_optimizer.step()
		
		if (epoch + 1) % 5 == 0 or epoch == 0:
			print(f"  Época {epoch+1:2d}/{epochs_fase2} | Proj Loss (MSE): {loss.item():.4f}")
			
	# Guardar checkpoints
	save_dir = os.path.join(base_dir, "storage", "checkpoints", "EXP_072")
	os.makedirs(save_dir, exist_ok=True)
	torch.save(model.state_dict(), os.path.join(save_dir, "model_final.pt"))
	print(f"\n[Checkpoint] Modelo guardado en: {save_dir}/model_final.pt")
	
	# ═══════════════════════════════════════════════════════════════════
	# FASE 3: Validación rápida del diálogo
	# ═══════════════════════════════════════════════════════════════════
	print("\n--- Fase 3: Pruebas de generación del diálogo ---")
	test_prompts = [
		["yo", "hola", "amigo"],
		["yo", "vamos", "al", "parque"],
		["yo", "tengo", "hambre"]
	]
	
	for p_words in test_prompts:
		p_tokens = [word_to_idx.get(w, 1) for w in p_words]
		response = generate_response(model, p_tokens, word_to_idx, idx_to_word, device)
		print(f"Prompt: {' '.join(p_words)}")
		print(f"Bit: {' '.join(response)}\n")

if __name__ == "__main__":
	run_training()
