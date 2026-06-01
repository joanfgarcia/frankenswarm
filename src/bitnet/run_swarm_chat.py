import os
import json
import re
import torch
import torch.nn.functional as F
from src.bitnet.modeling_bitnet import BitNet4LayerModel

def generate_step(model, context_tokens, device, word_to_idx, idx_to_word, max_len=12, temperature=0.7, penalty_val=1.2):
	model.eval()
	input_ids = list(context_tokens)
	generated = []
	
	for _ in range(max_len):
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
			
		# Muestreo
		probs = F.softmax(next_token_logits / temperature, dim=-1)
		next_token = torch.multinomial(probs, num_samples=1).item()
		
		# Si genera pad o fin de turno, parar
		if next_token == 0 or next_token == word_to_idx.get("yo") or next_token == word_to_idx.get("tú"):
			break
			
		input_ids.append(next_token)
		generated.append(next_token)
		
	return generated

def run_chat_simulation():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print("═══ 🏟️ Simulación de Swarm Chat: Dos Bits Conversando (EXP_072) ═══")
	print(f"[Device]: {device}")
	
	base_dir = "/home/joan/Documents/IA/frankenswarm"
	expanded_glyphs_path = os.path.join(base_dir, "configs", "expanded_glyphs.json")
	model_path = os.path.join(base_dir, "storage", "checkpoints", "EXP_072", "model_final.pt")
	
	if not os.path.exists(model_path):
		print(f"❌ No se encontró el modelo entrenado en: {model_path}. Por favor ejecuta train_tiny_dialogues.py primero.")
		return
		
	# Cargar vocabulario
	with open(expanded_glyphs_path, "r", encoding="utf-8") as f:
		vocab_data = json.load(f)
		words = vocab_data["words"]
		glyphs = np.array(vocab_data["glyphs"], dtype=np.float32)
		
	word_to_idx = {w: i for i, w in enumerate(words)}
	idx_to_word = {i: w for i, w in enumerate(words)}
	
	# Inicializar modelo
	model = BitNet4LayerModel(
		use_glyphs=True,
		glyph_table=glyphs,
		hidden_dim=512,
		num_layers=6,
		use_pos_embedding=True,
		is_causal=True,
		max_seq_len=64
	).to(device)
	
	model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
	print("🧠 Modelo cargado exitosamente.")
	
	# Historias para cada Bit
	story_a_path = os.path.join(base_dir, "configs", "story_bit_a.txt")
	story_b_path = os.path.join(base_dir, "configs", "story_bit_b.txt")
	
	nico_story_str = "yo soy nico y vivo cerca del río con perro grande"
	if os.path.exists(story_a_path):
		with open(story_a_path, "r", encoding="utf-8") as f:
			nico_story_str = f.read().strip()
			
	sofi_story_str = "yo soy sofi y tengo gato pequeño en casa cueva"
	if os.path.exists(story_b_path):
		with open(story_b_path, "r", encoding="utf-8") as f:
			sofi_story_str = f.read().strip()
	
	# Tokenizar historias
	def tokenize_words(text):
		w_list = re.findall(r'[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_<>\-]+', text.lower())
		return [word_to_idx.get(w, 1) for w in w_list]
		
	nico_story = tokenize_words(nico_story_str)
	sofi_story = tokenize_words(sofi_story_str)
	
	print("\n📜 Historias de Identidad:")
	print(f"  [Bit-A (Nico)]: {nico_story_str}")
	print(f"  [Bit-B (Sofi)]: {sofi_story_str}")
	print("-" * 50)
	
	# Historial compartido de la conversación (como tuplas de (autor, tokens_del_mensaje))
	# autor: 'A' o 'B'
	shared_history = []
	
	# Primer mensaje: Comienza Nico
	# Nico habla de su perro o saluda
	current_speaker = 'A'
	
	print("\n💬 Inicio del Diálogo Autónomo:\n")
	
	for turn_idx in range(12): # 12 turnos de conversación
		if current_speaker == 'A':
			# Nico está hablando
			# Construir contexto para Nico:
			# story + history (de forma que los mensajes propios sean 'yo', y los de Sofi sean 'tú')
			context = list(nico_story)
			for author, msg_tokens in shared_history:
				speaker_token = word_to_idx.get("yo") if author == 'A' else word_to_idx.get("tú")
				context.append(speaker_token)
				context.extend(msg_tokens)
			context.append(word_to_idx.get("yo")) # Nico va a hablar
			
			# Si el contexto es demasiado largo, deslizarlo conservando la historia (nico_story)
			max_history_len = 64 - len(nico_story) - 2
			if len(context) > 63:
				history_part = context[len(nico_story):-1]
				context = nico_story + history_part[-max_history_len:] + [word_to_idx.get("yo")]
				
			generated_tokens = generate_step(model, context, device, word_to_idx, idx_to_word, temperature=0.7)
			msg_str = " ".join([idx_to_word[t] for t in generated_tokens])
			print(f"👦 Nico (Bit-A) > {msg_str}")
			
			shared_history.append(('A', generated_tokens))
			current_speaker = 'B'
		else:
			# Sofi está hablando
			# Construir contexto para Sofi:
			# story + history (de forma que los mensajes propios sean 'yo', y los de Nico sean 'tú')
			context = list(sofi_story)
			for author, msg_tokens in shared_history:
				speaker_token = word_to_idx.get("yo") if author == 'B' else word_to_idx.get("tú")
				context.append(speaker_token)
				context.extend(msg_tokens)
			context.append(word_to_idx.get("yo")) # Sofi va a hablar
			
			# Ajustar longitud del contexto
			max_history_len = 64 - len(sofi_story) - 2
			if len(context) > 63:
				history_part = context[len(sofi_story):-1]
				context = sofi_story + history_part[-max_history_len:] + [word_to_idx.get("yo")]
				
			generated_tokens = generate_step(model, context, device, word_to_idx, idx_to_word, temperature=0.7)
			msg_str = " ".join([idx_to_word[t] for t in generated_tokens])
			print(f"👧 Sofi (Bit-B) > {msg_str}")
			
			shared_history.append(('B', generated_tokens))
			current_speaker = 'A'

if __name__ == "__main__":
	# Importar numpy localmente por consistencia
	import numpy as np
	run_chat_simulation()
