import os
import json
import re
import torch
import numpy as np
from src.bitnet.modeling_bitnet import BitNet4LayerModel
from src.bitnet.dictionary_tool import SovereignDictionary

def tokenize(text: str, word_to_idx: dict) -> list[int]:
	words = re.findall(r'[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_<>\-]+', text.lower())
	return [word_to_idx.get(w, word_to_idx.get("<unk>", 0)) for w in words]

def run_chat():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	expanded_glyphs_path = os.path.join(base_dir, "configs", "expanded_glyphs.json")
	checkpoint_path = os.path.join(base_dir, "storage", "checkpoints", "EXP_071", "model_final.pt")

	if not os.path.exists(checkpoint_path):
		print(f"❌ No se encontró el checkpoint en {checkpoint_path}. Por favor ejecuta primero train_dictionary_learner.py")
		return

	# Cargar vocabulario base
	with open(expanded_glyphs_path, "r", encoding="utf-8") as f:
		vocab_data = json.load(f)
		words = vocab_data["words"]
		glyphs = np.array(vocab_data["glyphs"], dtype=np.float32)
		theta = vocab_data["theta"]

	word_to_idx = {w: i for i, w in enumerate(words)}
	idx_to_word = {i: w for w, i in word_to_idx.items()}

	# Inicializar modelo y cargar pesos
	print("🤖 Cargando modelo BitNet...")
	model = BitNet4LayerModel(
		use_glyphs=True,
		glyph_table=glyphs,
		hidden_dim=256,
		num_layers=4,
		use_pos_embedding=True
	).to(device)
	model.load_state_dict(torch.load(checkpoint_path, map_location=device))
	model.eval()

	# Inicializar el Diccionario Soberano
	print("📖 Inicializando Diccionario Soberano (Samantha)...")
	dictionary = SovereignDictionary(expanded_glyphs_path)

	learned_words_history = []

	print("\n" + "═"*60)
	print("💬 CHAT INTERACTIVO CON BIT (Ternary GLYPH-LM)")
	print("   Bit conoce 1,008 palabras base en español.")
	print("   Si usas una palabra desconocida, Bit consultará a Samantha,")
	print("   generará su glifo de 65 trits y la registrará al vuelo.")
	print("   Escribe 'salir' para finalizar.")
	print("═"*60 + "\n")

	while True:
		try:
			user_input = input("Tú > ").strip()
			if not user_input:
				continue
			if user_input.lower() in ["salir", "exit", "quit"]:
				break

			# Encontrar palabras desconocidas en la entrada del usuario
			raw_words = re.findall(r'[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]+', user_input.lower())
			unknown_words = [w for w in raw_words if w not in word_to_idx]

			# Proceso de aprendizaje dinámico (unfrozen)
			for w in unknown_words:
				print(f"\n🔍 [Bit] ¿Qué significa '{w}'? Consultando diccionario...")
				definition = dictionary.buscar(w)
				print(f"📖 [Diccionario] '{w}' es: '{definition}'")
				
				# Tokenizar la definición
				def_tokens = tokenize(definition, word_to_idx)
				# Pad a seq_len_def = 8
				seq_len_def = 8
				if len(def_tokens) < seq_len_def:
					def_tokens = def_tokens + [word_to_idx["<pad>"]] * (seq_len_def - len(def_tokens))
				else:
					def_tokens = def_tokens[:seq_len_def]

				# Procesar definición para generar glifo
				with torch.no_grad():
					tokens_tensor = torch.tensor([def_tokens], dtype=torch.long, device=device)
					h = model._embed_input(tokens_tensor)
					for layer in model.core_layers:
						h = layer(h)
					h = model.norm(h)
					h_mean = h.mean(dim=1)
					
					pred_continuous = model.glyph_projection_head(h_mean)[0].cpu().numpy()
					
					new_glyph = np.zeros(65, dtype=np.int8)
					new_glyph[pred_continuous > theta] = 1
					new_glyph[pred_continuous < -theta] = -1
					
					new_glyph_tensor = torch.tensor(new_glyph, dtype=torch.float, device=device)
					model.register_new_word(w, new_glyph_tensor)

				# Registrar localmente
				new_idx = len(words)
				word_to_idx[w] = new_idx
				idx_to_word[new_idx] = w
				words.append(w)
				
				# Añadir al historial de la sesión
				learned_words_history.append({
					"word": w,
					"definition": definition,
					"glyph": new_glyph.tolist()
				})
				print(f"✨ [Bit] ¡Entendido! He aprendido el glifo de '{w}' y lo he registrado.\n")

			# Tokenizar la entrada completa (ahora todas las palabras están registradas)
			input_tokens = tokenize(user_input, word_to_idx)
			
			# Autoregressive generation with temperature sampling and repetition penalty
			response_tokens = []
			current_input = torch.tensor([input_tokens], dtype=torch.long, device=device)
			
			temperature = 0.7
			penalty_val = 15.0

			with torch.no_grad():
				for _ in range(5): # Generar hasta 5 tokens
					logits = model(current_input) # (1, seq_len, vocab_size)
					next_token_logits = logits[0, -1, :].clone()
					
					# Filtrar tokens especiales (<pad>, <unk>) de la salida
					next_token_logits[word_to_idx.get("<pad>", 0)] = -1e9
					next_token_logits[word_to_idx.get("<unk>", 0)] = -1e9
					
					# Aplicar penalización por repetición para evitar bucles infinitos
					for prev_tok in set(response_tokens):
						next_token_logits[prev_tok] -= penalty_val
					
					# Muestreo estocástico con temperatura
					probs = torch.softmax(next_token_logits / temperature, dim=-1)
					next_token = torch.multinomial(probs, num_samples=1).item()
					response_tokens.append(next_token)
					
					# Concatenar para el siguiente paso
					next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
					current_input = torch.cat([current_input, next_token_tensor], dim=1)
					
					if next_token == word_to_idx.get("<pad>", 0):
						break

			response_words = [idx_to_word.get(t, "?") for t in response_tokens if t in idx_to_word]
			response_text = " ".join([w for w in response_words if w not in ["<pad>", "<unk>"]])
			print(f"Bit > {response_text}\n")

		except KeyboardInterrupt:
			break
		except Exception as e:
			print(f"⚠️ Error: {e}")

	# Manejo del ciclo de sueño al salir si hubo palabras aprendidas
	if learned_words_history:
		print("\n💾 Guardando palabras aprendidas en la sesión...")
		session_words_path = os.path.join(base_dir, "configs", "session_new_words.json")
		with open(session_words_path, "w", encoding="utf-8") as f:
			json.dump(learned_words_history, f, indent=4, ensure_ascii=False)
		
		run_sleep = input("💤 ¿Deseas activar el ciclo de sueño/consolidación para estas palabras nuevas? (s/n): ").strip().lower()
		if run_sleep in ["s", "si", "y", "yes"]:
			import subprocess
			print("💤 [Sueño] Iniciando consolidación neuronal...")
			subprocess.run([os.path.join(base_dir, ".venv", "bin", "python"), os.path.join(base_dir, "src", "bitnet", "consolidate_sleep.py")])

if __name__ == "__main__":
	run_chat()
