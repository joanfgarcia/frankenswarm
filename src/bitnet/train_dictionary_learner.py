import os
import json
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.bitnet.modeling_bitnet import BitNet4LayerModel
from src.bitnet.dictionary_tool import SovereignDictionary

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
	return [word_to_idx.get(w, word_to_idx.get("<unk>", 0)) for w in words]

def generate_grammar_dataset(word_to_idx: dict, num_samples: int = 1000) -> list[list[int]]:
	subjects = ["perro", "gato", "hombre", "mujer", "niño", "niña", "persona", "agente", "jefe"]
	verbs = ["come", "bebe", "ve", "busca", "quiere", "sabe", "pensar", "tiene", "hace", "amar", "temer"]
	objects = ["comida", "agua", "árbol", "casa", "búnker", "peligro", "seguridad", "bosque", "río", "piedra", "código"]
	
	fixed_sentences = [
		"el fuego es caliente",
		"el agua es buena",
		"la noche es fría",
		"la noche es oscura",
		"el sol es grande",
		"el sol tiene luz",
		"la tormenta es peligro",
		"el búnker es seguro",
		"el agua limpia el cuerpo",
		"el amigo da seguridad",
		"el médico cura la herida",
		"el libro tiene palabras",
		"el agente escribe código"
	]
	
	dataset = []
	# Agregar oraciones fijas repetidas
	for _ in range(num_samples // 10):
		for s in fixed_sentences:
			dataset.append(tokenize(s, word_to_idx))
			
	# Generar oraciones aleatorias con estructura básica
	while len(dataset) < num_samples:
		subj = np.random.choice(subjects)
		verb = np.random.choice(verbs)
		obj = np.random.choice(objects)
		art = "la" if subj in ["mujer", "niña", "persona"] else "el"
		sentence = f"{art} {subj} {verb} {obj}"
		dataset.append(tokenize(sentence, word_to_idx))
		
	return dataset

def run_dictionary_training():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"═══ 📖 Entrenamiento de Aprendizaje por Diccionario (EXP_071) ═══")
	print(f"[Device]: {device}")
	
	# Cargar vocabulario y glifos base
	base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	expanded_glyphs_path = os.path.join(base_dir, "configs", "expanded_glyphs.json")
	
	with open(expanded_glyphs_path, "r", encoding="utf-8") as f:
		vocab_data = json.load(f)
		words = vocab_data["words"]
		glyphs = np.array(vocab_data["glyphs"], dtype=np.float32)
		theta = vocab_data["theta"]
		
	word_to_idx = {w: i for i, w in enumerate(words)}
	vocab_size = len(words)
	print(f"Vocabulario cargado: {vocab_size} palabras. Umbral calibrado (theta): {theta:.3f}")
	
	# Inicializar modelo BitNet con modo Glifos
	model = BitNet4LayerModel(
		use_glyphs=True,
		glyph_table=glyphs,
		hidden_dim=256,
		num_layers=4,
		use_pos_embedding=True
	).to(device)
	
	# ═══════════════════════════════════════════════════════════════════
	# FASE 1: Pre-entrenamiento Causal del Specialist Core
	# ═══════════════════════════════════════════════════════════════════
	print("\n--- Fase 1: Pre-entrenamiento de Gramática Causal ---")
	grammar_data = generate_grammar_dataset(word_to_idx, num_samples=1200)

	# Cargar el currículo de Samantha si existe
	samantha_curriculum_path = os.path.join(base_dir, "configs", "samantha_curriculum.json")
	if os.path.exists(samantha_curriculum_path):
		print(f"📖 [Currículo] Cargando frases generadas por Samantha desde {samantha_curriculum_path}...")
		with open(samantha_curriculum_path, "r", encoding="utf-8") as f:
			s_sentences = json.load(f)
		print(f"📖 [Currículo] Mezclando {len(s_sentences)} frases de Samantha con {len(grammar_data)} frases sintéticas.")
		for s in s_sentences:
			grammar_data.append(tokenize(s, word_to_idx))
	else:
		print("⚠️ [Currículo] No se encontró configs/samantha_curriculum.json. Usando solo frases sintéticas.")

	# Pad/Truncate a seq_len = 5
	seq_len = 5
	padded_data = []
	for seq in grammar_data:
		if len(seq) < seq_len:
			seq = seq + [word_to_idx["<pad>"]] * (seq_len - len(seq))
		else:
			seq = seq[:seq_len]
		padded_data.append(seq)
		
	x_train = torch.tensor(padded_data, dtype=torch.long, device=device)
	
	# Optimizer
	optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
	
	model.train()
	epochs_fase1 = 25  # Aumentado de 15 para asimilar la riqueza de Samantha
	batch_size = 32
	
	for epoch in range(epochs_fase1):
		epoch_loss = 0.0
		permutation = torch.randperm(x_train.size(0))
		
		for i in range(0, x_train.size(0), batch_size):
			indices = permutation[i:i+batch_size]
			batch_x = x_train[indices] # (batch, seq_len)
			
			optimizer.zero_grad()
			
			# Causal language modeling target
			inputs = batch_x[:, :-1]
			targets = batch_x[:, 1:]
			
			logits = model(inputs) # (batch, seq_len - 1, vocab_size)
			
			loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
			loss.backward()
			optimizer.step()
			
			epoch_loss += loss.item() * batch_x.size(0)
			
		epoch_loss /= x_train.size(0)
		if (epoch + 1) % 3 == 0 or epoch == 0:
			print(f"  Época {epoch+1:2d}/{epochs_fase1} | Causal Loss: {epoch_loss:.4f}")
			
	# ═══════════════════════════════════════════════════════════════════
	# FASE 2: Entrenamiento de glyph_projection_head con definiciones conocidas
	# ═══════════════════════════════════════════════════════════════════
	print("\n--- Fase 2: Entrenamiento de Proyección de Glifos ---")
	
	# Preparar las definiciones conocidas tokenizadas
	def_inputs = []
	def_targets = []
	
	for word_name, def_text in KNOWN_DEFINITIONS.items():
		if word_name in word_to_idx:
			tokens = tokenize(def_text, word_to_idx)
			# Pad a seq_len_def = 8
			seq_len_def = 8
			if len(tokens) < seq_len_def:
				tokens = tokens + [word_to_idx["<pad>"]] * (seq_len_def - len(tokens))
			else:
				tokens = tokens[:seq_len_def]
				
			def_inputs.append(tokens)
			def_targets.append(model.glyph_embedding.glyph_table[word_to_idx[word_name]].cpu().numpy())
			
	x_def = torch.tensor(def_inputs, dtype=torch.long, device=device)
	y_def = torch.tensor(np.array(def_targets), dtype=torch.float, device=device) # (N_def, 65)
	
	epochs_fase2 = 30
	# Optimizamos sólo la cabeza de proyección y el core del modelo para la tarea semántica
	proj_optimizer = torch.optim.AdamW(
		list(model.glyph_projection_head.parameters()) + list(model.core_layers.parameters()), 
		lr=1e-3
	)
	
	for epoch in range(epochs_fase2):
		proj_optimizer.zero_grad()
		
		# Obtener hidden state del core de la definición
		# Reutilizamos _embed_input y las capas del core para obtener el hidden state final de la definición
		h = model._embed_input(x_def)
		for layer in model.core_layers:
			h = layer(h)
		h = model.norm(h)
		
		# mean pooling sobre la dimensión de secuencia (dim=1)
		h_mean = h.mean(dim=1) # (N_def, hidden_dim)
		
		# Proyectar a primos semánticos
		pred_glyphs = model.glyph_projection_head(h_mean) # (N_def, 65)
		
		# MSE loss contra los glifos objetivo
		loss = F.mse_loss(pred_glyphs, y_def)
		loss.backward()
		proj_optimizer.step()
		
		if (epoch + 1) % 5 == 0 or epoch == 0:
			print(f"  Época {epoch+1:2d}/{epochs_fase2} | Proj Loss (MSE): {loss.item():.4f}")
			
	# Guardar checkpoints
	save_dir = os.path.join(base_dir, "storage", "checkpoints", "EXP_071")
	os.makedirs(save_dir, exist_ok=True)
	torch.save(model.state_dict(), os.path.join(save_dir, "model_final.pt"))
	print(f"\n[Checkpoint] Modelo guardado en: {save_dir}/model_final.pt")
	
	# ═══════════════════════════════════════════════════════════════════
	# FASE 3: Autonomía / Simulación de Aprendizaje Dinámico (Unfrozen)
	# ═══════════════════════════════════════════════════════════════════
	print("\n--- Fase 3: Evaluación de Aprendizaje Dinámico ---")
	dictionary = SovereignDictionary(expanded_glyphs_path)
	
	# Definimos una palabra nueva
	new_word = "computadora"
	print(f"Presentando palabra desconocida: '{new_word}'")
	
	# 1. Consultar el Diccionario Soberano (Samantha + fastembed cleaning)
	print("Consultando Diccionario Soberano...")
	clean_definition = dictionary.buscar(new_word)
	print(f"Definición obtenida: '{clean_definition}'")
	
	# 2. Tokenizar la definición
	def_tokens = tokenize(clean_definition, word_to_idx)
	# Pad a seq_len_def = 8
	seq_len_def = 8
	if len(def_tokens) < seq_len_def:
		def_tokens = def_tokens + [word_to_idx["<pad>"]] * (seq_len_def - len(def_tokens))
	else:
		def_tokens = def_tokens[:seq_len_def]
		
	# 3. Pasar por el modelo para extraer el glifo proyectado
	model.eval()
	with torch.no_grad():
		tokens_tensor = torch.tensor([def_tokens], dtype=torch.long, device=device)
		h = model._embed_input(tokens_tensor)
		for layer in model.core_layers:
			h = layer(h)
		h = model.norm(h)
		h_mean = h.mean(dim=1) # (1, hidden_dim)
		
		# Proyectar
		pred_continuous = model.glyph_projection_head(h_mean)[0].cpu().numpy() # (65,)
		
		# Ternarizar usando theta
		new_glyph = np.zeros(65, dtype=np.int8)
		new_glyph[pred_continuous > theta] = 1
		new_glyph[pred_continuous < -theta] = -1
		
		# 4. Registrar la nueva palabra en el modelo
		new_glyph_tensor = torch.tensor(new_glyph, dtype=torch.float, device=device)
		model.register_new_word(new_word, new_glyph_tensor)
		
		# Añadir al mapa de vocabulario local
		word_to_idx[new_word] = len(words)
		words.append(new_word)
		
	print(f"¡Palabra '{new_word}' registrada con éxito en el modelo!")
	print(f"Glifo generado (65 trits): {new_glyph.tolist()}")
	
	# 5. Medir la comprensión del modelo
	# Analizamos la similitud semántica del nuevo glifo con palabras conocidas
	comp_words = ["máquina", "código", "lógica", "libro", "sol"]
	print("\nSimilitud coseno del nuevo glifo de 'computadora' con palabras conocidas:")
	
	for cw in comp_words:
		if cw in word_to_idx:
			g_cw = model.glyph_embedding.glyph_table[word_to_idx[cw]].cpu().numpy()
			# Cosine sim
			norm_a = np.linalg.norm(new_glyph)
			norm_b = np.linalg.norm(g_cw)
			if norm_a > 0 and norm_b > 0:
				cos = np.dot(new_glyph, g_cw) / (norm_a * norm_b)
				print(f"  computadora ↔ {cw:<10} cos_sim = {cos:.4f}")

if __name__ == "__main__":
	run_dictionary_training()
