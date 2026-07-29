import json
import os
import re

import numpy as np
import torch
import torch.nn.functional as F

from src.bitnet.model.modeling_bitnet import BitNet4LayerModel


def tokenize(text: str, word_to_idx: dict) -> list[int]:
	words = re.findall(r'[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_<>\-]+', text.lower())
	return [word_to_idx.get(w, word_to_idx.get("<unk>", 0)) for w in words]

def generate_consolidation_sentences(new_word: str, definition: str, word_to_idx: dict) -> list[list[int]]:
	# Extraer palabras clave de la definición para componer oraciones de contexto
	def_words = re.findall(r'[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]+', definition.lower())
	templates = [
		f"{new_word} es una cosa",
		f"el agente tiene {new_word}",
		f"el jefe usa la {new_word}",
		f"la {new_word} es útil",
		f"el código está en la {new_word}"
	]
	# Agregar oraciones que unan la palabra nueva con su definición
	for w in def_words:
		if w in ["es", "un", "una", "que", "en", "el", "la", "los", "las", "por", "para"]:
			continue
		templates.append(f"{new_word} hace {w}")
		templates.append(f"el agente ve la {new_word}")
		
	return [tokenize(t, word_to_idx) for t in templates]

def run_sleep_cycle():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
	
	session_words_path = os.path.join(base_dir, "configs", "session_new_words.json")
	expanded_glyphs_path = os.path.join(base_dir, "configs", "expanded_glyphs.json")
	checkpoint_path = os.path.join(base_dir, "storage", "checkpoints", "EXP_071", "model_final.pt")
	
	if not os.path.exists(session_words_path):
		print("💤 [Sueño] No hay palabras nuevas en la sesión para consolidar.")
		return
		
	with open(session_words_path, encoding="utf-8") as f:
		session_data = json.load(f)
		
	if not session_data:
		print("💤 [Sueño] No hay palabras nuevas en la sesión para consolidar.")
		return
		
	print(f"\n💤 [Sueño] Iniciando ciclo de consolidación para {len(session_data)} palabras nuevas...")
	
	# Cargar vocabulario base actual
	with open(expanded_glyphs_path, encoding="utf-8") as f:
		vocab_data = json.load(f)
		words = vocab_data["words"]
		glyphs = list(vocab_data["glyphs"])
		theta = vocab_data["theta"]
		
	# 1. Integrar permanentemente en configs/expanded_glyphs.json
	new_vocab_added = False
	for item in session_data:
		w = item["word"]
		g = item["glyph"]
		if w not in words:
			words.append(w)
			glyphs.append(g)
			new_vocab_added = True
			print(f"  [Sueño] Consolidando '{w}' en el vocabulario base.")
			
	if new_vocab_added:
		with open(expanded_glyphs_path, "w", encoding="utf-8") as f:
			json.dump({
				"words": words,
				"glyphs": glyphs,
				"theta": theta
			}, f, indent=4, ensure_ascii=False)
			
	# Re-crear el mapa de vocabulario
	word_to_idx = {w: i for i, w in enumerate(words)}
	vocab_size = len(words)
	
	# 2. Inicializar modelo con la tabla expandida
	model = BitNet4LayerModel(
		use_glyphs=True,
		glyph_table=np.array(glyphs, dtype=np.float32),
		hidden_dim=256,
		num_layers=4,
		use_pos_embedding=True
	).to(device)
	
	# Cargar pesos omitiendo el buffer glyph_table (ya cargado en constructor)
	state_dict = torch.load(checkpoint_path, map_location=device)
	if "glyph_embedding.glyph_table" in state_dict:
		del state_dict["glyph_embedding.glyph_table"]
		
	model.load_state_dict(state_dict, strict=False)
	
	# 3. Generar oraciones de consolidación
	consolidation_data = []
	for item in session_data:
		w = item["word"]
		d = item["definition"]
		sentences = generate_consolidation_sentences(w, d, word_to_idx)
		consolidation_data.extend(sentences)
		
	# Pad/Truncate a seq_len = 5
	seq_len = 5
	padded_data = []
	for seq in consolidation_data:
		seq = seq + [word_to_idx["<pad>"]] * (seq_len - len(seq)) if len(seq) < seq_len else seq[:seq_len]
		padded_data.append(seq)
		
	# Ajustar tamaño de lote mínimo
	while len(padded_data) < 64:
		padded_data.extend(padded_data)
	padded_data = padded_data[:128]
	
	x_train = torch.tensor(padded_data, dtype=torch.long, device=device)
	
	# 4. Ajustar pesos (fine-tuning fino con lr bajo)
	optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
	model.train()
	
	epochs = 10
	batch_size = 16
	print("  [Sueño] Ajustando pesos neuronales para asimilar los nuevos conceptos...")
	for _epoch in range(epochs):
		epoch_loss = 0.0
		permutation = torch.randperm(x_train.size(0))
		for i in range(0, x_train.size(0), batch_size):
			indices = permutation[i:i+batch_size]
			batch_x = x_train[indices]
			
			optimizer.zero_grad()
			inputs = batch_x[:, :-1]
			targets = batch_x[:, 1:]
			logits = model(inputs)
			loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
			loss.backward()
			optimizer.step()
			epoch_loss += loss.item() * batch_x.size(0)
		epoch_loss /= x_train.size(0)
		
	# 5. Guardar el nuevo checkpoint consolidado
	torch.save(model.state_dict(), checkpoint_path)
	
	# 6. Eliminar el archivo temporal de la sesión
	os.remove(session_words_path)
	print("💤 [Sueño] ¡Consolidación de memoria completada! El modelo ha dormido y asimilado las nuevas palabras.")

def consolidate_latent_resonance(student_model, teaching_buffer, device, lr=1e-4, epochs=100, n_think=2):
	"""
	Realiza la destilación de resonancia latente durante el ciclo de sueño del alumno.
	Minimiza la distancia coseno entre el estado oculto del alumno y el del maestro.
	"""
	if not teaching_buffer:
		return []

	# Guardar el estado original de requires_grad para poder restaurarlo después
	orig_grad_states = {}
	for name, param in student_model.named_parameters():
		orig_grad_states[name] = param.requires_grad
		if "glyph_table" not in name:
			param.requires_grad = True

	# Agrupar muestras por habilidad
	skills_in_buffer = {sample["skill"] for sample in teaching_buffer}
	graduated_skills = []

	# Entrenar para cada habilidad en el buffer
	for skill in skills_in_buffer:
		skill_samples = [s for s in teaching_buffer if s["skill"] == skill]
		print(f"  💤 [Sueño] Destilando resonancia latente para habilidad '{skill}' ({len(skill_samples)} muestras)...")

		# Preparar tensores
		inputs = torch.cat([s["input"] for s in skill_samples], dim=0).to(device) # (N, 6)
		emotions = torch.tensor([s["emotion"] for s in skill_samples], dtype=torch.long, device=device) # (N,)
		targets = torch.cat([s["teacher_hidden"] for s in skill_samples], dim=0).to(device) # (N, hidden_dim)

		# Optimizador dedicado al sueño de destilación
		optimizer = torch.optim.AdamW(student_model.parameters(), lr=lr, weight_decay=0.01)
		student_model.train()

		final_loss = 1.0
		for epoch in range(epochs):
			optimizer.zero_grad()
			_, meta = student_model.forward_resonance(
				inputs, n_steps=n_think, pos_mode="clock", emotion_ids=emotions
			)
			student_hidden = meta["hidden"][:, 2, :] # shape (N, hidden_dim)

			# Similitud de coseno loss: 1.0 - cos_sim
			cos_sim = F.cosine_similarity(student_hidden, targets, dim=-1)
			loss = (1.0 - cos_sim).mean()

			loss.backward()
			optimizer.step()
			final_loss = loss.item()
			if final_loss < 0.02:
				break

		print(f"    [Sueño] Pérdida de resonancia latente final para '{skill}': {final_loss:.4f} (época {epoch+1}/{epochs})")
		if final_loss < 0.02:
			print(f"    🏆 [Sueño] ¡Habilidad '{skill}' graduada exitosamente!")
			graduated_skills.append(skill)
		else:
			print(f"    ⏳ [Sueño] Habilidad '{skill}' aún no dominada (pérdida >= 0.02).")

	# Restaurar el estado original de requires_grad para el PPO
	for name, param in student_model.named_parameters():
		param.requires_grad = orig_grad_states[name]

	return graduated_skills

if __name__ == "__main__":
	run_sleep_cycle()
