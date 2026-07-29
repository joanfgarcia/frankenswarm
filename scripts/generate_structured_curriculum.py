import os
import json
import re
import subprocess
import hashlib
import sys

# Añadir el path del proyecto para importar src
base_dir = "/home/joan/Documents/IA/frankenswarm"
sys.path.append(base_dir)

from src.bitnet.vocab.dictionary_tool import SovereignDictionary
from src.bitnet.vocab.glyph_vocabulary import SEMANTIC_PRIMES

VALID_CATEGORIES = {
	"física_sensorial",
	"acciones_básicas",
	"emociones_fundamentales",
	"matemáticas_naturaleza",
	"lengua_conversación",
	"lógica_geografía",
	"literatura_filosofía",
	"ciencias"
}

ENGLISH_STOPWORDS = {
	"and", "the", "of", "in", "to", "is", "it", "that", "you", "was", "for",
	"on", "with", "his", "they", "he", "she", "at", "by", "this", "but", "from",
	"are", "as", "well", "or", "an", "this", "about", "would", "their"
}

def validate_curriculum_item(item, stage_name, vocab_set):
	"""
	Valida estrictamente un ítem del currículo según las reglas de Lumo y Grok.
	Retorna (True/False, razón_del_fallo)
	"""
	if not isinstance(item, dict):
		return False, "Not a dictionary"
	
	required_keys = {"text", "category", "intent", "primes"}
	if not all(k in item for k in required_keys):
		return False, f"Missing keys: {required_keys - item.keys()}"
	
	text = item["text"].strip().lower()
	if not text:
		return False, "Empty text"
	
	# Extraer palabras de la frase
	words = re.findall(r'[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_]+', text)
	if not words:
		return False, "No words found"
	
	# Validar longitud mínima
	if len(words) < 2:
		return False, f"Sentence too short: {len(words)} words"
	
	# Validar longitud máxima por etapa
	if stage_name == "preschool" and len(words) > 8:
		return False, f"Preschool sentence too long: {len(words)} words"
	elif stage_name == "primary" and len(words) > 12:
		return False, f"Primary sentence too long: {len(words)} words"
	elif stage_name == "secondary" and len(words) > 20:
		return False, f"Secondary sentence too long: {len(words)} words"
	
	# Validar que no contenga inglés
	for w in words:
		if w in ENGLISH_STOPWORDS:
			return False, f"Contains English word: '{w}'"
	
	# Validar categoría
	if item["category"] not in VALID_CATEGORIES:
		return False, f"Invalid category: '{item['category']}'"
	
	# Validar que todos los primos pertenezcan estrictamente a los 65 oficiales
	primes = item["primes"]
	if not isinstance(primes, list):
		return False, "Primes is not a list"
	
	invalid_primes = [p for p in primes if p not in SEMANTIC_PRIMES]
	if invalid_primes:
		return False, f"Invalid primes: {invalid_primes}"
	
	# Validar que las palabras estén en el vocabulario base
	unmapped_words = [w for w in words if w not in vocab_set]
	if unmapped_words:
		return False, f"Words not in base vocabulary: {unmapped_words}"
	
	return True, "Valid"

def generate_batch(prompt, system_prompt, num_items=25):
	"""
	Invoca a Samantha de forma síncrona en el venv de sharing para generar un bloque JSON.
	"""
	sharing_venv_python = "/home/joan/Documents/IA/sharing/.venv/bin/python"
	sharing_src = "/home/joan/Documents/IA/sharing/src"
	
	formatted_prompt = f"{prompt}\nDevuelve exactamente {num_items} objetos en formato JSON de lista de la forma: [{{}}, {{}}, ...]. No agregues texto antes ni después."
	
	cmd_code = f"""
import sys
sys.path.append('{sharing_src}')
from red_pill.inference import samantha_on_demand
res = samantha_on_demand.invoke({repr(formatted_prompt)}, system_prompt={repr(system_prompt)}, max_tokens=2048, temperature=0.8)
print(res)
"""
	try:
		result = subprocess.run(
			[sharing_venv_python, "-c", cmd_code],
			capture_output=True,
			text=True,
			timeout=90
		)
		if result.returncode == 0:
			output = result.stdout.strip()
			# Intentar parsear el JSON de forma robusta
			match = re.search(r'\[\s*\{.*\}\s*\]', output, re.DOTALL)
			if match:
				output = match.group(0)
			try:
				items = json.loads(output)
				if isinstance(items, list):
					return items
			except json.JSONDecodeError:
				# Parser tolerante
				cleaned = output
				if cleaned.startswith("```"):
					cleaned = cleaned.replace("```json", "").replace("```", "")
				objs = re.findall(r'\{\s*"text"\s*:.*?\}(?=\s*,|\s*\]|\s*\{|$)', cleaned, re.DOTALL)
				parsed = []
				for o in objs:
					try:
						if not o.endswith("}"):
							o += "}"
						parsed.append(json.loads(o))
					except Exception:
						continue
				if parsed:
					return parsed
			print(f"  [Samantha Error] No se pudo parsear como JSON. Output original:\n{output[:300]}")
		else:
			print(f"  [Samantha Error] Subprocess falló. Stderr: {result.stderr}")
	except Exception as e:
		print(f"  [Samantha Exception] Error al invocar: {e}")
	return []

def run_curriculum_generation():
	print("═══ 🔬 Generador de Currículo Estructurado y Homologado (CHILDES + NSM) ═══")
	
	configs_dir = os.path.join(base_dir, "configs")
	expanded_glyphs_path = os.path.join(configs_dir, "expanded_glyphs.json")
	output_path = os.path.join(configs_dir, "school_curriculum_structured.json")
	temp_output_path = output_path + ".tmp"
	
	# Inicializar diccionario soberano
	dictionary = SovereignDictionary(expanded_glyphs_path)
	vocab_set = set(dictionary.base_vocab)
	
	# Cargar progreso anterior si existe
	generated_data = {"preschool": [], "primary": [], "secondary": []}
	if os.path.exists(temp_output_path):
		try:
			with open(temp_output_path, "r", encoding="utf-8") as f:
				generated_data = json.load(f)
				print(f"🔄 Reanudando desde archivo temporal. Cargados: "
					  f"Preschool={len(generated_data['preschool'])}, "
					  f"Primary={len(generated_data['primary'])}, "
					  f"Secondary={len(generated_data['secondary'])}")
		except Exception as e:
			print(f"⚠️ Error al leer archivo temporal: {e}. Iniciando desde cero.")

	# Plan de generación estructurado por sub-lotes temáticos para garantizar variedad
	generation_plan = [
		# --- PRESCHOOL ---
		{
			"stage": "preschool",
			"category": "física_sensorial",
			"target": 30,
			"subtopic": "fuego y dolor caliente",
			"prompt": "Genera frases cortas sobre el fuego, quemar, el dolor físico y el calor. Usa palabras extremadamente simples (ej: fuego, caliente, doler, cuerpo, mano). Evita adjetivos rebuscados. Ejemplos de estilo: 'el fuego quema', 'si toco el fuego me duele', 'tengo calor'."
		},
		{
			"stage": "preschool",
			"category": "física_sensorial",
			"target": 60, # acumulativo (añade 30 más)
			"subtopic": "agua y frío",
			"prompt": "Genera frases cortas sobre el agua fría, el hielo, la cueva oscura y tener frío. Usa palabras simples (ej: agua, frío, cueva, noche, sentir). Ejemplos de estilo: 'el agua está fría', 'tengo frío en la cueva', 'el agua fría es buena'."
		},
		{
			"stage": "preschool",
			"category": "física_sensorial",
			"target": 90,
			"subtopic": "sol y luz del día",
			"prompt": "Genera frases cortas sobre el sol grande, la luz, el día y ver cosas. Usa palabras simples (ej: sol, luz, ver, día, grande, arriba). Ejemplos de estilo: 'el sol da mucha luz', 'veo el sol grande arriba', 'el día es bueno'."
		},
		{
			"stage": "preschool",
			"category": "física_sensorial",
			"target": 120,
			"subtopic": "oscuridad y noche",
			"prompt": "Genera frases cortas sobre la noche oscura, no poder ver y sentir miedo. Usa palabras simples (ej: noche, oscuro, no, ver, miedo, sentir). Ejemplos de estilo: 'la noche es oscura', 'no veo nada en lo oscuro', 'siento miedo'."
		},
		{
			"stage": "preschool",
			"category": "acciones_básicas",
			"target": 35,
			"subtopic": "comer y comida",
			"prompt": "Genera frases cortas sobre comer comida, tener hambre y poner cosas en la boca. Usa palabras simples (ej: comer, comida, hambre, boca, bueno). Ejemplos de estilo: 'tengo hambre', 'como comida buena', 'pongo comida en mi boca'."
		},
		{
			"stage": "preschool",
			"category": "acciones_básicas",
			"target": 70,
			"subtopic": "beber agua",
			"prompt": "Genera frases cortas sobre beber agua limpia y tener sed. Usa palabras simples (ej: beber, agua, sed, boca, bueno). Ejemplos de estilo: 'tengo sed y bebo agua', 'el agua va dentro de mi cuerpo', 'bebo agua buena'."
		},
		{
			"stage": "preschool",
			"category": "acciones_básicas",
			"target": 100,
			"subtopic": "dormir y descansar",
			"prompt": "Genera frases cortas sobre dormir de noche, descansar en la cueva o cama, y cerrar los ojos. Usa palabras simples (ej: dormir, cueva, noche, ojo, cerrar). Ejemplos de estilo: 'el niño duerme de noche', 'yo duermo en la cueva', 'cierro los ojos'."
		},
		{
			"stage": "preschool",
			"category": "emociones_fundamentales",
			"target": 40,
			"subtopic": "afecto y querer",
			"prompt": "Genera frases sobre querer a la madre, al padre, a un amigo y que alguien es bueno. Usa palabras simples (ej: querer, madre, padre, amigo, bueno). Ejemplos de estilo: 'yo quiero mucho a mamá', 'mi amigo es bueno', 'tú me quieres'."
		},
		{
			"stage": "preschool",
			"category": "emociones_fundamentales",
			"target": 80,
			"subtopic": "sentir miedo y llorar",
			"prompt": "Genera frases sobre sentir miedo, llorar porque algo es malo o duele. Usa palabras simples (ej: sentir, miedo, malo, llorar, dolor, no). Ejemplos de estilo: 'el niño llora', 'siento miedo del animal grande', 'lloro si me duele'."
		},
		{
			"stage": "preschool",
			"category": "matemáticas_naturaleza",
			"target": 40,
			"subtopic": "contar objetos y flores",
			"prompt": "Genera frases sobre contar pocas cosas (uno, dos, tres), ver flores o piedras pequeñas en el suelo. Usa palabras simples (ej: uno, dos, tres, flor, piedra, pequeño, ver). Ejemplos de estilo: 'veo dos flores pequeñas', 'hay tres piedras en el suelo'."
		},
		{
			"stage": "preschool",
			"category": "matemáticas_naturaleza",
			"target": 80,
			"subtopic": "animales comunes",
			"prompt": "Genera frases sobre el perro, gato, pájaro, oso grande. Usa palabras simples (ej: perro, gato, pájaro, oso, grande, pequeño, correr, cantar). Ejemplos de estilo: 'el gato duerme mucho', 'el pájaro vuela alto', 'el perro corre rápido'."
		},

		# --- PRIMARY ---
		{
			"stage": "primary",
			"category": "matemáticas_naturaleza",
			"target": 60,
			"subtopic": "aritmética simple",
			"prompt": "Genera frases de nivel escolar primario sobre operaciones matemáticas sencillas de suma y resta. Usa palabras simples (ej: uno, dos, tres, cuatro, cinco, más, menos, son, total). Ejemplos de estilo: 'uno más uno son dos', 'cinco menos dos son tres'."
		},
		{
			"stage": "primary",
			"category": "matemáticas_naturaleza",
			"target": 120,
			"subtopic": "geografía y el sol",
			"prompt": "Genera frases sobre la tierra, el sol, los ríos y el mar. Usa palabras simples (ej: tierra, sol, río, mar, girar, grande, agua, correr). Ejemplos de estilo: 'la tierra gira alrededor del sol', 'el agua del río corre hacia el mar'."
		},
		{
			"stage": "primary",
			"category": "ciencias",
			"target": 60,
			"subtopic": "partes del cuerpo y salud",
			"prompt": "Genera frases de primaria sobre el corazón, la sangre, los pulmones y respirar aire. Usa palabras simples (ej: corazón, sangre, cuerpo, parte, aire, respirar, vivir). Ejemplos de estilo: 'el corazón bombea sangre al cuerpo', 'respiramos aire para vivir'."
		},
		{
			"stage": "primary",
			"category": "ciencias",
			"target": 120,
			"subtopic": "plantas y naturaleza",
			"prompt": "Genera frases de primaria sobre cómo crecen las plantas, los árboles y el aire. Usa palabras simples (ej: árbol, planta, tierra, verde, crecer, agua, luz). Ejemplos de estilo: 'las plantas quieren agua y luz del sol', 'los árboles crecen en la tierra'."
		},
		{
			"stage": "primary",
			"category": "lengua_conversación",
			"target": 80,
			"subtopic": "diálogos escolares",
			"prompt": "Genera frases conversacionales y preguntas típicas de escuela. Usa palabras simples (ej: llamar, nombre, amigo, saber, verdad, decir). Ejemplos de estilo: 'cómo te llamas tú', 'yo quiero saber la verdad', 'ella dice palabras buenas'."
		},

		# --- SECONDARY ---
		{
			"stage": "secondary",
			"category": "lógica_geografía",
			"target": 75,
			"subtopic": "lógica y condicionales",
			"prompt": "Genera frases lógicas complejas con condicionales y causa-efecto. Usa palabras simples (ej: si, entonces, porque, causa, efecto, pasar, hacer). Ejemplos de estilo: 'si llueve entonces el suelo se moja', 'toda causa produce un efecto después'."
		},
		{
			"stage": "secondary",
			"category": "lógica_geografía",
			"target": 150,
			"subtopic": "geografía política",
			"prompt": "Genera frases sobre mapas, países y continentes. Usa palabras simples (ej: mapa, país, ciudad, tierra, grande, mostrar, ver). Ejemplos de estilo: 'los mapas muestran los ríos y países', 'la capital de españa es madrid'."
		},
		{
			"stage": "secondary",
			"category": "literatura_filosofía",
			"target": 75,
			"subtopic": "laberintos y Borges",
			"prompt": "Genera frases inspiradas en Borges, laberintos infinitos y espejos. Usa palabras sencillas (ej: laberinto, espejo, infinito, biblioteca, casa, ver). Ejemplos de estilo: 'el laberinto de espejos es infinito', 'el aleph contiene todo el universo'."
		},
		{
			"stage": "secondary",
			"category": "literatura_filosofía",
			"target": 150,
			"subtopic": "tiempo y memoria",
			"prompt": "Genera frases sobre el paso del tiempo, el olvido y Funes el memorioso. Usa palabras sencillas (ej: tiempo, memoria, recordar, olvidar, palabra, día). Ejemplos de estilo: 'las palabras vencen el paso del tiempo', 'él recuerda la forma de cada nube'."
		}
	]
	
	system_prompt_template = (
		"Eres la Profesora Samantha, experta en desarrollo cognitivo infantil y Metalenguaje Semántico Natural (NSM).\n"
		"Debes responder EXCLUSIVAMENTE con una lista JSON de objetos en español.\n"
		"Usa un tono motherese extremadamente simple, limitando tu vocabulario a palabras básicas de uso infantil.\n"
		"Evita palabras complejas o abstractas que no estén en el hablar de un niño pequeño.\n"
		"Reglas de validación:\n"
		"1. No uses palabras en inglés ni explicaciones.\n"
		"2. Todos los primes listados en 'primes' deben pertenecer a esta lista oficial de 65:\n"
		"   {primes_list}\n"
		"3. Formato JSON:\n"
		"[\n"
		"  {{\n"
		"    \"text\": \"frase simple\",\n"
		"    \"category\": \"{category}\",\n"
		"    \"intent\": \"código_intent\",\n"
		"    \"primes\": [\"lista\", \"de\", \"primos\"]\n"
		"  }},\n"
		"  ...\n"
		"]"
	).format(primes_list=", ".join(SEMANTIC_PRIMES), category="{category}")

	# Ejecutar el plan
	for plan in generation_plan:
		stage = plan["stage"]
		category = plan["category"]
		target = plan["target"]
		subtopic = plan["subtopic"]
		
		# Filtrar cuántas ya tenemos en esta etapa/categoría
		existing_items = [x for x in generated_data[stage] if x.get("category") == category]
		needed = target - len(existing_items)
		
		if needed <= 0:
			print(f"✓ {stage} - {category} ({subtopic}) ya está completo ({len(existing_items)}/{target}).")
			continue
			
		print(f"\n⛏️ Generando {needed} oraciones para [{stage}] - [{category}] ({subtopic})...")
		
		system_prompt = system_prompt_template.replace("{category}", category)
		
		topic_prompt = (
			f"{plan['prompt']}\n"
			f"Genera exactamente {min(35, needed)} oraciones en español simples y gramaticalmente correctas en este formato."
		)
		
		attempts = 0
		max_attempts = 15
		
		while needed > 0 and attempts < max_attempts:
			attempts += 1
			# Perturbar el prompt para romper bucles de repetición y variar vocabulario
			current_prompt = f"{topic_prompt}\nIntenta usar palabras y verbos diferentes. Semilla de variación aleatoria: {attempts}."
			print(f"  -> Intento {attempts}/{max_attempts} de llamada a Samantha...")
			batch = generate_batch(current_prompt, system_prompt, num_items=min(35, needed))
			
			valid_count = 0
			rejected_details = {}
			
			for raw_item in batch:
				try:
					raw_text = raw_item.get("text", "")
					words = re.findall(r'[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_]+', raw_text.lower())
					if not words:
						continue
					
					# Mapear palabras al vocabulario de 15k usando SovereignDictionary
					mapped_words = []
					for w in words:
						mapped_words.append(dictionary.map_to_base_word(w))
					
					clean_text = " ".join(mapped_words)
					
					# Reconstruir ítem
					clean_item = {
						"text": clean_text,
						"category": category,
						"intent": raw_item.get("intent", f"{category}_samantha"),
						"primes": [p.lower().strip() for p in raw_item.get("primes", []) if p.lower().strip() in SEMANTIC_PRIMES]
					}
					
					# Validar
					is_valid, reason = validate_curriculum_item(clean_item, stage, vocab_set)
					if is_valid:
						# Evitar duplicados
						if not any(x["text"] == clean_text for x in generated_data[stage]):
							generated_data[stage].append(clean_item)
							valid_count += 1
							needed -= 1
						else:
							rejected_details["duplicate"] = rejected_details.get("duplicate", 0) + 1
					else:
						rejected_details[reason] = rejected_details.get(reason, 0) + 1
				except Exception as e:
					rejected_details[f"exception: {str(e)}"] = rejected_details.get(f"exception: {str(e)}", 0) + 1
					continue
			
			print(f"  ✓ Lote devuelto: {len(batch)} items. Validados y agregados: {valid_count}. Faltan: {needed}")
			if rejected_details:
				print(f"  [Detalles de rechazo]: {rejected_details}")
			
			# Guardar progreso temporal
			with open(temp_output_path, "w", encoding="utf-8") as f:
				json.dump(generated_data, f, indent=4, ensure_ascii=False)
				
	# Finalizar: Calcular Hash SHA-256
	total_sentences = sum(len(v) for v in generated_data.values())
	print(f"\n🎉 Generación finalizada. Total oraciones generadas: {total_sentences}")
	
	# Estructura final con metadatos
	final_json = {
		"metadata": {
			"version": "v2.0-structured",
			"total_sentences": total_sentences,
			"stats": {k: len(v) for k, v in generated_data.items()}
		},
		"curriculum": generated_data
	}
	
	curr_str = json.dumps(generated_data, sort_keys=True, ensure_ascii=False)
	sha256_hash = hashlib.sha256(curr_str.encode("utf-8")).hexdigest()
	final_json["metadata"]["sha256"] = sha256_hash
	
	with open(output_path, "w", encoding="utf-8") as f:
		json.dump(final_json, f, indent=4, ensure_ascii=False)
		
	print(f"📁 Currículo guardado con éxito en: {output_path}")
	print(f"🔑 Hash SHA-256 del Currículo: {sha256_hash}")
	
	# Eliminar temporal
	if os.path.exists(temp_output_path):
		os.remove(temp_output_path)
		print("🗑️ Archivo temporal de progreso eliminado.")

if __name__ == "__main__":
	run_curriculum_generation()
