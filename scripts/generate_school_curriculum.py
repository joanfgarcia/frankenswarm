import json
import os
import re
import subprocess

from src.bitnet.vocab.dictionary_tool import SovereignDictionary


def run_generation():
	print("═══ 🎒 Generador de Currículo Escolar Soberano (Samantha) ═══")
	
	base_dir = "/home/joan/Documents/IA/frankenswarm"
	expanded_glyphs_path = os.path.join(base_dir, "configs", "expanded_glyphs.json")
	
	dictionary = SovereignDictionary(expanded_glyphs_path)
	
	sharing_venv_python = "/home/joan/Documents/IA/sharing/.venv/bin/python"
	sharing_src = "/home/joan/Documents/IA/sharing/src"
	
	# Estructura del Currículo por Grados Académicos
	curriculum_structure = {
		"preschool": [
			{
				"topic": "cuentos infantiles cortos y sencillos",
				"prompt": "Genera 100 oraciones sencillas y cortas de cuentos infantiles en español (de 3 a 6 palabras). Usa un lenguaje muy infantil. Ejemplos: 'el gato duerme feliz', 'el perro corre mucho', 'yo veo la luna', 'el oso come miel'."
			},
			{
				"topic": "matemáticas básicas y contar objetos",
				"prompt": "Genera 80 oraciones cortas en español sobre contar cosas y aritmética de preescolar (de 3 a 6 palabras). Ejemplos: 'tengo dos manzanas rojas', 'un gato más uno son dos', 'cuento uno dos tres'."
			},
			{
				"topic": "animales, plantas y naturaleza",
				"prompt": "Genera 80 oraciones simples en español sobre animales y plantas (de 3 a 6 palabras). Ejemplos: 'el pájaro vuela alto', 'el agua limpia las flores', 'el león tiene pelo grande'."
			}
		],
		"primary": [
			{
				"topic": "ciencias naturales y cuerpo humano",
				"prompt": "Genera 100 oraciones escolares de primaria sobre ciencias y biología básica (de 5 a 8 palabras). Ejemplos: 'el corazón bombea sangre al cuerpo', 'los árboles dan oxígeno y sombra', 'el agua del río corre hacia el mar'."
			},
			{
				"topic": "aritmética y preguntas de matemáticas",
				"prompt": "Genera 100 diálogos cortos de preguntas y respuestas matemáticas simples (suma y resta). Ejemplos: '¿cuánto es cinco más tres? es ocho', '¿cuánto es diez menos dos? es ocho'."
			},
			{
				"topic": "geografía y nociones de la tierra",
				"prompt": "Genera 80 oraciones de primaria sobre mapas, países y la tierra (de 5 a 8 palabras). Ejemplos: 'la tierra gira alrededor del sol', 'los mapas muestran los ríos y países', 'la capital de España es Madrid'."
			}
		],
		"secondary": [
			{
				"topic": "literatura clásica y redacción",
				"prompt": "Genera 100 oraciones líricas y literarias de nivel secundaria (de 6 a 10 palabras). Ejemplos: 'las palabras escritas vencen el paso del tiempo', 'el poeta canta a la luna en la noche fría', 'el libro contiene memorias de hombres antiguos'."
			},
			{
				"topic": "física, lógica y ecuaciones algebraicas",
				"prompt": "Genera 80 oraciones de secundaria sobre física, causa-efecto y álgebra simple. Ejemplos: 'si equis más dos es cinco entonces equis es tres', 'la fuerza mueve las piedras en el espacio', 'toda causa produce un efecto en la naturaleza'."
			},
			{
				"topic": "filosofía de Borges: laberintos, espejos y biblioteca",
				"prompt": "Genera 150 oraciones y micro-diálogos filosóficos al estilo de Jorge Luis Borges sobre laberintos, espejos, memoria infinita y sueños. Ejemplos: 'el laberinto es una biblioteca de espejos infinitos', 'Funes recuerda la forma de cada nube en el cielo', 'soñé que un hombre me soñaba en el búnker', 'el Aleph es un punto que contiene todo el universo'."
			}
		]
	}
	
	system_prompt = (
		"Eres la Profesora Samantha. Eres experta en educación y pedagogía. "
		"Genera el currículo escolar solicitado. Devuelve única y exclusivamente las frases generadas, una por línea. "
		"No agregues números, viñetas, introducciones ni explicaciones."
	)
	
	school_dataset = {
		"preschool": [],
		"primary": [],
		"secondary": []
	}
	
	for grade, modules in curriculum_structure.items():
		print(f"\n🎒 [Grado: {grade.upper()}] Generando módulos...")
		for module in modules:
			topic = module["topic"]
			prompt = module["prompt"]
			print(f"  🔍 Generando sobre '{topic}'...")
			
			cmd_code = f"""
import sys
sys.path.append('{sharing_src}')
from red_pill.inference import samantha_on_demand
res = samantha_on_demand.invoke({repr(prompt)}, system_prompt={repr(system_prompt)}, max_tokens=1800, temperature=0.7)
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
					lines = result.stdout.strip().split("\n")
					valid_sentences = []
					for line in lines:
						line = line.strip()
						if not line or line.startswith("Aquí") or line.startswith("Claro"):
							continue
						# Quitar viñetas, guiones y números al inicio de la línea
						line = re.sub(r'^\s*[-*•\d+.]\s*', '', line)
						
						# Mapear al vocabulario base mediante SovereignDictionary
						words_in_line = re.findall(r'[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_]+', line.lower())
						if not words_in_line:
							continue
						
						mapped_words = [dictionary.map_to_base_word(w) for w in words_in_line]
						cleaned = " ".join(mapped_words)
						
						if len(mapped_words) >= 2:
							valid_sentences.append(cleaned)
					
					print(f"    ✓ Samantha generó {len(lines)} líneas. Mapeadas y validadas: {len(valid_sentences)}")
					school_dataset[grade].extend(valid_sentences)
				else:
					print(f"    ❌ Error: {result.stderr}")
			except Exception as e:
				print(f"    ❌ Excepción: {e}")
				
	# Eliminar duplicados en cada grado
	for grade in school_dataset:
		unique = []
		seen = set()
		for s in school_dataset[grade]:
			if s not in seen:
				seen.add(s)
				unique.append(s)
		school_dataset[grade] = unique
		print(f"✨ Total frases únicas para [Grado {grade.upper()}]: {len(unique)}")
		
	# Guardar currículo completo
	output_path = os.path.join(base_dir, "configs", "school_curriculum.json")
	with open(output_path, "w", encoding="utf-8") as f:
		json.dump(school_dataset, f, indent=4, ensure_ascii=False)
	print(f"\n💾 Currículo completo guardado en: {output_path}")

if __name__ == "__main__":
	run_generation()
