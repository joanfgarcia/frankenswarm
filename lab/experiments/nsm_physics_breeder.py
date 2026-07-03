import os
import json
import random
from src.bitnet.dictionary_tool import SovereignDictionary

class NSMPhysicsDatasetBreeder:
	def __init__(self, dictionary: SovereignDictionary):
		self.dictionary = dictionary
		
		# Sujetos primitivos y comunes
		self.subjects = ["yo", "tú", "él", "ella", "alguien", "gente", "nosotros", "ellos"]
		
		# Acciones y transiciones físicas básicas (Goddard Semantic Templates)
		# Cada plantilla contiene una lista de pasos causales secuenciales
		self.templates = {
			"caida": [
				"{entidad} se mueve hacia abajo",
				"{entidad} toca el suelo",
				"{entidad} no se mueve más",
				"{sujeto} ve esto y se siente mal"
			],
			"quemar": [
				"{sujeto} toca el fuego",
				"el fuego está muy caliente",
				"{sujeto} siente dolor",
				"esto es malo"
			],
			"comer": [
				"{sujeto} tiene hambre",
				"{sujeto} pone comida en su boca",
				"la comida va hacia adentro",
				"{sujeto} se siente bien"
			],
			"beber": [
				"{sujeto} tiene sed",
				"{sujeto} pone agua en su boca",
				"el agua va hacia adentro",
				"{sujeto} se siente bien"
			],
			"fabricar": [
				"{sujeto} hace una lanza",
				"{sujeto} usa ramas y piedras",
				"la lanza es buena",
				"{sujeto} quiere usar la lanza"
			],
			"construir": [
				"{sujeto} hace un refugio",
				"{sujeto} usa ramas y piedras",
				"el refugio es grande y bueno",
				"{sujeto} duerme en el refugio después"
			],
			"encender": [
				"{sujeto} hace fuego",
				"{sujeto} usa ramas secas",
				"el fuego calienta la cueva",
				"el fuego es bueno"
			],
			"cazar": [
				"{sujeto} ve un animal grande",
				"el animal se mueve rápido",
				"{sujeto} siente miedo",
				"{sujeto} usa la lanza para cazar"
			]
		}
		
		# Entidades físicas y estados corporales
		self.entities = ["objeto", "rama", "piedra", "comida", "agua", "fuego", "lanza", "refugio", "animal"]

	def clean_and_map_sentence(self, sentence: str) -> str:
		# Convertir a minúsculas y extraer palabras limpias
		words = sentence.lower().replace(",", "").replace(".", "").split()
		mapped_words = [self.dictionary.map_to_base_word(w) for w in words]
		return " ".join(mapped_words)

	def breed_corpus(self, num_samples_per_template: int = 50) -> list[str]:
		corpus = []
		
		for name, steps in self.templates.items():
			for _ in range(num_samples_per_template):
				# Seleccionar sujeto y entidad aleatorios para esta variación
				sujeto = random.choice(self.subjects)
				entidad = random.choice(self.entities)
				
				# Construir la secuencia de oraciones causales
				for step in steps:
					formatted = step.format(sujeto=sujeto, entidad=entidad)
					cleaned = self.clean_and_map_sentence(formatted)
					corpus.append(cleaned)
					
		# Añadir combinaciones simples de sujeto + operador + estado emocional/físico
		extra_states = [
			"{sujeto} sentir {emocion}",
			"yo saber que {sujeto} sentir {emocion}",
			"yo querer hacer {accion}",
			"{sujeto} hacer {accion} porque sentir {emocion}"
		]
		emotions = ["alegría", "tristeza", "miedo", "ira", "dolor", "hambre"]
		acciones = ["comer", "beber", "dormir", "fabricar", "construir", "encender"]
		
		for _ in range(num_samples_per_template * 3):
			sujeto = random.choice(self.subjects)
			emocion = random.choice(emotions)
			accion = random.choice(acciones)
			
			template = random.choice(extra_states)
			formatted = template.format(sujeto=sujeto, emocion=emocion, accion=accion)
			cleaned = self.clean_and_map_sentence(formatted)
			corpus.append(cleaned)
			
		return corpus

def run_nsm_breeding():
	print("═══ 🧬 Generador de Explicaciones NSM (Física Semántica) ═══")
	
	base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	configs_dir = os.path.join(base_dir, "configs")
	
	expanded_glyphs_path = os.path.join(configs_dir, "expanded_glyphs.json")
	dictionary = SovereignDictionary(expanded_glyphs_path)
	
	breeder = NSMPhysicsDatasetBreeder(dictionary)
	
	# Generar corpus
	corpus = breeder.breed_corpus(num_samples_per_template=60)
	print(f"  ✓ Generadas {len(corpus)} frases de física semántica.")
	
	# Eliminar duplicados manteniendo orden
	unique_corpus = []
	seen = set()
	for s in corpus:
		if s not in seen:
			seen.add(s)
			unique_corpus.append(s)
			
	print(f"  ✓ Filtradas {len(unique_corpus)} frases únicas.")
	
	# Guardar en configs/nsm_physics_pre_school.json
	output_path = os.path.join(configs_dir, "nsm_physics_pre_school.json")
	with open(output_path, "w", encoding="utf-8") as f:
		json.dump(unique_corpus, f, indent=4, ensure_ascii=False)
		
	print(f"\n📁 Archivo guardado con éxito: {output_path}")

if __name__ == "__main__":
	run_nsm_breeding()
