import json
import os

import numpy as np
from fastembed import TextEmbedding

from src.bitnet.vocab.glyph_vocabulary import SEMANTIC_PRIMES, VOCABULARY

# Mapeo de claves internas a palabras naturales para fastembed.
# BIT-003: el vocabulario es INGLÉS (corpus EN de cero) → las 28 palabras de
# referencia se proyectan con sus equivalentes EN para que la regresión Ridge
# aprenda en el espacio semántico del corpus real (RULE 7: sin OOV ciego).
VOCAB_MAP = {
	"agua": "water",
	"comida": "food",
	"fuego": "fire",
	"sol": "sun",
	"noche": "night",
	"cueva": "cave",
	"yo_palabra": "i",
	"peligro": "danger",
	"comer": "eat",
	"beber": "drink",
	"mover_accion": "move",
	"ver_accion": "see",
	"dormir": "sleep",
	"dar": "give",
	"enseñar": "teach",
	"aprender": "learn",
	"bosque": "forest",
	"río": "river",
	"piedra": "stone",
	"árbol": "tree",
	"tierra": "earth",
	"lluvia": "rain",
	"depredador": "predator",
	"tormenta": "storm",
	"herida": "wound",
	"seguro": "safe",
	"saciado": "full",
	"grupo": "group",
}

# Lista de palabras base en español para completar hasta 1,000 términos
BASE_WORDS = [
	"<unk>", "<pad>",
	# --- Sustantivos (Nouns) ---
	"hombre", "mujer", "niño", "niña", "bebé", "persona", "gente", "padre", "madre", "hijo", "hija",
	"hermano", "hermana", "abuelo", "abuela", "tío", "tía", "primo", "prima", "amigo", "enemigo",
	"familia", "casa", "hogar", "puerta", "ventana", "pared", "techo", "suelo", "mesa", "silla",
	"cama", "habitación", "libro", "papel", "lápiz", "pluma", "carta", "llave", "reloj", "caja",
	"bolsa", "ropa", "zapato", "comida", "pan", "leche", "fruta", "manzana", "carne", "huevo",
	"sal", "azúcar", "agua", "vino", "cerveza", "cielo", "estrella", "luna", "planeta", "sol",
	"nube", "viento", "lluvia", "nieve", "hielo", "calor", "frío", "luz", "oscuridad", "sombra",
	"fuego", "humo", "ceniza", "tierra", "barro", "polvo", "arena", "piedra", "roca", "cueva",
	"mar", "océano", "río", "lago", "bosque", "árbol", "flor", "planta", "hoja", "rama",
	"semilla", "hierba", "animal", "perro", "gato", "caballo", "vaca", "oveja", "cabra", "cerdo",
	"león", "oso", "lobo", "zorro", "ratón", "pájaro", "pez", "serpiente", "insecto", "mosca",
	"abeja", "araña", "hormiga", "cuerpo", "cabeza", "pelo", "cara", "ojo", "oreja", "nariz",
	"boca", "diente", "lengua", "cuello", "hombro", "brazo", "codo", "mano", "dedo", "pecho",
	"espalda", "corazón", "sangre", "estómago", "pierna", "rodilla", "pie", "piel", "hueso",
	"mente", "cerebro", "pensamiento", "vida", "muerte", "tiempo", "espacio", "verdad", "mentira",
	"peligro", "seguridad", "ayuda", "dolor", "hambre", "sed", "sueño", "miedo", "alegría",
	"ira", "tristeza", "sorpresa", "amor", "odio", "paz", "guerra", "batalla", "arma", "soldado",
	"rey", "reina", "jefe", "médico", "doctor", "maestro", "profesor", "escuela", "hospital",
	"tienda", "mercado", "trabajo", "dinero", "oro", "plata", "hierro", "metal", "madera",
	"ciudad", "pueblo", "calle", "camino", "plaza", "puente", "barco", "carro", "rueda",
	"juego", "música", "canto", "baile", "voz", "sonido", "ruido", "silencio", "olor", "sabor",
	"búnker", "agente", "código", "sistema", "máquina", "red", "córtex", "datos", "número",
	"cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez",
	"cien", "mil", "lógica", "ciencia", "historia", "lenguaje", "palabra", "frase", "carta",
	"vuela", "beben", "bombea", "oxigeno", "oxígeno", "capital", "españa", "madrid", "equis",
	"produce", "laberinto", "biblioteca", "infinito", "infinitos", "funes", "aleph", "contiene",


	# --- Verbos (Verbs) ---
	"ser", "estar", "tener", "haber", "hacer", "decir", "ir", "venir", "ver", "oír", "tocar",
	"sentir", "pensar", "saber", "querer", "poder", "deber", "necesitar", "comer", "beber",
	"dormir", "vivir", "morir", "dar", "tomar", "coger", "poner", "traer", "llevar", "buscar",
	"encontrar", "perder", "ganar", "comprar", "vender", "pagar", "trabajar", "jugar", "cantar",
	"bailar", "correr", "caminar", "saltar", "volar", "nadar", "subir", "bajar", "entrar",
	"salir", "abrir", "cerrar", "empezar", "terminar", "esperar", "recordar", "olvidar",
	"aprender", "enseñar", "escribir", "leer", "hablar", "llamar", "pedir", "ayudar", "amar",
	"odiar", "temer", "luchar", "correr", "destruir", "crear", "cambiar", "romper", "limpiar",

	# --- Adjetivos (Adjectives) ---
	"bueno", "malo", "grande", "pequeño", "alto", "bajo", "largo", "corto", "ancho", "estrecho",
	"rápido", "lento", "fuerte", "débil", "fácil", "difícil", "caliente", "frío", "seco", "mojado",
	"limpio", "sucio", "claro", "oscuro", "seguro", "peligro", "rico", "pobre", "libre", "preso",
	"alegre", "triste", "cansado", "enfermo", "sano", "inteligente", "hermoso", "feo", "nuevo",
	"viejo", "joven", "primero", "último", "diferente", "mismo", "solo", "acompañado", "lleno",
	"vacío", "pesado", "ligero", "suave", "duro", "dulce", "amargo", "ácido", "salado",

	# --- Conectores y otros ---
	"sí", "no", "quizás", "muy", "bastante", "poco", "mucho", "más", "menos", "tan", "como",
	"aquí", "allí", "cerca", "lejos", "arriba", "abajo", "dentro", "fuera", "antes", "ahora",
	"después", "siempre", "nunca", "y", "o", "pero", "porque", "si", "entonces", "aunque",
	"con", "sin", "de", "para", "por", "en", "sobre", "bajo", "a", "hacia", "desde", "hasta",
	"este", "esta", "estos", "estas", "ese", "esa", "esos", "esas", "aquel", "aquella",
	"mi", "mis", "tu", "tus", "su", "sus", "nuestro", "nuestra", "yo", "tú", "él", "ella",
	"nosotros", "ellos", "ellas", "mío", "tuyo", "suyo", "quién", "qué", "dónde", "cuándo",
	"cómo", "por_qué", "cada", "todo", "todos", "algunos", "ninguno", "otro", "otra", "otros",
	# --- Artículos, relativos y pronombres críticos ---
	"el", "la", "los", "las", "un", "una", "unos", "unas", "que", "lo", "le", "se", "me", "te", "nos",
	# --- Conjugaciones comunes de verbos auxiliares (ser, estar, haber, hacer) ---
	"es", "son", "era", "eran", "será", "serán", "sea", "está", "están", "estaba", "estaban", "estará",
	"estarán", "ha", "han", "hay", "había", "habían", "hace", "hacen", "hacía", "hacían", "hecho"
]

# Ampliar dinámicamente la lista con variaciones sistemáticas para llegar a exactamente 1,000 palabras únicas
categories = {
	"colores": ["rojo", "azul", "verde", "amarillo", "blanco", "negro", "gris", "marrón", "naranja", "rosa", "morado"],
	"cuerpo_extra": ["cuello", "pecho", "espalda", "hombro", "brazo", "mano", "dedo", "rodilla", "pie", "uña", "pelo", "piel", "hueso", "sangre", "corazón", "estómago", "diente", "boca", "ojo", "oreja", "nariz", "frente", "mejilla", "barba"],
	"frutas_extra": ["plátano", "naranja", "limón", "fresa", "uva", "pera", "piña", "sandía", "melón", "melocotón", "cereza", "ciruela", "tomate", "patata", "zanahoria", "cebolla", "ajo", "lechuga"],
	"animales_extra": ["elefante", "jirafa", "mono", "león", "tigre", "oso", "lobo", "zorro", "conejo", "ratón", "águila", "pato", "pollo", "gallina", "pez", "ballena", "delfín", "tiburón", "rana", "tortuga", "serpiente", "araña", "mosca", "mosquito", "abeja", "hormiga"],
	"verbos_extra": ["cantar", "bailar", "pintar", "dibujar", "escribir", "leer", "hablar", "escuchar", "mirar", "tocar", "sentir", "pensar", "saber", "comprender", "aprender", "enseñar", "ayudar", "trabajar", "estudiar", "viajar", "caminar", "correr", "saltar", "volar", "nadar", "dormir", "despertar", "comer", "beber", "cocinar", "comprar", "vender", "pagar", "abrir", "cerrar", "entrar", "salir", "subir", "bajar", "perder", "ganar", "llegar", "partir", "esperar", "buscar", "encontrar", "cortar", "pegar", "romper", "reparar", "limpiar", "sucio", "lavar", "planchar", "vestir"],
	"adjetivos_extra": ["rápido", "lento", "fuerte", "débil", "pesado", "ligero", "duro", "suave", "áspero", "liso", "caliente", "frío", "tibio", "seco", "mojado", "húmedo", "limpio", "sucio", "ordenado", "desordenado", "bonito", "feo", "hermoso", "horrible", "bueno", "malo", "excelente", "pésimo", "fácil", "difícil", "simple", "complejo", "claro", "oscuro", "brillante", "apagado", "alegre", "triste", "feliz", "infeliz", "divertido", "aburrido", "interesante", "inteligente", "sabio", "tonto", "rico", "pobre", "caro", "barato", "nuevo", "viejo", "joven", "anciano"],
	"hogar_extra": ["cocina", "baño", "salón", "comedor", "dormitorio", "jardín", "patio", "garaje", "sótano", "ático", "escalera", "pasillo", "puerta", "ventana", "pared", "suelo", "techo", "chimenea", "mesa", "silla", "sofá", "cama", "armario", "espejo", "lámpara", "reloj", "cuadro", "alfombra", "cortina", "plato", "vaso", "taza", "tenedor", "cuchara", "cuchillo", "olla", "sartén", "nevera", "horno", "microondas"],
	"profesiones_extra": ["médico", "enfermero", "dentista", "maestro", "profesor", "estudiante", "policía", "bombero", "soldado", "abogado", "juez", "ingeniero", "arquitecto", "programador", "científico", "escritor", "periodista", "artista", "pintor", "músico", "actor", "cantante", "cocinero", "camarero", "conductor", "piloto", "marinero", "agricultor", "jardinero", "carpintero", "electricista", "fontanero", "mecánico", "sastre", "peluquero"],
	"geografia_extra": ["continente", "país", "nación", "estado", "provincia", "ciudad", "pueblo", "aldea", "calle", "avenida", "plaza", "camino", "carretera", "vía", "puente", "túnel", "estación", "aeropuerto", "puerto", "frontera", "costa", "playa", "isla", "península", "bahía", "golfo", "estrecho", "canal", "río", "arroyo", "lago", "laguna", "pantano", "cascada", "mar", "océano", "montaña", "cordillera", "pico", "valle", "llanura", "desierto", "selva", "bosque", "prado", "campo"],
	"tiempo_extra": ["segundo", "minuto", "hora", "día", "noche", "mañana", "tarde", "semana", "mes", "año", "década", "siglo", "estación", "primavera", "verano", "otoño", "invierno", "fecha", "calendario", "reloj", "momento", "instante", "época", "era", "pasado", "presente", "futuro", "temprano", "tarde", "pronto", "luego", "antes", "después", "siempre", "nunca", "jamás", "mientras", "durante"],
	"abstracciones_extra": ["idea", "concepto", "teoría", "opinión", "creencia", "duda", "certeza", "verdad", "mentira", "secreto", "misterio", "problema", "solución", "pregunta", "respuesta", "causa", "efecto", "motivo", "razón", "propósito", "meta", "objetivo", "plan", "proyecto", "trabajo", "esfuerzo", "éxito", "fracaso", "error", "acierto", "culpa", "perdón", "justicia", "ley", "regla", "norma", "deber", "derecho", "libertad", "paz", "guerra", "orden", "caos", "peligro", "seguridad", "riesgo", "suerte", "destino", "azar", "magia", "milagro"],
	"sentimientos_extra": ["amor", "cariño", "afecto", "amistad", "odio", "rencor", "rabia", "ira", "enfado", "molestia", "miedo", "temor", "terror", "pánico", "tristeza", "pena", "dolor", "sufrimiento", "alegría", "felicidad", "gozo", "placer", "orgullo", "vergüenza", "culpa", "sorpresa", "asombro", "curiosidad", "aburrimiento", "interés", "calma", "tranquilidad", "paz", "esperanza", "desesperación", "celos", "envidia", "compasión", "empatía"]
}

# Cargar el vocabulario limpio (rebuild_clean_vocabulary.py) si existe, de lo contrario usar el fallback base
clean_vocab_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "configs/clean_vocabulary_words.json")

if os.path.exists(clean_vocab_path):
	print(f"Cargando vocabulario limpio desde {clean_vocab_path}...")
	with open(clean_vocab_path, encoding="utf-8") as f:
		FINAL_VOCAB = json.load(f)["words"]
else:
	print("⚠️ No se encontró configs/clean_vocabulary_words.json, utilizando fallback de 15,000 palabras...")
	# Agregar palabras de las categorías adicionales
	word_set = set(BASE_WORDS)

	# Asegurar que todos los primos semánticos estén en el vocabulario base
	for p in SEMANTIC_PRIMES:
		word_set.add(VOCAB_MAP.get(p, p))

	for _cat, list_words in categories.items():
		for w in list_words:
			word_set.add(w)

	# Agregar palabras de alta frecuencia de HermitDave
	freq_words_path = "/home/joan/.gemini/antigravity/scratch/spanish_words.txt"
	if os.path.exists(freq_words_path):
		print(f"Cargando frecuencias de español desde {freq_words_path}...")
		import re
		with open(freq_words_path, encoding="utf-8") as f:
			for line in f:
				parts = line.strip().split()
				if parts:
					w = parts[0].lower().strip()
					# Validar que es una palabra alfabética real de al menos 2 letras
					if re.match(r'^[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_]{2,}$', w):
						word_set.add(w)
						if len(word_set) >= 15000:
							break

	# Si no llegamos a 15000, autocompletamos con variaciones numeradas (sólo como fallback)
	i = 0
	while len(word_set) < 15000:
		word_set.add(f"palabra_{i}")
		i += 1

	# Asegurar que las 26 palabras de referencia están incluidas EXACTAMENTE con sus nombres clave de glyph_vocabulary
	for k in VOCABULARY:
		word_set.add(k)

	# Convertir a lista y ordenar
	FINAL_VOCAB = sorted(word_set)

print(f"Total palabras en el vocabulario base: {len(FINAL_VOCAB)}")

def calibrate_and_project():
	print("Inicializando modelo TextEmbedding...")
	model = TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

	# 1. Obtener embeddings para las 26 palabras de referencia
	ref_keys = list(VOCABULARY.keys())
	# Mapeamos a palabras naturales para obtener mejores embeddings semánticos
	ref_natural_words = [VOCAB_MAP.get(k, k) for k in ref_keys]
	
	print(f"Generando embeddings para {len(ref_keys)} palabras de referencia...")
	ref_embeddings = np.array(list(model.embed(ref_natural_words)), dtype=np.float32)
	ref_glyphs = np.stack([VOCABULARY[k] for k in ref_keys], axis=0).astype(np.float32) # (26, 65)

	# 2. Ajustar la regresión Ridge (Dual Ridge en el espacio de muestras)
	# W = E.T @ inv(E @ E.T + alpha * I) @ Y
	E = ref_embeddings # (26, 384)
	Y = ref_glyphs     # (26, 65)
	alpha = 1.0        # Regularización Ridge

	print("Entrenando proyección Ridge lineal...")
	K = E @ E.T # (26, 26)
	I = np.eye(len(ref_keys), dtype=np.float32)
	A = np.linalg.solve(K + alpha * I, Y) # (26, 65)
	
	# W = E.T @ A es de forma (384, 65)
	# Para proyectar un nuevo embedding x: x @ W = (x @ E.T) @ A

	# 3. Optimizar el umbral theta en base a la reconstrucción de los glifos de referencia
	best_theta = 0.25
	best_hamming = 0.0
	
	preds_continuous = (E @ E.T) @ A # (26, 65)
	
	for theta in np.arange(0.05, 0.6, 0.01):
		# Ternarizar
		preds_ternary = np.zeros_like(preds_continuous, dtype=np.int8)
		preds_ternary[preds_continuous > theta] = 1
		preds_ternary[preds_continuous < -theta] = -1
		
		# Calcular precisión/Hamming contra Y original
		accuracy = np.mean(preds_ternary == Y)
		if accuracy > best_hamming:
			best_hamming = accuracy
			best_theta = theta

	print(f"Mejor umbral calibrado: theta={best_theta:.3f} (Hamming Accuracy reconstrucción: {best_hamming * 100:.2f}%)")

	# 4. Proyectar todas las 1,000 palabras base
	print(f"Generando embeddings para las {len(FINAL_VOCAB)} palabras base...")
	# Mapeamos a palabras naturales antes de pasar a fastembed
	natural_vocab = [VOCAB_MAP.get(w, w) for w in FINAL_VOCAB]
	all_embeddings = np.array(list(model.embed(natural_vocab)), dtype=np.float32) # (1000+, 384)

	# Proyectar
	continuous_projections = (all_embeddings @ E.T) @ A # (1000+, 65)
	
	# Ternarizar usando el mejor umbral
	all_glyphs = np.zeros_like(continuous_projections, dtype=np.int8)
	all_glyphs[continuous_projections > best_theta] = 1
	all_glyphs[continuous_projections < -best_theta] = -1

	# Conservar EXACTAMENTE los glifos de referencia originales para los tokens de referencia
	for idx, w in enumerate(FINAL_VOCAB):
		if w in VOCABULARY:
			all_glyphs[idx] = VOCABULARY[w]

	# ── DESEMPATADOR DE GLIFOS DUPLICADOS (Asegura unicidad semántica) ──
	print("Resolviendo duplicados de glifos mediante desempate determinista iterativo...")
	iteration = 0
	while iteration < 10:
		glyph_to_indices = {}
		for idx, g in enumerate(all_glyphs):
			g_tuple = tuple(g.tolist())
			if g_tuple not in glyph_to_indices:
				glyph_to_indices[g_tuple] = []
			glyph_to_indices[g_tuple].append(idx)

		duplicates = {g: idxs for g, idxs in glyph_to_indices.items() if len(idxs) > 1}
		if not duplicates:
			break

		print(f"  [Iteración {iteration+1}] Resolviendo {len(duplicates)} grupos de duplicados ({sum(len(v) for v in duplicates.values())} palabras)...")

		for g_tuple, indices in duplicates.items():
			ref_indices = [idx for idx in indices if FINAL_VOCAB[idx] in VOCABULARY]
			non_ref_indices = [idx for idx in indices if FINAL_VOCAB[idx] not in VOCABULARY]

			g_arr = np.array(g_tuple)
			zero_dims = np.where(g_arr == 0)[0]

			n_to_resolve = len(non_ref_indices)
			start_counter = 1 if len(ref_indices) > 0 else 0
			n_bits = int(np.ceil(np.log2(n_to_resolve + start_counter)))

			if len(zero_dims) < n_bits:
				n_bits = len(zero_dims)

			for i, word_idx in enumerate(non_ref_indices):
				counter = i + start_counter + iteration * 13  # Desplazar el contador por iteración para evitar colisiones repetidas
				for j in range(min(n_bits, len(zero_dims))):
					val = 1 if (counter & (1 << j)) else -1
					all_glyphs[word_idx, zero_dims[j]] = val
		iteration += 1

	# Verificar resultado final
	final_glyph_to_indices = {}
	for idx, g in enumerate(all_glyphs):
		g_tuple = tuple(g.tolist())
		if g_tuple not in final_glyph_to_indices:
			final_glyph_to_indices[g_tuple] = []
		final_glyph_to_indices[g_tuple].append(idx)
	print(f"✓ Post-procesamiento completado: {len(final_glyph_to_indices)} glifos únicos para {len(FINAL_VOCAB)} palabras en {iteration} iteraciones.")


	# Guardar en configs/expanded_glyphs.json
	output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "configs")

	os.makedirs(output_dir, exist_ok=True)
	
	output_dict = {
		"words": FINAL_VOCAB,
		"glyphs": all_glyphs.tolist(),
		"theta": float(best_theta)
	}
	
	output_path = os.path.join(output_dir, "expanded_glyphs.json")
	with open(output_path, "w", encoding="utf-8") as f:
		json.dump(output_dict, f, indent=4, ensure_ascii=False)
		
	print(f"¡Hito completado! Glifos expandidos guardados en: {output_path}")

if __name__ == "__main__":
	calibrate_and_project()
