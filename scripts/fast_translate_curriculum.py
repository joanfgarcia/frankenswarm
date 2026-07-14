import json
import os
import re

base_dir = "/home/joan/Documents/IA/frankenswarm"

# Comprehensive dictionary for words in the structured curriculum
WORD_MAP = {
    # Nouns
    "fuego": "fire", "comida": "food", "agua": "water", "hielo": "ice", "cueva": "cave",
    "noche": "night", "sol": "sun", "día": "day", "dolor": "pain", "calor": "heat",
    "frío": "cold", "cuerpo": "body", "mano": "hand", "nene": "baby", "nena": "baby",
    "perro": "dog", "gato": "cat", "pájaro": "bird", "oso": "bear", "flor": "flower",
    "piedra": "stone", "tierra": "earth", "río": "river", "mar": "sea", "árbol": "tree",
    "planta": "plant", "luz": "light", "corazón": "heart", "sangre": "blood", "pulmones": "lungs",
    "aire": "air", "vida": "life", "verdad": "truth", "mentira": "lie", "amigo": "friend",
    "nombre": "name", "mapa": "map", "país": "country", "ciudad": "city", "laberinto": "labyrinth",
    "espejo": "mirror", "biblioteca": "library", "tiempo": "time", "memoria": "memory",
    "nube": "cloud", "universo": "universe", "búnker": "bunker", "sistema": "system",
    "mamá": "mom", "papá": "dad", "bebé": "baby", "niño": "boy", "juguetes": "toys",
    "balón": "ball", "flores": "flowers", "piedras": "stones", "animales": "animals",
    "hambre": "hunger", "sed": "thirst", "boca": "mouth", "ojos": "eyes", "madre": "mother",
    "padre": "father", "oxígeno": "oxygen", "sombra": "shade", "plantas": "plants",
    "árboles": "trees", "capital": "capital", "mapas": "maps", "países": "countries",
    "espejos": "mirrors", "palabras": "words", "causa": "cause", "efecto": "effect",
    "nubes": "clouds",

    # Verbs
    "es": "is", "son": "are", "siente": "feels", "sentir": "feel", "tocar": "touch",
    "toco": "touch", "tocas": "touch", "quema": "burns", "duele": "hurts", "tengo": "have",
    "tienes": "have", "tiene": "has", "quieres": "want", "quiere": "wants", "pasa": "happens",
    "hace": "does", "hacen": "do", "come": "eats", "duerme": "sleeps", "corre": "runs",
    "vuela": "flies", "beben": "drink", "bebe": "drinks", "bombea": "pumps", "respiramos": "breathe",
    "respirar": "breathe", "vivir": "live", "vive": "lives", "viven": "live", "crecen": "grow",
    "crecer": "grow", "llamas": "call", "llamar": "call", "saber": "know", "decir": "say",
    "dice": "says", "gira": "rotates", "muestran": "show", "muestra": "shows", "vencen": "defeat",
    "recuerda": "remembers", "contiene": "contains", "soñé": "dreamed", "soñaba": "dreamed",
    "gusta": "likes", "pongo": "put", "bebo": "drink", "descansar": "rest", "cerrar": "close",
    "querer": "want", "llora": "cries", "lloro": "cry", "volar": "fly", "cantar": "sing",
    "mueve": "moves", "saltar": "jump", "nadie": "nobody", "duermen": "sleep", "trabaja": "works",
    "girar": "rotate", "correr": "run", "respiramos": "breathe", "obtiene": "obtains",
    "muestra": "shows", "muestran": "show", "llueve": "rains", "moja": "wets", "produce": "produces",
    "veo": "see", "ves": "see", "ve": "sees", "vemos": "see", "ven": "see", "puedo": "can",
    "está": "is", "están": "are", "estamos": "are", "brilla": "shines", "manda": "sends",

    # Adjectives
    "caliente": "hot", "buena": "good", "bueno": "good", "fría": "cold", "frío": "cold",
    "dolorosa": "painful", "doloroso": "painful", "peligrosa": "dangerous", "peligroso": "dangerous",
    "sabia": "wise", "sabio": "wise", "sabroso": "tasty", "grande": "big", "grand": "big",
    "gran": "big", "pequeña": "small", "pequeño": "small", "oscuro": "dark", "oscura": "dark",
    "brillante": "bright", "alto": "high", "limpia": "clean", "limpio": "clean", "dulce": "sweet",
    "rápida": "fast", "rápido": "fast", "fuerte": "strong", "larga": "long", "largo": "long",
    "verdes": "green", "verde": "green", "sencillas": "simple", "sencillo": "simple",
    "infinitos": "infinite", "infinito": "infinite", "fresca": "fresh", "frio": "cold",
    "caliente": "hot", "saciado": "full", "seguro": "safe", "malo": "bad", "pocas": "few",
    "comunes": "common", "sencillas": "simple", "típicas": "typical", "complejas": "complex",
    "política": "political",

    # Pronouns & Determiners
    "el": "the", "la": "the", "los": "the", "las": "the", "un": "a", "una": "a",
    "unos": "some", "unas": "some", "yo": "i", "tú": "you", "él": "he", "ella": "she",
    "nosotros": "we", "ellos": "they", "mi": "my", "mis": "my", "tu": "your", "tus": "your",
    "su": "his", "sus": "his", "nuestro": "our", "nuestra": "our", "mío": "mine",
    "quién": "who", "qué": "what", "dónde": "where", "cuándo": "when", "cómo": "how",
    "este": "this", "esta": "this", "estos": "these", "estas": "these", "ese": "that",
    "esa": "that", "esos": "those", "esas": "those",

    # Prepositions & Adverbs
    "de": "of", "en": "in", "para": "for", "por": "by", "con": "with", "sin": "without",
    "a": "to", "hacia": "towards", "arriba": "up", "abajo": "down", "lejos": "far",
    "cerca": "near", "dentro": "inside", "fuera": "outside", "antes": "before", "ahora": "now",
    "después": "after", "siempre": "always", "nunca": "never", "alrededor": "around",
    "atrás": "behind", "adelante": "forward", "hoy": "today", "muy": "very", "más": "more",
    "menos": "less", "tan": "so", "como": "as", "aquí": "here", "allí": "there",

    # Conjunctions & Logic
    "y": "and", "o": "or", "pero": "but", "porque": "because", "si": "if", "entonces": "then",
    "aunque": "although", "no": "not", "quizá": "maybe", "sí": "yes",

    # Numbers
    "uno": "one", "dos": "two", "tres": "three", "cuatro": "four", "cinco": "five",
    "seis": "six", "siete": "seven", "ocho": "eight", "nueve": "nine", "diez": "ten",

    # Specific names
    "españa": "spain", "madrid": "madrid", "aleth": "aleth", "maureen": "maureen", "susan": "susan",
    "borges": "borges", "funes": "funes", "aleph": "aleph", "equis": "x"
}

def translate_sentence(text: str) -> str:
    # 1. Tokenize words
    words = re.findall(r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_]+", text.lower())
    
    # 2. Translate word by word
    translated = []
    for w in words:
        translated.append(WORD_MAP.get(w, w))
        
    # 3. Simple grammar fixes (e.g. "fire hot" -> "hot fire")
    # Let's search for typical noun-adjective pairs and swap them:
    # "fire hot" -> "hot fire", "water cold" -> "cold water", "cave dark" -> "dark cave",
    # "night dark" -> "dark night", "earth big" -> "big earth", "blood red" -> "red blood",
    # "water hot" -> "hot water", "water clean" -> "clean water", "stone small" -> "small stone",
    # "river big" -> "big river", "cueva oscura" -> "dark cave", "fuego caliente" -> "hot fire",
    # "agua fría" -> "cold water", "agua caliente" -> "hot water"
    
    res = " ".join(translated)
    
    # Basic swaps
    swaps = [
        ("fire hot", "hot fire"),
        ("water cold", "cold water"),
        ("cave dark", "dark cave"),
        ("night dark", "dark night"),
        ("earth big", "big earth"),
        ("water hot", "hot water"),
        ("water clean", "clean water"),
        ("stone small", "small stone"),
        ("river big", "big river"),
        ("flowers small", "small flowers"),
        ("stones small", "small stones"),
        ("bear big", "big bear"),
        ("animal big", "big animal"),
        ("dog small", "small dog"),
        ("cat small", "small cat"),
        ("bird small", "small bird"),
        ("water fresh", "fresh water"),
        ("day good", "good day"),
        ("friend good", "good friend"),
        ("people good", "good people"),
        ("night cold", "cold night"),
        ("maps show the rivers and countries", "maps show rivers and countries"),
        ("maps show the rivers and the countries", "maps show rivers and countries")
    ]
    for src, dst in swaps:
        res = res.replace(src, dst)
        
    return res

def main():
    input_path = os.path.join(base_dir, "configs", "school_curriculum_structured.json")
    output_path = os.path.join(base_dir, "configs", "school_curriculum_structured_en.json")
    
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)
        
    curriculum = data.get("curriculum", {})
    translated_curriculum = {}
    
    for stage, items in curriculum.items():
        translated_items = []
        for item in items:
            new_item = item.copy()
            new_item["text"] = translate_sentence(item["text"])
            translated_items.append(new_item)
        translated_curriculum[stage] = translated_items
        
    # Save structured translation
    out_data = {
        "metadata": {
            "version": "v2.0-structured-en-fast",
            "total_sentences": sum(len(v) for v in translated_curriculum.values()),
            "stats": {k: len(v) for k, v in translated_curriculum.items()}
        },
        "curriculum": translated_curriculum
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=4, ensure_ascii=False)
        
    print(f"Instantly translated structured curriculum to: {output_path}")

if __name__ == "__main__":
    main()
