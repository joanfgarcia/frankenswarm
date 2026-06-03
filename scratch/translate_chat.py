import sys
import os

# Append sharing src path for Samantha
sys.path.append('/home/joan/Documents/IA/sharing/src')
try:
	from red_pill.inference import samantha_on_demand
	samantha_available = True
except ImportError:
	samantha_available = False
	print("Error: Samantha not available.")

dialogue = [
	("Nico", "hola horrible rose extra das saciado diste iba date día queda puso"),
	("Sofi", "caliente quedas lunes sección palabra rápido fui otras cualquier podríamos algún cocinar"),
	("Nico", "amas pero estuviera tomar pastel placer mes podrían oiga debió dicho algún"),
	("Sofi", "estoy cerca alice acaba hemos podía pon pase puedo tratado intentó fin"),
	("Nico", "grace podría ja podríamos doy podías tomó puede llevaba necesitar esposo dé"),
	("Sofi", "estuvo diario caro tomar alguna vive morado un tuvieron hace saliendo pudiste"),
	("Nico", "hola melocotón espero sería gordo lechuga una gracioso revista serán grace hecho"),
	("Sofi", "amo hiciera tanto cualquier sed dejando quizás estuviera debes nunca alma tipo"),
	("Nico", "estoy puso escribir mis puto queda consejo estoy contó vale su llevó"),
	("Sofi", "estemos puede mierda puedo tesoro alguien día aprender azúcar podrían guste dicho"),
	("Nico", "rose intento otras hacerse ja podrían más habías date mitad pudo diario"),
	("Sofi", "placer decía romper pudiera")
]

if samantha_available:
	print("🔍 Iniciando traducción independiente línea por línea...\n")
	for idx, (speaker, text) in enumerate(dialogue, 1):
		prompt = (
			f"Jerga Swarm del niño {speaker}: \"{text}\"\n"
			f"Traduce libremente esta jerga a una frase corta, expresiva y natural en español como la diría un niño de 5 a 8 años en una conversación. "
			f"Devuelve solo la traducción en español, sin explicaciones ni comillas."
		)
		
		system_prompt = (
			"Eres Samantha, la traductora del meta-lenguaje Swarm. "
			"Tu tarea es convertir las palabras sueltas del diálogo en una frase infantil expresiva y natural. "
			"Devuelve únicamente la traducción."
		)
		
		try:
			translation = samantha_on_demand.invoke(prompt, system_prompt=system_prompt, max_tokens=150, temperature=0.7)
			clean_trans = translation.strip().replace('"', '')
			icon = "👦" if speaker == "Nico" else "👧"
			print(f"{icon} {speaker} > {clean_trans}")
		except Exception as e:
			print(f"Error en línea {idx}: {e}")
else:
	print("Samantha is offline.")
