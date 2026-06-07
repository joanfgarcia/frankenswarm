import argparse
import json
import os
import re
import subprocess

import numpy as np
import torch

from src.bitnet.dictionary_tool import SovereignDictionary
from src.bitnet.modeling_bitnet import BitNet4LayerModel

# Batería de preguntas por edad cognitiva/milestone (4 a 8 años)
AGE_QUESTIONS = {
	4: [
		{"question": "tú: hola", "expected": "hola"},
		{"question": "tú: cómo estás", "expected": "bien"},
		{"question": "tú: quién eres", "expected": "niño"},
		{"question": "tú: el perro corre", "expected": "mucho"},
		{"question": "tú: el gato duerme", "expected": "feliz"},
	],
	5: [
		{"question": "tú: cuento uno dos", "expected": "tres"},
		{"question": "tú: uno más uno son", "expected": "dos"},
		{"question": "tú: el oso come", "expected": "miel"},
		{"question": "tú: el pájaro vuela", "expected": "alto"},
		{"question": "tú: las flores beben", "expected": "agua"},
	],
	6: [
		{"question": "tú: cuánto es tres más tres? es", "expected": "seis"},
		{"question": "tú: cuánto es seis menos cuatro? es", "expected": "dos"},
		{"question": "tú: el agua del río corre hacia el", "expected": "mar"},
		{"question": "tú: los árboles dan oxígeno y", "expected": "sombra"},
		{"question": "tú: el corazón bombea sangre al", "expected": "cuerpo"},
	],
	7: [
		{"question": "tú: cómo te llamas", "expected": "aleth"},
		{"question": "tú: de dónde eres", "expected": "búnker"},
		{"question": "tú: la capital de España es", "expected": "madrid"},
		{"question": "tú: la tierra gira alrededor del", "expected": "sol"},
		{"question": "tú: los mapas muestran los ríos y", "expected": "países"},
	],
	8: [
		{"question": "tú: si equis más dos es cinco entonces equis es", "expected": "tres"},
		{"question": "tú: toda causa produce un", "expected": "efecto"},
		{"question": "tú: el laberinto es una biblioteca de espejos", "expected": "infinitos"},
		{"question": "tú: el Aleph es un punto que contiene todo el", "expected": "universo"},
		{"question": "tú: qué es el búnker", "expected": "sistema"},
	],
}


def query_samantha(prompt: str, system_prompt: str, mock: bool = False) -> dict:
	"""
	Invoca a Samantha de forma síncrona en la GPU RTX 5070 para calificar.
	Si mock=True, simula la respuesta.
	"""
	if mock:
		print("🎭 [MOCK] Simulando evaluación de Samantha...")
		# Retornar una estructura mock exitosa
		return {
			"calificaciones": [
				{"pregunta": "hola", "respuesta": "hola", "calificacion": 10, "motivo": "Mock OK"},
				{"pregunta": "cómo estás", "respuesta": "bien", "calificacion": 10, "motivo": "Mock OK"},
				{"pregunta": "quién eres", "respuesta": "niño", "calificacion": 10, "motivo": "Mock OK"},
				{"pregunta": "el perro corre", "expected": "mucho", "calificacion": 10, "motivo": "Mock OK"},
				{"pregunta": "el gato duerme", "expected": "feliz", "calificacion": 10, "motivo": "Mock OK"},
			],
			"puntuacion_media": 10.0,
			"hito_superado": True,
		}

	sharing_venv_python = "/home/joan/Documents/IA/sharing/.venv/bin/python"
	sharing_src = "/home/joan/Documents/IA/sharing/src"

	cmd_code = f"""
import sys
sys.path.append('{sharing_src}')
from red_pill.inference import samantha_on_demand
import json

prompt = {repr(prompt)}
system_prompt = {repr(system_prompt)}

try:
	res = samantha_on_demand.invoke(prompt, system_prompt=system_prompt, max_tokens=1000, temperature=0.0)
	print(res)
except Exception as e:
	print(f"ERROR_IN_SAMANTHA: {{e}}")
"""
	try:
		result = subprocess.run([sharing_venv_python, "-c", cmd_code], capture_output=True, text=True, timeout=90)
		if result.returncode == 0:
			output = result.stdout.strip()
			if "ERROR_IN_SAMANTHA" in output:
				print(f"❌ Error en ejecución de Samantha: {output}")
				return None

			# Buscar el bloque JSON en el stdout
			match = re.search(r"(\{.*\})", output, re.DOTALL)
			if match:
				json_str = match.group(1)
				try:
					return json.loads(json_str)
				except json.JSONDecodeError as je:
					print(f"❌ Error al decodificar JSON devuelto por Samantha: {je}. Contenido: {json_str}")
			else:
				print(f"❌ No se encontró formato JSON en la salida de Samantha: {output}")
		else:
			print(f"[Samantha Subprocess Error] stdout: {result.stdout} stderr: {result.stderr}")
	except Exception as e:
		print(f"[Samantha Exception] Fallo al invocar Samantha: {e}")

	return None


def run_evaluation(args):
	device = torch.device(args.device)
	base_dir = "/home/joan/Documents/IA/frankenswarm"
	expanded_glyphs_path = os.path.join(base_dir, "configs", "expanded_glyphs.json")

	if not os.path.exists(args.model_path):
		print(f"❌ Error: No se encontró el modelo en {args.model_path}")
		return False, 0.0

	# 1. Cargar vocabulario y glifos
	with open(expanded_glyphs_path, encoding="utf-8") as f:
		vocab_data = json.load(f)
		words = vocab_data["words"]
		glyphs = np.array(vocab_data["glyphs"], dtype=np.float32)

	word_to_idx = {w: i for i, w in enumerate(words)}
	idx_to_word = dict(enumerate(words))

	# 2. Inicializar Diccionario Soberano
	dictionary = SovereignDictionary(expanded_glyphs_path)

	# 3. Inicializar y cargar modelo (en la CPU o iGPU según args.device)
	print(f"🧠 Inicializando modelo BitNet en [{device}] ({args.hidden_dim} dim, {args.num_layers} capas)...")
	model = BitNet4LayerModel(
		use_glyphs=True,
		glyph_table=glyphs,
		hidden_dim=args.hidden_dim,
		num_layers=args.num_layers,
		use_pos_embedding=True,
		is_causal=True,
		max_seq_len=128,
	).to(device)

	model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
	model.eval()

	target_age = args.target_age
	questions = AGE_QUESTIONS.get(target_age, [])
	if not questions:
		print(f"❌ Error: No hay preguntas definidas para la edad {target_age}")
		return False, 0.0

	print(f"\n📝 [Evaluación] Haciendo preguntas del hito de {target_age} años al alumno...")
	qa_pairs = []
	for qa in questions:
		raw_q = qa["question"]
		expected = dictionary.map_to_base_word(qa["expected"])

		# Extraer contenido de la pregunta
		q_match = re.match(r"^(yo|tú)\s*:\s*(.*)$", raw_q, re.IGNORECASE)
		q_content = q_match.group(2) if q_match else raw_q

		# Limpiar y mapear la pregunta
		q_words = re.findall(r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_]+", q_content.lower())
		mapped_q = [dictionary.map_to_base_word(w) for w in q_words]

		# Tokenizar
		context = [word_to_idx.get(w, 1) for w in mapped_q]

		# Pad a 128
		padded_input = list(context)
		padded_input = padded_input + [0] * (128 - len(padded_input)) if len(padded_input) < 128 else padded_input[-128:]

		x = torch.tensor([padded_input], dtype=torch.long, device=device)

		with torch.no_grad():
			logits = model(x)
			last_token_idx = len(context) - 1
			pred_token = logits[0, last_token_idx].argmax(dim=-1).item()

		pred_word = idx_to_word.get(pred_token, "<unk>")
		print(f"   💬 P: '{q_content}' | R: '{pred_word}' (Esperado: '{expected}')")
		qa_pairs.append({"question": q_content, "answer": pred_word, "expected": expected})

	# Liberar el modelo y los tensores de memoria antes de llamar a Samantha
	del model
	if args.device == "cuda":
		torch.cuda.empty_cache()

	# 4. Formular el prompt evaluador para Samantha (Mistral 7B)
	system_prompt = (
		"Eres la Profesora Samantha. Tu única tarea es comparar mecánicamente si la 'Respuesta del alumno' coincide exactamente con la 'Respuesta correcta esperada'. "
		"Si ambas palabras son iguales (por ejemplo: 'tres' y 'tres', o 'dos' y 'dos'), la calificación DEBE ser 10 y el motivo 'Respuesta correcta'. "
		"Bajo ninguna circunstancia califiques con menos de 10 si las palabras coinciden exactamente. No intentes resolver ni juzgar la pregunta. "
		"Debes responder ÚNICAMENTE con un objeto JSON válido que siga exactamente este formato:\n"
		"{\n"
		'  "calificaciones": [\n'
		'    {"pregunta": "...", "respuesta": "...", "esperada": "...", "calificacion": 10, "motivo": "..."}\n'
		"  ],\n"
		'  "puntuacion_media": 10.0,\n'
		'  "hito_superado": true\n'
		"}\n"
		"Nota: La puntuacion_media es el promedio de las calificaciones de las 5 preguntas. El hito se considera superado si la puntuación media es igual o superior a 8.0."
	)

	prompt = f"Edad de evaluación cognitiva: {target_age} años.\n\nPreguntas del examen y respuestas dadas por el alumno:\n"
	for idx, qa in enumerate(qa_pairs):
		prompt += f'{idx + 1}. Pregunta: "{qa["question"]}"\n'
		prompt += f'   Respuesta del alumno: "{qa["answer"]}"\n'
		prompt += f'   Respuesta correcta esperada: "{qa["expected"]}"\n\n'

	print(f"\n📡 Enviando examen del hito de {target_age} años a la Profesora Samantha para calificar...")
	print("--- DEBUG PROMPT ---")
	print(prompt)
	print("--- DEBUG SYSTEM PROMPT ---")
	print(system_prompt)
	print("---------------------------")
	result_json = query_samantha(prompt, system_prompt, mock=args.test_mock)

	if not result_json:
		print("❌ Fallo en la evaluación. No se pudo obtener calificación de Samantha.")
		return False, 0.0

	print("\n════════════════════════════════════════════════════════════")
	print(f"📊 REPORT DE EVALUACIÓN COGNITIVA: {target_age} AÑOS")
	print("════════════════════════════════════════════════════════════")
	for idx, cal in enumerate(result_json.get("calificaciones", [])):
		print(f'{idx + 1}. P: "{cal.get("pregunta")}"')
		print(f'   R: "{cal.get("respuesta")}" (Esperaba: "{cal.get("esperada")}")')
		print(f"   ⭐ Nota: {cal.get('calificacion')}/10 | Motivo: {cal.get('motivo')}")
		print("-" * 40)

	avg_score = result_json.get("puntuacion_media", 0.0)
	passed = result_json.get("hito_superado", False)

	status = "✅ APROBADO (Hito Superado)" if passed else "❌ SUSPENDIDO"
	print(f"🏆 PUNTUACIÓN MEDIA: {avg_score:.2f}/10 | ESTADO: {status}")
	print("════════════════════════════════════════════════════════════\n")

	return passed, avg_score


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Samantha Age Grader Evaluator")
	parser.add_argument("--model_path", type=str, required=True, help="Ruta del checkpoint a evaluar")
	parser.add_argument("--hidden_dim", type=int, default=256, help="Dimensión oculta del modelo")
	parser.add_argument("--num_layers", type=int, default=6, help="Capas del modelo")
	parser.add_argument("--target_age", type=int, required=True, choices=[4, 5, 6, 7, 8], help="Edad objetivo a evaluar")
	parser.add_argument("--device", type=str, default="cpu", help="Dispositivo para correr el modelo (default: cpu)")
	parser.add_argument("--test_mock", action="store_true", help="Simular respuestas de Samantha de forma mock")

	args = parser.parse_args()
	passed, score = run_evaluation(args)
	# Retornar exit code según el resultado
	exit(0 if passed else 1)
