"""Samantha Story Factory — controlled-vocabulary synthetic corpus for School v3.

TinyStories lesson applied to Spanish: the corpus, not the architecture, was Bit's
bottleneck (~864K tokens total, >60% template-monotone). This factory turns Samantha
from examiner into data generator: batches of vocabulary-graded mini-stories and
dialogues, post-filtered against the clean vocabulary, deduplicated, written as JSONL.

Designed to run unattended under the Sovereign Wake Gate (short batches, resumable,
zero interactivity). Example:

	PYTHONPATH=. .venv/bin/python scripts/samantha_story_factory.py \
		--stage preschool --count 500 --mock

Volume plan: preschool 2M tokens, primary 8M, secondary 20M (adjust per hardware).
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
from datetime import UTC, datetime

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

STAGE_SPECS = {
	"preschool": {
		"age": "3 a 5 años", "sentences": "2 a 4 frases muy cortas (3-8 palabras)",
		"topics": ["la familia", "los animales", "la comida", "jugar en el parque", "el sol y la lluvia", "dormir y soñar", "el cuerpo", "los colores"],
	},
	"primary": {
		"age": "6 a 8 años", "sentences": "4 a 8 frases cortas (5-12 palabras)",
		"topics": ["la escuela", "los amigos", "un pequeño problema y su solución", "ayudar en casa", "un animal perdido", "el miedo y la valentía", "compartir", "un día de lluvia"],
	},
	"secondary": {
		"age": "9 a 12 años", "sentences": "6 a 12 frases (8-16 palabras)",
		"topics": ["una aventura en el bosque", "un malentendido entre amigos", "aprender algo difícil", "un viaje", "decir la verdad", "un invento", "cuidar de alguien", "una promesa"],
	},
}

SYSTEM_PROMPT = (
	"Eres una maestra de escuela española experta en lecturas graduadas. Generas mini-historias "
	"en español sencillo para niños. Reglas ESTRICTAS: solo vocabulario cotidiano infantil, sin "
	"nombres de marcas, sin violencia, sin cine ni televisión, sin tecnicismos. Frases simples "
	"(sujeto-verbo-objeto), presente o pasado simple. Responde SOLO con las historias, una por "
	"línea, sin numerar y sin comentarios."
)


def words_of(text):
	return re.findall(r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ\-]+", text.lower())


def load_clean_vocab():
	path = os.path.join(base_dir, "configs/clean_vocabulary_words.json")
	if not os.path.exists(path):
		return None
	with open(path, encoding="utf-8") as f:
		return set(json.load(f)["words"])


def build_prompt(stage_spec, rng):
	topic = rng.choice(stage_spec["topics"])
	return (
		f"Escribe 10 mini-historias distintas sobre '{topic}' para niños de {stage_spec['age']}. "
		f"Cada historia: {stage_spec['sentences']}. Personajes con nombres españoles comunes. "
		f"Una historia por línea."
	)


def mock_generate(prompt):
	seeds = [
		"el gato de ana tiene hambre. ana le da leche. el gato está contento.",
		"luis juega en el parque. ve un perro pequeño. el perro quiere jugar también.",
		"mamá hace pan en casa. huele muy bien. toda la familia come pan caliente.",
	]
	return "\n".join(seeds)


def generate_batch(prompt, mock):
	if mock:
		return mock_generate(prompt)
	from red_pill.inference import samantha_on_demand
	res = samantha_on_demand.invoke(prompt, system_prompt=SYSTEM_PROMPT, max_tokens=900, temperature=0.9)
	return res if isinstance(res, str) else res.get("text", "")


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--stage", choices=list(STAGE_SPECS), required=True)
	ap.add_argument("--count", type=int, default=100, help="historias objetivo en esta ejecución")
	ap.add_argument("--max-oov", type=float, default=0.02, help="tasa máxima de palabras fuera del vocabulario limpio")
	ap.add_argument("--seed", type=int, default=770)
	ap.add_argument("--mock", action="store_true", help="sin LLM, para probar el pipeline")
	args = ap.parse_args()

	spec = STAGE_SPECS[args.stage]
	vocab = load_clean_vocab()
	rng = random.Random(args.seed)

	out_dir = os.path.join(base_dir, "storage/curriculum/factory")
	os.makedirs(out_dir, exist_ok=True)
	out_path = os.path.join(out_dir, f"{args.stage}.jsonl")

	seen = set()
	if os.path.exists(out_path):
		with open(out_path, encoding="utf-8") as f:
			for line in f:
				try:
					seen.add(json.loads(line)["hash"])
				except (json.JSONDecodeError, KeyError):
					continue
	print(f"Fábrica: stage={args.stage}, objetivo={args.count}, ya en disco={len(seen)}, vocab_limpio={'sí' if vocab else 'NO (sin filtro OOV)'}")

	accepted, rejected_oov, rejected_dup, batches = 0, 0, 0, 0
	with open(out_path, "a", encoding="utf-8") as out:
		while accepted < args.count and batches < args.count:
			batches += 1
			raw = generate_batch(build_prompt(spec, rng), args.mock)
			for line in raw.splitlines():
				text = line.strip().lower()
				# Limpiar numeración inicial (ej. "1. ", "1.- ", "2) ")
				text = re.sub(r'^\d+[\.\-\)\s]+', '', text).strip()
				if len(words_of(text)) < 5:
					continue
				h = hashlib.sha256(re.sub(r"\W+", "", text).encode()).hexdigest()[:16]
				if h in seen:
					rejected_dup += 1
					continue
				if vocab is not None:
					ws = words_of(text)
					oov = sum(1 for w in ws if w not in vocab) / len(ws)
					if oov > args.max_oov:
						rejected_oov += 1
						continue
				seen.add(h)
				accepted += 1
				out.write(json.dumps({
					"text": text, "stage": args.stage, "hash": h,
					"generated": datetime.now(UTC).astimezone().isoformat(),
					"generator": "mock" if args.mock else "samantha_on_demand",
				}, ensure_ascii=False) + "\n")
				if accepted >= args.count:
					break
			if args.mock and batches >= 1:
				break

	print(f"✅ aceptadas={accepted} | duplicadas={rejected_dup} | OOV>{args.max_oov * 100:.0f}%={rejected_oov} | fichero={out_path}")
	print("Nota: tras ampliar la fábrica, re-ejecutar rebuild_clean_vocabulary.py para incorporar palabras nuevas y re-derivar glifos.")


if __name__ == "__main__":
	main()
