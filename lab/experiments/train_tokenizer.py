import glob
import json
import os

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


def train_dialog_tokenizer(lang="es", pretrain=False, vocab_size=24000):
	# 1. Cargar corpus o diálogos
	texts = []
	
	if pretrain:
		corpus_path = f"storage/curriculum/pretrain_corpus_{lang}.txt"
		if not os.path.exists(corpus_path):
			# Fallback a sin idioma por si acaso (el español es el base)
			corpus_path = "storage/curriculum/pretrain_corpus_es.txt"
		if not os.path.exists(corpus_path):
			print(f"Error: El corpus de pre-entrenamiento no existe en {corpus_path}")
			return
		print(f"Entrenando tokenizer sobre el corpus de pre-entrenamiento: {corpus_path}...")
		with open(corpus_path, encoding="utf-8") as f:
			for line in f:
				if line.strip():
					texts.append(line.strip())
	else:
		files = glob.glob("configs/tiny_dialogues*.json")
		for fpath in files:
			if lang == "en":
				if "_en" not in fpath:
					continue
			else:
				if "_en" in fpath:
					continue
			with open(fpath, encoding="utf-8") as f:
				data = json.load(f)
				# data es una lista de diálogos, cada uno es una lista de turnos
				for dialogue in data:
					for turn in dialogue:
						# Guardamos tanto el texto limpio como el texto con prefijos
						clean_text = turn.replace("me: ", "").replace("you: ", "") if lang == "en" else turn.replace("yo: ", "").replace("tú: ", "")
						texts.append(clean_text)
						texts.append(turn)

	# 2. Configurar el tokenizador BPE
	# Usamos [UNK] para tokens desconocidos
	tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
	tokenizer.pre_tokenizer = Whitespace()
	
	# Entrenamos con vocabularios y todos los tokens especiales de conversación
	special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "yo:", "tú:", "me:", "you:"]

	trainer = BpeTrainer(
		vocab_size=vocab_size,
		special_tokens=special_tokens,
		min_frequency=2
	)
	
	# Escribir a un archivo temporal para el lector de tokenizers
	temp_path = f"storage/temp_tokenizer_input_{lang}.txt"
	os.makedirs("storage", exist_ok=True)
	with open(temp_path, "w", encoding="utf-8") as f:
		for line in texts:
			f.write(line + "\n")
			
	tokenizer.train([temp_path], trainer)
	
	# Guardar tokenizador
	os.makedirs("storage/experiments", exist_ok=True)
	out_path = "storage/experiments/conversational_tokenizer_en.json" if lang == "en" else "storage/experiments/conversational_tokenizer.json"
	tokenizer.save(out_path)
	os.remove(temp_path)
	print(f"Tokenizer {lang} entrenado y guardado en {out_path} (Vocab size: {tokenizer.get_vocab_size()})")

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--lang", type=str, default="es", choices=["es", "en"], help="Idioma a entrenar")
	parser.add_argument("--pretrain", action="store_true", help="Entrenar sobre el corpus de pre-entrenamiento")
	parser.add_argument("--vocab-size", type=int, default=None, help="Tamaño del vocabulario")
	args = parser.parse_args()
	
	vocab_size = args.vocab_size
	if vocab_size is None:
		vocab_size = 24000 if args.pretrain else 8000
		
	train_dialog_tokenizer(args.lang, args.pretrain, vocab_size)
