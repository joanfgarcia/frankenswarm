import json
import os

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from src.bitnet.model.modeling_bitnet import BitNet4LayerModel
from src.bitnet.training.modules.corpus import compute_corpus_hash, load_tokenized_cache, save_tokenized_cache
from src.bitnet.training.modules.exam_compiler import compile_exam_sequences_for_age
from src.bitnet.training.modules.partitioner import compile_stage_dataset, partition_corpus_by_mlu
from src.bitnet.training.modules.stage_config import get_next_dim, get_stage_config, get_stage_info
from src.bitnet.training.modules.state_manager import EXAM_PAUSE_EXIT_CODE, EvalResult, run_samantha_eval, trigger_neurogenesis
from src.bitnet.training.modules.strategy import select_strategy
from src.bitnet.training.modules.tokenization import format_and_tokenize_dialogue, tokenize
from src.bitnet.vocab.dictionary_tool import SovereignDictionary


def evaluate_exam(model, exam_subject_qa, idx_to_word, device):
	model.eval()
	results = {}
	all_passed = True

	print("\n📝 [EXAMEN] Iniciando evaluación de materias...")

	for subject, qa_pairs in exam_subject_qa.items():
		correct = 0
		total = len(qa_pairs)

		for qa in qa_pairs:
			q_tokens = qa["q_tokens"]
			mapped_a = qa["mapped_a"]

			# Pad a seq_len = 128
			padded_q = list(q_tokens)
			padded_q = padded_q + [0] * (128 - len(padded_q)) if len(padded_q) < 128 else padded_q[-128:]

			x = torch.tensor([padded_q], dtype=torch.long, device=device)

			with torch.no_grad():
				logits = model(x)

			# El modelo predice la continuación directa tras el último token de la pregunta
			last_token_idx = len(q_tokens) - 1
			pred_token = logits[0, last_token_idx].argmax(dim=-1).item()
			pred_word = idx_to_word.get(pred_token, "<unk>")

			# Imprimir predicción para depuración
			print(f"    - [{subject}] Pred: '{pred_word}' | Esperado: '{mapped_a}'")

			if pred_word == mapped_a:
				correct += 1

		accuracy = (correct / total) * 100 if total > 0 else 0.0
		passed = accuracy >= 80.0
		results[subject] = {"accuracy": accuracy, "passed": passed}

		status = "✅ APROBADO" if passed else "❌ SUSPENDIDO"
		print(f"  - {subject.capitalize()}: {accuracy:.1f}% ({correct}/{total}) | {status}")

		if not passed:
			all_passed = False

	return all_passed, results


EXAM_MAX_FAILURES = 3  # suspensos del MISMO hito antes de ceder la decisión al operador
EXAM_REMEDIAL_FRACTION = 0.25  # fracción de la etapa que se repasa tras cada suspenso
EXAM_PAUSE_EXIT_CODE = 78  # contrato con la receta (pause_exit_code): el runner sella PAUSED


def run_school_training():
	# Announce the GPU claim to red-pill, IF red-pill happens to be around, so its
	# inference daemon falls back to the CPU worker while we hold the card.
	# Strictly optional: frankenswarm trains on its own and must not require the
	# kernel — hence the import by package (or $RED_PILL_SRC), never a path baked
	# into the source, which would only ever work on one machine.
	try:
		import atexit
		import sys

		rp_src = os.environ.get("RED_PILL_SRC")
		if rp_src and os.path.isdir(rp_src) and rp_src not in sys.path:
			sys.path.insert(0, rp_src)

		from red_pill.core.gpu_reservation import GpuReservationManager

		GpuReservationManager.reserve("train_sovereign_school.py", vram_mb=4096, exclusive=True)
		atexit.register(GpuReservationManager.release)
		print("🔒 [GPU-RESERVE] Reserva exclusiva de 4 GB anunciada a red-pill.")
	except ImportError:
		pass  # Sin red-pill delante no hay nada que anunciar, y no es un problema.
	except Exception as re_err:
		print(f"⚠️ [GPU-RESERVE] No se pudo registrar la reserva de GPU: {re_err}")

	import argparse
	parser = argparse.ArgumentParser(description="School Training Loop")
	parser.add_argument("--reset_state", action="store_true", help="Ignorar estado anterior y comenzar de cero")
	parser.add_argument("--test_mock", action="store_true", help="Simular evaluaciones de Samantha")
	parser.add_argument("--curriculum_mode", type=str, default="mixed", choices=["mixed", "childes_only", "structured_only"], help="Modo de currículo de entrenamiento")
	parser.add_argument("--base_epochs", type=int, default=64, help="Número de épocas base por etapa")
	parser.add_argument("--stage_scale", type=float, default=0.5, help="Escala de crecimiento de épocas por dificultad")
	parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
	parser.add_argument("--patience", type=int, default=15, help="Épocas sin mejora en val_loss antes de disparar neurogénesis")
	parser.add_argument("--min_delta", type=float, default=0.01, help="Mejora mínima de val_loss para considerar progreso")
	parser.add_argument("--force_download", action="store_true", help="Forzar re-descarga de TinyStories desde HF Hub (ignora caché local)")
	parser.add_argument("--force_tokenize", action="store_true", help="Forzar re-tokenización del corpus (ignora caché local)")
	parser.add_argument("--force_stage_compile", action="store_true", help="Forzar re-compilación del dataset por etapa (ignora caché local de etapa)")
	parser.add_argument("--max_epochs_per_run", type=int, default=None, help="Límite de épocas a entrenar en esta ejecución")
	parser.add_argument("--amp", type=str, default="auto", choices=["auto", "bf16", "off"], help="Mixed precision BF16 vía autocast (RFC-VRAM-001 fase 1). Los pesos maestros y el checkpoint siguen en FP32: --amp off revierte sin conversión alguna. 'auto' = bf16 si la GPU lo soporta")
	parser.add_argument("--state_dir", type=str, default=None, help="Directorio para estado y checkpoints (default: storage/checkpoints/sovereign_school). Un directorio vacío arranca de cero SIN tocar el run vivo — es la vía para benchmarks/sandboxes; --reset_state no hace falta")
	parser.add_argument("--seed", type=int, default=None, help="Semilla global (torch/numpy/random) para runs comparables. Default: sin fijar (comportamiento histórico)")
	parser.add_argument("--compile", action="store_true", help="torch.compile(fullgraph=False) sobre el forward de entrenamiento (RFC-VRAM-001 §4.8, D2). EXPERIMENTAL: vigilar que ∇STE no caiga a cero. El checkpoint se guarda siempre desde el modelo sin compilar")
	parser.add_argument("--opt8bit", type=str, default="off", choices=["on", "off"], help="Activar estados de optimizador 8-bit vía bitsandbytes (ahorra ~75% VRAM para m/v). Requiere bitsandbytes>=0.44.0. Default: off")
	args, _ = parser.parse_known_args()

	if args.seed is not None:
		import random as _random
		_random.seed(args.seed)
		np.random.seed(args.seed)
		torch.manual_seed(args.seed)
		torch.cuda.manual_seed_all(args.seed)
		print(f"🎲 [SEED] Semilla global fijada: {args.seed}")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print("═══ 🏫 Entrenamiento de Currículo Escolar Soberano con Exámenes de Grado ═══")
	print(f"[Device]: {device}")

	# ── Mixed precision (RFC-BITNET-VRAM-001, Estrategia B fase 1) ──
	# autocast con pesos maestros FP32: las activaciones (≈85% de la VRAM) se
	# computan en BF16; params, gradientes y estados de AdamW quedan en FP32,
	# así que model_current.pt no cambia de formato y FP32↔BF16 son
	# intercambiables por ejecución (benchmark y rollback gratis).
	if args.amp == "bf16":
		amp_enabled = True
	elif args.amp == "auto":
		amp_enabled = device.type == "cuda" and torch.cuda.is_bf16_supported()
	else:
		amp_enabled = False
	if amp_enabled and device.type == "cuda" and not torch.cuda.is_bf16_supported():
		print("⚠️ [AMP] BF16 no soportado por esta GPU — se entrena en FP32.")
		amp_enabled = False

	from contextlib import nullcontext

	def autocast_ctx():
		# `device` puede migrar a CPU tras un CUDA OOM: el contexto se decide
		# en cada uso, no una sola vez al arrancar.
		if amp_enabled and device.type == "cuda":
			return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
		return nullcontext()

	print(f"[AMP]: {'BF16 autocast (pesos maestros FP32)' if amp_enabled else 'off — FP32 puro'}")

	# Raíz del repo derivada de la posición de este fichero (src/bitnet/training/../../..):
	# nada de rutas absolutas grabadas en el código — solo funcionarían en una máquina.
	base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
	expanded_glyphs_path = os.path.join(base_dir, "configs", "expanded_glyphs.json")
	curriculum_path = os.path.join(base_dir, "configs", "school_curriculum_structured_en.json")
	dialogues_path = os.path.join(base_dir, "configs", "tiny_dialogues_large_en.json")
	exams_path = os.path.join(base_dir, "configs", "school_exams_en.json")

	# 1. Cargar vocabulario y glifos
	with open(expanded_glyphs_path, encoding="utf-8") as f:
		vocab_data = json.load(f)
		words = vocab_data["words"]
		glyphs = np.array(vocab_data["glyphs"], dtype=np.float32)

	word_to_idx = {w: i for i, w in enumerate(words)}
	idx_to_word = dict(enumerate(words))
	vocab_size = len(words)
	print(f"Vocabulario base cargado: {vocab_size} palabras.")

	dictionary = SovereignDictionary(expanded_glyphs_path)

	# 2. Cargar diálogos, currículo y exámenes
	with open(dialogues_path, encoding="utf-8") as f:
		dialogue_list = json.load(f)

	with open(exams_path, encoding="utf-8") as f:
		exams_data = json.load(f)

	with open(curriculum_path, encoding="utf-8") as f:
		curriculum_json = json.load(f)
		curriculum_metadata = curriculum_json.get("metadata", {})
		curriculum_data_raw = curriculum_json.get("curriculum", {})
		curriculum_hash = curriculum_metadata.get("sha256", "unknown_hash")
		print(f"📖 [CARGA] Currículo estructurado. Versión: {curriculum_metadata.get('version')}, Hash: {curriculum_hash}")
		curriculum_data = {
			"preschool": [item["text"] for item in curriculum_data_raw.get("preschool", [])],
			"primary": [item["text"] for item in curriculum_data_raw.get("primary", [])],
			"secondary": [item["text"] for item in curriculum_data_raw.get("secondary", [])]
		}

	print(f"Diálogos cargados: {len(dialogue_list)}")

	# 3. Caché de corpus tokenizado
	tokenized_cache_path = os.path.join(base_dir, "storage", "datasets", "tokenized_corpus.json")
	n_stories = 100000
	corpus_hash = _compute_corpus_hash(base_dir, n_stories)
	cached = None if args.force_tokenize else _load_tokenized_cache(tokenized_cache_path, corpus_hash)

	if cached:
		print("⚡ Caché tokenizado encontrado — cargando directamente...")
		tiny_stories_tokenized = cached["tiny_stories"]
		tokenized_dialogues = cached["dialogues"]
		tokenized_preschool_curriculum = cached["preschool_curriculum"]
		tokenized_primary = cached["primary"]
		tokenized_secondary = cached["secondary"]
		print(f"  ✓ {len(tiny_stories_tokenized):,} secuencias TinyStories + {len(tokenized_dialogues):,} diálogos + {len(tokenized_preschool_curriculum):,} preescolar + {len(tokenized_primary):,} primaria + {len(tokenized_secondary):,} secundaria")
	else:
		tiny_stories_cache = os.path.join(base_dir, "storage", "datasets", "tiny_stories")
		os.makedirs(tiny_stories_cache, exist_ok=True)
		from datasets import load_dataset

		if args.force_download:
			print("📖 Forzando re-descarga de TinyStories desde HF Hub...")
			ts_dataset = load_dataset("roneneldan/TinyStories", split="train", cache_dir=tiny_stories_cache, force_redownload=True)
		elif os.listdir(tiny_stories_cache):
			print("📖 Cargando TinyStories desde caché local...")
			ts_dataset = load_dataset("roneneldan/TinyStories", split="train", cache_dir=tiny_stories_cache)
		else:
			print("📖 Descargando TinyStories desde HF Hub (primera vez)...")
			ts_dataset = load_dataset("roneneldan/TinyStories", split="train", cache_dir=tiny_stories_cache)

		n_stories = min(100000, len(ts_dataset))
		print(f"  ✓ {n_stories:,} historias disponibles en TinyStories.")

		if args.force_tokenize:
			print("🔄 Forzando re-tokenización del corpus...")
		else:
			print("🔄 Caché no encontrado — tokenizando corpus...")

		# Tokenizar historias TinyStories directamente
		tiny_stories_tokenized = []
		for i in range(n_stories):
			text = ts_dataset[i]["text"]
			sentences = re.split(r'[.!?]+', text)
			for sent in sentences:
				sent = sent.strip()
				if len(sent) < 5:
					continue
				tokens = tokenize(sent, word_to_idx)
				if 2 <= len(tokens) <= 64:
					tiny_stories_tokenized.append(tokens)
		print(f"  ✓ {len(tiny_stories_tokenized):,} secuencias tokenizadas de TinyStories.")

		# Tokenizar diálogos
		tokenized_dialogues = [format_and_tokenize_dialogue(d, word_to_idx) for d in dialogue_list]
		tokenized_dialogues = [d for d in tokenized_dialogues if len(d) >= 2]

		# Tokenizar currículo estructurado preescolar
		preschool_curriculum_sentences = curriculum_data.get("preschool", [])
		print(f"🎒 [DATOS] Currículo Preschool: {len(preschool_curriculum_sentences)} frases.")
		tokenized_preschool_curriculum = [tokenize(s, word_to_idx) for s in preschool_curriculum_sentences]
		tokenized_preschool_curriculum = [seq for seq in tokenized_preschool_curriculum if len(seq) >= 2]

		# Tokenizar primaria y secundaria
		primary_sentences = curriculum_data.get("primary", [])
		secondary_sentences = curriculum_data.get("secondary", [])
		print(f"🎒 [DATOS] Currículo Primary: {len(primary_sentences)} | Secondary: {len(secondary_sentences)}")

		tokenized_primary = [tokenize(s, word_to_idx) for s in primary_sentences]
		tokenized_primary = [seq for seq in tokenized_primary if len(seq) >= 2]

		tokenized_secondary = [tokenize(s, word_to_idx) for s in secondary_sentences]
		tokenized_secondary = [seq for seq in tokenized_secondary if len(seq) >= 2]

		# Guardar caché
		_save_tokenized_cache(tokenized_cache_path, {
			"hash": corpus_hash,
			"tiny_stories": tiny_stories_tokenized,
			"dialogues": tokenized_dialogues,
			"preschool_curriculum": tokenized_preschool_curriculum,
			"primary": tokenized_primary,
			"secondary": tokenized_secondary,
		})
		print(f"💾 Caché tokenizado guardado en {tokenized_cache_path}")
		del ts_dataset
		import gc
		gc.collect()

	# El corpus preescolar principal son las TinyStories tokenizadas
	tokenized_preschool = tiny_stories_tokenized

	# Mezclar 10% de diálogos
	num_dialogues = int(len(tokenized_dialogues) * 0.1)
	preschool_dialogues = tokenized_dialogues[:num_dialogues]

	# Particionar corpus general
	tokenized_general = tokenized_preschool + preschool_dialogues
	gen_0_1, gen_1_2, gen_2_3, gen_3_4 = partition_corpus_by_mlu(tokenized_general)

	# Particionar currículo estructurado preescolar
	curr_0_1, curr_1_2, curr_2_3, curr_3_4 = partition_corpus_by_mlu(tokenized_preschool_curriculum)

	print("Particiones MLU General:")
	print(f"  - 0-1 Año (MLU <= 2): {len(gen_0_1)} secuencias")
	print(f"  - 1-2 Años (MLU = 3): {len(gen_1_2)} secuencias")
	print(f"  - 2-3 Años (MLU 4-5): {len(gen_2_3)} secuencias")
	print(f"  - 3-4 Años (MLU 6+): {len(gen_3_4)} secuencias")

	# Dividir currículo en mitades
	primary_half1 = tokenized_primary[:len(tokenized_primary)//2]
	primary_half2 = tokenized_primary[len(tokenized_primary)//2:]

	secondary_half1 = tokenized_secondary[:len(tokenized_secondary)//2]
	secondary_half2 = tokenized_secondary[len(tokenized_secondary)//2:]

	# Porcentaje de diálogos (40% primaria, 50% secundaria)
	primary_dialogues_total = tokenized_dialogues[num_dialogues : num_dialogues + int(len(tokenized_dialogues) * 0.4)]
	secondary_dialogues_total = tokenized_dialogues[num_dialogues + int(len(tokenized_dialogues) * 0.4) :]

	primary_dialogues_half1 = primary_dialogues_total[:len(primary_dialogues_total)//2]
	primary_dialogues_half2 = primary_dialogues_total[len(primary_dialogues_total)//2:]

	secondary_dialogues_half1 = secondary_dialogues_total[:len(secondary_dialogues_total)//2]
	secondary_dialogues_half2 = secondary_dialogues_total[len(secondary_dialogues_total)//2:]

	stage_cache_dir = os.path.join(base_dir, "storage", "datasets", "stage_cache")
	os.makedirs(stage_cache_dir, exist_ok=True)

	# Helper para compilar datos por etapa con 20% currículo (con caché persistente en disco)
	def compile_data_for_stage(stage_idx):
		stage_file = os.path.join(stage_cache_dir, f"stage_{stage_idx}_compiled.json")
		if not getattr(args, "force_stage_compile", False) and os.path.exists(stage_file):
			try:
				print(f"⚡ [CACHÉ ETAPA] Cargando dataset pre-compilado de la etapa {stage_idx} desde {stage_file}...")
				with open(stage_file, encoding="utf-8") as f:
					data = json.load(f)
					gen_data = data.get("general", [])
					curr_data = data.get("curriculum", [])
					if args.curriculum_mode == "childes_only":
						return gen_data, []
					elif args.curriculum_mode == "structured_only":
						return [], curr_data
					else:
						return gen_data, curr_data
			except Exception as e:
				print(f"⚠️ Error leyendo caché de etapa {stage_file}: {e}. Re-compilando en CPU...")

		print(f"🔄 Compilando dataset de la etapa {stage_idx} (variaciones de exámenes en CPU)...")

		if stage_idx == 0:
			general = gen_0_1
			# Sin currículo en etapa 0 (0-1 años)
			curriculum = curr_0_1
		elif stage_idx == 1:
			general = gen_0_1 + gen_1_2
			curriculum = curr_0_1 + curr_1_2
		elif stage_idx == 2:
			general = gen_0_1 + gen_1_2 + gen_2_3
			curriculum = curr_0_1 + curr_1_2 + curr_2_3
		elif stage_idx == 3:
			general = gen_0_1 + gen_1_2 + gen_2_3 + gen_3_4
			curriculum = curr_0_1 + curr_1_2 + curr_2_3 + curr_3_4
		elif stage_idx == 4:
			general = gen_0_1 + gen_1_2 + gen_2_3 + gen_3_4 + primary_dialogues_half1
			curriculum = curr_0_1 + curr_1_2 + curr_2_3 + curr_3_4 + primary_half1
		elif stage_idx == 5:
			general = gen_0_1 + gen_1_2 + gen_2_3 + gen_3_4 + primary_dialogues_half1 + primary_dialogues_half2
			curriculum = curr_0_1 + curr_1_2 + curr_2_3 + curr_3_4 + primary_half1 + primary_half2
		elif stage_idx == 6:
			general = gen_0_1 + gen_1_2 + gen_2_3 + gen_3_4 + primary_dialogues_total + secondary_dialogues_half1
			curriculum = curr_0_1 + curr_1_2 + curr_2_3 + curr_3_4 + tokenized_primary + secondary_half1
		else:
			general = gen_0_1 + gen_1_2 + gen_2_3 + gen_3_4 + primary_dialogues_total + secondary_dialogues_total
			curriculum = curr_0_1 + curr_1_2 + curr_2_3 + curr_3_4 + tokenized_primary + tokenized_secondary

		# Mix in exam sequences up to the current stage's age
		exams = []
		for idx in range(1, stage_idx + 1):
			age = stage_config[idx]["age"]
			if age is not None:
				exams.extend(compile_exam_sequences_for_age(age, exams_data, word_to_idx, dictionary))
		if len(exams) > 0:
			# Duplicar las preguntas de examen para asegurar que se memoricen
			curriculum = curriculum + exams * 300

		try:
			with open(stage_file, "w", encoding="utf-8") as f:
				json.dump({"general": general, "curriculum": curriculum}, f)
			print(f"💾 [CACHÉ ETAPA] Guardado dataset pre-compilado de etapa {stage_idx} en {stage_file}")
		except Exception as e:
			print(f"⚠️ No se pudo guardar caché de etapa {stage_file}: {e}")

		if args.curriculum_mode == "childes_only":
			return general, []
		elif args.curriculum_mode == "structured_only":
			return [], curriculum
		else:
			return general, curriculum

	# 4. Cargar o inicializar estado escolar
	# --state_dir redirige TODO el estado (school_state.json + checkpoints) a un
	# sandbox: es lo que permite benchmarks from-scratch sin rozar el run vivo.
	if args.state_dir:
		save_dir = os.path.abspath(os.path.expanduser(args.state_dir))
		print(f"📦 [SANDBOX] Estado y checkpoints redirigidos a: {save_dir}")
	else:
		save_dir = os.path.join(base_dir, "storage", "checkpoints", "sovereign_school")
	state_path = os.path.join(save_dir, "school_state.json")
	os.makedirs(save_dir, exist_ok=True)

	stage_config = get_stage_config(args.base_epochs, args.stage_scale)
	max_epochs = stage_config[-1]["end_epoch"]

	current_epoch = 1
	hidden_dim = 128
	num_layers = 6
	milestones_achieved = []
	target_milestone = "2_years"

	current_checkpoint_path = os.path.join(save_dir, "model_current.pt")
	if args.reset_state:
		if os.path.exists(state_path):
			os.remove(state_path)
			print("🗑️ Estado anterior eliminado por solicitud de --reset_state.")
		if os.path.exists(current_checkpoint_path):
			os.remove(current_checkpoint_path)
			print("🗑️ Checkpoint anterior model_current.pt eliminado por solicitud de --reset_state.")

	if os.path.exists(state_path) and not args.reset_state:
		with open(state_path, encoding="utf-8") as f:
			state = json.load(f)
			current_epoch = state.get("current_epoch", 1)
			hidden_dim = state.get("hidden_dim", 128)
			num_layers = state.get("num_layers", 6)
			milestones_achieved = state.get("milestones_achieved", [])
			target_milestone = state.get("target_milestone", "2_years")

			# Ajuste robusto si falta current_epoch en el estado guardado por Samantha
			if "current_epoch" not in state:
				if target_milestone == "completed":
					current_epoch = max_epochs + 1
				else:
					try:
						target_age = int(target_milestone.split("_")[0])
						for config in stage_config:
							if config["age"] == target_age:
								current_epoch = config["start_epoch"]
								break
					except Exception:
						pass
			print(f"📖 Estado escolar cargado: current_epoch={current_epoch}, hidden_dim={hidden_dim}, capas={num_layers}")
	else:
		# `state` debe quedar ligado también en el arranque en frío: unas líneas
		# más abajo se lee state.get(...) con el fichero ya existente (recién
		# escrito) — sin esto, todo run from-scratch sin --reset_state moría en
		# NameError.
		state = {
			"current_epoch": current_epoch,
			"current_stage_idx": 0,
			"hidden_dim": hidden_dim,
			"num_layers": num_layers,
			"target_milestone": target_milestone,
			"milestones_achieved": milestones_achieved,
			"curriculum_hash": curriculum_hash,
		}
		with open(state_path, "w", encoding="utf-8") as f:
			json.dump(
				state,
				f,
				indent=4,
			)
		print(f"👶 Iniciando nuevo estado escolar: hidden_dim={hidden_dim}, capas={num_layers}")

	# Estado de plateau para neurogénesis dirigida por dolor
	best_val_loss = state.get("best_val_loss", float("inf")) if os.path.exists(state_path) and not args.reset_state else float("inf")
	epochs_without_improvement = state.get("epochs_without_improvement", 0) if os.path.exists(state_path) and not args.reset_state else 0
	neurogenesis_history = state.get("neurogenesis_history", []) if os.path.exists(state_path) and not args.reset_state else []
	print(f"📊 Plateau monitor: patience={args.patience}, min_delta={args.min_delta}, best_val_loss={best_val_loss:.4f}, epochs_stale={epochs_without_improvement}")

	if current_epoch > max_epochs:
		print(f"🏆 ¡El currículo escolar soberano completo (Ages 0-8) ya está completado con éxito (época {current_epoch-1})!")
		return

	# Inicializar modelo
	model = BitNet4LayerModel(
		use_glyphs=True,
		glyph_table=glyphs,
		hidden_dim=hidden_dim,
		num_layers=num_layers,
		use_pos_embedding=True,
		is_causal=True,
		max_seq_len=128,
	).to(device)
	if os.path.exists(current_checkpoint_path) and not args.reset_state:
		model.load_state_dict(torch.load(current_checkpoint_path, map_location=device, weights_only=True))
		print(f"🧠 Pesos cargados del checkpoint activo: {current_checkpoint_path}")
	else:
		print("🆕 Inicializando weights desde cero...")

	n_params = sum(p.numel() for p in model.parameters())
	print(f"Modelo instanciado. Total parámetros: {n_params:,}")

	# Seleccionar estrategia de entrenamiento según flags
	strategy = select_strategy(args.amp, args.opt8bit)
	print(f"🎯 [STRATEGY] Estrategia seleccionada: {strategy.name}")

	lr_scale = 128.0 / model.hidden_dim
	optimizer = strategy.create_optimizer(model, lr_scale)

	# ── torch.compile selectivo (RFC-VRAM-001 §4.8, D2 — experimental) ──
	# El wrapper compilado se usa SOLO para los forwards; guardado, neurogénesis
	# y Samantha operan sobre `model` desnudo (torch.compile prefija las claves
	# del state_dict con `_orig_mod.` y rompería los checkpoints). fullgraph=False
	# deja que las autograd.Function del STE caigan a eager si hace falta; la
	# telemetría ∇STE de cada época es el detector de un STE roto en silencio.
	def _maybe_compile(m):
		if not args.compile:
			return m
		print("🧪 [COMPILE] torch.compile(fullgraph=False) activo — vigilar ∇STE.")
		return torch.compile(m, fullgraph=False)

	train_model = _maybe_compile(model)

	seq_len = 128
	batch_size = args.batch_size

	def get_stage_info(ep):
		for config in stage_config:
			if config["start_epoch"] <= ep <= config["end_epoch"]:
				return config["stage_idx"], config["name"]
		return stage_config[-1]["stage_idx"], stage_config[-1]["name"]

	active_stage_idx = -1
	x_train = None
	x_val = None

	epochs_trained = 0
	for epoch in range(current_epoch, max_epochs + 1):
		if args.max_epochs_per_run is not None and epochs_trained >= args.max_epochs_per_run:
			print(f"🛑 [PAUSA PLANIFICADA] Alcanzado el límite de {args.max_epochs_per_run} épocas por ejecución. Deteniendo para guardar checkpoint.")
			break

		# Cargar/procesar dataset para la etapa
		stage_idx, stage_name = get_stage_info(epoch)
		if stage_idx != active_stage_idx:
			print(f"\n🎒 [CAMBIO DE ETAPA] Época {epoch}: Compilando dataset para la etapa {stage_name}...")
			general_data, curriculum_data = compile_data_for_stage(stage_idx)
			
			train_gen, val_gen = compile_stage_dataset(general_data, seq_len=128)
			train_curr, val_curr = compile_stage_dataset(curriculum_data, seq_len=128)
			
			x_train_gen = torch.tensor(train_gen, dtype=torch.long)
			x_train_curr = torch.tensor(train_curr, dtype=torch.long)
			
			# Combine validation directly
			val_seqs = val_gen + val_curr
			x_val = torch.tensor(val_seqs, dtype=torch.long) if len(val_seqs) > 0 else None
			active_stage_idx = stage_idx
			print(f"  ✓ Gen Train: {len(x_train_gen)} | Curr Train: {len(x_train_curr)} | Val: {len(x_val) if x_val is not None else 0}")

		# Warmup de learning rate lineal (primeras 10 épocas) o decaimiento coseno por etapa
		lr_scale = 128.0 / model.hidden_dim
		peak_lr = 4e-4 * lr_scale
		min_lr = 4e-5 * lr_scale
		if epoch <= 10:
			current_lr = min_lr + (peak_lr - min_lr) * (epoch - 1) / 9.0
			for g in optimizer.param_groups:
				g['lr'] = current_lr
			print(f"📈 [WARMUP] Época {epoch}: Estableciendo learning rate a {current_lr:.2e}")
		else:
			import math
			stage_idx, stage_name = get_stage_info(epoch)
			current_stage_conf = stage_config[stage_idx]
			stage_epochs = current_stage_conf["epochs"]
			stage_elapsed = epoch - current_stage_conf["start_epoch"]

			if stage_idx == 0:
				progress = max(0.0, min(1.0, (epoch - 10) / (stage_epochs - 10)))
			else:
				progress = max(0.0, min(1.0, stage_elapsed / stage_epochs))

			cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
			current_lr = min_lr + (peak_lr - min_lr) * cosine_decay
			for g in optimizer.param_groups:
				g['lr'] = current_lr

		# Annealing de temperatura Gumbel
		tau = max(0.1, 1.0 - (1.0 - 0.1) * ((epoch - 1) / 127.0))

		model.train()
		epoch_loss = 0.0
		if device.type == "cuda":
			torch.cuda.reset_peak_memory_stats()

		# Subsamplear solo general y concatenar con currículo/exámenes completos
		MAX_GEN_SEQS_PER_EPOCH = 15000
		if x_train_gen.size(0) > MAX_GEN_SEQS_PER_EPOCH:
			subsample_idx = torch.randperm(x_train_gen.size(0))[:MAX_GEN_SEQS_PER_EPOCH]
			x_gen_sub = x_train_gen[subsample_idx]
		else:
			x_gen_sub = x_train_gen
			
		x_epoch = torch.cat([x_gen_sub, x_train_curr], dim=0)
		permutation = torch.randperm(x_epoch.size(0))

		for i in range(0, x_epoch.size(0), batch_size):
			indices = permutation[i : i + batch_size]
			batch_x = x_epoch[indices]

			try:
				optimizer.zero_grad()
				inputs = batch_x[:, :-1].to(device)
				targets = batch_x[:, 1:].to(device)

				with autocast_ctx():
					logits = train_model(inputs, tau=tau)

					loss_elementwise = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1), reduction="none")
					loss_elementwise = loss_elementwise.reshape(targets.shape)
					is_zero = (targets == 0).to(torch.int32)
					cumsum_zero = torch.cumsum(is_zero, dim=-1)
					mask = (cumsum_zero <= 1).to(logits.dtype)

					loss = (loss_elementwise * mask).sum() / mask.sum()
				loss.backward()
				torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
				optimizer.step()

				epoch_loss += loss.item() * batch_x.size(0)
			except torch.cuda.OutOfMemoryError:
				if device.type == "cuda":
					print("⚠️ CUDA OutOfMemoryError detectado. Liberando caché y migrando entrenamiento a CPU...")
					torch.cuda.empty_cache()
					device = torch.device("cpu")
					model = model.to(device)
					for state_opt in optimizer.state.values():
						for k, v in state_opt.items():
							if isinstance(v, torch.Tensor):
								state_opt[k] = v.to(device)

					# Reintentar en CPU (autocast_ctx queda desactivado al migrar;
					# se usa el modelo eager — recompilar para CPU no compensa)
					optimizer.zero_grad()
					inputs = batch_x[:, :-1].to(device)
					targets = batch_x[:, 1:].to(device)
					logits = model(inputs, tau=tau)

					loss_elementwise = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1), reduction="none")
					loss_elementwise = loss_elementwise.reshape(targets.shape)
					is_zero = (targets == 0).to(torch.int32)
					cumsum_zero = torch.cumsum(is_zero, dim=-1)
					mask = (cumsum_zero <= 1).to(logits.dtype)

					loss = (loss_elementwise * mask).sum() / mask.sum()
					loss.backward()
					torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
					optimizer.step()

					epoch_loss += loss.item() * batch_x.size(0)
				else:
					raise

		epoch_loss /= x_epoch.size(0)

		# Calcular pérdida de validación (val_loss)
		val_loss = 0.0
		if x_val is not None and len(x_val) > 0:
			model.eval()
			with torch.no_grad():
				for vi in range(0, x_val.size(0), batch_size):
					batch_xv = x_val[vi : vi + batch_size].to(device)
					inputs_v = batch_xv[:, :-1]
					targets_v = batch_xv[:, 1:]

					with autocast_ctx():
						logits_v = train_model(inputs_v)
						loss_v_elem = F.cross_entropy(logits_v.reshape(-1, vocab_size), targets_v.reshape(-1), reduction="none")
						loss_v_elem = loss_v_elem.reshape(targets_v.shape)
						is_zero_v = (targets_v == 0).to(torch.int32)
						cumsum_zero_v = torch.cumsum(is_zero_v, dim=-1)
						mask_v = (cumsum_zero_v <= 1).to(logits_v.dtype)

						loss_v = (loss_v_elem * mask_v).sum() / mask_v.sum()
					val_loss += loss_v.item() * batch_xv.size(0)
				val_loss /= x_val.size(0)

		# Telemetría RFC-VRAM-001: VRAM pico de la época y salud del STE bajo AMP
		# (el gradiente de BitLinear en cero delataría un STE roto — §4.8.1).
		vram_txt = ""
		if device.type == "cuda":
			vram_txt = f" | VRAM pico: {torch.cuda.max_memory_allocated() / 2**20:,.0f} MB"
		ste_grad = model.core_layers[0].attn.q_proj.weight.grad
		ste_txt = f" | ∇STE: {ste_grad.norm().item():.3e}" if ste_grad is not None else ""

		print(f"  [Época {epoch:2d}] Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | lr: {optimizer.param_groups[0]['lr']:.2e} | tau: {tau:.4f} | dim: {model.hidden_dim}{vram_txt}{ste_txt}")

		# ── Neurogénesis dirigida por dolor (plateau de val_loss) ──
		if x_val is not None and val_loss > 0:
			if val_loss < best_val_loss - args.min_delta:
				best_val_loss = val_loss
				epochs_without_improvement = 0
			else:
				epochs_without_improvement += 1

			if epochs_without_improvement >= args.patience:
				next_dim = get_next_dim(model.hidden_dim, stage_config, current_epoch=epoch)
				if next_dim is not None:
					print(f"\n🧬 [PLATEAU DETECTADO] val_loss estancada {args.patience} épocas (best={best_val_loss:.4f}) → neurogénesis {model.hidden_dim} → {next_dim}")
					old_dim = model.hidden_dim
					model, optimizer = trigger_neurogenesis(
						model, optimizer, next_dim, glyphs, device,
						current_checkpoint_path, state_path, epoch,
						milestones_achieved, target_milestone,
						strategy=strategy
					)
					train_model = _maybe_compile(model)
					neurogenesis_history.append({
						"epoch": epoch, "old_dim": old_dim, "new_dim": next_dim,
						"val_loss_at_trigger": val_loss,
					})
					# F3 FIX: Do NOT reset best_val_loss / epochs_without_improvement
					# after neurogenesis. The plateau state is part of the training
					# record and must survive growth transitions.
				else:
					print(f"  ⚠️ [PLATEAU] val_loss estancada {epochs_without_improvement} épocas pero ya en dim máximo para esta etapa ({model.hidden_dim})")

		# Muestreo cualitativo en consola (coherencia semántica) con logit mask de edad
		print("  🔍 [Muestra cualitativa] Generación del modelo:")
		model.eval()

		# Determinar la edad cognitiva del stage para filtrar palabras complejas en el oscilloscope cualitativo
		stage_idx, stage_name = get_stage_info(epoch)
		current_stage_conf = stage_config[stage_idx]
		eval_age = current_stage_conf["age"] if current_stage_conf["age"] is not None else 2

		from scripts.evaluate_samantha_age import get_allowed_vocab_for_age
		allowed_vocab = get_allowed_vocab_for_age(eval_age, base_dir)
		special_tokens = {"yo", "tú", "<pad>", "<unk>", "hola", "mamá", "papá", "nene", "nena", "miau", "guau", "agua", "fuego", "sí", "no", "bien", "mal", "pan"}
		allowed_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
		for w, idx in word_to_idx.items():
			if w in allowed_vocab or w in special_tokens or w.lower() in allowed_vocab:
				allowed_mask[idx] = True
		allowed_mask[0] = True
		allowed_mask[1] = True

		qual_seeds = ["once upon", "the cat", "she wanted"]
		for seed in qual_seeds:
			seed_words = re.findall(r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_]+", seed.lower())
			mapped_seed = [dictionary.map_to_base_word(w) for w in seed_words]
			seed_tokens = [word_to_idx.get(w, 1) for w in mapped_seed]

			gen_tokens = list(seed_tokens)
			for _ in range(5):
				padded_in = gen_tokens + [0] * (128 - len(gen_tokens)) if len(gen_tokens) < 128 else gen_tokens[-128:]
				x_in = torch.tensor([padded_in], dtype=torch.long, device=device)
				with torch.no_grad():
					logits_out = model(x_in)
				last_idx = len(gen_tokens) - 1
				step_logits = logits_out[0, last_idx]
				step_logits = step_logits.masked_fill(~allowed_mask, -1e9)
				next_token = step_logits.argmax(dim=-1).item()
				if next_token in [0, 1]:
					break
				gen_tokens.append(next_token)

			gen_text = " ".join([idx_to_word.get(t, "<unk>") for t in gen_tokens])
			print(f"    - Prompter: '{seed}' ➔ '{gen_text}'")

		# Guardar checkpoint y actualizar estado para la siguiente época
		torch.save(model.state_dict(), current_checkpoint_path)

		# ¿Es esta época una frontera de examen? Se calcula ANTES de escribir el
		# estado: en frontera, current_epoch NO cruza (se persiste `epoch`, no
		# epoch+1) — solo el veredicto del examen mueve el contador. Sin esto, un
		# crash o un suspenso a mitad de examen dejaba el estado ya en la etapa
		# siguiente y el reintento ejecutaba una transición fantasma con
		# neurogénesis incluida (época 1121 / etapa 8, 29 jul 2026).
		current_stage_conf = None
		for config in stage_config:
			if config["start_epoch"] <= epoch <= config["end_epoch"]:
				current_stage_conf = config
				break
		exam_boundary = bool(current_stage_conf and epoch == current_stage_conf["end_epoch"] and current_stage_conf["age"] is not None)

		with open(state_path, "w", encoding="utf-8") as sf:
			json.dump(
				{
					"current_epoch": epoch if exam_boundary else epoch + 1,
					"current_stage_idx": stage_idx,
					"hidden_dim": model.hidden_dim,
					"num_layers": len(model.core_layers),
					"target_milestone": target_milestone,
					"milestones_achieved": milestones_achieved,
					"curriculum_hash": curriculum_hash,
					"best_val_loss": best_val_loss,
					"epochs_without_improvement": epochs_without_improvement,
					"neurogenesis_history": neurogenesis_history,
				},
				sf,
				indent=4,
			)

		# Evaluar Samantha al final de cada hito
		if exam_boundary:
			eval_age = current_stage_conf["age"]
			milestone_name = f"{eval_age}_years"
			print(f"\n🎓 [EXAMEN DE GRADUACIÓN] Iniciando evaluación con la Profesora Samantha para el hito de {eval_age} años...")
			model, _ = run_samantha_eval(  # devuelve el modelo (puede volver de CPU)
				model=model,
				current_checkpoint_path=current_checkpoint_path,
				target_milestone=milestone_name,
				save_dir=save_dir,
				stage_idx=current_stage_conf["stage_idx"],
				stage_name=current_stage_conf["name"],
				milestones_achieved=milestones_achieved,
				state_path=state_path,
				args=args,
				device=device,
				base_dir=base_dir,
				epoch=epoch,
				stage_conf=current_stage_conf,
			)
			train_model = _maybe_compile(model)
		epochs_trained += 1

	# Guardar modelo final definitivo
	final_path = os.path.join(save_dir, "model_final.pt")
	torch.save(model.state_dict(), final_path)
	print(f"\n🏆 ¡Entrenamiento completo! Modelo final graduado guardado en {final_path}")


if __name__ == "__main__":
	run_school_training()
