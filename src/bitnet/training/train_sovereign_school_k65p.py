"""Entrenador Soberano de Bit sobre K-65P — Escuela v3, protocolo adaptativo DL-006.

Brazos experimentales sobre el MISMO corpus, escuela y exámenes:
- --embedding glyph    → Bit v2 (embedding composicional de primos, GlyphEmbedding)
- --embedding standard → Bit v0 (tabla one-hot congelada + proyecciones entrenables,
                         matemáticamente equivalente a nn.Embedding + cabeza lineal:
                         el enfoque estándar actual)

Protocolo adaptativo (DL-006 — la edad se mide en hitos superados, no en épocas):
- Cada etapa entrena HASTA PLATEAU (patience épocas sin mejora de val_loss) o hasta
  el tope de seguridad (max_stage_epochs). El calendario fijo de v1 (1408 épocas)
  queda retirado para este brazo: sobreentrenaba por construcción (DL-005 §evidencia).
- Al plateau, el examen de hito se hace sobre el MEJOR checkpoint de la etapa
  (best-val), no sobre el último (que ya derrapó).
- Aprobado → avanza de etapa DESDE el mejor checkpoint. Suspenso → neurogénesis
  como remediación (si el techo de dim de la etapa lo permite) y repite; sin techo
  disponible → pausa rc=78 para revisión del operador.
- La neurogénesis ya NO se dispara por aburrimiento a mitad de etapa: solo como
  respuesta a un examen suspendido. (En v2 el plateau significa "corpus digerido",
  no "necesito más neuronas"; dárselas solo aceleraba la memorización.)

Umbrales de examen congelados (pre-registro DL-004, intactos):
	EXAM_MIN_GEN_VALID = 0.60   (generaciones greedy válidas según k65p.validator)
	EXAM_BIGRAM_MARGIN = 0.10   (val_loss ≤ bigrama_loss − margen)
La precisión next-token se reporta pero no umbraliza (techo estructural en corpus
composicional; el oráculo del generador marca 2.09 nats/token en preescolar).
"""

import argparse
import atexit
import json
import math
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

# Add project root and k65p src to sys.path
base_dir = Path(__file__).resolve().parents[3]
sys.path.append(str(base_dir))

k65p_src = base_dir.parent / "k65p" / "src"
if k65p_src.exists():
	sys.path.append(str(k65p_src))

try:
	from k65p.lexicon import load_lexicon, molecule_names
	from k65p.primes import N_PRIMES, PRIMES
	from k65p.validator import is_valid
except ImportError:
	print("ERROR: Módulo k65p no encontrado. Ejecutar con PYTHONPATH=.:../k65p/src")
	sys.exit(1)

from src.bitnet.growth.net2net import net2wider_model
from src.bitnet.model.modeling_bitnet import BitNet4LayerModel
from src.bitnet.training.modules.stage_config import get_stage_config

DEFAULT_STATE_DIR = base_dir / "storage" / "checkpoints" / "sovereign_school_k65p"

EXAM_PAUSE_EXIT_CODE = 78
EXAM_MIN_GEN_VALID = 0.60
EXAM_BIGRAM_MARGIN = 0.10
EXAM_N_PROMPTS = 25
MAX_LEN = 48

# Solo nombres, edades y techos de dim; los rangos de épocas del calendario v1 se ignoran (DL-006).
STAGE_CONFIG = get_stage_config(base_epochs=64, stage_scale=0.5)
ALL_DIMS = sorted({c["dim"] for c in STAGE_CONFIG})

# Firmas ternarias (trit −1) para los tokens estructurales (DL-005). Sin esto los
# seis especiales comparten el glifo todo-ceros → embedding CERO idéntico → '[' y
# ']' indistinguibles: sintaxis inaprendible por construcción. <pad> conserva el
# glifo cero a propósito: jamás es target (ignore_index) y la generación para por
# balance de corchetes, no por emitir pad.
STRUCT_TRITS = {"<unk>": 4, "<stop>": 3, "[": 0, "]": 1, "G": 2}


def build_k65p_vocab_and_glyphs(lang: str = "es") -> tuple[dict[str, int], dict[int, str], np.ndarray]:
	lexicon = load_lexicon()
	tokens = ["<pad>", "<unk>", "<stop>", "[", "]", "G"]
	glyphs_list = []
	for t in tokens:
		g = [0] * N_PRIMES
		if t in STRUCT_TRITS:
			g[STRUCT_TRITS[t]] = -1
		glyphs_list.append(g)

	# Primos en forma canónica + simbólica (sin aliasing: cada primo tiene su
	# propio glifo; añadir el nombre simbólico no reintroduce el bug DL-005
	# porque no hay ambigüedad entre primos distintos). La forma simbólica se
	# añade primero → el token resultante es el nombre legible.
	for pid, row in enumerate(PRIMES):
		g_sym = [0] * N_PRIMES
		g_sym[pid] = 1
		if lang == "en":
			tokens.append(row[0].lower())
			glyphs_list.append(g_sym)
		tokens.append(str(pid))
		glyphs_list.append(g_sym)

	for name, entry in sorted(lexicon.items()):
		mol_name = name.casefold() if lang == "es" else entry.get("en", name).casefold()
		tokens.append(mol_name)
		glyphs_list.append(entry["glyph"])

	vocab_words = []
	final_glyphs = []
	seen = set()
	for t, g in zip(tokens, glyphs_list, strict=False):
		if t not in seen:
			seen.add(t)
			vocab_words.append(t)
			final_glyphs.append(g)

	word_to_idx = {w: i for i, w in enumerate(vocab_words)}
	idx_to_word = dict(enumerate(vocab_words))
	glyph_table = np.array(final_glyphs, dtype=np.float32)

	return word_to_idx, idx_to_word, glyph_table


def build_model(embedding_mode: str, hidden_dim: int, glyph_table: np.ndarray, num_layers: int = 6) -> BitNet4LayerModel:
	"""Fábrica de los brazos experimentales: v2 (glyph) y v0 (standard)."""
	if embedding_mode == "glyph":
		return BitNet4LayerModel(
			use_glyphs=True,
			glyph_table=glyph_table,
			hidden_dim=hidden_dim,
			num_layers=num_layers,
			use_pos_embedding=True,
			is_causal=True,
			max_seq_len=128,
		)
	if embedding_mode == "standard":
		# Tabla one-hot CONGELADA (buffer) + inbound/outbound entrenables ≡ embedding
		# estándar entrenado desde cero (cada token = una columna libre de inbound_proj)
		# con cabeza de salida lineal. La doctrina EXP_005b (tabla congelada) se cumple
		# trivialmente: la identidad no tiene nada que aprender.
		vocab_size = glyph_table.shape[0]
		return BitNet4LayerModel(
			vocab_embeddings=np.eye(vocab_size, dtype=np.float32),
			use_glyphs=False,
			hidden_dim=hidden_dim,
			num_layers=num_layers,
			use_pos_embedding=True,
			is_causal=True,
			max_seq_len=128,
		)
	raise ValueError(f"embedding_mode desconocido: {embedding_mode!r}")


def tokenize_k65p(text: str, word_to_idx: dict[str, int], max_len: int = MAX_LEN) -> list[int]:
	# OJO (DL-004): sin alternativa `G|g` en el regex — partía "grupo" en "g"+"rupo".
	# Lookup exacto antes que casefold: "G" (marcador de grupo) vive en mayúscula en el vocab.
	raw_tokens = re.findall(r"\[|\]|-?\d+|[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_]+", text)
	ids = []
	for t in raw_tokens:
		idx = word_to_idx.get(t, word_to_idx.get(t.casefold(), word_to_idx["<unk>"]))
		ids.append(idx)

	ids = ids + [word_to_idx["<pad>"]] * (max_len - len(ids)) if len(ids) < max_len else ids[:max_len]
	return ids


def detokenize_k65p(ids: list[int], idx_to_word: dict[int, str]) -> str:
	"""IDs → expresión K-65P textual (los corchetes recuperan su adherencia)."""
	words = []
	for i in ids:
		w = idx_to_word.get(i, "")
		if w in {"<pad>", "<stop>"}:
			break
		words.append(w)
	text = " ".join(words)
	return text.replace("[ ", "[").replace(" ]", "]")


def stage_group_for(stage_name: str) -> str:
	if any(tag in stage_name for tag in ("0-1", "1-2", "2-3", "3-4")):
		return "preschool"
	if "primary" in stage_name:
		return "primary"
	return "secondary"


def load_dataset_stage(
	stage_name: str,
	word_to_idx: dict[str, int],
	max_len: int = MAX_LEN,
	seed: int = 770,
	corpus: str = "factory_k65p",
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
	"""Carga el corpus de la etapa. Split train/val 85/15 BARAJADO (semilla fija) y sin duplicados."""
	ds_name = stage_group_for(stage_name)
	jsonl_path = base_dir / "storage" / "curriculum" / corpus / f"{ds_name}.jsonl"

	samples: list[str] = []
	seen: set[str] = set()
	if jsonl_path.exists():
		with open(jsonl_path, encoding="utf-8") as f:
			for line in f:
				if line.strip():
					d = json.loads(line)
					expr = d.get("k65p_canonical") or d.get("k65p_raw")
					if expr and expr not in seen:
						seen.add(expr)
						samples.append(expr)

	if len(samples) < 20:
		print(f"✗ Corpus insuficiente para '{ds_name}': {len(samples)} muestras (<20). "
			f"Genera el corpus primero: scripts/generate_semantic_corpus.py")
		sys.exit(1)

	rng = random.Random(seed)
	rng.shuffle(samples)

	tokenized = [tokenize_k65p(s, word_to_idx, max_len=max_len) for s in samples]
	split_idx = int(len(tokenized) * 0.85)
	train_t = torch.tensor(tokenized[:split_idx], dtype=torch.long)
	val_t = torch.tensor(tokenized[split_idx:], dtype=torch.long)
	val_exprs = samples[split_idx:]
	return train_t, val_t, val_exprs


def build_stage_logit_mask(stage_idx: int, vocab_size: int, word_to_idx: dict[str, int], lang: str = "es") -> torch.Tensor:
	# <unk> vetado: el corpus no lo contiene (verificado por assert) y emitirlo
	# en generación es ruido puro. <pad> se permite como señal implícita de stop.
	mask = torch.full((vocab_size,), float("-inf"))
	allowed = {"<pad>", "<stop>", "[", "]", "G"}

	for pid, row in enumerate(PRIMES):
		allowed.add(row[0].lower())
		allowed.add(str(pid))

	molecules = sorted(molecule_names(lang=lang))
	# Escuela semántica (en): todas las moléculas desde el principio
	# Escuela sintáctica (es): desbloqueo progresivo por etapa
	if lang != "es" or stage_idx >= 6:
		allowed.update(molecules)
	elif stage_idx <= 3:
		allowed.update(molecules[:10])
	elif stage_idx <= 5:
		allowed.update(molecules[:20])

	for w in allowed:
		if w in word_to_idx:
			mask[word_to_idx[w]] = 0.0

	return mask


def assert_mask_covers_dataset(
	datasets: list[torch.Tensor],
	logit_mask: torch.Tensor,
	idx_to_word: dict[int, str],
	stage_name: str,
	pad_idx: int,
) -> None:
	"""Gate DL-004: la máscara de etapa jamás puede vetar un target real del corpus."""
	banned = (logit_mask == float("-inf")).nonzero(as_tuple=True)[0]
	banned_set = set(banned.tolist())
	for ds in datasets:
		present = set(torch.unique(ds).tolist()) - {pad_idx}
		conflict = present & banned_set
		if conflict:
			words = sorted(idx_to_word[i] for i in conflict)
			print(f"✗ INCONSISTENCIA MÁSCARA↔CORPUS en etapa '{stage_name}': "
				f"los tokens {words} aparecen en el corpus pero la máscara los veta. Abortando.")
			sys.exit(1)


@torch.no_grad()
def greedy_generate(
	model,
	prompt_ids: list[int],
	logit_mask: torch.Tensor,
	device,
	max_new_tokens: int = MAX_LEN,
	stop_idx: int | None = None,
	pad_idx: int | None = None,
	open_idx: int | None = None,
	close_idx: int | None = None,
) -> list[int]:
	"""Decodificación greedy autoregresiva. Para al cerrar el árbol, en <stop>/<pad> o al agotar longitud."""
	ids = list(prompt_ids)
	balance = sum(1 if i == open_idx else -1 if i == close_idx else 0 for i in ids)
	for _ in range(max_new_tokens):
		if len(ids) >= MAX_LEN:
			break
		x = torch.tensor([ids], dtype=torch.long, device=device)
		logits = model(x) + logit_mask.view(1, 1, -1)
		next_id = int(torch.argmax(logits[0, -1]).item())
		if next_id in {stop_idx, pad_idx}:
			break
		ids.append(next_id)
		if next_id == open_idx:
			balance += 1
		elif next_id == close_idx:
			balance -= 1
			if balance <= 0:
				break
	return ids


@torch.no_grad()
def evaluate_model(model, val_dataset, logit_mask, vocab_size, pad_idx, batch_size) -> tuple[float, float]:
	"""val_loss y precisión next-token, ambas con pads excluidos (ignore_index)."""
	model.eval()
	total_loss, total_correct, total_tokens = 0.0, 0, 0
	for i in range(0, len(val_dataset), batch_size):
		batch = val_dataset[i : i + batch_size]
		if len(batch) == 0:
			continue
		x = batch[:, :-1]
		y = batch[:, 1:]
		logits = model(x) + logit_mask.view(1, 1, -1)
		flat_logits = logits.reshape(-1, vocab_size)
		flat_y = y.reshape(-1)
		loss = F.cross_entropy(flat_logits, flat_y, ignore_index=pad_idx, reduction="sum")
		real = flat_y != pad_idx
		n_real = int(real.sum().item())
		if n_real == 0:
			continue
		total_loss += loss.item()
		total_tokens += n_real
		preds = flat_logits.argmax(dim=-1)
		total_correct += int((preds[real] == flat_y[real]).sum().item())

	if total_tokens == 0:
		return float("inf"), 0.0
	return total_loss / total_tokens, total_correct / total_tokens


def bigram_baseline(train_t: torch.Tensor, val_t: torch.Tensor, vocab_size: int, pad_idx: int) -> tuple[float, float]:
	"""Loss y precisión next-token de un bigrama Laplace entrenado en train (pads excluidos)."""
	from collections import defaultdict
	counts = defaultdict(lambda: defaultdict(int))
	for row in train_t.tolist():
		for a, b in zip(row[:-1], row[1:], strict=False):
			if b != pad_idx:
				counts[a][b] += 1

	total_loss, total_correct, total_tokens = 0.0, 0, 0
	for row in val_t.tolist():
		for a, b in zip(row[:-1], row[1:], strict=False):
			if b == pad_idx:
				continue
			row_counts = counts.get(a, {})
			denom = sum(row_counts.values()) + vocab_size
			total_loss += -math.log((row_counts.get(b, 0) + 1) / denom)
			if row_counts and max(row_counts, key=row_counts.get) == b:
				total_correct += 1
			total_tokens += 1

	if total_tokens == 0:
		return float("inf"), 0.0
	return total_loss / total_tokens, total_correct / total_tokens


def run_milestone_exam(
	model,
	train_dataset,
	val_dataset,
	val_exprs,
	logit_mask,
	vocab_size,
	word_to_idx,
	idx_to_word,
	device,
	batch_size,
	milestone: str,
) -> dict:
	"""Examen de hito con umbrales congelados (DL-004). Devuelve el acta del examen."""
	pad_idx = word_to_idx["<pad>"]
	val_loss, token_acc = evaluate_model(model, val_dataset, logit_mask, vocab_size, pad_idx, batch_size)
	bigram_loss, bigram_acc = bigram_baseline(train_dataset.cpu(), val_dataset.cpu(), vocab_size, pad_idx)

	prompts = val_exprs[:EXAM_N_PROMPTS]
	n_valid = 0
	samples = []
	for expr in prompts:
		full_ids = [i for i in tokenize_k65p(expr, word_to_idx) if i != pad_idx]
		prompt_ids = full_ids[:2] if len(full_ids) >= 2 else full_ids
		gen_ids = greedy_generate(
			model, prompt_ids, logit_mask, device,
			stop_idx=word_to_idx["<stop>"], pad_idx=pad_idx,
			open_idx=word_to_idx["["], close_idx=word_to_idx["]"],
		)
		gen_text = detokenize_k65p(gen_ids, idx_to_word)
		try:
			ok = is_valid(gen_text)
		except Exception:
			ok = False
		n_valid += int(ok)
		if len(samples) < 5:
			samples.append({"prompt": detokenize_k65p(prompt_ids, idx_to_word), "generated": gen_text, "valid": ok})

	gen_valid_rate = n_valid / max(1, len(prompts))
	passed = (
		math.isfinite(val_loss)
		and gen_valid_rate >= EXAM_MIN_GEN_VALID
		and val_loss <= bigram_loss - EXAM_BIGRAM_MARGIN
	)
	return {
		"milestone": milestone,
		"val_loss": val_loss,
		"token_acc_no_pad": round(token_acc, 4),
		"gen_valid_rate": round(gen_valid_rate, 4),
		"bigram_loss": round(bigram_loss, 4),
		"bigram_acc": round(bigram_acc, 4),
		"n_prompts": len(prompts),
		"thresholds": {"gen_valid": EXAM_MIN_GEN_VALID, "bigram_margin": EXAM_BIGRAM_MARGIN},
		"passed": passed,
		"samples": samples,
	}


def trigger_k65p_neurogenesis(model, optimizer, new_dim, device):
	"""Amplía la dimensión oculta vía Net2WiderNet preservando funciones (ambos modos de embedding)."""
	print(f"\n🧬 [NEUROGÉNESIS Net2WiderNet — remediación post-examen] {model.hidden_dim}d ➔ {new_dim}d...")

	model = model.cpu()
	for state_opt in optimizer.state.values():
		for k, v in state_opt.items():
			if isinstance(v, torch.Tensor):
				state_opt[k] = v.cpu()

	dummy = build_model("glyph" if model.use_glyphs else "standard", new_dim,
		model.glyph_embedding.glyph_table.cpu().numpy() if model.use_glyphs else np.eye(model.vocab_size, dtype=np.float32))
	lr_scale = 128.0 / new_dim
	new_optimizer = torch.optim.AdamW(dummy.parameters(), lr=1e-4 * lr_scale, weight_decay=0.01)
	del dummy

	model = net2wider_model(
		model,
		new_hidden_dim=new_dim,
		noise_std=0.01,
		old_optimizer=optimizer,
		new_optimizer=new_optimizer,
	)

	model = model.to(device)
	if torch.cuda.is_available():
		torch.cuda.empty_cache()

	print(f"✓ Neurogénesis completada. Parámetros: {sum(p.numel() for p in model.parameters()):,}\n")
	return model, new_optimizer


def save_checkpoint(path: Path, model, optimizer, epoch: int, val_loss: float) -> None:
	torch.save({
		"epoch": epoch,
		"model_state_dict": model.state_dict(),
		"optimizer_state_dict": optimizer.state_dict(),
		"loss": val_loss,
	}, path)


def load_weights(path: Path, model, optimizer, device) -> bool:
	try:
		checkpoint = torch.load(path, map_location=device, weights_only=True)
		model.load_state_dict(checkpoint["model_state_dict"])
		if optimizer is not None and "optimizer_state_dict" in checkpoint:
			try:
				optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
			except Exception as exc:
				print(f"⚠️ Optimizer no recargado (se reinicia AdamW): {exc}")
		return True
	except Exception as exc:
		print(f"⚠️ Error cargando checkpoint {path.name}: {exc}")
		return False


def run_school_training_k65p():
	try:
		rp_src = os.environ.get("RED_PILL_SRC")
		if rp_src and os.path.isdir(rp_src) and rp_src not in sys.path:
			sys.path.insert(0, rp_src)
		from red_pill.core.gpu_reservation import GpuReservationManager
		GpuReservationManager.reserve("train_sovereign_school_k65p.py", vram_mb=4096, exclusive=True)
		atexit.register(GpuReservationManager.release)
	except Exception:
		pass

	parser = argparse.ArgumentParser(description="School Training Loop K-65P (protocolo adaptativo DL-006)")
	parser.add_argument("--batch_size", type=int, default=32, help="Tamaño de lote")
	parser.add_argument("--max_epochs_per_run", type=int, default=1, help="Épocas por ejecución de paso atómico")
	parser.add_argument("--patience", type=int, default=15, help="Épocas sin mejora de val para cerrar la etapa (plateau)")
	parser.add_argument("--min_delta", type=float, default=0.005, help="Delta mínimo para considerar mejora")
	parser.add_argument("--max_stage_epochs", type=int, default=200, help="Tope de seguridad de épocas por etapa")
	parser.add_argument("--seed", type=int, default=770, help="Semilla global (split, init, shuffle)")
	parser.add_argument("--embedding", choices=["glyph", "standard"], default="glyph", help="Brazo: glyph=Bit v2, standard=Bit v0")
	parser.add_argument("--state_dir", type=str, default=str(DEFAULT_STATE_DIR), help="Directorio de estado/checkpoints (sandboxing)")
	parser.add_argument("--reset_state", action="store_true", help="Reiniciar entrenamiento desde cero")
	parser.add_argument("--corpus", type=str, default="factory_k65p", help="Subdirectorio del corpus (factory_k65p | factory_semantic)")
	parser.add_argument("--lang", type=str, default="es", help="Idioma del lexicon (es | en)")
	args = parser.parse_args()

	state_dir = Path(args.state_dir)
	state_file = state_dir / "school_state_k65p.json"
	model_current_path = state_dir / "model_current_k65p.pt"
	model_best_path = state_dir / "model_best_k65p.pt"
	state_dir.mkdir(parents=True, exist_ok=True)

	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	random.seed(args.seed)

	word_to_idx, idx_to_word, glyph_table = build_k65p_vocab_and_glyphs(lang=args.lang)
	vocab_size = len(word_to_idx)
	pad_idx = word_to_idx["<pad>"]

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	if state_file.exists() and not args.reset_state:
		state = json.loads(state_file.read_text(encoding="utf-8"))
		if state.get("embedding_mode", "glyph") != args.embedding:
			print(f"✗ El estado en {state_dir} es del brazo '{state.get('embedding_mode')}' y pediste '{args.embedding}'. "
				f"Usa otro --state_dir o --reset_state.")
			sys.exit(1)
	else:
		state = {
			"protocol": "adaptive_dl006",
			"embedding_mode": args.embedding,
			"current_epoch": 0,
			"epoch_in_stage": 0,
			"current_stage_idx": 0,
			"hidden_dim": 128,
			"num_layers": 6,
			"target_milestone": "in_progress",
			"milestones_achieved": [],
			"best_stage_val_loss": float("inf"),
			"epochs_since_best": 0,
			"stage_history": [],
			"neurogenesis_history": [],
			"exam_history": [],
			"exam_failures": {},
			"seed": args.seed,
		}

	model = build_model(state["embedding_mode"], state["hidden_dim"], glyph_table, state["num_layers"]).to(device)
	lr_scale = 128.0 / model.hidden_dim
	optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4 * lr_scale, weight_decay=0.01)

	if model_current_path.exists() and not args.reset_state:
		load_weights(model_current_path, model, optimizer, device)

	st_cfg = STAGE_CONFIG[state["current_stage_idx"]]
	n_params = sum(p.numel() for p in model.parameters())
	print(f"⚡ Brazo: {state['embedding_mode']} | Dispositivo: {device} | Vocab: {vocab_size} | Params: {n_params:,} | Semilla: {args.seed}")
	print(f"📖 Estado: Época {state['current_epoch']} (etapa {st_cfg['name']}: {state['epoch_in_stage']} ép., best={state['best_stage_val_loss']:.4f}), Dim {model.hidden_dim}d")

	def load_stage(idx: int):
		cfg = STAGE_CONFIG[idx]
		tr, va, ve = load_dataset_stage(cfg["name"], word_to_idx, seed=args.seed, corpus=args.corpus)
		mask = build_stage_logit_mask(cfg["stage_idx"], vocab_size, word_to_idx, lang=args.lang)
		assert_mask_covers_dataset([tr, va], mask, idx_to_word, cfg["name"], pad_idx)
		return cfg, tr.to(device), va.to(device), ve, mask.to(device)

	st_cfg, train_dataset, val_dataset, val_exprs, logit_mask = load_stage(state["current_stage_idx"])

	epochs_run = 0
	while epochs_run < args.max_epochs_per_run:
		if state["target_milestone"] == "completed":
			print("🎓 Escuela completada: nada que entrenar.")
			break

		state["current_epoch"] += 1
		state["epoch_in_stage"] += 1
		epochs_run += 1

		# ── Entrenar una época ──
		model.train()
		total_loss = 0.0
		total_real_tokens = 0
		perm = torch.randperm(len(train_dataset))
		shuffled_data = train_dataset[perm]

		for i in range(0, len(shuffled_data), args.batch_size):
			batch = shuffled_data[i : i + args.batch_size]
			if len(batch) == 0:
				continue
			x = batch[:, :-1]
			y = batch[:, 1:]
			optimizer.zero_grad()
			logits = model(x)
			masked_logits = logits + logit_mask.view(1, 1, -1)
			flat_y = y.reshape(-1)
			loss = F.cross_entropy(masked_logits.reshape(-1, vocab_size), flat_y, ignore_index=pad_idx)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			optimizer.step()
			n_real = int((flat_y != pad_idx).sum().item())
			total_loss += loss.item() * n_real
			total_real_tokens += n_real

		train_loss = total_loss / max(1, total_real_tokens)
		val_loss, val_acc = evaluate_model(model, val_dataset, logit_mask, vocab_size, pad_idx, args.batch_size)
		print(f"▶ [Época {state['current_epoch']} | etapa {st_cfg['name']} ép.{state['epoch_in_stage']}] Dim: {model.hidden_dim}d | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Acc: {val_acc:.4f}")

		# ── Seguimiento del mejor checkpoint de la etapa ──
		if math.isfinite(val_loss) and val_loss < state["best_stage_val_loss"] - args.min_delta:
			state["best_stage_val_loss"] = val_loss
			state["epochs_since_best"] = 0
			save_checkpoint(model_best_path, model, optimizer, state["current_epoch"], val_loss)
		else:
			state["epochs_since_best"] += 1

		# ── ¿Fin de etapa? (plateau o tope de seguridad) ──
		stage_end = (
			state["epochs_since_best"] >= args.patience
			or state["epoch_in_stage"] >= args.max_stage_epochs
		)

		exam_pause = False
		if stage_end:
			reason = "plateau" if state["epochs_since_best"] >= args.patience else "max_stage_epochs"
			print(f"\n🏁 Fin de etapa {st_cfg['name']} por {reason} tras {state['epoch_in_stage']} épocas (best val={state['best_stage_val_loss']:.4f}).")

			# El examen y el avance parten del MEJOR checkpoint de la etapa, no del último.
			if model_best_path.exists():
				load_weights(model_best_path, model, optimizer, device)

			advance = False
			if st_cfg["age"]:
				m_name = f"{st_cfg['age']}_years"
				print(f"📝 [EXAMEN DE HITO {m_name} sobre best-val] Umbrales congelados: gen_valid≥{EXAM_MIN_GEN_VALID}, val_loss ≤ bigrama − {EXAM_BIGRAM_MARGIN}")
				exam = run_milestone_exam(
					model, train_dataset, val_dataset, val_exprs, logit_mask, vocab_size,
					word_to_idx, idx_to_word, device, args.batch_size, m_name,
				)
				exam["epoch"] = state["current_epoch"]
				exam["epochs_in_stage"] = state["epoch_in_stage"]
				exam["hidden_dim"] = model.hidden_dim
				state["exam_history"].append(exam)
				summary = (f"gen_valid={exam['gen_valid_rate']}, val_loss={exam['val_loss']:.4f} vs bigrama={exam['bigram_loss']}, "
					f"acc={exam['token_acc_no_pad']} (informativa)")
				if exam["passed"]:
					print(f"🎓 ¡Hito {m_name} APROBADO en {state['epoch_in_stage']} épocas con {model.hidden_dim}d! {summary}")
					state["milestones_achieved"].append(m_name)
					advance = True
				else:
					state["exam_failures"][m_name] = state["exam_failures"].get(m_name, 0) + 1
					# Remediación DL-006: la neurogénesis SOLO responde a un suspenso.
					next_dim = next((d for d in ALL_DIMS if d > model.hidden_dim and d <= st_cfg["dim"]), None)
					if next_dim is not None:
						old_dim = model.hidden_dim
						print(f"✗ Hito {m_name} SUSPENDIDO ({summary}). Remediación: neurogénesis y repetición de etapa.")
						model, optimizer = trigger_k65p_neurogenesis(model, optimizer, next_dim, device)
						state["hidden_dim"] = next_dim
						state["neurogenesis_history"].append({
							"epoch": state["current_epoch"],
							"old_dim": old_dim,
							"new_dim": next_dim,
							"reason": f"exam_failed:{m_name}",
							"val_loss": exam["val_loss"],
						})
						save_checkpoint(model_best_path, model, optimizer, state["current_epoch"], float("inf"))
					else:
						print(f"✗ Hito {m_name} SUSPENDIDO ({summary}) y sin techo de dim disponible en la etapa. "
							f"Pausa para revisión del operador (rc={EXAM_PAUSE_EXIT_CODE}).")
						exam_pause = True
			else:
				print(f"✓ Etapa {st_cfg['name']} (guardería, sin examen) superada por plateau.")
				advance = True

			if advance:
				state["stage_history"].append({
					"stage": st_cfg["name"],
					"epochs": state["epoch_in_stage"],
					"best_val_loss": state["best_stage_val_loss"],
					"hidden_dim": model.hidden_dim,
				})
				if state["current_stage_idx"] + 1 < len(STAGE_CONFIG):
					state["current_stage_idx"] += 1
					st_cfg, train_dataset, val_dataset, val_exprs, logit_mask = load_stage(state["current_stage_idx"])
					print(f"➡️ Avanza a la etapa {st_cfg['name']} desde el mejor checkpoint.")
				else:
					state["target_milestone"] = "completed"
					print("🎓 ¡ESCUELA COMPLETADA! Todos los hitos aprobados con examen.")

			if advance or (st_cfg["age"] and not exam_pause):
				state["epoch_in_stage"] = 0
				state["best_stage_val_loss"] = float("inf")
				state["epochs_since_best"] = 0

		save_checkpoint(model_current_path, model, optimizer, state["current_epoch"], val_loss)
		state_file.write_text(json.dumps(state, indent=4, ensure_ascii=False), encoding="utf-8")

		if exam_pause:
			sys.exit(EXAM_PAUSE_EXIT_CODE)

	print(f"✅ Run completada limpiamente: {epochs_run} época(s) ejecutadas.")


if __name__ == "__main__":
	run_school_training_k65p()
