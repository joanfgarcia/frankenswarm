"""Entrenador Soberano de Bit v2 (Formación Nativa K-65P) — Escuela v3, instrumentos DL-004.

Entrena a Bit v2 sobre expresiones K-65P generadas composicionalmente
(storage/curriculum/factory_k65p/<stage>.jsonl), aplicando:
- Curriculum-Gated Softmax (logit_mask por etapa) con ASSERT de consistencia
  máscara↔corpus: ningún target real puede estar vetado (bug raíz de DL-004).
- Loss y precisión con ignore_index=<pad> (el padding no puntúa).
- Neurogénesis dirigida por plateau SOLO sobre val_loss finita.
- Examen de hito (gate real, umbrales congelados pre-run): un hito NO se
  otorga por cronómetro; exige val_loss finita, precisión token (sin pads)
  y tasa de generaciones sintácticamente válidas. Suspenso → repetición de
  curso (retention) + pausa con exit code 78 para revisión del operador.
- Checkpoints atómicos por época con estado del optimizer RECARGADO al resumir.

Umbrales de examen congelados (pre-registro DL-004, 2026-08-03 — NO tocar a mitad de run):
	EXAM_MIN_GEN_VALID = 0.60   (fracción de generaciones greedy válidas según k65p.validator;
	                             criterio primario: la tesis es que Bit aprende la GRAMÁTICA)
	EXAM_BIGRAM_MARGIN = 0.10   (val_loss del modelo debe ser ≤ bigrama_loss − margen;
	                             criterio secundario: información más allá de estadística trivial)
	EXAM_N_PROMPTS = 25         (prefijos de val usados como prompts)
La precisión next-token (sin pads) se REPORTA pero no umbraliza: en un corpus
composicional los átomos concretos son impredecibles por diseño (techo estructural
~50-65%; el bigrama ya puntúa 35-48%), así que un umbral absoluto mediría ruido.
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
from src.bitnet.training.modules.stage_config import get_next_dim, get_stage_config

DEFAULT_STATE_DIR = base_dir / "storage" / "checkpoints" / "sovereign_school_k65p"

EXAM_PAUSE_EXIT_CODE = 78
EXAM_MIN_GEN_VALID = 0.60
EXAM_BIGRAM_MARGIN = 0.10
EXAM_N_PROMPTS = 25
RETENTION_EPOCHS = 16
MAX_LEN = 48
MIN_VAL_SAMPLES_FOR_GROWTH = 30

STAGE_CONFIG = get_stage_config(base_epochs=64, stage_scale=0.5)


def get_stage_info_for_epoch(epoch: int) -> dict:
	for cfg in STAGE_CONFIG:
		if cfg["start_epoch"] <= epoch <= cfg["end_epoch"]:
			return cfg
	return STAGE_CONFIG[-1]


# Firmas ternarias (trit −1) para los tokens estructurales. Sin esto, los seis
# especiales comparten el glifo todo-ceros → embedding CERO idéntico → el modelo
# no puede distinguir '[' de ']' ni a la entrada ni a la salida: la sintaxis es
# inaprendible por construcción (hallazgo DL-005, destapado por el examen de hito).
# <pad> conserva el glifo cero a propósito: jamás es target (ignore_index) y la
# generación para por balance de corchetes, no por emitir pad.
STRUCT_TRITS = {"<unk>": 4, "<stop>": 3, "[": 0, "]": 1, "G": 2}


def build_k65p_vocab_and_glyphs() -> tuple[dict[str, int], dict[int, str], np.ndarray]:
	lexicon = load_lexicon()
	tokens = ["<pad>", "<unk>", "<stop>", "[", "]", "G"]
	glyphs_list = []
	for t in tokens:
		g = [0] * N_PRIMES
		if t in STRUCT_TRITS:
			g[STRUCT_TRITS[t]] = -1
		glyphs_list.append(g)

	# Primos SOLO en forma canónica (dígito). La forma símbolo compartía glifo
	# idéntico con el dígito → logits empatados y precisión estructuralmente ~0
	# (aliasing DL-005). El corpus canónico usa dígitos; los símbolos sobraban.
	for pid, _row in enumerate(PRIMES):
		g_sym = [0] * N_PRIMES
		g_sym[pid] = 1
		tokens.append(str(pid))
		glyphs_list.append(g_sym)

	for name, entry in sorted(lexicon.items()):
		tokens.append(name.casefold())
		glyphs_list.append(entry["glyph"])

	vocab_words = []
	final_glyphs = []
	seen = set()
	for t, g in zip(tokens, glyphs_list):
		if t not in seen:
			seen.add(t)
			vocab_words.append(t)
			final_glyphs.append(g)

	word_to_idx = {w: i for i, w in enumerate(vocab_words)}
	idx_to_word = {i: w for i, w in enumerate(vocab_words)}
	glyph_table = np.array(final_glyphs, dtype=np.float32)

	return word_to_idx, idx_to_word, glyph_table


def tokenize_k65p(text: str, word_to_idx: dict[str, int], max_len: int = MAX_LEN) -> list[int]:
	# OJO (DL-004): sin alternativa `G|g` en el regex — partía "grupo" en "g"+"rupo".
	# Lookup exacto antes que casefold: "G" (marcador de grupo) vive en mayúscula en el vocab.
	raw_tokens = re.findall(r"\[|\]|-?\d+|[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_]+", text)
	ids = []
	for t in raw_tokens:
		idx = word_to_idx.get(t, word_to_idx.get(t.casefold(), word_to_idx["<unk>"]))
		ids.append(idx)

	if len(ids) < max_len:
		ids = ids + [word_to_idx["<pad>"]] * (max_len - len(ids))
	else:
		ids = ids[:max_len]
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
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
	"""Carga el corpus de la etapa. Split train/val 85/15 BARAJADO (semilla fija) y sin duplicados."""
	ds_name = stage_group_for(stage_name)
	jsonl_path = base_dir / "storage" / "curriculum" / "factory_k65p" / f"{ds_name}.jsonl"

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

	if len(samples) < 40:
		print(f"✗ Corpus insuficiente para '{ds_name}': {len(samples)} muestras (<40). "
			f"Genera el corpus primero: scripts/generate_k65p_corpus.py")
		sys.exit(1)

	rng = random.Random(seed)
	rng.shuffle(samples)

	tokenized = [tokenize_k65p(s, word_to_idx, max_len=max_len) for s in samples]
	split_idx = int(len(tokenized) * 0.85)
	train_t = torch.tensor(tokenized[:split_idx], dtype=torch.long)
	val_t = torch.tensor(tokenized[split_idx:], dtype=torch.long)
	val_exprs = samples[split_idx:]
	return train_t, val_t, val_exprs


def build_stage_logit_mask(stage_idx: int, vocab_size: int, word_to_idx: dict[str, int]) -> torch.Tensor:
	# <unk> vetado: el corpus no lo contiene (verificado por assert) y emitirlo
	# en generación es ruido puro. <pad> se permite como señal implícita de stop.
	mask = torch.full((vocab_size,), float("-inf"))
	allowed = {"<pad>", "<stop>", "[", "]", "G"}

	for pid, row in enumerate(PRIMES):
		allowed.add(row[0].lower())
		allowed.add(str(pid))

	molecules = sorted(molecule_names())
	if stage_idx <= 3:
		allowed.update(molecules[:10])
	elif stage_idx <= 5:
		allowed.update(molecules[:20])
	else:
		allowed.update(molecules)

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
	"""Gate DL-004: la máscara de etapa jamás puede vetar un target real del corpus.

	El bug raíz de la run del 2-ago: targets con logit −inf → loss infinita →
	neurogénesis disparada por artefacto. Aquí se aborta ANTES de entrenar.
	"""
	banned = (logit_mask == float("-inf")).nonzero(as_tuple=True)[0]
	banned_set = set(banned.tolist())
	for ds in datasets:
		present = set(torch.unique(ds).tolist()) - {pad_idx}
		conflict = present & banned_set
		if conflict:
			words = sorted(idx_to_word[i] for i in conflict)
			print(f"✗ INCONSISTENCIA MÁSCARA↔CORPUS en etapa '{stage_name}': "
				f"los tokens {words} aparecen en el corpus pero la máscara los veta. "
				f"Regenera el corpus con los tiers correctos o corrige la máscara. Abortando.")
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
	"""Decodificación greedy autoregresiva. Para al cerrar el árbol ([...] balanceado), en <stop>/<pad> o al agotar longitud."""
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
		for a, b in zip(row[:-1], row[1:]):
			if b != pad_idx:
				counts[a][b] += 1

	total_loss, total_correct, total_tokens = 0.0, 0, 0
	for row in val_t.tolist():
		for a, b in zip(row[:-1], row[1:]):
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
	"""Examen de hito con umbrales congelados. Devuelve el acta del examen."""
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


def trigger_k65p_neurogenesis(model, optimizer, new_dim, glyph_table, device):
	"""Amplía la dimensión oculta del modelo vía Net2WiderNet preservando funciones aprendidas."""
	print(f"\n🧬 [NEUROGÉNESIS Net2WiderNet] Ampliando dimensión oculta: {model.hidden_dim} ➔ {new_dim}d...")

	new_model = BitNet4LayerModel(
		use_glyphs=True,
		glyph_table=glyph_table,
		hidden_dim=new_dim,
		num_layers=len(model.core_layers),
		use_pos_embedding=True,
		is_causal=True,
		max_seq_len=128,
	).cpu()

	model = model.cpu()
	for state_opt in optimizer.state.values():
		for k, v in state_opt.items():
			if isinstance(v, torch.Tensor):
				state_opt[k] = v.cpu()

	lr_scale = 128.0 / new_dim
	new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-4 * lr_scale, weight_decay=0.01)

	model = net2wider_model(
		model,
		new_hidden_dim=new_dim,
		noise_std=0.01,
		old_optimizer=optimizer,
		new_optimizer=new_optimizer,
	)

	model = model.to(device)
	del new_model
	if torch.cuda.is_available():
		torch.cuda.empty_cache()

	print(f"✓ Neurogénesis Net2WiderNet completada. Nuevos parámetros: {sum(p.numel() for p in model.parameters()):,}\n")
	return model, new_optimizer


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

	parser = argparse.ArgumentParser(description="School Training Loop K-65P")
	parser.add_argument("--batch_size", type=int, default=32, help="Tamaño de lote")
	parser.add_argument("--max_epochs_per_run", type=int, default=1, help="Épocas por ejecución de paso atómico")
	parser.add_argument("--patience", type=int, default=15, help="Épocas sin mejora para gatillar neurogénesis")
	parser.add_argument("--min_delta", type=float, default=0.005, help="Delta mínimo para considerar mejora")
	parser.add_argument("--seed", type=int, default=770, help="Semilla global (split, init, shuffle)")
	parser.add_argument("--state_dir", type=str, default=str(DEFAULT_STATE_DIR), help="Directorio de estado/checkpoints (sandboxing)")
	parser.add_argument("--reset_state", action="store_true", help="Reiniciar entrenamiento desde cero")
	args = parser.parse_args()

	state_dir = Path(args.state_dir)
	state_file = state_dir / "school_state_k65p.json"
	model_current_path = state_dir / "model_current_k65p.pt"
	state_dir.mkdir(parents=True, exist_ok=True)

	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	random.seed(args.seed)

	word_to_idx, idx_to_word, glyph_table = build_k65p_vocab_and_glyphs()
	vocab_size = len(word_to_idx)
	pad_idx = word_to_idx["<pad>"]

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	if state_file.exists() and not args.reset_state:
		state = json.loads(state_file.read_text(encoding="utf-8"))
	else:
		state = {
			"current_epoch": 0,
			"current_stage_idx": 0,
			"hidden_dim": 128,  # Arrancar estrictamente en 128d (Etapa 0-1)
			"num_layers": 6,
			"target_milestone": "in_progress",
			"milestones_achieved": [],
			"best_val_loss": float("inf"),
			"epochs_without_improvement": 0,
			"retention_epochs": 0,
			"neurogenesis_history": [],
			"exam_history": [],
			"exam_failures": {},
			"seed": args.seed,
		}

	for key, default in (("neurogenesis_history", []), ("exam_history", []), ("exam_failures", {}), ("retention_epochs", 0)):
		state.setdefault(key, default)

	effective_epoch = state["current_epoch"] + 1 - state["retention_epochs"]
	st_cfg = get_stage_info_for_epoch(max(1, effective_epoch))
	state["current_stage_idx"] = st_cfg["stage_idx"]

	model = BitNet4LayerModel(
		use_glyphs=True,
		glyph_table=glyph_table,
		hidden_dim=state["hidden_dim"],
		num_layers=state["num_layers"],
		use_pos_embedding=True,
		is_causal=True,
		max_seq_len=128,
	).to(device)

	lr_scale = 128.0 / model.hidden_dim
	optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4 * lr_scale, weight_decay=0.01)

	if model_current_path.exists() and not args.reset_state:
		try:
			checkpoint = torch.load(model_current_path, map_location=device, weights_only=True)
			model.load_state_dict(checkpoint["model_state_dict"])
			if "optimizer_state_dict" in checkpoint:
				try:
					optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
				except Exception as exc:
					print(f"⚠️ Optimizer no recargado (se reinicia AdamW): {exc}")
		except Exception as exc:
			print(f"⚠️ Error cargando checkpoint: {exc}")

	print(f"⚡ Dispositivo: {device} | Vocab K-65P: {vocab_size} tokens | Glifos: {glyph_table.shape} | Semilla: {args.seed}")
	print(f"📖 Estado: Época {state['current_epoch']}, Etapa {st_cfg['stage_idx']} ({st_cfg['name']}), Dim {model.hidden_dim}d, Retención {state['retention_epochs']} ép.")

	train_dataset, val_dataset, val_exprs = load_dataset_stage(st_cfg["name"], word_to_idx, seed=args.seed)
	logit_mask = build_stage_logit_mask(st_cfg["stage_idx"], vocab_size, word_to_idx)
	assert_mask_covers_dataset([train_dataset, val_dataset], logit_mask, idx_to_word, st_cfg["name"], pad_idx)
	train_dataset = train_dataset.to(device)
	val_dataset = val_dataset.to(device)
	logit_mask = logit_mask.to(device)

	epochs_run = 0
	target_epochs = args.max_epochs_per_run

	while epochs_run < target_epochs:
		state["current_epoch"] += 1
		epochs_run += 1

		effective_epoch = state["current_epoch"] - state["retention_epochs"]
		st_cfg = get_stage_info_for_epoch(max(1, effective_epoch))
		if state["current_stage_idx"] != st_cfg["stage_idx"]:
			state["current_stage_idx"] = st_cfg["stage_idx"]
			# NOTA: Transición de etapa NO gatilla neurogénesis; solo el plateau (patience) la exige.
			train_dataset, val_dataset, val_exprs = load_dataset_stage(st_cfg["name"], word_to_idx, seed=args.seed)
			logit_mask = build_stage_logit_mask(st_cfg["stage_idx"], vocab_size, word_to_idx)
			assert_mask_covers_dataset([train_dataset, val_dataset], logit_mask, idx_to_word, st_cfg["name"], pad_idx)
			train_dataset = train_dataset.to(device)
			val_dataset = val_dataset.to(device)
			logit_mask = logit_mask.to(device)

		# Entrenar una época
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
		print(f"▶ [Época {state['current_epoch']}/1408] Etapa {st_cfg['stage_idx']+1}/8 ({st_cfg['name']}) | Dim: {model.hidden_dim}d | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc (sin pads): {val_acc:.4f}")

		# ── Monitor de Plateau: EXCLUSIVO activador de Neurogénesis (solo métricas finitas) ──
		if not math.isfinite(val_loss):
			print("⚠️ val_loss no finita: el plateau NO avanza (esto sería el bug DL-004; revisar máscara/corpus).")
		elif val_loss < state["best_val_loss"] - args.min_delta:
			state["best_val_loss"] = val_loss
			state["epochs_without_improvement"] = 0
		else:
			state["epochs_without_improvement"] += 1

		if state["epochs_without_improvement"] >= args.patience:
			if len(val_dataset) < MIN_VAL_SAMPLES_FOR_GROWTH:
				print(f"⚠️ Plateau con val de {len(val_dataset)} muestras (<{MIN_VAL_SAMPLES_FOR_GROWTH}): señal no fiable, neurogénesis VETADA.")
				state["epochs_without_improvement"] = 0
			else:
				next_dim = get_next_dim(model.hidden_dim, STAGE_CONFIG, current_epoch=max(1, effective_epoch))
				if next_dim is not None and next_dim <= 1024:
					old_dim = model.hidden_dim
					print(f"\n🧬 [PLATEAU] val_loss estancada {state['epochs_without_improvement']} épocas (best={state['best_val_loss']:.4f}) ➔ Neurogénesis {old_dim}d ➔ {next_dim}d")
					model, optimizer = trigger_k65p_neurogenesis(model, optimizer, next_dim, glyph_table, device)
					state["hidden_dim"] = next_dim
					state["best_val_loss"] = float("inf")
					state["epochs_without_improvement"] = 0
					state["neurogenesis_history"].append({
						"epoch": state["current_epoch"],
						"old_dim": old_dim,
						"new_dim": next_dim,
						"reason": "plateau",
						"val_loss": val_loss,
					})

		# ── Examen de hito: el diploma se gana, no se cumple por cronómetro ──
		exam_pause = False
		if st_cfg["end_epoch"] == effective_epoch and st_cfg["age"]:
			m_name = f"{st_cfg['age']}_years"
			if m_name not in state["milestones_achieved"]:
				print(f"\n📝 [EXAMEN DE HITO {m_name}] Umbrales congelados: gen_valid≥{EXAM_MIN_GEN_VALID}, val_loss ≤ bigrama − {EXAM_BIGRAM_MARGIN}")
				exam = run_milestone_exam(
					model, train_dataset, val_dataset, val_exprs, logit_mask, vocab_size,
					word_to_idx, idx_to_word, device, args.batch_size, m_name,
				)
				exam["epoch"] = state["current_epoch"]
				state["exam_history"].append(exam)
				summary = (f"gen_valid={exam['gen_valid_rate']}, val_loss={exam['val_loss']:.4f} vs bigrama={exam['bigram_loss']}, "
					f"acc={exam['token_acc_no_pad']} (informativa)")
				if exam["passed"]:
					state["milestones_achieved"].append(m_name)
					print(f"🎓 ¡Hito {m_name} APROBADO! {summary}")
					if m_name == "8_years":
						state["target_milestone"] = "completed"
				else:
					state["exam_failures"][m_name] = state["exam_failures"].get(m_name, 0) + 1
					state["retention_epochs"] += RETENTION_EPOCHS
					exam_pause = True
					print(f"✗ Hito {m_name} SUSPENDIDO ({summary}). "
						f"Repetición de curso: +{RETENTION_EPOCHS} épocas de {st_cfg['name']}. Pausa para revisión del operador (rc={EXAM_PAUSE_EXIT_CODE}).")

		torch.save({
			"epoch": state["current_epoch"],
			"model_state_dict": model.state_dict(),
			"optimizer_state_dict": optimizer.state_dict(),
			"loss": val_loss,
		}, model_current_path)

		state_file.write_text(json.dumps(state, indent=4, ensure_ascii=False), encoding="utf-8")

		if exam_pause:
			sys.exit(EXAM_PAUSE_EXIT_CODE)

	print(f"✅ Run completada limpiamente: {epochs_run} época(s) ejecutadas.")


if __name__ == "__main__":
	run_school_training_k65p()
