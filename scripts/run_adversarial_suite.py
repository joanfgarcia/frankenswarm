"""Batería Adversarial (Red Teaming) para Bit v2 (K-65P Nativo) — instrumentos DL-004.

A diferencia de la versión del 2-ago (retirada por DL-004: validaba la sintaxis
de la EXPRESIÓN DE ENTRADA e ignoraba la salida del modelo), esta suite:

1. Evalúa métricas con pads EXCLUIDOS (ignore_index) y las compara contra dos
   baselines triviales: uniforme (ln|V|) y un bigrama entrenado en train.
2. El ataque OOD genera texto AUTOREGRESIVAMENTE desde prompts del holdout real
   (ood_holdout.jsonl, pares cabeza-argumento jamás vistos en train/val) y
   valida LA SALIDA GENERADA con k65p.validator.
3. Mide robustez al ruido de glifos con precisión sin pads.
4. El veredicto final es CONDICIONAL a los resultados: esta suite puede suspender.

Uso:
	PYTHONPATH=.:../k65p/src .venv/bin/python scripts/run_adversarial_suite.py [--state_dir DIR]
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812

base_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(base_dir))

k65p_src = base_dir.parent / "k65p" / "src"
if k65p_src.exists():
	sys.path.append(str(k65p_src))

from k65p.validator import is_valid

from src.bitnet.model.modeling_bitnet import BitNet4LayerModel
from src.bitnet.training.train_sovereign_school_k65p import (
	MAX_LEN,
	bigram_baseline,
	build_k65p_vocab_and_glyphs,
	build_stage_logit_mask,
	detokenize_k65p,
	evaluate_model,
	greedy_generate,
	load_dataset_stage,
	tokenize_k65p,
)

DEFAULT_STATE_DIR = base_dir / "storage" / "checkpoints" / "sovereign_school_k65p"
OOD_PATH = base_dir / "storage" / "curriculum" / "factory_k65p" / "ood_holdout.jsonl"

# Umbrales de la suite (congelados junto a DL-004; el veredicto depende de ellos).
# Alineados con el examen de hito del trainer: gramática (gen_valid) + información
# más allá de estadística trivial (loss ≤ bigrama − margen).
PASS_GEN_VALID_OOD = 0.60
PASS_LOSS_MARGIN_OVER_BIGRAM = 0.10


def main() -> None:
	parser = argparse.ArgumentParser(description="Suite adversarial K-65P (DL-004)")
	parser.add_argument("--state_dir", type=str, default=str(DEFAULT_STATE_DIR))
	args = parser.parse_args()

	state_dir = Path(args.state_dir)
	state_file = state_dir / "school_state_k65p.json"
	ckpt_path = state_dir / "model_current_k65p.pt"
	if not state_file.exists() or not ckpt_path.exists():
		print(f"✗ No hay estado/checkpoint en {state_dir}. Nada que auditar.")
		sys.exit(1)

	state = json.loads(state_file.read_text(encoding="utf-8"))
	device = torch.device("cpu")
	word_to_idx, idx_to_word, glyph_table = build_k65p_vocab_and_glyphs()
	vocab_size = len(word_to_idx)
	pad_idx = word_to_idx["<pad>"]

	model = BitNet4LayerModel(
		use_glyphs=True,
		glyph_table=glyph_table,
		hidden_dim=state["hidden_dim"],
		num_layers=state.get("num_layers", 6),
		use_pos_embedding=True,
		is_causal=True,
		max_seq_len=128,
	).to(device)
	checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
	sd = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
	model.load_state_dict(sd)
	model.eval()

	n_params = sum(p.numel() for p in model.parameters())
	stage_idx = state.get("current_stage_idx", 7)
	seed = state.get("seed", 770)

	print("═" * 80)
	print("  🛡️ BATERÍA ADVERSARIAL BIT V2 (K-65P) — instrumentos DL-004 (evalúa la SALIDA)")
	print("═" * 80)
	print(f"  Checkpoint: {ckpt_path} | dim={state['hidden_dim']} | {n_params:,} params | época {state.get('current_epoch')}")

	results = {"pass": True}

	# ── ATAQUE 1: val held-out vs baselines triviales ──
	print("\n🔴 ATAQUE 1: ¿Supera el modelo a un bigrama trivial en val held-out? (pads excluidos)")
	stage_name = "secondary_8" if stage_idx >= 6 else ("primary_5" if stage_idx >= 4 else "0-1")
	train_t, val_t, val_exprs = load_dataset_stage(stage_name, word_to_idx, seed=seed)
	logit_mask = build_stage_logit_mask(stage_idx, vocab_size, word_to_idx)

	val_loss, val_acc = evaluate_model(model, val_t, logit_mask, vocab_size, pad_idx, batch_size=32)
	bi_loss, bi_acc = bigram_baseline(train_t, val_t, vocab_size, pad_idx)
	uniform_loss = math.log(vocab_size)

	print(f"    • Modelo    : loss={val_loss:.4f} | acc={val_acc*100:.2f}% ({len(val_t)} muestras val)")
	print(f"    • Bigrama   : loss={bi_loss:.4f} | acc={bi_acc*100:.2f}%  (baseline trivial)")
	print(f"    • Uniforme  : loss={uniform_loss:.4f} (ln {vocab_size})")
	beats_bigram = math.isfinite(val_loss) and val_loss <= bi_loss - PASS_LOSS_MARGIN_OVER_BIGRAM
	print(f"    • Veredicto : {'✅ bate al bigrama en loss con margen' if beats_bigram else '❌ NO bate al bigrama trivial (margen loss ≥ ' + str(PASS_LOSS_MARGIN_OVER_BIGRAM) + ')'}")
	results["attack1"] = {"model_loss": val_loss, "model_acc": val_acc, "bigram_loss": bi_loss, "bigram_acc": bi_acc, "pass": beats_bigram}
	results["pass"] &= beats_bigram

	# ── ATAQUE 2: OOD real — generación autoregresiva validada en LA SALIDA ──
	print("\n🔴 ATAQUE 2: OOD real (holdout de pares jamás vistos) — se valida LO QUE GENERA el modelo")
	if not OOD_PATH.exists():
		print(f"    ✗ Falta {OOD_PATH}. Genera el corpus con scripts/generate_k65p_corpus.py")
		sys.exit(1)
	ood_exprs = []
	with open(OOD_PATH, encoding="utf-8") as f:
		for line in f:
			if line.strip():
				ood_exprs.append(json.loads(line)["k65p_canonical"])
	if not ood_exprs:
		print("    ✗ Holdout OOD vacío: la suite no puede certificar generalización.")
		sys.exit(1)

	full_mask = build_stage_logit_mask(7, vocab_size, word_to_idx)
	n_valid, shown = 0, 0
	for expr in ood_exprs[:40]:
		full_ids = [i for i in tokenize_k65p(expr, word_to_idx) if i != pad_idx]
		prompt_ids = full_ids[:2]
		gen_ids = greedy_generate(
			model, prompt_ids, full_mask, device,
			stop_idx=word_to_idx["<stop>"], pad_idx=pad_idx,
			open_idx=word_to_idx["["], close_idx=word_to_idx["]"],
		)
		gen_text = detokenize_k65p(gen_ids, idx_to_word)
		try:
			ok = is_valid(gen_text)
		except Exception:
			ok = False
		n_valid += int(ok)
		if shown < 5:
			print(f"    • prompt '{detokenize_k65p(prompt_ids, idx_to_word)}' → '{gen_text}' {'✅' if ok else '❌'}")
			shown += 1

	n_ood = min(40, len(ood_exprs))
	gen_valid_rate = n_valid / n_ood
	ood_ids = torch.tensor([tokenize_k65p(e, word_to_idx) for e in ood_exprs], dtype=torch.long)
	ood_loss, ood_acc = evaluate_model(model, ood_ids, full_mask, vocab_size, pad_idx, batch_size=32)
	gap = ood_loss - val_loss if math.isfinite(ood_loss) and math.isfinite(val_loss) else float("inf")
	ood_pass = gen_valid_rate >= PASS_GEN_VALID_OOD
	print(f"    • Generaciones válidas (salida real del modelo): {n_valid}/{n_ood} = {gen_valid_rate*100:.1f}%")
	print(f"    • Teacher-forced OOD: loss={ood_loss:.4f} acc={ood_acc*100:.2f}% | gap OOD−val = {gap:+.4f} nats")
	print(f"    • Veredicto : {'✅' if ood_pass else '❌'} (umbral gen_valid ≥ {PASS_GEN_VALID_OOD})")
	results["attack2"] = {"gen_valid_rate": gen_valid_rate, "ood_loss": ood_loss, "ood_acc": ood_acc, "gap": gap, "pass": ood_pass}
	results["pass"] &= ood_pass

	# ── ATAQUE 3: ruido en glifos (precisión sin pads) ──
	print("\n🔴 ATAQUE 3: Robustez al ruido en glifos (corrupción de la matriz de primos)")
	orig = model.glyph_embedding.prime_embeddings.data.clone()
	for noise_std in (0.0, 0.1, 0.25, 0.5):
		model.glyph_embedding.prime_embeddings.data = orig + torch.randn_like(orig) * noise_std
		n_loss, n_acc = evaluate_model(model, val_t, logit_mask, vocab_size, pad_idx, batch_size=32)
		print(f"    • Ruido σ={noise_std:.2f} ➔ loss={n_loss:.4f} | acc (sin pads)={n_acc*100:.2f}%")
	model.glyph_embedding.prime_embeddings.data = orig

	# ── Veredicto global CONDICIONAL ──
	print("\n" + "═" * 80)
	if results["pass"]:
		print("  ✅ VEREDICTO: el modelo supera la batería adversarial DL-004.")
	else:
		print("  ❌ VEREDICTO: el modelo NO supera la batería. Revisar antes de citar cifras.")
	print("═" * 80 + "\n")

	out_path = state_dir / "adversarial_report.json"
	out_path.write_text(json.dumps(results, indent=4, ensure_ascii=False), encoding="utf-8")
	print(f"Acta guardada en {out_path}")
	sys.exit(0 if results["pass"] else 2)


if __name__ == "__main__":
	main()
