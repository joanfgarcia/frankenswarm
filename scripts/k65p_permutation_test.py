"""Test de permutación K-65P (2-sep): ¿ha aprendido la Escuela Soberana la
semántica posicional de las S-expressions?

Compara la loss teacher-forced del modelo sobre secuencias canónicas del
corpus factory_k65p vs permutaciones de las mismas (esqueleto de corchetes
fijo, contenido reordenado). Si la molécula es orden-sensible (la visión
K-65P: permutar cambia el significado), la canónica debe costar menos que
cualquier permutación. Se registra también la validez gramatical de cada
permutación según el validador de referencia, para separar sensibilidad
a la gramática de sensibilidad al significado.

Uso:
	PYTHONPATH=.:../k65p/src .venv/bin/python scripts/k65p_permutation_test.py \
		[--n 300] [--k 5] [--checkpoint model_best_k65p.pt]
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

base_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(base_dir))

k65p_src = base_dir.parent / "k65p" / "src"
if k65p_src.exists():
	sys.path.append(str(k65p_src))

from k65p.core import parse_k65p  # noqa: E402
from k65p.validator import is_valid  # noqa: E402

from src.bitnet.training.train_sovereign_school_k65p import (  # noqa: E402
	build_k65p_vocab_and_glyphs,
	build_model,
	tokenize_k65p,
)

BRACKETS = {"[", "]"}
TOKEN_RE = re.compile(r"\[|\]|-?\d+|[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_]+")


def permute_content(text: str, rng: random.Random) -> str | None:
	"""Reordena los tokens de contenido manteniendo fijo el esqueleto de corchetes."""
	raw = TOKEN_RE.findall(text)
	content_pos = [i for i, t in enumerate(raw) if t not in BRACKETS]
	if len(content_pos) < 2:
		return None
	content = [raw[i] for i in content_pos]
	perm = content[:]
	while perm == content:
		rng.shuffle(perm)
	out = list(raw)
	for pos, tok in zip(content_pos, perm):
		out[pos] = tok
	return " ".join(out)


def seq_loss(model, text: str, word_to_idx: dict, pad_idx: int, device, max_len: int = 128) -> tuple[float, int]:
	ids = tokenize_k65p(text, word_to_idx, max_len=max_len)
	if len(ids) < 2:
		return float("nan"), 0
	x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
	y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
	with torch.no_grad():
		logits = model(x)
	loss = F.cross_entropy(
		logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=pad_idx, reduction="sum"
	)
	n_real = int((y != pad_idx).sum().item())
	return loss.item() / max(n_real, 1), n_real


def rebuild_vocab_era(corpus_dir: Path, lang: str = "es") -> tuple[dict, dict, np.ndarray]:
	"""Reconstruye el vocabulario de la época del checkpoint: el léxico actual
	creció tras el entrenamiento (165 tokens vs 99), así que las moléculas se
	derivan del corpus factory (28) y se ordenan como hacía el builder
	(sorted casefold). 6 estructurales + 65 primos + 28 moléculas = 99."""
	names: set[str] = set()
	for f in list(corpus_dir.glob("*.jsonl")):
		for line in f.read_text(encoding="utf-8").splitlines():
			if not line.strip():
				continue
			text = json.loads(line)["k65p_canonical"]
			for t in TOKEN_RE.findall(text):
				if t not in BRACKETS and not t.lstrip("-").isdigit():
					names.add(t.casefold())
	names.discard("g")  # marcador estructural, no molécula
	return build_k65p_vocab_and_glyphs(lang=lang, molecule_order=sorted(names))


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--n", type=int, default=300, help="Secuencias muestreadas por etapa")
	ap.add_argument("--k", type=int, default=5, help="Permutaciones por secuencia")
	ap.add_argument("--checkpoint", type=str, default="model_best_k65p.pt")
	ap.add_argument("--state_dir", type=str, default=str(base_dir / "storage" / "checkpoints" / "sovereign_school_k65p"))
	ap.add_argument("--corpus_dir", type=str, default=str(base_dir / "storage" / "curriculum" / "factory_k65p"))
	ap.add_argument("--seed", type=int, default=42)
	args = ap.parse_args()

	rng = random.Random(args.seed)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	state_dir = Path(args.state_dir)

	word_to_idx, idx_to_word, glyph_table = rebuild_vocab_era(Path(args.corpus_dir))
	sch = json.loads((state_dir / "school_state_k65p.json").read_text(encoding="utf-8"))
	model = build_model(sch["embedding_mode"], sch["hidden_dim"], glyph_table, sch["num_layers"])
	ckpt = torch.load(state_dir / args.checkpoint, map_location=device)
	ckpt_table = ckpt["model_state_dict"]["glyph_embedding.glyph_table"].cpu().numpy()
	assert ckpt_table.shape == glyph_table.shape, (
		f"vocabulario desajustado: checkpoint {ckpt_table.shape} vs reconstruido {glyph_table.shape}")
	assert np.allclose(ckpt_table, glyph_table), "la tabla reconstruida no coincide con la del checkpoint"
	print(f"✓ vocabulario de la época verificado contra el checkpoint ({glyph_table.shape[0]} tokens)")
	model.load_state_dict(ckpt["model_state_dict"])
	model.to(device).eval()
	pad_idx = word_to_idx["<pad>"]
	print(f"⚡ {args.checkpoint} | {sch['embedding_mode']} {sch['hidden_dim']}d | {device} | vocab {len(word_to_idx)}")

	stages = ["preschool", "primary", "secondary", "ood_holdout"]
	results = {"checkpoint": args.checkpoint, "hidden_dim": sch["hidden_dim"], "n": args.n, "k": args.k, "stages": {}}

	for stage in stages:
		path = Path(args.corpus_dir) / f"{stage}.jsonl"
		lines = [json.loads(l)["k65p_canonical"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
		sample = rng.sample(lines, min(args.n, len(lines)))

		rows = []
		for text in sample:
			can_loss, n_tok = seq_loss(model, text, word_to_idx, pad_idx, device)
			if n_tok == 0:
				continue
			perms = []
			while len(perms) < args.k:
				ptext = permute_content(text, rng)
				if ptext is None:
					break
				perms.append(ptext)
			if not perms:
				continue
			perm_losses = []
			perm_valid = []
			for ptext in perms:
				pl, _ = seq_loss(model, ptext, word_to_idx, pad_idx, device)
				perm_losses.append(pl)
				try:
					perm_valid.append(bool(is_valid(parse_k65p(ptext))))
				except SyntaxError:
					perm_valid.append(False)
			rows.append({
				"text": text,
				"can_loss": can_loss,
				"perm_loss_mean": float(np.mean(perm_losses)),
				"perm_loss_min": float(np.min(perm_losses)),
				"perm_valid_frac": float(np.mean(perm_valid)),
			})

		n_rows = len(rows)
		can = np.array([r["can_loss"] for r in rows])
		pm = np.array([r["perm_loss_mean"] for r in rows])
		pmin = np.array([r["perm_loss_min"] for r in rows])
		valid_frac = np.array([r["perm_valid_frac"] for r in rows])
		discrim = float(np.mean(pm > can)) if n_rows else 0.0
		hard = float(np.mean((pmin > can) & (valid_frac == 0))) if n_rows else 0.0

		results["stages"][stage] = {
			"n_seq": n_rows,
			"can_loss_mean": float(can.mean()),
			"perm_loss_mean": float(pm.mean()),
			"delta": float(pm.mean() - can.mean()),
			"discrimination_rate": discrim,
			"all_perms_invalid_frac": float(np.mean(valid_frac == 0)) if n_rows else 0.0,
			"hard_discrimination": hard,
			"perm_valid_frac_mean": float(valid_frac.mean()) if n_rows else 0.0,
		}
		r = results["stages"][stage]
		print(f"  {stage:12s} n={r['n_seq']:4d} | can {r['can_loss_mean']:.4f} vs perm {r['perm_loss_mean']:.4f} "
			f"| Δ {r['delta']:+.4f} | discrimina {discrim*100:.1f}% | perms válidas {r['perm_valid_frac_mean']*100:.0f}%")

	out = base_dir / "storage" / "reports"
	out.mkdir(exist_ok=True)
	(out / "k65p_permutation_test.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
	print(f"→ {out / 'k65p_permutation_test.json'}")


if __name__ == "__main__":
	main()
