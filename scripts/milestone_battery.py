"""Milestone Battery v1 — operational, judge-free milestone certification (School v3).

Replaces age analogies with three measurable exams. Thresholds are PRE-REGISTERED
here, in code, before training School v3 — changing them after seeing results
requires a documented decision, not an edit.

Exams:
  A. Grammar preference — logprob of well-formed novel sentences vs scrambled (no crutches).
  B. Cloze — next-token top-1 accuracy on a HELD-OUT file (never trained on).
  C. Production health — free generation: mean length, natural-stop rate, ghost-word rate.

Usage:
	PYTHONPATH=. .venv/bin/python scripts/milestone_battery.py \
		--checkpoint storage/checkpoints/sovereign_school_v3/model_milestone_A.pt \
		--hidden 256 --layers 6 --milestone M1 \
		--heldout storage/curriculum/heldout_preschool.json
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import torch

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, "src"))

from bitnet.modeling_bitnet import BitNet4LayerModel  # noqa: E402

# ── PRE-REGISTERED THRESHOLDS (School v3, frozen 2026-07-03) ──
MILESTONES = {
	"M1": {"grammar": 0.80, "cloze": 0.30, "min_len": 3.0, "stop_rate": 0.80, "ghost_rate": 0.05},
	"M2": {"grammar": 0.85, "cloze": 0.40, "min_len": 5.0, "stop_rate": 0.85, "ghost_rate": 0.03},
	"M3": {"grammar": 0.90, "cloze": 0.50, "min_len": 7.0, "stop_rate": 0.90, "ghost_rate": 0.02},
	"M4": {"grammar": 0.95, "cloze": 0.60, "min_len": 9.0, "stop_rate": 0.90, "ghost_rate": 0.01},
}

GRAM_PAIRS = [
	("el niño bebe agua", "agua el bebe niño"),
	("la niña come pan", "pan come la niña no"),
	("el perro ve al gato", "al ve gato perro el"),
	("mamá me da un beso", "beso un da me mamá"),
	("el sol es grande", "grande sol es el"),
	("yo quiero jugar contigo", "contigo jugar quiero yo"),
	("el fuego quema mucho", "mucho quema fuego el"),
	("la luna sale de noche", "noche de sale luna la"),
	("papá tiene una casa", "casa una tiene papá"),
	("el agua está fría", "fría está agua el"),
	("quiero comer una manzana", "manzana una comer quiero"),
	("el gato bebe leche", "leche bebe gato el"),
	("tengo mucho frío hoy", "hoy frío mucho tengo"),
	("la niña duerme en su cama", "cama su en duerme niña la"),
	("el pájaro puede volar", "volar puede pájaro el"),
	("mi hermano corre muy rápido", "rápido muy corre hermano mi"),
	("la mesa tiene cuatro patas", "patas cuatro tiene mesa la"),
	("hoy vamos a la escuela", "escuela la a vamos hoy"),
	("el bebé llora por la noche", "noche la por llora bebé el"),
	("quiero un vaso de leche", "leche de vaso un quiero"),
]

GEN_PROMPTS = ["hola", "cómo estás", "qué quieres comer", "dónde está el gato", "por qué lloras", "cuéntame algo del sol", "tienes frío", "vamos a jugar"]


def tokenize(text, word_to_idx):
	words = re.findall(r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_<>\-]+", text.lower())
	return [word_to_idx.get(w, word_to_idx.get("<unk>", 1)) for w in words]


@torch.no_grad()
def mean_logprob(model, tokens, device):
	if len(tokens) < 3:
		return None
	x = torch.tensor([tokens], dtype=torch.long, device=device)
	logits = model(x)
	logp = torch.log_softmax(logits[0, :-1, :], dim=-1)
	targets = x[0, 1:]
	return logp[torch.arange(len(targets)), targets].mean().item()


@torch.no_grad()
def cloze_accuracy(model, sentences, word_to_idx, device):
	hits, total = 0, 0
	for s in sentences:
		toks = tokenize(s, word_to_idx)
		if len(toks) < 3:
			continue
		x = torch.tensor([toks], dtype=torch.long, device=device)
		logits = model(x)
		preds = logits[0, :-1, :].argmax(-1)
		targets = x[0, 1:]
		hits += (preds == targets).sum().item()
		total += len(targets)
	return hits / max(total, 1)


@torch.no_grad()
def production_health(model, word_to_idx, idx_to_word, clean_vocab, device, max_new=20):
	pad = word_to_idx.get("<pad>", 0)
	lengths, stops, ghost, n_words = [], 0, 0, 0
	torch.manual_seed(770)
	for p in GEN_PROMPTS:
		x = torch.tensor([tokenize("tú: " + p, word_to_idx)], dtype=torch.long, device=device)
		out = []
		for _ in range(max_new):
			logits = model(x)[0, -1, :]
			nxt = torch.multinomial(torch.softmax(logits / 0.7, -1), 1).item()
			if nxt == pad:
				stops += 1
				break
			out.append(nxt)
			x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
		lengths.append(len(out))
		for t in out:
			n_words += 1
			if clean_vocab is not None and idx_to_word.get(t, "") not in clean_vocab:
				ghost += 1
	return float(np.mean(lengths)), stops / len(GEN_PROMPTS), ghost / max(n_words, 1)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--checkpoint", required=True)
	ap.add_argument("--hidden", type=int, required=True)
	ap.add_argument("--layers", type=int, default=6)
	ap.add_argument("--milestone", choices=list(MILESTONES), required=True)
	ap.add_argument("--heldout", help="JSON list de frases held-out (nunca entrenadas)")
	ap.add_argument("--glyphs", default="configs/expanded_glyphs.json")
	args = ap.parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	with open(os.path.join(base_dir, args.glyphs), encoding="utf-8") as f:
		vocab_data = json.load(f)
	words = vocab_data["words"]
	glyphs = np.array(vocab_data["glyphs"], dtype=np.float32)
	word_to_idx = {w: i for i, w in enumerate(words)}
	idx_to_word = dict(enumerate(words))

	clean_path = os.path.join(base_dir, "configs/clean_vocabulary_words.json")
	clean_vocab = None
	if os.path.exists(clean_path):
		with open(clean_path, encoding="utf-8") as f:
			clean_vocab = set(json.load(f)["words"])

	model = BitNet4LayerModel(
		use_glyphs=True, glyph_table=glyphs, hidden_dim=args.hidden,
		num_layers=args.layers, use_pos_embedding=True, is_causal=True, max_seq_len=128,
	).to(device)
	model.load_state_dict(torch.load(os.path.join(base_dir, args.checkpoint), map_location=device, weights_only=True), strict=True)
	model.eval()

	th = MILESTONES[args.milestone]
	results = {}

	usable, wins = 0, 0
	unk = word_to_idx.get("<unk>", 1)
	for good, bad in GRAM_PAIRS:
		gt, bt = tokenize(good, word_to_idx), tokenize(bad, word_to_idx)
		if unk in gt or unk in bt:
			continue
		g, b = mean_logprob(model, gt, device), mean_logprob(model, bt, device)
		if g is None or b is None:
			continue
		usable += 1
		wins += g > b
	results["grammar"] = wins / max(usable, 1)

	if args.heldout:
		with open(os.path.join(base_dir, args.heldout), encoding="utf-8") as f:
			results["cloze"] = cloze_accuracy(model, json.load(f), word_to_idx, device)
	else:
		results["cloze"] = None
		print("⚠️ Sin --heldout: el examen B no puntúa (y sin held-out NO se certifica).")

	mean_len, stop_rate, ghost_rate = production_health(model, word_to_idx, idx_to_word, clean_vocab, device)
	results.update({"min_len": mean_len, "stop_rate": stop_rate, "ghost_rate": ghost_rate})

	print(f"\n── BATERÍA {args.milestone} | {args.checkpoint} ──")
	verdicts = {
		"grammar": results["grammar"] >= th["grammar"],
		"cloze": results["cloze"] is not None and results["cloze"] >= th["cloze"],
		"min_len": results["min_len"] >= th["min_len"],
		"stop_rate": results["stop_rate"] >= th["stop_rate"],
		"ghost_rate": results["ghost_rate"] <= th["ghost_rate"],
	}
	for k, ok in verdicts.items():
		shown = f"{results[k]:.3f}" if results[k] is not None else "N/A"
		print(f"  {'✅' if ok else '❌'} {k}: {shown} (umbral {th[k]})")
	passed = all(verdicts.values())
	print(f"\n{'🎓 HITO CERTIFICADO' if passed else '🛑 HITO NO ALCANZADO'} — {sum(verdicts.values())}/5 exámenes")
	sys.exit(0 if passed else 1)


if __name__ == "__main__":
	main()
