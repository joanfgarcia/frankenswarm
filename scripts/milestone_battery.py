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
# Los umbrales NO se tocan (doctrina). Lo único versionable son los DATOS de
# examen (pares/prompts), que deben hablar el idioma del corpus vigente.
MILESTONES = {
	"M1": {"grammar": 0.80, "cloze": 0.30, "min_len": 3.0, "stop_rate": 0.80, "ghost_rate": 0.05},
	"M2": {"grammar": 0.85, "cloze": 0.40, "min_len": 5.0, "stop_rate": 0.85, "ghost_rate": 0.03},
	"M3": {"grammar": 0.90, "cloze": 0.50, "min_len": 7.0, "stop_rate": 0.90, "ghost_rate": 0.02},
	"M4": {"grammar": 0.95, "cloze": 0.60, "min_len": 9.0, "stop_rate": 0.90, "ghost_rate": 0.01},
}

# ── EXAM DATA v2-en (2026-07-28) ──
# v1 (2026-07-03) estaba en castellano; el corpus pasó a inglés el 14-jul
# (dd09ba4) y la batería quedó examinando en un idioma que el modelo ya no ve:
# grammar puntuaba 0.000 para cualquier checkpoint post-transición (EXP_079).
# v2 es la traducción fiel de los 20 pares y 8 prompts de v1 — mismo diseño de
# examen, mismo vocabulario preescolar, mismos umbrales. Las puntuaciones NO son
# comparables entre versiones de datos de examen: toda cifra publicada debe
# citar la versión (se imprime en la cabecera del informe).
EXAM_DATA_VERSION = "v2-en (2026-07-28)"

GRAM_PAIRS = [
	("the boy drinks water", "water the drinks boy"),
	("the girl eats bread", "bread eats the girl no"),
	("the dog sees the cat", "the sees cat dog the"),
	("mom gives me a kiss", "kiss a me gives mom"),
	("the sun is big", "big sun is the"),
	("I want to play with you", "you with play to want I"),
	("the fire burns a lot", "lot a burns fire the"),
	("the moon comes out at night", "night at out comes moon the"),
	("dad has a house", "house a has dad"),
	("the water is cold", "cold is water the"),
	("I want to eat an apple", "apple an eat to want I"),
	("the cat drinks milk", "milk drinks cat the"),
	("I am very cold today", "today cold very am I"),
	("the girl sleeps in her bed", "bed her in sleeps girl the"),
	("the bird can fly", "fly can bird the"),
	("my brother runs very fast", "fast very runs brother my"),
	("the table has four legs", "legs four has table the"),
	("today we go to school", "school to go we today"),
	("the baby cries at night", "night at cries baby the"),
	("I want a glass of milk", "milk of glass a want I"),
]

GEN_PROMPTS = ["hello", "how are you", "what do you want to eat", "where is the cat", "why are you crying", "tell me about the sun", "are you cold", "do you want to play"]


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
		x = torch.tensor([tokenize("you: " + p, word_to_idx)], dtype=torch.long, device=device)
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

	print(f"\n── BATERÍA {args.milestone} | exam data {EXAM_DATA_VERSION} | {args.checkpoint} ──")
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
