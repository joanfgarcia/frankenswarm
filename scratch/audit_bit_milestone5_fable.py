"""Cold audit of Bit at milestone 5 (Fable review, 2026-07-03).

Four measurements, no narrative:
1. Corpus census — how many tokens is Bit actually trained on.
2. Teacher-forced next-token accuracy + perplexity on curriculum samples.
3. Grammatical preference — logprob of well-formed vs scrambled novel sentences (no crutches).
4. Generation: raw (no penalties, no filters) vs assisted (verify_milestone_4 crutches).
"""

import json
import math
import os
import re
import sys

import numpy as np
import torch

base_dir = "/home/joan/Documents/IA/frankenswarm"
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, "src"))

from bitnet.modeling_bitnet import BitNet4LayerModel  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT = os.path.join(base_dir, "storage/checkpoints/sovereign_school/model_milestone_5_years.pt")
HIDDEN, LAYERS = 640, 6


def tokenize(text, word_to_idx):
	words = re.findall(r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_<>\-]+", text.lower())
	return [word_to_idx.get(w, word_to_idx.get("<unk>", 1)) for w in words]


def load_vocab():
	with open(os.path.join(base_dir, "configs/expanded_glyphs.json"), encoding="utf-8") as f:
		vocab_data = json.load(f)
	words = vocab_data["words"]
	glyphs = np.array(vocab_data["glyphs"], dtype=np.float32)
	return words, glyphs, {w: i for i, w in enumerate(words)}


def corpus_census(word_to_idx):
	sources = {
		"tiny_dialogues_large.json": None,
		"childes_pre_school.json": None,
		"nsm_physics_pre_school.json": None,
		"school_curriculum_structured.json": None,
	}
	total = 0
	print("── 1. CENSO DEL CORPUS ──")
	for name in sources:
		path = os.path.join(base_dir, "configs", name)
		if not os.path.exists(path):
			print(f"  {name}: NO EXISTE")
			continue
		with open(path, encoding="utf-8") as f:
			data = json.load(f)
		texts = []
		if name.startswith("school_curriculum"):
			for stage_items in data.get("curriculum", {}).values():
				texts += [it["text"] for it in stage_items]
		elif name.startswith("tiny_dialogues"):
			texts = [" ".join(d) if isinstance(d, list) else str(d) for d in data]
		else:
			texts = [str(s) for s in data]
		n_tokens = sum(len(tokenize(t, word_to_idx)) for t in texts)
		total += n_tokens
		print(f"  {name}: {len(texts):,} items, ~{n_tokens:,} tokens")
	print(f"  TOTAL ≈ {total:,} tokens")
	return total


def load_model(glyphs):
	model = BitNet4LayerModel(
		use_glyphs=True, glyph_table=glyphs,
		hidden_dim=HIDDEN, num_layers=LAYERS, use_pos_embedding=True,
		is_causal=True, max_seq_len=128,
	).to(DEVICE)
	sd = torch.load(CKPT, map_location=DEVICE, weights_only=True)
	model.load_state_dict(sd, strict=True)
	model.eval()
	return model


@torch.no_grad()
def seq_logprob_and_acc(model, tokens):
	"""Teacher-forced total logprob, next-token accuracy over a token sequence."""
	if len(tokens) < 3:
		return None
	x = torch.tensor([tokens], dtype=torch.long, device=DEVICE)
	logits = model(x)
	logp = torch.log_softmax(logits[0, :-1, :], dim=-1)
	targets = x[0, 1:]
	tok_logp = logp[torch.arange(len(targets)), targets]
	acc = (logits[0, :-1, :].argmax(-1) == targets).float().mean().item()
	return tok_logp.sum().item(), tok_logp.mean().item(), acc, len(targets)


def teacher_forced_eval(model, word_to_idx):
	print("\n── 2. TEACHER-FORCED (muestra del currículo, 200 frases) ──")
	with open(os.path.join(base_dir, "configs/childes_pre_school.json"), encoding="utf-8") as f:
		childes = json.load(f)
	rng = np.random.RandomState(770)
	sample = [childes[i] for i in rng.choice(len(childes), size=min(200, len(childes)), replace=False)]
	accs, mean_lps, n_tok = [], [], 0
	for s in sample:
		res = seq_logprob_and_acc(model, tokenize(s, word_to_idx))
		if res:
			_, mean_lp, acc, n = res
			accs.append(acc)
			mean_lps.append(mean_lp)
			n_tok += n
	ppl = math.exp(-np.mean(mean_lps))
	print(f"  next-token accuracy: {np.mean(accs)*100:.1f}%  |  perplexity: {ppl:.1f}  |  tokens evaluados: {n_tok:,}")
	print("  (OJO: distribución de entrenamiento — mide memorización+ajuste, no generalización)")


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
]


def grammatical_preference(model, word_to_idx):
	print("\n── 3. PREFERENCIA GRAMATICAL (frases nuevas vs desordenadas, sin muletas) ──")
	usable, wins = 0, 0
	for good, bad in GRAM_PAIRS:
		gt, bt = tokenize(good, word_to_idx), tokenize(bad, word_to_idx)
		unk = word_to_idx.get("<unk>", 1)
		if unk in gt or unk in bt:
			continue
		g = seq_logprob_and_acc(model, gt)
		b = seq_logprob_and_acc(model, bt)
		if not g or not b:
			continue
		usable += 1
		win = g[1] > b[1]
		wins += win
		print(f"  {'✅' if win else '❌'} '{good}'  (lp/tok {g[1]:.2f} vs {b[1]:.2f})")
	if usable:
		print(f"  RESULTADO: {wins}/{usable} = {wins/usable*100:.0f}% (azar = 50%)")


@torch.no_grad()
def generate(model, word_to_idx, idx_to_word, prompt, crutches, max_new=15, temp=0.7):
	tokens = tokenize("tú: " + prompt, word_to_idx)
	x = torch.tensor([tokens], dtype=torch.long, device=DEVICE)
	out = []
	pad = word_to_idx.get("<pad>", 0)
	for _ in range(max_new):
		logits = model(x)[0, -1, :].clone()
		if crutches:
			for w in ("<pad>", "<unk>", "yo", "tú"):
				logits[word_to_idx.get(w, 0)] = -1e9
			for t in set(out):
				logits[t] -= 2.0
		probs = torch.softmax(logits / temp, dim=-1)
		nxt = torch.multinomial(probs, 1).item()
		if nxt == pad:
			out.append(nxt)
			break
		out.append(nxt)
		x = torch.cat([x, torch.tensor([[nxt]], device=DEVICE)], dim=1)
	words = [idx_to_word.get(t, "?") for t in out]
	stopped = "<pad>" in words
	return " ".join(words), stopped


def generation_probes(model, word_to_idx, idx_to_word):
	prompts = [
		"hola", "cómo estás", "quién eres", "el sol brilla", "si toco el fuego",
		"tengo frío", "quiero jugar",
		"qué comes cuando tienes hambre", "dónde duerme el gato", "por qué lloras",
	]
	torch.manual_seed(770)
	print("\n── 4a. GENERACIÓN CRUDA (sin filtros, sin penalización) ──")
	stops = 0
	for p in prompts:
		text, stopped = generate(model, word_to_idx, idx_to_word, p, crutches=False)
		stops += stopped
		print(f"  Q: '{p}' -> '{text}'")
	print(f"  paradas naturales (<pad>): {stops}/{len(prompts)}")
	torch.manual_seed(770)
	print("\n── 4b. GENERACIÓN ASISTIDA (muletas del verify: filtros + rep-penalty) ──")
	for p in prompts:
		text, _ = generate(model, word_to_idx, idx_to_word, p, crutches=True)
		print(f"  Q: '{p}' -> '{text}'")


if __name__ == "__main__":
	words, glyphs, word_to_idx = load_vocab()
	idx_to_word = dict(enumerate(words))
	print(f"AUDIT Bit milestone_5 | {HIDDEN}-dim x {LAYERS} capas | vocab {len(words):,} | device {DEVICE}\n")
	corpus_census(word_to_idx)
	model = load_model(glyphs)
	teacher_forced_eval(model, word_to_idx)
	grammatical_preference(model, word_to_idx)
	generation_probes(model, word_to_idx, idx_to_word)
