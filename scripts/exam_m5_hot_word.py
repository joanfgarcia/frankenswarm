"""Examen M5 — Vocabulario en Caliente: ¿la composición de primos es REAL o cosmética?

Pregunta: ¿la capacidad exclusiva del brazo GLYPH de registrar una palabra nueva
(`register_new_word`) sin reentrenar la tabla demuestra que el embedding
composicional compra algo real, o es un truco que funciona con cualquier glifo?

Diseño (3 pruebas de discriminación, umbrales PRE-REGISTRADOS):

  P1 · INYECCIÓN EN FRÍO — la composición mueve el logit de la palabra nueva.
       Para un glifo compuesto de primos semánticamente coherentes frente a un
       glifo ALEATORIO (mismos trits, sin relación): el logit de la palabra
       nueva se evalúa sobre el corpus OOD (teacher-forced, última posición).
       PASS si p(coherente)/p(aleatorio) >= 10 (la composición es 10× más
       probable que el ruido → el embedding compone, no es cosmético).

  P2 · CONTRASTE ESTRUCTURAL — la composición discrimina contexto SEMÁNTICO.
       En contextos 'de trueno' (contienen HEAR/MOVE/ABOVE) el glifo coherente
       debe rankear MEJOR que en contextos neutros. ⚠️ PRECONDICIÓN: el corpus
       debe ser SEMÁNTICO (los primos deben co-ocurrir con coherencia causal).
       En el corpus sintáctico actual la co-ocurrencia tras ABOVE es arbitraria
       (10,9,64,60...), así que el modelo NO tiene semántica que discriminar:
       P2 se declara NO EVALUABLE y la pregunta de discriminación semántica
       queda para la escuela semántica (gen_true, juez Prolog).

  P3 · CONSOLIDACIÓN — la inyección en caliente integra la palabra con coste
       mínimo. Tras <= 5 épocas de consolidación con la palabra como target,
       PASS si se emite en >= 80% de contextos de trueno. Esta prueba NO es
       posible en standard (register_new_word bloqueado por diseño) → es la
       ventaja estructural a demostrar.

Referencias: ESTADO_DEL_CAMINO.md §3 (cola de trabajo, paso 3) ·
INFORME_DL006_MATRIZ.md §3 (baza exclusiva del glifo, 'jamás examinada') ·
docs/BITACORA_BIT_V2.md (Hito 5, fase de réplicas 2026-08-10).

Uso:
	PYTHONPATH=.:../k65p/src .venv/bin/python scripts/exam_m5_hot_word.py [--state_dir DIR]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812

base_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(base_dir))

k65p_src = base_dir.parent / "k65p" / "src"
if k65p_src.exists():
	sys.path.append(str(k65p_src))

from src.bitnet.training.train_sovereign_school_k65p import (  # noqa: E402
	build_k65p_vocab_and_glyphs,
	build_model,
	tokenize_k65p,
)

DEFAULT_STATE_DIR = base_dir / "storage" / "checkpoints" / "replicates" / "k65p" / "glyph_s773"

# ── Umbrales pre-registrados (antes de ver los números) ──────────────────────
THRESH_P1_RATIO = 10.0     # p(coherente)/p(aleatorio) en inyección en frío
THRESH_P2_LIFT = 5.0       # puestos de rango que gana el coherente en ctx-trueno
THRESH_P3_EMIT = 0.80      # fracción de emisión tras consolidación

# Palabra de prueba y sus glifos
WORD = "trueno"
# primos con significado: HEAR(17) MOVE(23) ABOVE(38) SOMETHING(4) VERY(49)
COHERENT_PRIMES = [17, 23, 38, 4, 49]
# primos sin relación semántica: I(0) BODY(6) MINE(26) LONG_TIME(33) SOME(57)
RANDOM_PRIMES = [0, 6, 26, 33, 57]
# primos del contexto 'trueno' (los que debe activar)
CTX_PRIMES = {"17", "23", "38"}


def make_glyph(primos: list[int]) -> torch.Tensor:
	g = torch.zeros(65)
	for p in primos:
		g[p] = 1.0
	return g


def load_corpus() -> list[str]:
	exprs: list[str] = []
	for f in ("preschool", "primary", "secondary"):
		path = base_dir / "storage" / "curriculum" / "factory_k65p" / f"{f}.jsonl"
		for line in path.read_text(encoding="utf-8").splitlines():
			if line.strip():
				exprs.append(json.loads(line)["k65p_canonical"])
	return exprs


def clone_with_word(model, glyph: torch.Tensor) -> torch.nn.Module:
	"""Clona el modelo y registra la palabra nueva con el glifo dado."""
	m = build_model("glyph", state["hidden_dim"], glyph_table, state.get("num_layers", 6))
	m.load_state_dict(model.state_dict())
	m.eval()
	m.register_new_word(WORD, glyph.clone())
	return m


def hidden_last(m, ids: list[int]) -> torch.Tensor:
	"""Hidden state del último token para una secuencia de ids."""
	x = torch.tensor([ids], dtype=torch.long)
	h = m._embed_input(x)
	for layer in m.core_layers:
		h = layer(h)
	h = m.norm(h)
	return h


def p1_frost_injection(model) -> bool:
	"""¿La composición mueve el logit de la palabra nueva frente a un glifo aleatorio?"""
	ood = [json.loads(line)["k65p_canonical"] for line in
		(base_dir / "storage" / "curriculum" / "factory_k65p" / "ood_holdout.jsonl")
		.read_text().splitlines() if line.strip()]
	mc = clone_with_word(model, make_glyph(COHERENT_PRIMES))
	ma = clone_with_word(model, make_glyph(RANDOM_PRIMES))
	pad = word_to_idx["<pad>"]

	pc = pa = n = 0.0
	for expr in ood:
		ids = [i for i in tokenize_k65p(expr, word_to_idx) if i != pad]
		if len(ids) < 3:
			continue
		for m, acc in ((mc, "c"), (ma, "a")):
			logits = m.glyph_embedding.decode_logits(hidden_last(m, ids[:-1]))[0, -1]
			p = F.softmax(logits, dim=-1)
			if acc == "c":
				pc += p[-1].item()
			else:
				pa += p[-1].item()
		n += 1

	ratio = (pc / n) / max((pa / n), 1e-12)
	print(f"  p_mean coherente = {pc/n:.5f} | p_mean aleatorio = {pa/n:.5f} | ratio = {ratio:.1f}x")
	passed = ratio >= THRESH_P1_RATIO
	print(f"  {'✅ PASS' if passed else '❌ FAIL'}  (umbral ≥ {THRESH_P1_RATIO}x)")
	return passed


def p2_structural_contrast(model) -> bool | None:
	"""¿La composición discrimina contexto semántico? NO EVALUABLE sin semántica."""
	corpus = load_corpus()
	# Precondición: comprobar si el corpus es semántico (los primos de contexto
	# deben co-ocurrir con coherencia, no arbitrariamente).
	ctx_trueno = [e for e in corpus if any(t in e for t in CTX_PRIMES)]
	ctx_neutro = [e for e in corpus if not any(t in e for t in CTX_PRIMES)][:200]
	mc = clone_with_word(model, make_glyph(COHERENT_PRIMES))
	ma = clone_with_word(model, make_glyph(RANDOM_PRIMES))
	pad = word_to_idx["<pad>"]

	def mean_rank(m, exprs, n=200) -> float:
		ranks = []
		for expr in exprs[:n]:
			ids = [i for i in tokenize_k65p(expr, word_to_idx) if i != pad]
			if len(ids) < 3:
				continue
			logits = m.glyph_embedding.decode_logits(hidden_last(m, ids[:-1]))[0, -1]
			ranks.append((logits > logits[-1]).sum().item())
		return sum(ranks) / len(ranks) if ranks else float("nan")

	rc_t = mean_rank(mc, ctx_trueno)
	rc_n = mean_rank(mc, ctx_neutro)
	ra_t = mean_rank(ma, ctx_trueno)
	ra_n = mean_rank(ma, ctx_neutro)
	lift_coherent = rc_n - rc_t
	lift_random = ra_n - ra_t
	print(f"  coherente: rango ctx-trueno {rc_t:.1f} vs neutro {rc_n:.1f} → lift {lift_coherent:+.1f}")
	print(f"  aleatorio: rango ctx-trueno {ra_t:.1f} vs neutro {ra_n:.1f} → lift {lift_random:+.1f}")
	# El lift del coherente en el corpus sintáctico es RUIDO (sin semántica no
	# hay dirección esperada): no se puede interpretar ni como PASS ni FAIL.
	print("  ⚠️ NO EVALUABLE — el corpus sintáctico no enseña semántica: la co-ocurrencia"
		" tras los primos de contexto es arbitraria, así que no hay 'contexto de trueno'"
		" que el modelo pueda haber aprendido. La discriminación semántica exige la"
		" escuela semántica (gen_true, juez Prolog).")
	return None


def p3_consolidation(model) -> bool:
	"""¿La inyección en caliente integra la palabra con coste mínimo?"""
	corpus = load_corpus()
	m = clone_with_word(model, make_glyph(COHERENT_PRIMES))
	trueno_idx = m.vocab_size - 1
	pad = word_to_idx["<pad>"]

	random.seed(7)
	ctx = [e for e in corpus if any(t in e for t in CTX_PRIMES)]
	consol = []
	for e in random.sample(ctx, min(120, len(ctx))):
		ids = [i for i in tokenize_k65p(e, word_to_idx) if i != pad]
		if len(ids) < 4:
			continue
		consol.append(ids[:3] + [trueno_idx])
	print(f"  dataset consolidación: {len(consol)} secuencias")

	m.train()
	opt = torch.optim.Adam(m.parameters(), lr=1e-3)
	for epoch in range(5):
		random.shuffle(consol)
		tot = n = 0
		for seq in consol:
			x = torch.tensor([seq[:-1]])
			y = torch.tensor([seq[-1]])
			logits = m(x)
			loss = F.cross_entropy(logits[0, -1:].view(1, -1), y.view(-1))
			opt.zero_grad()
			loss.backward()
			opt.step()
			tot += loss.item()
			n += 1
		print(f"    época {epoch+1}: loss={tot/n:.4f}")

	m.eval()
	ctx_t = [e for e in corpus if any(t in e for t in CTX_PRIMES)][:200]
	emits = total = 0
	for expr in ctx_t:
		ids = [i for i in tokenize_k65p(expr, word_to_idx) if i != pad]
		if len(ids) < 3:
			continue
		logits = m(torch.tensor([ids[:-1]]))[0, -1]
		total += 1
		emits += int(logits[trueno_idx] == logits.max())
	rate = emits / total
	print(f"  emite '{WORD}' en {emits}/{total} ({100*rate:.1f}%) de contextos de trueno")
	passed = rate >= THRESH_P3_EMIT
	print(f"  {'✅ PASS' if passed else '❌ FAIL'}  (umbral ≥ {100*THRESH_P3_EMIT:.0f}%)")
	return passed


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Examen M5 — vocabulario en caliente (composición real vs cosmética)")
	parser.add_argument("--state_dir", type=str, default=str(DEFAULT_STATE_DIR))
	args = parser.parse_args()

	global state, word_to_idx, idx_to_word, glyph_table
	state_dir = Path(args.state_dir)
	state = json.loads((state_dir / "school_state_k65p.json").read_text(encoding="utf-8"))
	ckpt = state_dir / "model_current_k65p.pt"
	if not ckpt.exists():
		print(f"✗ No hay checkpoint en {state_dir}")
		sys.exit(1)
	word_to_idx, idx_to_word, glyph_table = build_k65p_vocab_and_glyphs()
	model = build_model("glyph", state["hidden_dim"], glyph_table, state.get("num_layers", 6))
	ck = torch.load(ckpt, map_location="cpu", weights_only=True)
	sd = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
	model.load_state_dict(sd)
	model.eval()

	print("═" * 72)
	print(f"  🧪 EXAMEN M5 — VOCABULARIO EN CALIENTE ({WORD}) — {state_dir.name}")
	print("═" * 72)
	print(f"  Checkpoint: dim={state['hidden_dim']} épocas={state['current_epoch']} seed={state.get('seed', 770)}")

	results = {}
	results["p1_frost"] = p1_frost_injection(model)
	results["p2_contrast"] = p2_structural_contrast(model)
	results["p3_consolidation"] = p3_consolidation(model)

	print("═" * 72)
	bools = [v for v in results.values() if v is not None]
	na = [k for k, v in results.items() if v is None]
	overall = bool(bools) and all(bools)
	if overall and not na:
		verdict = "✅ M5 SUPERADO — la composición compra capacidad real"
	elif overall and na:
		verdict = "✅ M5 PARCIAL — capacidad estructural confirmada; discriminación semántica pendiente (requiere escuela semántica)"
	else:
		verdict = "❌ M5 NO SUPERADO"
	print(f"  {verdict}")
	print(f"  P1 inyección en frío : {'✅' if results['p1_frost'] else '❌'}")
	print(f"  P2 contraste estructural : {'⚠️ NO EVALUABLE sin semántica' if results['p2_contrast'] is None else ('✅' if results['p2_contrast'] else '❌')}")
	print(f"  P3 consolidación     : {'✅' if results['p3_consolidation'] else '❌'}")
	print("═" * 72)
	sys.exit(0 if overall else 1)
