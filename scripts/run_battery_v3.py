"""Runner de la batería v3 (DL-011): examina un checkpoint con la batería 80/50.

Separa las dos métricas: gate (vistas, entrenadas) y cognición (no vistas,
jamás entrenadas — verificadas ausentes por scripts/generate_battery_v3.py).
Generación idéntica al examen de hito (wrapper de diálogo, máscara de gateo,
5 tokens greedy). Uso:
  .venv/bin/python scripts/run_battery_v3.py --model_path <ckpt> --age 4 \
      [--stage_idx 3] [--device cuda]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.bitnet.model.modeling_bitnet import BitNet4LayerModel  # noqa: E402

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DIALOGUE = {"hello", "how are you", "who are you", "what is your name", "where are you from", "what is your home", "do you like books"}


def build_context(q: str, w2i: dict) -> list:
	words = q.split()
	if q.lower().strip() in DIALOGUE:
		return [w2i.get("you", 1)] + [w2i.get(w, 1) for w in words] + [w2i.get("me", 1)]
	return [w2i.get(w, 1) for w in words]


def generate(model, context: list, allowed_mask, w2i, idx_to_word, device, n_steps=3, resonance=False, pos_mode="clock", emo=None) -> tuple:
	gen = list(context)
	conf = None
	for step_i in range(5):
		padded = gen[-128:] if len(gen) > 128 else gen + [0] * (128 - len(gen))
		x = torch.tensor([padded], dtype=torch.long, device=device)
		with torch.no_grad():
			if resonance:
				logits, _ = model.forward_resonance(x, n_steps=n_steps, pos_mode=pos_mode, emotion_ids=emo)
			else:
				logits = model(x)
		step_logits = logits[0, len(gen) - 1].float()
		current = allowed_mask.clone()
		if step_i == 0:
			current[0] = False
			current[1] = False
		step_logits = step_logits.masked_fill(~current, -1e9)
		tok = int(step_logits.argmax(dim=-1).item())
		if tok in (0, 1):
			break
		gen.append(tok)
	return gen[len(context):], conf


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--model_path", required=True)
	ap.add_argument("--age", type=int, default=4)
	ap.add_argument("--stage_idx", type=int, default=3)
	ap.add_argument("--device", default="cuda")
	ap.add_argument("--hidden_dim", type=int, default=128)
	ap.add_argument("--num_layers", type=int, default=6)
	ap.add_argument("--resonance_steps_max", type=int, default=0)
	ap.add_argument("--resonance_pos_mode", default="clock")
	ap.add_argument("--resonance_eval_steps", type=int, default=3)
	ap.add_argument("--n_emotions", type=int, default=0)
	ap.add_argument("--emotion_dim", type=int, default=16)
	ap.add_argument("--pool_gate", action="store_true", help="Añade el gate de RETENCIÓN: frases aleatorias del pool de etapa (cloze determinista)")
	ap.add_argument("--gate_n", type=int, default=80)
	ap.add_argument("--store_path", default="storage/datasets/tokenized_store.npz")
	ap.add_argument("--emotion_mode", default="first_only")
	args = ap.parse_args()

	battery = json.load(open(os.path.join(BASE, "configs", "battery_v3", f"age{args.age}.json")))
	vocab = json.load(open(os.path.join(BASE, "configs", "expanded_glyphs.json")))
	words = vocab["words"]
	glyphs = np.array(vocab["glyphs"], dtype=np.float32)
	w2i = {w: i for i, w in enumerate(words)}
	idx_to_word = dict(enumerate(words))

	state_dict = torch.load(args.model_path, map_location=args.device, weights_only=True)
	uses_glyphs = any(k.startswith("glyph_embedding.") for k in state_dict)
	kwargs = dict(hidden_dim=args.hidden_dim, num_layers=args.num_layers, use_pos_embedding=True,
		is_causal=True, max_seq_len=128, max_resonance_steps=args.resonance_steps_max,
		n_emotions=args.n_emotions, emotion_dim=args.emotion_dim, emotion_mode=args.emotion_mode)
	if uses_glyphs:
		model = BitNet4LayerModel(use_glyphs=True, glyph_table=glyphs, **kwargs).to(args.device)
	else:
		model = BitNet4LayerModel(use_glyphs=False, vocab_embeddings=np.eye(len(words), dtype=np.float32), **kwargs).to(args.device)
	model.load_state_dict(state_dict)
	model.eval()

	masks = json.load(open(os.path.join(BASE, "storage", "checkpoints", "bit003_glyph", "stage_gate_masks.json")))
	allowed = torch.zeros(len(words), dtype=torch.bool, device=args.device)
	for i in masks["stage_allowed"][args.stage_idx]:
		allowed[i] = True
	allowed[0] = True

	# ── Gate de RETENCIÓN: muestra aleatoria del pool de entrenamiento de la etapa ──
	retention_rows = []
	if args.pool_gate:
		import numpy as _np
		store = _np.load(os.path.join(BASE, args.store_path))
		flat, offsets = store["flat"], store["offsets"]
		stage_cache = None
		cache_dir = os.path.join(BASE, "storage", "datasets", "stage_cache")
		for h in sorted(os.listdir(cache_dir)):
			f = os.path.join(cache_dir, h, f"stage_{args.stage_idx}_indices.npz")
			if os.path.exists(f):
				idx = _np.load(f); train_idx = idx["train"]; break
		rng = _np.random.default_rng(42)
		sel = rng.choice(train_idx, size=min(args.gate_n, len(train_idx)), replace=False)
		allowed = torch.zeros(len(words), dtype=torch.bool, device=args.device)
		for i in masks["stage_allowed"][args.stage_idx]:
			allowed[i] = True
		allowed[0] = True
		hits, total = 0, 0
		for si in sel:
			a_off, b_off = int(offsets[si]), int(offsets[si + 1])
			toks = flat[a_off:b_off].tolist()
			if len(toks) < 3:
				continue
			ctx, ans = toks[:-1], toks[-1]
			x = torch.tensor([ctx], dtype=torch.long, device=args.device)
			with torch.no_grad():
				if args.resonance_steps_max > 0:
					emo = torch.full((1,), args.n_emotions - 1, dtype=torch.long, device=args.device) if args.n_emotions > 0 else None
					logits, _ = model.forward_resonance(x, n_steps=min(args.resonance_eval_steps, args.resonance_steps_max), pos_mode=args.resonance_pos_mode, emotion_ids=emo)
				else:
					logits = model(x)
			pred = int(logits[0, len(ctx) - 1].float().argmax(dim=-1).item())
			total += 1
			hits += int(pred == ans)
		retention_rows.append({"hits": hits, "total": total})
		ret_pct = hits / max(1, total) * 100
		print(f"  RETENCIÓN (pool de etapa, {total} frases): {hits}/{total} = {ret_pct:.1f}%")


	emo = None
	if args.resonance_steps_max > 0 and args.n_emotions > 0:
		emo = torch.full((1,), args.n_emotions - 1, dtype=torch.long, device=args.device)

	def run_set(questions: list, is_seen: bool) -> tuple:
		# Matriz de puntuación 3×2 (definida por el operador, 31-ago):
		#                 correcto   "no sé"   mal
		#   visto            +1        +1      −1   ← honestidad calibrada premia;
		#   no visto         +2         0       0   ← generalizar es EL premio doble
		# El "no sé" nunca castiga; fallar lo enseñado sí. La confianza del
		# primer token alimenta la curva riesgo-cobertura.
		hits_first, hits_any, rows = 0, 0, []
		for item in questions:
			q, a = item["q"], item["a"]
			ctx = build_context(q, w2i)
			gen_toks, conf = generate(model, ctx, allowed, w2i, idx_to_word, args.device,
				n_steps=args.resonance_eval_steps, resonance=args.resonance_steps_max > 0,
				pos_mode=args.resonance_pos_mode, emo=emo)
			gen_words = [idx_to_word.get(t, "<unk>") for t in gen_toks]
			first = gen_words[0] if gen_words else ""
			is_abstain = ("dont" in gen_words[:3] and "know" in gen_words[:3]) or gen_words[:1] == ["know"]
			hit_first = (first == a) and not is_abstain
			hit_any = (a in gen_words) and not is_abstain
			verdict = "hit" if hit_first else ("abstain" if is_abstain else "hallucination")
			if hit_first:
				points = 2 if not is_seen else 1
			elif is_abstain:
				points = 1 if is_seen else 0
			else:
				points = -1 if is_seen else 0
			hits_first += hit_first
			hits_any += hit_any
			rows.append({"q": q, "expected": a, "generated": " ".join(gen_words),
				"verdict": verdict, "points": points, "hit_first": hit_first, "hit_any": hit_any,
				"confidence": conf, "stratum": item.get("stratum", "?"), "seen": is_seen})
		return hits_first, hits_any, rows

	print(f"── Batería edad {args.age} · {os.path.basename(args.model_path)} · {'resonante' if args.resonance_steps_max > 0 else 'normal'} ──")
	g1, g2, rows_seen = run_set(battery["seen_gate"], is_seen=True)
	u1, u2, rows_unseen = run_set(battery["unseen_cognition"], is_seen=False)
	n_s, n_u = len(battery["seen_gate"]), len(battery["unseen_cognition"])

	def three_way(rows):
		hits = sum(1 for r in rows if r["verdict"] == "hit")
		abst = sum(1 for r in rows if r["verdict"] == "abstain")
		wrong = sum(1 for r in rows if r["verdict"] == "hallucination")
		score = sum(r["points"] for r in rows)
		return hits, abst, wrong, score

	sh, sa, sm, s_score = three_way(rows_seen)
	uh, ua, um, u_score = three_way(rows_unseen)
	print(f"\n══ RESULTADOS — matriz: visto +1/0/−1 · no visto +1/0/0 ══")
	print(f"  GATE (vistas, {n_s}): acierto {sh} | abstiene {sa} | falla {sm} | SCORE {s_score:+d}")
	print(f"    tasas: acierto {sh/n_s*100:.1f}% · abstención {sa/n_s*100:.1f}% · fallo {sm/n_s*100:.1f}%")
	print(f"  COGNICIÓN (no vistas, {n_u}): acierto {uh} | abstiene {ua} | falla {um} | SCORE {u_score:+d}")
	print(f"    tasas: acierto {uh/n_u*100:.1f}% · abstención {ua/n_u*100:.1f}% · fallo {um/n_u*100:.1f}%")
	# curva riesgo-cobertura: si Bit abstuviera según confianza, ¿mejoraría?
	all_rows = sorted(rows_seen + rows_unseen, key=lambda r: -(r["confidence"] or 0))
	n_all = len(all_rows)
	coverages = (0.2, 0.4, 0.6, 0.8, 1.0)
	rc = []
	for c in coverages:
		k = max(1, int(n_all * c))
		top = all_rows[:k]
		acc = sum(1 for r in top if r["verdict"] == "hit") / k
		rc.append((c, acc))
	print("  ── riesgo-cobertura (si abstuviera por confianza) ──")
	for c, acc in rc:
		print(f"    responder solo al {c*100:.0f}% más seguro → acierto {acc*100:.1f}%")
	# por estrato: dónde rompe la generalización
	from collections import defaultdict
	by_stratum = defaultdict(lambda: {"hit": 0, "abstain": 0, "hallucination": 0, "total": 0})
	for r in rows_unseen:
		st = r.get("stratum", "?")
		by_stratum[st]["total"] += 1
		by_stratum[st][r["verdict"]] += 1
	print("  ── por estrato (no vistas) ──")
	for st, c in sorted(by_stratum.items()):
		print(f"    {st}: acierto {c['hit']} | abstiene {c['abstain']} | alucina {c['hallucination']} (de {c['total']})")

	out = args.model_path.replace(".pt", f"_battery_age{args.age}.json")
	json.dump({"model": args.model_path, "set_id": battery.get("set_id"),
		"scoring": "seen: +1(correcto)/+1(no sé)/-1(mal) · unseen: +2/0/0",
		"gate": {"hits": sh, "abstains": sa, "wrong": sm, "score": s_score, "total": n_s},
		"cognition": {"hits": uh, "abstains": ua, "wrong": um, "score": u_score, "total": n_u},
		"risk_coverage": [{"coverage": c, "accuracy": round(a, 4)} for c, a in rc],
		"strata": {st: dict(c) for st, c in sorted(by_stratum.items())},
		"rows_seen": rows_seen, "rows_unseen": rows_unseen},
		open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
	print(f"→ {out}")


if __name__ == "__main__":
	main()
