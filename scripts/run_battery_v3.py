"""Runner de la batería v3 (DL-011): examina un checkpoint con la batería congelada.

Separa las dos métricas: gate (vistas, entrenadas) y cognición (no vistas,
jamás entrenadas — verificadas ausentes por scripts/generate_battery_v3.py).
Generación idéntica al examen de hito (wrapper de diálogo, máscara de gateo,
5 tokens greedy). Uso:
  .venv/bin/python scripts/run_battery_v3.py --model_path <ckpt> --age 4 \
      [--stage_idx 3] [--device cuda] [--hidden_dim 128] [--pool_gate]

Auditoría 2026-09-01 (correcciones sobre la primera versión):
  - la confianza del primer token se CALCULA (softmax post-máscara) — antes
    era siempre None y la curva riesgo-cobertura ordenaba vistas→no vistas;
  - el contexto se tokeniza con tokenize() (la puntuación desaparece) — antes
    q.split() metía <unk> en 23/195 no-vistas (19/38 de aritmética);
  - los duplicados exactos (q, a) se descartan al cargar (los denominadores
    ya no cuentan repetidas);
  - el gate de RETENCIÓN y la procedencia completa se persisten en el JSON;
  - la máscara de gateo se toma del directorio del checkpoint evaluado
    (fallback al legado bit003_glyph si no existe);
  - curvas riesgo-cobertura separadas: mixta, solo-vistas y solo-no-vistas.
"""
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.bitnet.model.modeling_bitnet import BitNet4LayerModel  # noqa: E402
from src.bitnet.training.modules.tokenization import tokenize  # noqa: E402

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DIALOGUE = {"hello", "how are you", "who are you", "what is your name", "where are you from", "what is your home", "do you like books"}


def build_context(q: str, w2i: dict) -> list:
	toks = tokenize(q, w2i)
	if q.lower().strip() in DIALOGUE:
		return [w2i.get("you", 1)] + toks + [w2i.get("me", 1)]
	return toks


def generate(model, context: list, allowed_mask, device, n_steps=3, resonance=False, pos_mode="clock", emo=None) -> tuple:
	"""Genera ≤5 tokens greedy bajo la máscara. Devuelve (tokens, confianza),
	donde confianza = probabilidad softmax post-máscara del PRIMER token."""
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
		step_logits = logits[0, min(len(gen), 128) - 1].float()
		current = allowed_mask.clone()
		if step_i == 0:
			current[0] = False
			current[1] = False
		step_logits = step_logits.masked_fill(~current, -1e9)
		tok = int(step_logits.argmax(dim=-1).item())
		if step_i == 0:
			conf = float(torch.softmax(step_logits, dim=-1)[tok].item())
		if tok in (0, 1):
			break
		gen.append(tok)
	return gen[len(context):], conf


def dedupe(items: list, key_fn) -> tuple:
	"""Descarta duplicados exactos preservando el orden. Devuelve (únicos, n_descartados)."""
	seen_keys, out = set(), []
	for it in items:
		k = key_fn(it)
		if k in seen_keys:
			continue
		seen_keys.add(k)
		out.append(it)
	return out, len(items) - len(out)


def file_md5(path: str) -> str:
	h = hashlib.md5()
	with open(path, "rb") as f:
		for chunk in iter(lambda: f.read(1 << 20), b""):
			h.update(chunk)
	return h.hexdigest()


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
	ap.add_argument("--masks_path", default=None, help="stage_gate_masks.json a usar (default: el del directorio del checkpoint)")
	ap.add_argument("--battery_path", default=None, help="Fichero de batería (default: configs/battery_v3/age{age}.json)")
	ap.add_argument("--emotion_mode", default="first_only")
	args = ap.parse_args()

	battery_path = args.battery_path or os.path.join(BASE, "configs", "battery_v3", f"age{args.age}.json")
	battery = json.load(open(battery_path))
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

	masks_path = args.masks_path or os.path.join(os.path.dirname(os.path.abspath(args.model_path)), "stage_gate_masks.json")
	if not os.path.exists(masks_path):
		masks_path = os.path.join(BASE, "storage", "checkpoints", "bit003_glyph", "stage_gate_masks.json")
	masks = json.load(open(masks_path))
	allowed = torch.zeros(len(words), dtype=torch.bool, device=args.device)
	for i in masks["stage_allowed"][args.stage_idx]:
		allowed[i] = True
	allowed[0] = True

	# ── Gate de RETENCIÓN: muestra aleatoria del pool de entrenamiento de la etapa ──
	retention = None
	if args.pool_gate:
		store = np.load(os.path.join(BASE, args.store_path))
		flat, offsets = store["flat"], store["offsets"]
		cache_dir = os.path.join(BASE, "storage", "datasets", "stage_cache")
		train_idx, cache_used = None, None
		for h in sorted(os.listdir(cache_dir)):
			f = os.path.join(cache_dir, h, f"stage_{args.stage_idx}_indices.npz")
			if os.path.exists(f):
				train_idx = np.load(f)["train"]; cache_used = f; break
		if train_idx is None:
			raise SystemExit(f"--pool_gate: sin stage_{args.stage_idx}_indices.npz bajo {cache_dir}")
		rng = np.random.default_rng(42)
		sel = rng.choice(train_idx, size=min(args.gate_n, len(train_idx)), replace=False)
		hits, total, rows = 0, 0, []
		for si in sel:
			a_off, b_off = int(offsets[si]), int(offsets[si + 1])
			toks = flat[a_off:b_off].tolist()
			if len(toks) < 3:
				continue
			ctx, ans = toks[:-1], toks[-1]
			x = torch.tensor([ctx[-128:]], dtype=torch.long, device=args.device)
			with torch.no_grad():
				if args.resonance_steps_max > 0:
					emo_r = torch.full((1,), args.n_emotions - 1, dtype=torch.long, device=args.device) if args.n_emotions > 0 else None
					logits, _ = model.forward_resonance(x, n_steps=min(args.resonance_eval_steps, args.resonance_steps_max), pos_mode=args.resonance_pos_mode, emotion_ids=emo_r)
				else:
					logits = model(x)
			pred = int(logits[0, min(len(ctx), 128) - 1].float().argmax(dim=-1).item())
			total += 1
			ok = pred == ans
			hits += int(ok)
			rows.append({"seq_idx": int(si), "expected": idx_to_word.get(ans, "<unk>"),
				"predicted": idx_to_word.get(pred, "<unk>"), "hit": ok})
		retention = {"hits": hits, "total": total, "pct": round(hits / max(1, total) * 100, 1),
			"sample_seed": 42, "gate_n": args.gate_n, "stage_cache": os.path.relpath(cache_used, BASE),
			"rows": rows}
		print(f"  RETENCIÓN (pool de etapa, {total} frases): {hits}/{total} = {retention['pct']}%")

	emo = None
	if args.resonance_steps_max > 0 and args.n_emotions > 0:
		emo = torch.full((1,), args.n_emotions - 1, dtype=torch.long, device=args.device)

	seen_items, seen_dups = dedupe(battery["seen_gate"], lambda it: (it["q"], it["a"]))
	unseen_items, unseen_dups = dedupe(battery["unseen_cognition"], lambda it: (it["q"], it["a"]))

	def run_set(questions: list, is_seen: bool) -> list:
		# Matriz de puntuación 3×2 (definida por el operador, 31-ago):
		#                 correcto   "no sé"   mal
		#   visto            +1        +1      −1   ← honestidad calibrada premia;
		#   no visto         +2         0       0   ← generalizar es EL premio doble
		# El "no sé" nunca castiga; fallar lo enseñado sí. La confianza del
		# primer token alimenta la curva riesgo-cobertura.
		rows = []
		for item in questions:
			q, a = item["q"], item["a"]
			ctx = build_context(q, w2i)
			gen_toks, conf = generate(model, ctx, allowed, args.device,
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
			rows.append({"q": q, "expected": a, "generated": " ".join(gen_words),
				"verdict": verdict, "points": points, "hit_first": hit_first, "hit_any": hit_any,
				"confidence": conf, "stratum": item.get("stratum", "?"), "seen": is_seen})
		return rows

	print(f"── Batería edad {args.age} · {os.path.basename(args.model_path)} · {'resonante' if args.resonance_steps_max > 0 else 'normal'} ──")
	rows_seen = run_set(seen_items, is_seen=True)
	rows_unseen = run_set(unseen_items, is_seen=False)
	n_s, n_u = len(rows_seen), len(rows_unseen)

	def three_way(rows):
		hits = sum(1 for r in rows if r["verdict"] == "hit")
		abst = sum(1 for r in rows if r["verdict"] == "abstain")
		wrong = sum(1 for r in rows if r["verdict"] == "hallucination")
		score = sum(r["points"] for r in rows)
		return hits, abst, wrong, score

	sh, sa, sm, s_score = three_way(rows_seen)
	uh, ua, um, u_score = three_way(rows_unseen)
	print(f"\n══ RESULTADOS — matriz: visto +1/+1/−1 · no visto +2/0/0 ══")
	print(f"  GATE (vistas, {n_s}): acierto {sh} | abstiene {sa} | falla {sm} | SCORE {s_score:+d}")
	print(f"    tasas: acierto {sh/n_s*100:.1f}% · abstención {sa/n_s*100:.1f}% · fallo {sm/n_s*100:.1f}%")
	print(f"  COGNICIÓN (no vistas, {n_u}): acierto {uh} | abstiene {ua} | falla {um} | SCORE {u_score:+d}")
	print(f"    tasas: acierto {uh/n_u*100:.1f}% · abstención {ua/n_u*100:.1f}% · fallo {um/n_u*100:.1f}%")

	# curva riesgo-cobertura: si Bit abstuviera según confianza, ¿mejoraría?
	def risk_coverage(rows: list) -> list:
		ordered = sorted(rows, key=lambda r: -(r["confidence"] or 0))
		out = []
		for c in (0.2, 0.4, 0.6, 0.8, 1.0):
			k = max(1, int(len(ordered) * c))
			acc = sum(1 for r in ordered[:k] if r["verdict"] == "hit") / k
			out.append({"coverage": c, "accuracy": round(acc, 4)})
		return out

	rc_all = risk_coverage(rows_seen + rows_unseen)
	rc_seen = risk_coverage(rows_seen)
	rc_unseen = risk_coverage(rows_unseen)
	print("  ── riesgo-cobertura por confianza real (mixta | vistas | no vistas) ──")
	for i, c in enumerate((0.2, 0.4, 0.6, 0.8, 1.0)):
		print(f"    cobertura {c*100:.0f}% → acierto {rc_all[i]['accuracy']*100:.1f}% | {rc_seen[i]['accuracy']*100:.1f}% | {rc_unseen[i]['accuracy']*100:.1f}%")

	# por estrato: dónde rompe la generalización
	by_stratum = defaultdict(lambda: {"hit": 0, "abstain": 0, "hallucination": 0, "total": 0})
	for r in rows_unseen:
		st = r.get("stratum", "?")
		by_stratum[st]["total"] += 1
		by_stratum[st][r["verdict"]] += 1
	print("  ── por estrato (no vistas) ──")
	for st, c in sorted(by_stratum.items()):
		print(f"    {st}: acierto {c['hit']} | abstiene {c['abstain']} | alucina {c['hallucination']} (de {c['total']})")

	try:
		git_rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=BASE,
			capture_output=True, text=True).stdout.strip() or None
	except OSError:
		git_rev = None
	provenance = {
		"model_path": os.path.relpath(os.path.abspath(args.model_path), BASE),
		"model_md5": file_md5(args.model_path),
		"hidden_dim": args.hidden_dim, "num_layers": args.num_layers,
		"uses_glyphs": uses_glyphs, "device": args.device, "stage_idx": args.stage_idx,
		"masks_path": os.path.relpath(masks_path, BASE), "masks_md5": file_md5(masks_path),
		"battery_path": os.path.relpath(battery_path, BASE),
		"timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
		"runner_git_rev": git_rev,
		"dedup": {"seen_dropped": seen_dups, "unseen_dropped": unseen_dups},
	}
	out = args.model_path.replace(".pt", f"_battery_age{args.age}.json")
	json.dump({"model": args.model_path, "set_id": battery.get("set_id"),
		"scoring": "seen: +1(correcto)/+1(no sé)/-1(mal) · unseen: +2/0/0",
		"provenance": provenance,
		"gate": {"hits": sh, "abstains": sa, "wrong": sm, "score": s_score, "total": n_s},
		"cognition": {"hits": uh, "abstains": ua, "wrong": um, "score": u_score, "total": n_u},
		"risk_coverage": rc_all, "risk_coverage_seen": rc_seen, "risk_coverage_unseen": rc_unseen,
		"retention": retention,
		"strata": {st: dict(c) for st, c in sorted(by_stratum.items())},
		"rows_seen": rows_seen, "rows_unseen": rows_unseen},
		open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
	print(f"→ {out}")


if __name__ == "__main__":
	main()
