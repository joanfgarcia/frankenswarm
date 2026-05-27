"""
analyze_proto_syntax.py — Analyze Exp 005 telemetry for proto-language structure.

Questions this answers:
1. Consistency: Does "fuego" always get encoded as the same message tokens?
2. Confusion geometry: Are errors systematic? Do "agua" and "gato" cluster together?
3. Proto-syntax: Is there token order structure? Does concept always go first?
4. Affective stability: Does emotion stabilize faster than concept at scale?
"""

import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np


def _get_telemetry_dir() -> str:
	base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	return os.path.join(base, "storage", "telemetry")


CONCEPTS = [
	"gato", "perro", "casa", "árbol", "agua",
	"fuego", "tierra", "aire", "sol", "luna",
	"peligro", "seguridad", "búnker", "agente", "código",
]
EMOTIONS = ["miedo", "alegría", "ira", "tristeza", "dolor", "hambre"]


def load_steps(experiment_id: str) -> list[dict]:
	"""Load step records that have message_tokens from autonomy phase."""
	path = os.path.join(_get_telemetry_dir(), f"EXP_{experiment_id}.jsonl")
	steps = []
	with open(path, encoding="utf-8") as f:
		for line in f:
			record = json.loads(line)
			if record.get("type") == "step" and record.get("message_tokens") is not None:
				steps.append(record)
	return steps


def load_epochs(experiment_id: str) -> list[dict]:
	path = os.path.join(_get_telemetry_dir(), f"EXP_{experiment_id}.jsonl")
	epochs = []
	with open(path, encoding="utf-8") as f:
		for line in f:
			record = json.loads(line)
			if record.get("type") == "epoch":
				epochs.append(record)
	return epochs


def load_confusion(experiment_id: str) -> tuple[dict | None, dict | None]:
	path = os.path.join(_get_telemetry_dir(), f"EXP_{experiment_id}.jsonl")
	concept_conf = None
	emotion_conf = None
	with open(path, encoding="utf-8") as f:
		for line in f:
			record = json.loads(line)
			if record.get("type") == "event":
				if record.get("event") == "confusion_concept":
					concept_conf = record["data"]
				elif record.get("event") == "confusion_emotion":
					emotion_conf = record["data"]
	return concept_conf, emotion_conf


def analyze(experiment_id: str = "005"):
	print(f"=== 🔬 Proto-Syntax Analysis — EXP_{experiment_id} ===\n")

	# ── 1. Load autonomy steps with message tokens ──
	steps = load_steps(experiment_id)
	if not steps:
		print("❌ No steps with message_tokens found. Did the autonomy phase run?")
		return

	autonomy_steps = [s for s in steps if s.get("tf_ratio", 1.0) == 0.0]
	print(f"📊 Datos: {len(steps)} steps con mensajes | {len(autonomy_steps)} en autonomía pura\n")

	if not autonomy_steps:
		autonomy_steps = steps
		print("(Usando todos los steps con mensajes disponibles)\n")

	# ── 2. Message consistency ──
	print("=" * 60)
	print("1️⃣  CONSISTENCIA DE MENSAJES")
	print("   ¿El speaker usa siempre los mismos tokens para el mismo target?")
	print("=" * 60)

	# Group messages by target
	target_messages = defaultdict(list)
	is_3d = "target_homeostasis" in autonomy_steps[0]
	
	for s in autonomy_steps:
		if is_3d:
			key = (s["target_concept"], s["target_emotion"], s["target_homeostasis"])
		else:
			key = (s["target_concept"], s["target_emotion"])
		msg = tuple(s["message_tokens"])
		target_messages[key].append(msg)

	# For each target, count unique messages
	consistency_scores = {}
	for target, messages in sorted(target_messages.items()):
		counter = Counter(messages)
		total = len(messages)
		most_common_msg, most_common_count = counter.most_common(1)[0]
		consistency = most_common_count / total * 100
		consistency_scores[target] = consistency

	# Top 10 most consistent
	sorted_targets = sorted(consistency_scores.items(), key=lambda x: -x[1])
	print("\n  Top 10 más consistentes:")
	for target, score in sorted_targets[:10]:
		msgs = target_messages[target]
		top_msg = Counter(msgs).most_common(1)[0][0]
		target_str = ", ".join(target)
		print(f"    ({target_str}) → [{', '.join(top_msg)}] | {score:.1f}% ({len(msgs)} muestras)")

	# Bottom 5 least consistent
	print("\n  Bottom 5 menos consistentes:")
	for target, score in sorted_targets[-5:]:
		msgs = target_messages[target]
		top2 = Counter(msgs).most_common(2)
		msg_strs = [f"[{', '.join(m)}]×{c}" for m, c in top2]
		target_str = ", ".join(target)
		print(f"    ({target_str}) → {' | '.join(msg_strs)} | {score:.1f}%")

	avg_consistency = np.mean(list(consistency_scores.values()))
	print(f"\n  📈 Consistencia media: {avg_consistency:.1f}%")

	# ── 3. Proto-syntax: is there token order structure? ──
	print("\n" + "=" * 60)
	print("2️⃣  PROTO-SINTAXIS")
	print("   ¿Hay orden en los tokens del mensaje?")
	print("=" * 60)

	msg_len = len(autonomy_steps[0]["message_tokens"])
	
	pos_by_concept = [defaultdict(Counter) for _ in range(msg_len)]
	pos_by_emotion = [defaultdict(Counter) for _ in range(msg_len)]
	pos_by_homeostasis = [defaultdict(Counter) for _ in range(msg_len)] if is_3d else None

	for s in autonomy_steps:
		msg = s["message_tokens"]
		concept = s["target_concept"]
		emotion = s["target_emotion"]
		homeo = s.get("target_homeostasis", "") if is_3d else None
		
		for pos in range(min(len(msg), msg_len)):
			token = msg[pos]
			pos_by_concept[pos][concept][token] += 1
			pos_by_emotion[pos][emotion][token] += 1
			if is_3d:
				pos_by_homeostasis[pos][homeo][token] += 1

	def entropy_score(pos_dict_list):
		scores = []
		for pos in range(msg_len):
			counter_dict = pos_dict_list[pos]
			all_tokens = set()
			for c in counter_dict.values():
				all_tokens.update(c.keys())
			if len(all_tokens) <= 1:
				scores.append(0.0)
			else:
				uniques = [len(c) for c in counter_dict.values()]
				scores.append(np.mean(uniques))
		return scores

	concept_div = entropy_score(pos_by_concept)
	emotion_div = entropy_score(pos_by_emotion)
	homeo_div = entropy_score(pos_by_homeostasis) if is_3d else None

	print("\n  Diversidad de tokens por posición (más alto = más variado):")
	pos_headers = "      ".join([f"Pos {p}" for p in range(msg_len)])
	print(f"                      {pos_headers}")
	concept_div_str = "    ".join([f"{val:5.1f}" for val in concept_div])
	emotion_div_str = "    ".join([f"{val:5.1f}" for val in emotion_div])
	print(f"    Por concepto:     {concept_div_str}")
	print(f"    Por emoción:      {emotion_div_str}")
	if is_3d:
		homeo_div_str = "    ".join([f"{val:5.1f}" for val in homeo_div])
		print(f"    Por homeostasis:  {homeo_div_str}")

	print("\n  Proto-léxico por concepto (token más frecuente en cada posición):")
	pos_cols = "    ".join([f"{f'Pos {p}':12s}" for p in range(msg_len)])
	print(f"    {'Concepto':12s}  {pos_cols}")
	print(f"    {'─' * 12}  " + "    ".join([f"{'─' * 12}" for _ in range(msg_len)]))
	for concept in CONCEPTS:
		p_tokens = []
		for p in range(msg_len):
			tok = pos_by_concept[p][concept].most_common(1)[0][0] if pos_by_concept[p][concept] else "?"
			p_tokens.append(f"{tok:12s}")
		print(f"    {concept:12s}  " + "    ".join(p_tokens))

	print("\n  Proto-léxico por emoción:")
	print(f"    {'Emoción':12s}  {pos_cols}")
	print(f"    {'─' * 12}  " + "    ".join([f"{'─' * 12}" for _ in range(msg_len)]))
	for emotion in EMOTIONS:
		p_tokens = []
		for p in range(msg_len):
			tok = pos_by_emotion[p][emotion].most_common(1)[0][0] if pos_by_emotion[p][emotion] else "?"
			p_tokens.append(f"{tok:12s}")
		print(f"    {emotion:12s}  " + "    ".join(p_tokens))

	if is_3d:
		HOMEOSTASIS = ["neutral", "dolor", "hambre", "urgencia", "seguridad"]
		print("\n  Proto-léxico por homeostasis:")
		print(f"    {'Homeostasis':12s}  {pos_cols}")
		print(f"    {'─' * 12}  " + "    ".join([f"{'─' * 12}" for _ in range(msg_len)]))
		for homeo in HOMEOSTASIS:
			p_tokens = []
			for p in range(msg_len):
				tok = pos_by_homeostasis[p][homeo].most_common(1)[0][0] if pos_by_homeostasis[p][homeo] else "?"
				p_tokens.append(f"{tok:12s}")
			print(f"    {homeo:12s}  " + "    ".join(p_tokens))

	# ── 4. Learning stability ──
	print("\n" + "=" * 60)
	print("3️⃣  ESTABILIDAD DE APRENDIZAJE")
	print("   ¿Cuáles componentes se estabilizan antes?")
	print("=" * 60)

	epochs = load_epochs(experiment_id)
	autonomy_epochs = [e for e in epochs if e["epoch"] > 20]
	if autonomy_epochs:
		concept_accs = [e["acc_concept"] for e in autonomy_epochs]
		emotion_accs = [e["acc_emotion"] for e in autonomy_epochs]
		joint_accs = [e["acc_joint"] for e in autonomy_epochs]
		has_h = "acc_homeostasis" in autonomy_epochs[0]
		if has_h:
			homeo_accs = [e["acc_homeostasis"] for e in autonomy_epochs]

		print(f"\n  Épocas de autonomía ({len(autonomy_epochs)} epochs):")
		print(f"    Concepto:     μ={np.mean(concept_accs):.2f}%  σ={np.std(concept_accs):.2f}%  min={np.min(concept_accs):.2f}%  max={np.max(concept_accs):.2f}%")
		print(f"    Emoción:      μ={np.mean(emotion_accs):.2f}%  σ={np.std(emotion_accs):.2f}%  min={np.min(emotion_accs):.2f}%  max={np.max(emotion_accs):.2f}%")
		if has_h:
			print(f"    Homeostasis:  μ={np.mean(homeo_accs):.2f}%  σ={np.std(homeo_accs):.2f}%  min={np.min(homeo_accs):.2f}%  max={np.max(homeo_accs):.2f}%")
		print(f"    Conjunta:     μ={np.mean(joint_accs):.2f}%  σ={np.std(joint_accs):.2f}%  min={np.min(joint_accs):.2f}%  max={np.max(joint_accs):.2f}%")

	# ── 5. Confusion geometry ──
	print("\n" + "=" * 60)
	print("4️⃣  GEOMETRÍA DE CONFUSIÓN")
	print("   ¿Los errores son sistemáticos?")
	print("=" * 60)

	concept_conf, emotion_conf = load_confusion(experiment_id)
	if concept_conf:
		matrix = np.array(concept_conf["matrix"])
		labels = concept_conf["labels"]
		np.fill_diagonal(matrix, 0)

		print("\n  Top-10 confusiones conceptuales:")
		flat_idx = np.argsort(matrix.ravel())[::-1][:10]
		for idx in flat_idx:
			r, c = divmod(idx, len(labels))
			count = matrix[r, c]
			if count > 0:
				print(f"    {labels[r]:12s} → {labels[c]:12s}  ({count} veces)")

	if emotion_conf:
		matrix = np.array(emotion_conf["matrix"])
		labels = emotion_conf["labels"]
		np.fill_diagonal(matrix, 0)

		print("\n  Top-5 confusiones emocionales:")
		flat_idx = np.argsort(matrix.ravel())[::-1][:5]
		for idx in flat_idx:
			r, c = divmod(idx, len(labels))
			count = matrix[r, c]
			if count > 0:
				print(f"    {labels[r]:12s} → {labels[c]:12s}  ({count} veces)")

	print("\n" + "=" * 60)
	print("🔬 Análisis completo.")
	print("=" * 60)
if __name__ == "__main__":
	exp_id = sys.argv[1] if len(sys.argv) > 1 else "005"
	analyze(exp_id)
