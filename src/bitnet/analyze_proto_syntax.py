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

	# Group messages by target pair
	target_messages = defaultdict(list)
	for s in autonomy_steps:
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
	for (concept, emotion), score in sorted_targets[:10]:
		msgs = target_messages[(concept, emotion)]
		top_msg = Counter(msgs).most_common(1)[0][0]
		print(f"    ({concept:12s}, {emotion:8s}) → [{', '.join(top_msg)}] | {score:.1f}% ({len(msgs)} muestras)")

	# Bottom 5 least consistent
	print("\n  Bottom 5 menos consistentes:")
	for (concept, emotion), score in sorted_targets[-5:]:
		msgs = target_messages[(concept, emotion)]
		top2 = Counter(msgs).most_common(2)
		msg_strs = [f"[{', '.join(m)}]×{c}" for m, c in top2]
		print(f"    ({concept:12s}, {emotion:8s}) → {' | '.join(msg_strs)} | {score:.1f}%")

	avg_consistency = np.mean(list(consistency_scores.values()))
	print(f"\n  📈 Consistencia media: {avg_consistency:.1f}%")

	# ── 3. Proto-syntax: is there token order structure? ──
	print("\n" + "=" * 60)
	print("2️⃣  PROTO-SINTAXIS")
	print("   ¿Hay orden en los tokens del mensaje?")
	print("=" * 60)

	# Check if position 0 correlates with concept, position 1 with emotion, etc.
	pos0_by_concept = defaultdict(Counter)
	pos1_by_concept = defaultdict(Counter)
	pos2_by_concept = defaultdict(Counter)
	pos0_by_emotion = defaultdict(Counter)
	pos1_by_emotion = defaultdict(Counter)
	pos2_by_emotion = defaultdict(Counter)

	for s in autonomy_steps:
		msg = s["message_tokens"]
		concept = s["target_concept"]
		emotion = s["target_emotion"]
		pos0_by_concept[concept][msg[0]] += 1
		pos1_by_concept[concept][msg[1]] += 1
		pos2_by_concept[concept][msg[2]] += 1
		pos0_by_emotion[emotion][msg[0]] += 1
		pos1_by_emotion[emotion][msg[1]] += 1
		pos2_by_emotion[emotion][msg[2]] += 1

	# Mutual information proxy: for each position, does the token distribution
	# change significantly by concept vs by emotion?
	def entropy_score(counter_dict):
		"""How much does the token vary across different keys?"""
		all_tokens = set()
		for c in counter_dict.values():
			all_tokens.update(c.keys())
		if len(all_tokens) <= 1:
			return 0.0
		# Average number of unique tokens per key
		uniques = [len(c) for c in counter_dict.values()]
		return np.mean(uniques)

	print("\n  Diversidad de tokens por posición (más alto = más variado):")
	print(f"                      Pos 0    Pos 1    Pos 2")
	print(f"    Por concepto:     {entropy_score(pos0_by_concept):5.1f}    {entropy_score(pos1_by_concept):5.1f}    {entropy_score(pos2_by_concept):5.1f}")
	print(f"    Por emoción:      {entropy_score(pos0_by_emotion):5.1f}    {entropy_score(pos1_by_emotion):5.1f}    {entropy_score(pos2_by_emotion):5.1f}")

	# Show what token each concept uses most at each position
	print("\n  Proto-léxico por concepto (token más frecuente en cada posición):")
	print(f"    {'Concepto':12s}  {'Pos 0':12s}  {'Pos 1':12s}  {'Pos 2':12s}")
	print(f"    {'─' * 12}  {'─' * 12}  {'─' * 12}  {'─' * 12}")
	for concept in CONCEPTS:
		p0 = pos0_by_concept[concept].most_common(1)[0][0] if pos0_by_concept[concept] else "?"
		p1 = pos1_by_concept[concept].most_common(1)[0][0] if pos1_by_concept[concept] else "?"
		p2 = pos2_by_concept[concept].most_common(1)[0][0] if pos2_by_concept[concept] else "?"
		print(f"    {concept:12s}  {p0:12s}  {p1:12s}  {p2:12s}")

	print(f"\n  Proto-léxico por emoción:")
	print(f"    {'Emoción':12s}  {'Pos 0':12s}  {'Pos 1':12s}  {'Pos 2':12s}")
	print(f"    {'─' * 12}  {'─' * 12}  {'─' * 12}  {'─' * 12}")
	for emotion in EMOTIONS:
		p0 = pos0_by_emotion[emotion].most_common(1)[0][0] if pos0_by_emotion[emotion] else "?"
		p1 = pos1_by_emotion[emotion].most_common(1)[0][0] if pos1_by_emotion[emotion] else "?"
		p2 = pos2_by_emotion[emotion].most_common(1)[0][0] if pos2_by_emotion[emotion] else "?"
		print(f"    {emotion:12s}  {p0:12s}  {p1:12s}  {p2:12s}")

	# ── 4. Affective stability ──
	print("\n" + "=" * 60)
	print("3️⃣  ESTABILIDAD AFECTIVA")
	print("   ¿La emoción se estabiliza antes que el concepto?")
	print("=" * 60)

	epochs = load_epochs(experiment_id)
	autonomy_epochs = [e for e in epochs if e["epoch"] > 20]
	if autonomy_epochs:
		concept_accs = [e["acc_concept"] for e in autonomy_epochs]
		emotion_accs = [e["acc_emotion"] for e in autonomy_epochs]
		joint_accs = [e["acc_joint"] for e in autonomy_epochs]

		print(f"\n  Épocas de autonomía ({len(autonomy_epochs)} epochs):")
		print(f"    Concepto:  μ={np.mean(concept_accs):.2f}%  σ={np.std(concept_accs):.2f}%  min={np.min(concept_accs):.2f}%  max={np.max(concept_accs):.2f}%")
		print(f"    Emoción:   μ={np.mean(emotion_accs):.2f}%  σ={np.std(emotion_accs):.2f}%  min={np.min(emotion_accs):.2f}%  max={np.max(emotion_accs):.2f}%")
		print(f"    Conjunta:  μ={np.mean(joint_accs):.2f}%  σ={np.std(joint_accs):.2f}%  min={np.min(joint_accs):.2f}%  max={np.max(joint_accs):.2f}%")

		emo_advantage = np.mean(emotion_accs) - np.mean(concept_accs)
		print(f"\n  📊 Ventaja emocional: +{emo_advantage:.2f} puntos porcentuales")
		if emo_advantage > 0:
			print(f"  ✅ CONFIRMADO: La emoción se mantiene más estable que el concepto (DeepSeek)")
		else:
			print(f"  ❌ NO CONFIRMADO: El concepto es igual o más estable que la emoción")

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
