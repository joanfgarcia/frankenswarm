import json
import os
from collections import Counter, defaultdict

import numpy as np


def _get_telemetry_path(experiment_id: str) -> str:
	base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	new_path = os.path.join(base, "storage", "experiments", experiment_id, "telemetry.jsonl")
	if os.path.exists(new_path):
		return new_path
	return os.path.join(base, "storage", "telemetry", f"EXP_{experiment_id}.jsonl")


def analyze_divergence(experiment_id: str = "006"):
	path = _get_telemetry_path(experiment_id)
	if not os.path.exists(path):
		print(f"❌ Telemetry file not found: {path}")
		return

	steps = []
	with open(path, encoding="utf-8") as f:
		for line in f:
			record = json.loads(line)
			if record.get("type") == "step" and record.get("message_tokens") is not None:
				steps.append(record)

	# Try to find config to get tf_min dynamically
	base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	config_path = os.path.join(base, "configs", "experiments", f"{experiment_id}.json")
	if not os.path.exists(config_path):
		config_path = os.path.join(base, "configs", "experiments", f"EXP_{experiment_id}.json")

	tf_min = 0.05
	if os.path.exists(config_path):
		try:
			with open(config_path, encoding="utf-8") as cf:
				cfg = json.load(cf)
				tf_min = cfg.get("tf_min", 0.05)
		except Exception:
			pass

	# Filter for autonomy phase (TF <= tf_min)
	autonomy_steps = [s for s in steps if s.get("tf_ratio", 1.0) <= (tf_min + 1e-5)]
	if not autonomy_steps:
		print(f"❌ No steps in autonomy phase found (TF <= {tf_min:.2f}).")
		return

	print(f"=== 🔍 Dialect Divergence Analysis — EXP_{experiment_id} ===")
	print(f"📊 Analyzing {len(autonomy_steps)} autonomous steps across population.\n")

	# Group steps by speaker
	steps_by_speaker = defaultdict(list)
	for s in autonomy_steps:
		steps_by_speaker[s["speaker_id"]].append(s)

	speakers = sorted(steps_by_speaker.keys())

	# ── 1. Speaker Vocabularies & Consistencies ──
	print("=" * 65)
	print("1️⃣  INDIVIDUAL SPEAKER PROFILES")
	print("=" * 65)

	vocab_by_speaker = {}
	consistency_by_speaker = {}

	for spk in speakers:
		spk_steps = steps_by_speaker[spk]

		# Collect all tokens emitted by this speaker
		emitted_tokens = set()
		for s in spk_steps:
			emitted_tokens.update(s["message_tokens"])
		vocab_by_speaker[spk] = emitted_tokens

		# Group messages by target triplet
		target_messages = defaultdict(list)
		for s in spk_steps:
			key = (s["target_concept"], s["target_emotion"], s.get("target_homeostasis", ""))
			target_messages[key].append(tuple(s["message_tokens"]))

		# Calculate average consistency
		consistencies = []
		for target, messages in target_messages.items():
			counter = Counter(messages)
			most_common_count = counter.most_common(1)[0][1]
			consistencies.append(most_common_count / len(messages))

		avg_consistency = np.mean(consistencies) * 100 if consistencies else 0.0
		consistency_by_speaker[spk] = avg_consistency

		print(f"  👤 Agent_{spk}:")
		print(f"    - Unique Tokens Emitted: {len(emitted_tokens)}")
		print(f"    - Message Consistency:  {avg_consistency:.2f}%")
		print(f"    - Samples Spoken:       {len(spk_steps)}")
		print()

	# ── 2. Pairwise Vocabulary Overlap (Jaccard Similarity) ──
	print("=" * 65)
	print("2️⃣  PAIRWISE VOCABULARY OVERLAP (JACCARD)")
	print("   ¿Los agentes usan el mismo léxico o desarrollan dialectos separados?")
	print("=" * 65)

	for i in range(len(speakers)):
		for j in range(i + 1, len(speakers)):
			spk_a = speakers[i]
			spk_b = speakers[j]
			vocab_a = vocab_by_speaker[spk_a]
			vocab_b = vocab_by_speaker[spk_b]

			intersection = vocab_a.intersection(vocab_b)
			union = vocab_a.union(vocab_b)
			jaccard = (len(intersection) / len(union) * 100) if union else 0.0

			print(f"    Agent_{spk_a} ↔ Agent_{spk_b} Overlap: {jaccard:.2f}% ({len(intersection)} shared / {len(union)} total)")
	print()

	# ── 3. Dialect Alignment (Agreement on Targets) ──
	print("=" * 65)
	print("3️⃣  DIALECT ALIGNMENT")
	print("   ¿Los agentes usan el mismo mensaje para los mismos conceptos?")
	print("=" * 65)

	# Map target -> speaker -> most common message
	target_spk_msg = defaultdict(dict)
	for spk in speakers:
		spk_steps = steps_by_speaker[spk]
		target_messages = defaultdict(list)
		for s in spk_steps:
			key = (s["target_concept"], s["target_emotion"], s.get("target_homeostasis", ""))
			target_messages[key].append(tuple(s["message_tokens"]))

		for target, messages in target_messages.items():
			most_common_msg = Counter(messages).most_common(1)[0][0]
			target_spk_msg[target][spk] = most_common_msg

	# Count agreement levels
	agreement_counts = Counter()
	for target, spk_map in target_spk_msg.items():
		if len(spk_map) < 2:
			continue  # Not enough speakers tried this target

		messages = list(spk_map.values())
		msg_counts = Counter(messages)
		max_agreement = msg_counts.most_common(1)[0][1]
		agreement_counts[max_agreement] += 1

	total_targets_evaluated = sum(agreement_counts.values())
	print(f"  Total target combinations evaluated by multiple agents: {total_targets_evaluated}")
	if total_targets_evaluated > 0:
		for agreement_level, count in sorted(agreement_counts.items(), reverse=True):
			pct = count / total_targets_evaluated * 100
			print(f"    - Consensus level {agreement_level}/{len(speakers)}: {count:3d} targets ({pct:.2f}%)")
	print()

	# Show examples of agreement and disagreement
	print("=" * 65)
	print("4️⃣  DIALECT EXAMPLES")
	print("=" * 65)

	consensus_targets = []
	divergent_targets = []

	for target, spk_map in target_spk_msg.items():
		if len(spk_map) < 3:
			continue
		messages = list(spk_map.values())
		unique_msgs = len(set(messages))

		if unique_msgs == 1:
			consensus_targets.append((target, spk_map))
		elif unique_msgs >= 3:
			divergent_targets.append((target, spk_map))

	print("  🤝 Consensus Examples (Same message from all agents):")
	if consensus_targets:
		for target, spk_map in consensus_targets[:3]:
			target_str = ", ".join(target)
			msg_str = " ".join(list(spk_map.values())[0])
			print(f"    Target: ({target_str}) → Message: [{msg_str}]")
	else:
		print("    (None found)")
	print()

	print("  ⚡ Divergent Examples (Different dialects per agent):")
	if divergent_targets:
		for target, spk_map in divergent_targets[:3]:
			target_str = ", ".join(target)
			print(f"    Target: ({target_str})")
			for spk, msg in sorted(spk_map.items()):
				print(f"      - Agent_{spk}: [{' '.join(msg)}]")
	else:
		print("    (None found)")

	print("\n🔬 Analisis completo.")
	print("=" * 65)


if __name__ == "__main__":
	# Soporte para --config configs/experiments/EXP_XXX.json o ID directo
	import argparse

	parser = argparse.ArgumentParser(description="Analizador de Divergencia Frankenswarm")
	parser.add_argument("experiment_id", type=str, nargs="?", default="006", help="ID del experimento o ruta del config")
	parser.add_argument("--config", type=str, default=None, help="Ruta al JSON de configuración del experimento")
	args = parser.parse_args()

	exp_id = args.experiment_id
	if args.config:
		with open(args.config, encoding="utf-8") as f:
			config = json.load(f)
		exp_id = config.get("experiment_id", exp_id)
	elif exp_id.endswith(".json") and os.path.exists(exp_id):
		# El usuario pasó el config JSON como argumento posicional
		with open(exp_id, encoding="utf-8") as f:
			config = json.load(f)
		exp_id = config.get("experiment_id", exp_id)

	analyze_divergence(exp_id)
