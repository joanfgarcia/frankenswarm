"""
Inference Loop: El modelo razona en cadena consigo mismo.

Carga un checkpoint entrenado y lo usa como motor de inferencia
en bucle: dado un concepto inicial, descubre el camino completo
a través del grafo causal probando todas las implicaciones posibles.

Sin reentrenamiento. Solo inferencia encadenada.
"""

import json
import os

import torch
import torch.nn.functional as F

import src.bitnet.operators_logic as ol
from src.bitnet.modeling_bitnet import BitNet4LayerModel
from src.bitnet.translator import SovereignTranslator


def load_model(checkpoint_path: str, config_path: str, device: torch.device):
	"""Carga modelo y translator desde checkpoint."""
	with open(config_path, encoding="utf-8") as f:
		config = json.load(f)

	translator = SovereignTranslator()
	vocab_embeddings = translator.get_concept_embeddings()

	model = BitNet4LayerModel(
		vocab_embeddings=vocab_embeddings,
		hidden_dim=config["hidden_dim"],
		num_layers=config["num_layers"],
		use_pos_embedding=config.get("use_pos_embedding", True),
	).to(device)

	state_dict = torch.load(checkpoint_path, map_location=device)
	model.load_state_dict(state_dict)
	model.eval()

	# Logit mask
	logit_mask = None
	if config.get("use_logit_mask", True):
		logit_mask = torch.zeros(8192, dtype=torch.bool, device=device)
		for word in config["micro_vocab_words"]:
			tids = translator.encode(word)
			if tids:
				logit_mask[tids[0]] = True

	return model, translator, logit_mask, config


def get_concept_token_ids(translator: SovereignTranslator) -> dict[str, int]:
	"""Obtiene token IDs para todos los conceptos."""
	concepts = {}
	for name in ol.CONCEPT_NAMES:
		tids = translator.encode(name)
		if tids:
			concepts[name] = tids[0]
	return concepts


def query_implication(
	model,
	concept_a_tid: int,
	concept_b_tid: int,
	op_tid: int,
	logit_mask,
	device,
	result_tids: dict,
) -> tuple[str, float]:
	"""
	Pregunta al modelo: ¿A op B = verdad o falsedad?
	Devuelve (resultado, confianza).
	"""
	with torch.no_grad():
		inp = torch.tensor([[concept_a_tid, op_tid, concept_b_tid, 0]], dtype=torch.long, device=device)
		logits = model(inp, logit_mask=logit_mask)
		probs = F.softmax(logits[0, 3, :], dim=-1)

		verdad_prob = probs[result_tids["verdad"]].item()
		falsedad_prob = probs[result_tids["falsedad"]].item()

		if verdad_prob > falsedad_prob:
			return "verdad", verdad_prob
		else:
			return "falsedad", falsedad_prob


def find_implications(
	model,
	concept_a: str,
	concepts: dict,
	op_tid: int,
	logit_mask,
	device,
	result_tids: dict,
	threshold: float = 0.6,
) -> list[tuple[str, float]]:
	"""
	Dado un concepto A, encuentra todos los B donde A op B = verdad.
	"""
	a_tid = concepts[concept_a]
	results = []

	for b_name, b_tid in concepts.items():
		if b_name == concept_a:
			continue
		result, confidence = query_implication(model, a_tid, b_tid, op_tid, logit_mask, device, result_tids)
		if result == "verdad" and confidence > threshold:
			results.append((b_name, confidence))

	# Ordenar por confianza
	results.sort(key=lambda x: -x[1])
	return results


def inference_loop(
	model,
	start_concept: str,
	concepts: dict,
	op_tid: int,
	logit_mask,
	device,
	result_tids: dict,
	max_depth: int = 6,
	threshold: float = 0.6,
) -> list[tuple[str, list[tuple[str, float]]]]:
	"""
	Loop de inferencia: desde un concepto, sigue las implicaciones
	hasta que no queden más o se alcance la profundidad máxima.
	"""
	chain = []
	visited = {start_concept}
	current = start_concept

	for depth in range(max_depth):
		implications = find_implications(model, current, concepts, op_tid, logit_mask, device, result_tids, threshold)

		# Filtrar ya visitados
		new_implications = [(name, conf) for name, conf in implications if name not in visited]

		chain.append((current, new_implications))

		if not new_implications:
			break

		# Seguir el camino más seguro (mayor confianza)
		next_concept = new_implications[0][0]
		visited.add(next_concept)
		current = next_concept

	return chain


def main():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

	# Cargar el mejor modelo (EXP_028A — el Estoico)
	checkpoint = os.path.join(base_dir, "storage", "experiments", "EXP_028A", "best_agent.pt")
	config_path = os.path.join(base_dir, "storage", "experiments", "EXP_028A", "config.json")

	if not os.path.exists(checkpoint):
		# Fallback a EXP_027
		checkpoint = os.path.join(base_dir, "storage", "experiments", "EXP_027", "best_agent.pt")
		config_path = os.path.join(base_dir, "storage", "experiments", "EXP_027", "config.json")

	print("=== 🔄 Motor de Inferencia en Loop ===")
	print(f"Checkpoint: {checkpoint}")

	model, translator, logit_mask, config = load_model(checkpoint, config_path, device)
	concepts = get_concept_token_ids(translator)
	print(f"Conceptos: {list(concepts.keys())}")

	# Token IDs de resultados
	result_tids = {
		"verdad": translator.encode("verdad")[0],
		"falsedad": translator.encode("falsedad")[0],
	}

	# Token ID del operador "implica"
	op_implica = translator.encode("implica")[0]
	op_niega = translator.encode("niega")[0]
	op_cadena = translator.encode("cadena")[0]

	# ═══════════════════════════════════════════
	# FASE 1: EXPLORACIÓN FORWARD — ¿qué implica cada concepto?
	# ═══════════════════════════════════════════
	print("\n" + "=" * 60)
	print("FASE 1: MAPA DE IMPLICACIONES DIRECTAS")
	print("=" * 60)

	for concept in ol.CONCEPT_NAMES:
		implications = find_implications(model, concept, concepts, op_implica, logit_mask, device, result_tids, threshold=0.5)
		if implications:
			impl_str = ", ".join([f"{name} ({conf:.2f})" for name, conf in implications])
			print(f"  {concept:12s} → {impl_str}")
		else:
			print(f"  {concept:12s} → (nada)")

	# ═══════════════════════════════════════════
	# FASE 2: CADENAS EMERGENTES — loop de inferencia
	# ═══════════════════════════════════════════
	print("\n" + "=" * 60)
	print("FASE 2: CADENAS EMERGENTES (Loop de Inferencia)")
	print("=" * 60)

	start_concepts = ["luna", "fuego", "peligro", "árbol", "gato", "sol"]

	for start in start_concepts:
		chain = inference_loop(model, start, concepts, op_implica, logit_mask, device, result_tids, max_depth=8, threshold=0.5)

		# Formatear la cadena
		path = []
		for concept, next_options in chain:
			path.append(concept)

		arrow_str = " → ".join(path)
		if len(path) > 1:
			print(f"\n  🔗 {arrow_str}")
			# Detalles
			for concept, next_options in chain:
				if next_options:
					opts = ", ".join([f"{n}({c:.2f})" for n, c in next_options[:3]])
					print(f"     {concept:12s} podría ir a: {opts}")
		else:
			print(f"\n  ⛔ {start} → (sin implicaciones)")

	# ═══════════════════════════════════════════
	# FASE 3: MODUS TOLLENS EN LOOP — razonamiento backward
	# ═══════════════════════════════════════════
	print("\n" + "=" * 60)
	print("FASE 3: MODUS TOLLENS EN LOOP (Razonamiento Backward)")
	print("=" * 60)

	for absent_concept in ["seguridad", "agua", "peligro", "aire"]:
		print(f"\n  ❌ Si NO hay {absent_concept}, ¿qué podemos descartar?")
		negations = find_implications(model, absent_concept, concepts, op_niega, logit_mask, device, result_tids, threshold=0.5)
		if negations:
			for name, conf in negations:
				print(f"     → NO hay {name:12s} (confianza: {conf:.2f})")
		else:
			print("     → (no se puede descartar nada)")

	# ═══════════════════════════════════════════
	# FASE 4: COMPARAR LOOP vs CADENA DIRECTA
	# ═══════════════════════════════════════════
	print("\n" + "=" * 60)
	print("FASE 4: LOOP vs CADENA — ¿coinciden?")
	print("=" * 60)

	# Cadenas conocidas del grafo causal
	test_chains = [
		("luna", "seguridad", "luna→agua→seguridad"),
		("fuego", "agua", "fuego→tierra→agua"),
		("árbol", "sol", "árbol→aire→sol"),
		("luna", "casa", "luna→agua→seguridad→casa"),
		("peligro", "agua", "peligro→fuego→tierra→agua"),
	]

	for start, end, expected_path in test_chains:
		# Test directo con operador cadena
		result_direct, conf_direct = query_implication(model, concepts[start], concepts[end], op_cadena, logit_mask, device, result_tids)

		# Test por loop
		chain = inference_loop(model, start, concepts, op_implica, logit_mask, device, result_tids, max_depth=6, threshold=0.5)
		path_concepts = [c for c, _ in chain]
		loop_found = end in path_concepts

		match = "✅" if (result_direct == "verdad" and loop_found) or (result_direct == "falsedad" and not loop_found) else "❌"

		print(f"  {match} {expected_path}")
		print(f"     Cadena directa: {result_direct} ({conf_direct:.2f})")
		print(f"     Loop encontró:  {'→'.join(path_concepts)} {'(contiene ' + end + ')' if loop_found else '(NO contiene ' + end + ')'}")


if __name__ == "__main__":
	main()
