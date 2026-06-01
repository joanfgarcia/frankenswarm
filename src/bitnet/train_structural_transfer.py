"""
EXP_031: Test de Transferencia Estructural (propuesto por Reviewer #4).

Pregunta: ¿Los pesos del modelo codifican ESTRUCTURA causal o CONTENIDO léxico?

Método:
  1. Construir un grafo causal ISOMORFO al original con vocabulario nuevo.
  2. EXP_031A: Entrenar con hotstart desde EXP_028A (el Estoico).
  3. EXP_031B: Entrenar desde cero (sin hotstart).
  4. Comparar curvas de convergencia.

Interpretación:
  - Si 031A converge SIGNIFICATIVAMENTE más rápido → transferencia de estructura.
  - Si convergen a la misma velocidad → los pesos codificaban contenido.

Grafo original:                    Grafo isomorfo:
  fuego → peligro (0.9)             tormenta → inundación (0.9)
  fuego → tierra  (0.7)             tormenta → barro      (0.7)
  peligro → fuego (0.95)            inundación → tormenta  (0.95)
  luna → agua     (0.2)             estrella → río         (0.2)
  tierra → agua   (0.3)             barro → río            (0.3)
  agua → seguridad(0.1)             río → refugio          (0.1)
  seguridad → casa(0.0)             refugio → cueva        (0.0)
  casa → seguridad(0.1)             cueva → refugio        (0.1)
  gato → seguridad(0.1)             ratón → refugio        (0.1)
  perro → seguridad(0.1)            conejo → refugio       (0.1)
  sol → aire      (0.3)             nieve → viento         (0.3)
  árbol → aire    (0.2)             montaña → viento       (0.2)
  aire → sol      (0.1)             viento → nieve         (0.1)
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from src.bitnet.modeling_bitnet import BitNet4LayerModel
from src.bitnet.translator import SovereignTranslator

# ═══════════════════════════════════════════
# DOMINIO ISOMORFO
# ═══════════════════════════════════════════

# Mapeo: original → isomorfo (misma posición = misma topología)
CONCEPT_MAP = {
	"fuego": "tormenta",
	"peligro": "inundación",
	"tierra": "barro",
	"luna": "estrella",
	"agua": "río",
	"seguridad": "refugio",
	"casa": "cueva",
	"gato": "ratón",
	"perro": "conejo",
	"sol": "nieve",
	"árbol": "montaña",
	"aire": "viento",
}

ISO_CONCEPT_NAMES = list(CONCEPT_MAP.values())

# Mismas reglas causales, mismos fear scores, nuevos conceptos
ISO_CAUSAL_RULES = [
	("tormenta", "inundación", 0.9),
	("tormenta", "barro", 0.7),
	("inundación", "tormenta", 0.95),
	("estrella", "río", 0.2),
	("barro", "río", 0.3),
	("río", "refugio", 0.1),
	("refugio", "cueva", 0.0),
	("cueva", "refugio", 0.1),
	("ratón", "refugio", 0.1),
	("conejo", "refugio", 0.1),
	("nieve", "viento", 0.3),
	("montaña", "viento", 0.2),
	("viento", "nieve", 0.1),
]

ISO_CONCEPT_FEAR = {
	"tormenta": 0.7,
	"inundación": 0.95,
	"barro": 0.3,
	"estrella": 0.2,
	"río": 0.2,
	"refugio": 0.0,
	"cueva": 0.0,
	"ratón": 0.1,
	"conejo": 0.1,
	"nieve": 0.2,
	"montaña": 0.1,
	"viento": 0.1,
}

ISO_CONCEPT_EMOTION = {
	"tormenta": "miedo",
	"inundación": "miedo",
	"barro": "tristeza",
	"estrella": "alegría",
	"río": "alegría",
	"refugio": "alegría",
	"cueva": "alegría",
	"ratón": "alegría",
	"conejo": "alegría",
	"nieve": "tristeza",
	"montaña": "alegría",
	"viento": "tristeza",
}

OPERATORS = ["implica", "contradice", "cadena", "niega"]
RESULTS = ["verdad", "falsedad"]
EMOTIONS = ["miedo", "alegría", "ira", "tristeza", "dolor", "hambre"]


def build_iso_causal_graph():
	"""Construye grafo dirigido isomorfo."""
	graph = {}
	for a_name, b_name, fear in ISO_CAUSAL_RULES:
		a_idx = ISO_CONCEPT_NAMES.index(a_name)
		b_idx = ISO_CONCEPT_NAMES.index(b_name)
		if a_idx not in graph:
			graph[a_idx] = []
		graph[a_idx].append((b_idx, fear))
	return graph


def find_iso_chains(max_depth=3):
	"""Encuentra todas las cadenas transitivas en el grafo isomorfo."""
	graph = build_iso_causal_graph()
	chains = []

	def dfs(start, current, path, depth, max_fear):
		if depth >= 2:
			chains.append((start, current, list(path), max_fear))
		if depth >= max_depth:
			return
		for next_idx, fear in graph.get(current, []):
			if next_idx not in path:
				path.append(next_idx)
				dfs(start, next_idx, path, depth + 1, max(max_fear, fear))
				path.pop()

	for start_idx in range(len(ISO_CONCEPT_NAMES)):
		if start_idx in graph:
			for next_idx, fear in graph[start_idx]:
				dfs(start_idx, next_idx, [start_idx, next_idx], 1, fear)

	return chains


class IsomorphicBreeder:
	"""Genera ecuaciones lógicas sobre el dominio isomorfo."""

	def __init__(self, translator: SovereignTranslator, operators: dict, seed: int = 42):
		self.translator = translator
		self.rng = np.random.RandomState(seed)
		self.operators = operators
		self.graph = build_iso_causal_graph()
		self.chains = find_iso_chains(max_depth=3)

		# Token IDs
		self.concept_tids = {}
		for name in ISO_CONCEPT_NAMES:
			tids = translator.encode(name)
			self.concept_tids[name] = tids[0] if tids else 0

		self.op_tids = {}
		for op in OPERATORS:
			tids = translator.encode(op)
			self.op_tids[op] = tids[0] if tids else 0

		self.result_tids = {}
		for r in RESULTS:
			tids = translator.encode(r)
			self.result_tids[r] = tids[0] if tids else 0

	def generate_dataset(self):
		"""Genera todas las ecuaciones posibles."""
		equations = []

		n = len(ISO_CONCEPT_NAMES)

		for a_idx in range(n):
			a_name = ISO_CONCEPT_NAMES[a_idx]
			a_tid = self.concept_tids[a_name]

			for b_idx in range(n):
				if a_idx == b_idx:
					continue
				b_name = ISO_CONCEPT_NAMES[b_idx]
				b_tid = self.concept_tids[b_name]

				# Implica
				if self.operators.get("implica", {}).get("enabled", True):
					is_true = any(
						r[0] == a_idx and r[1] == b_idx
						for r in [(ISO_CONCEPT_NAMES.index(a), ISO_CONCEPT_NAMES.index(b)) for a, b, _ in ISO_CAUSAL_RULES]
					)
					fear = ISO_CONCEPT_FEAR.get(a_name, 0.1)
					emotion = ISO_CONCEPT_EMOTION.get(a_name, "alegría")
					equations.append(
						{
							"input": [a_tid, self.op_tids["implica"], b_tid, 0],
							"target": [
								a_tid,
								self.op_tids["implica"],
								b_tid,
								self.result_tids["verdad"] if is_true else self.result_tids["falsedad"],
							],
							"fear": fear,
							"emotion": emotion,
							"op": "implica",
						}
					)

				# Contradice
				if self.operators.get("contradice", {}).get("enabled", True):
					a_targets = set()
					for ra, rb, _ in ISO_CAUSAL_RULES:
						if ISO_CONCEPT_NAMES.index(ra) == a_idx:
							a_targets.add(ISO_CONCEPT_NAMES.index(rb))
					b_targets = set()
					for ra, rb, _ in ISO_CAUSAL_RULES:
						if ISO_CONCEPT_NAMES.index(ra) == b_idx:
							b_targets.add(ISO_CONCEPT_NAMES.index(rb))
					is_true = bool(a_targets & b_targets)
					fear = max(ISO_CONCEPT_FEAR.get(a_name, 0.1), ISO_CONCEPT_FEAR.get(b_name, 0.1))
					emotion = ISO_CONCEPT_EMOTION.get(a_name, "alegría")
					equations.append(
						{
							"input": [a_tid, self.op_tids["contradice"], b_tid, 0],
							"target": [
								a_tid,
								self.op_tids["contradice"],
								b_tid,
								self.result_tids["verdad"] if is_true else self.result_tids["falsedad"],
							],
							"fear": fear,
							"emotion": emotion,
							"op": "contradice",
						}
					)

			# Cadena
			if self.operators.get("cadena", {}).get("enabled", True):
				for start, end, path, max_fear in self.chains:
					if start == a_idx:
						end_name = ISO_CONCEPT_NAMES[end]
						end_tid = self.concept_tids[end_name]
						emotion = ISO_CONCEPT_EMOTION.get(a_name, "alegría")
						equations.append(
							{
								"input": [a_tid, self.op_tids["cadena"], end_tid, 0],
								"target": [a_tid, self.op_tids["cadena"], end_tid, self.result_tids["verdad"]],
								"fear": max_fear,
								"emotion": emotion,
								"op": "cadena",
							}
						)

			# Niega (modus tollens)
			if self.operators.get("niega", {}).get("enabled", True):
				for b_idx2 in range(n):
					if a_idx == b_idx2:
						continue
					b_name2 = ISO_CONCEPT_NAMES[b_idx2]
					# ¿Existe cadena b→...→a?
					is_true = any(start == b_idx2 and end == a_idx for start, end, _, _ in self.chains)
					# También regla directa
					if not is_true:
						is_true = any(
							ISO_CONCEPT_NAMES.index(ra) == b_idx2 and ISO_CONCEPT_NAMES.index(rb) == a_idx for ra, rb, _ in ISO_CAUSAL_RULES
						)
					if is_true:
						b_tid2 = self.concept_tids[b_name2]
						fear = ISO_CONCEPT_FEAR.get(a_name, 0.1)
						emotion = ISO_CONCEPT_EMOTION.get(a_name, "alegría")
						equations.append(
							{
								"input": [a_tid, self.op_tids["niega"], b_tid2, 0],
								"target": [a_tid, self.op_tids["niega"], b_tid2, self.result_tids["verdad"]],
								"fear": fear,
								"emotion": emotion,
								"op": "niega",
							}
						)

		return equations


def svd_crossover(parent_a, parent_b, child, alpha=0.5, sigma=0.01):
	with torch.no_grad():
		for name, param in child.named_parameters():
			if not param.requires_grad:
				continue
			p_a = parent_a.state_dict()[name]
			p_b = parent_b.state_dict()[name]
			if p_a.ndim == 2:
				w_avg = alpha * p_a + (1.0 - alpha) * p_b
				try:
					u, s, vh = torch.linalg.svd(w_avg, full_matrices=False)
					s_perturbed = (s + torch.randn_like(s) * sigma).clamp_(min=0.0)
					param.copy_(u @ torch.diag(s_perturbed) @ vh)
				except Exception:
					param.copy_(w_avg)
			else:
				param.copy_(alpha * p_a + (1.0 - alpha) * p_b + torch.randn_like(p_a) * sigma)


def run_structural_transfer():
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", type=str, required=True)
	args, _ = parser.parse_known_args()

	base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	config_path = args.config if os.path.isabs(args.config) else os.path.join(base_dir, args.config)

	with open(config_path, encoding="utf-8") as f:
		config = json.load(f)

	experiment_id = config["experiment_id"]
	use_hotstart = config.get("use_hotstart", True)

	print(f"=== 🧬 Transferencia Estructural — {experiment_id} ===")
	print(f"Hotstart: {'SÍ (desde EXP_028A)' if use_hotstart else 'NO (desde cero)'}")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"[Device]: {device}")

	seed = config.get("seed", 42)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)

	exp_dir = os.path.join(base_dir, "storage", "experiments", experiment_id)
	os.makedirs(exp_dir, exist_ok=True)
	with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf-8") as f:
		json.dump(config, f, ensure_ascii=False, indent=4)

	# Translator y dataset
	translator = SovereignTranslator()
	vocab_embeddings = translator.get_concept_embeddings()

	breeder = IsomorphicBreeder(translator, config.get("operators", {}), seed=seed)
	all_equations = breeder.generate_dataset()
	print(f"📊 Ecuaciones generadas: {len(all_equations)}")

	# Split train/test
	split = config.get("split_ratio", 0.88)
	np.random.shuffle(all_equations)
	split_idx = int(len(all_equations) * split)
	train_eqs = all_equations[:split_idx]
	test_eqs = all_equations[split_idx:]
	print(f"📊 Train={len(train_eqs)} | Test={len(test_eqs)}")

	# Población
	pop_size = config["pop_size"]
	population = [
		BitNet4LayerModel(
			vocab_embeddings=vocab_embeddings,
			hidden_dim=config["hidden_dim"],
			num_layers=config["num_layers"],
			use_pos_embedding=config.get("use_pos_embedding", True),
		).to(device)
		for _ in range(pop_size)
	]

	# Hotstart condicional
	if use_hotstart:
		resume = config.get("resume_checkpoint", "")
		resume_path = resume if os.path.isabs(resume) else os.path.join(base_dir, resume)
		if os.path.exists(resume_path):
			print(f"💾 [HOTSTART] Cargando estructura desde {resume_path}...")
			state_dict = torch.load(resume_path, map_location=device)
			for model in population:
				model.load_state_dict(state_dict)
			print("✅ Pesos del dominio original cargados. Vocabulario NUEVO.")
		else:
			print(f"⚠️ Checkpoint no encontrado: {resume_path}. Entrenando desde cero.")
	else:
		print("🆕 Entrenando desde CERO — control baseline.")

	lr = config["lr"]
	wd = config.get("weight_decay", 0.05)
	optimizers = [torch.optim.AdamW(filter(lambda p: p.requires_grad, m.parameters()), lr=lr, weight_decay=wd) for m in population]

	# Logit mask con vocabulario isomorfo
	logit_mask = None
	if config.get("use_logit_mask", True):
		logit_mask = torch.zeros(8192, dtype=torch.bool, device=device)
		for word in config["micro_vocab_words"]:
			tids = translator.encode(word)
			if tids:
				logit_mask[tids[0]] = True

	epochs = config["epochs"]
	steps_per_epoch = config["steps_per_epoch"]
	batch_size = config["batch_size"]
	tau_start, tau_min = config["tau_start"], config["tau_min"]
	total_steps = epochs * steps_per_epoch
	nursery_end = config["nursery_end"]
	transition_end = config["transition_end"]
	tf_min = config["tf_min"]
	fear_amplifier = config.get("fear_amplifier", 3.0)

	fitness = np.zeros(pop_size)
	current_step = 0
	convergence_log = []  # Para comparar velocidad de convergencia
	t_start = time.time()

	for epoch in range(epochs):
		if epoch < nursery_end:
			tf_ratio, phase = 1.0, "🍼 Guardería"
		elif epoch < transition_end:
			progress = (epoch - nursery_end) / (transition_end - nursery_end)
			tf_ratio, phase = 1.0 - progress * (1.0 - tf_min), "🎮 Recreo"
		else:
			tf_ratio, phase = tf_min, "🦅 Autonomía"

		print(f"\n--- Época {epoch + 1}/{epochs} [{phase}] TF={tf_ratio:.0%} ---")

		epoch_losses, epoch_correct, epoch_total = [], 0, 0
		epoch_fear_correct, epoch_fear_total = 0, 0
		interactions = np.zeros((pop_size, pop_size))
		successes = np.zeros((pop_size, pop_size))

		for model in population:
			model.train()

		for step in range(steps_per_epoch):
			tau = max(tau_min, tau_start * (1.0 - current_step / total_steps))

			# Sample batch
			batch = [train_eqs[np.random.randint(len(train_eqs))] for _ in range(batch_size)]
			inputs = torch.tensor([e["input"] for e in batch], dtype=torch.long, device=device)
			targets = torch.tensor([e["target"] for e in batch], dtype=torch.long, device=device)
			fears = torch.tensor([1.0 + e["fear"] * fear_amplifier for e in batch], dtype=torch.float32, device=device)

			# Selección de pareja
			idx_s = np.random.randint(0, pop_size)
			idx_l = np.random.randint(0, pop_size)
			while idx_l == idx_s:
				idx_l = np.random.randint(0, pop_size)

			speaker, listener = population[idx_s], population[idx_l]
			opt_s, opt_l = optimizers[idx_s], optimizers[idx_l]
			opt_s.zero_grad()
			opt_l.zero_grad()

			# Speaker forward
			speaker_logits = speaker(inputs, logit_mask=logit_mask)
			msg = speaker.generate_message(inputs, tau=tau, hard=False, logit_mask=logit_mask)

			if torch.rand(1).item() < tf_ratio:
				teacher_msg = F.one_hot(targets, num_classes=8192).float()
				msg_to_listener = teacher_msg
			else:
				msg_to_listener = msg

			listener_logits = listener(msg_to_listener, logit_mask=logit_mask)

			# Loss
			loss_speaker = F.cross_entropy(speaker_logits[:, 3, :], targets[:, 3], reduction="none")
			loss_listener = F.cross_entropy(listener_logits[:, 3, :], targets[:, 3], reduction="none")
			total_loss = ((loss_speaker + loss_listener) * fears).mean()

			total_loss.backward()
			torch.nn.utils.clip_grad_norm_(speaker.parameters(), max_norm=1.0)
			torch.nn.utils.clip_grad_norm_(listener.parameters(), max_norm=1.0)
			opt_s.step()
			opt_l.step()

			epoch_losses.append(total_loss.item())

			with torch.no_grad():
				pred = torch.argmax(speaker_logits[:, 3, :], dim=-1)
				correct = (pred == targets[:, 3]).sum().item()
				epoch_correct += correct
				epoch_total += batch_size

				for i in range(batch_size):
					if batch[i]["fear"] > 0.5:
						epoch_fear_total += 1
						if pred[i] == targets[i, 3]:
							epoch_fear_correct += 1

			interactions[idx_s, idx_l] += batch_size
			successes[idx_s, idx_l] += correct

			if step % 50 == 0:
				eq = batch[0]
				a_name = translator.decode([eq["input"][0]])
				op_name = translator.decode([eq["input"][1]])
				b_name = translator.decode([eq["input"][2]])
				target_name = translator.decode([eq["target"][3]])
				pred_name = translator.decode([pred[0].item()])
				ok = "✓" if pred_name == target_name else "✗"
				print(
					f"  step {step:3d} | τ={tau:.3f} TF={tf_ratio:.0%} | loss={total_loss.item():.3f} | "
					f"({a_name} {op_name} {b_name} = {target_name}) → pred={pred_name}({ok})"
				)

			current_step += 1

		# Test eval
		test_correct, test_total = 0, 0
		best_model = population[np.argmax(fitness)] if fitness.max() > 0 else population[0]
		best_model.eval()
		with torch.no_grad():
			for eq in test_eqs:
				inp = torch.tensor([eq["input"]], dtype=torch.long, device=device)
				logits = best_model(inp, logit_mask=logit_mask)
				pred = torch.argmax(logits[0, 3, :]).item()
				if pred == eq["target"][3]:
					test_correct += 1
				test_total += 1

		# Fitness
		for i in range(pop_size):
			sent_total = interactions[i, :].sum() + interactions[:, i].sum()
			sent_correct = successes[i, :].sum() + successes[:, i].sum()
			fitness[i] = sent_correct / (sent_total + 1e-10)

		avg_loss = np.mean(epoch_losses)
		train_acc = (epoch_correct / epoch_total) * 100
		test_acc = (test_correct / test_total) * 100 if test_total > 0 else 0
		survival = (epoch_fear_correct / epoch_fear_total) * 100 if epoch_fear_total > 0 else 0

		elapsed = time.time() - t_start
		convergence_log.append(
			{"epoch": epoch + 1, "train": train_acc, "test": test_acc, "survival": survival, "loss": avg_loss, "elapsed_s": elapsed}
		)

		print(f"Loss: {avg_loss:.4f} | Train: {train_acc:.2f}% | Test: {test_acc:.2f}% | 😱 Survival: {survival:.2f}% | ⏱️ {elapsed:.1f}s")
		print(f"Fitness: {[f'Agent_{i}: {f * 100:.2f}%' for i, f in enumerate(fitness)]}")

		# SVD evolución
		svd_interval = config.get("svd_interval", 3)
		svd_phases = config.get("svd_phases", ["recreo", "autonomia"])
		current_phase = "guarderia" if epoch < nursery_end else ("recreo" if epoch < transition_end else "autonomia")

		if current_phase in svd_phases and epoch % svd_interval == 0 and epoch > 0:
			worst_idx = int(np.argmin(fitness))
			best = np.argsort(fitness)[-2:]
			print(f"[Evolución] Reemplazando Agent_{worst_idx} con hijo SVD de Agent_{best[1]} y Agent_{best[0]}")
			svd_crossover(population[best[1]], population[best[0]], population[worst_idx])
			optimizers[worst_idx] = torch.optim.AdamW(filter(lambda p: p.requires_grad, population[worst_idx].parameters()), lr=lr, weight_decay=wd)

	# Guardar
	best_idx = np.argmax(fitness)
	save_path = os.path.join(exp_dir, "best_agent.pt")
	torch.save(population[best_idx].state_dict(), save_path)
	print(f"💾 Mejor agente guardado en {save_path}")

	# Guardar log de convergencia para comparación
	conv_path = os.path.join(exp_dir, "convergence.json")
	with open(conv_path, "w", encoding="utf-8") as f:
		json.dump(convergence_log, f, indent=2)
	print(f"📈 Log de convergencia guardado en {conv_path}")

	# Resumen final
	print("\n" + "=" * 60)
	print(f"RESUMEN — {experiment_id} ({'HOTSTART' if use_hotstart else 'FROM SCRATCH'})")
	print("  Época donde Train > 90%: ", end="")
	for entry in convergence_log:
		if entry["train"] > 90:
			print(f"Época {entry['epoch']} ({entry['elapsed_s']:.1f}s)")
			break
	else:
		print("NUNCA")

	print("  Época donde Test > 80%: ", end="")
	for entry in convergence_log:
		if entry["test"] > 80:
			print(f"Época {entry['epoch']} ({entry['elapsed_s']:.1f}s)")
			break
	else:
		print("NUNCA")

	print(f"  Test final: {convergence_log[-1]['test']:.2f}%")
	print(f"  Tiempo total: {convergence_log[-1]['elapsed_s']:.1f}s")
	print("=" * 60)


if __name__ == "__main__":
	run_structural_transfer()
