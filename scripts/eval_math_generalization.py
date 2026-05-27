import os
import json
import numpy as np
import torch

from src.bitnet.generalization_breeder import CompositionalMathDatasetBreeder
from src.bitnet.modeling_bitnet import BitNet4LayerModel
from src.bitnet.translator import SovereignTranslator


def eval_generalization(experiment_id: str = "EXP_013"):
	print(f"=== 🔍 Evaluación de Generalización Sistemática — {experiment_id} ===")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"[Device]: {device}")

	base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	config_path = os.path.join(base_dir, "configs", "experiments", f"{experiment_id}.json")
	if not os.path.exists(config_path):
		config_path = os.path.join(base_dir, "configs", "experiments", f"EXP_{experiment_id}.json")

	with open(config_path, encoding="utf-8") as f:
		config = json.load(f)

	# 1. Cargar Traductor y Breeder Composicional
	translator = SovereignTranslator()
	vocab_embeddings = translator.get_concept_embeddings()
	seed = config.get("seed", 42)
	split_ratio = config.get("split_ratio", 0.8)
	breeder = CompositionalMathDatasetBreeder(translator, split_ratio=split_ratio, seed=seed)

	# 2. Inicializar Modelo y cargar pesos
	hidden_dim = config["hidden_dim"]
	num_layers = config["num_layers"]
	model = BitNet4LayerModel(vocab_embeddings=vocab_embeddings, hidden_dim=hidden_dim, num_layers=num_layers).to(device)

	checkpoint_path = os.path.join(base_dir, "storage", "experiments", experiment_id, "best_agent.pt")
	if not os.path.exists(checkpoint_path):
		print(f"❌ Checkpoint no encontrado: {checkpoint_path}")
		return

	model.load_state_dict(torch.load(checkpoint_path, map_location=device))
	model.eval()
	print(f"✅ Pesos cargados desde {checkpoint_path}")

	# Logit Mask
	use_logit_mask = config["use_logit_mask"]
	logit_mask = None
	if use_logit_mask:
		micro_vocab_words = config["micro_vocab_words"]
		logit_mask = torch.zeros(8192, dtype=torch.bool, device=device)
		for word in micro_vocab_words:
			tids = translator.encode(word)
			if tids:
				logit_mask[tids[0]] = True

	# 3. Evaluar
	def run_eval_set(equations, set_name):
		correct_a = 0
		correct_op = 0
		correct_b = 0
		correct_r = 0
		correct_joint = 0
		total = len(equations)

		print(f"\n--- Evaluando {total} ecuaciones en el conjunto de {set_name} ---")

		for a, op, b, r in equations:
			a_token_id = breeder.operand_token_ids[a]
			op_token_id = breeder.operator_token_ids[op]
			b_token_id = breeder.operand_token_ids[b]
			r_token_id = breeder.operand_token_ids[r]

			# Input Speaker
			speaker_input = torch.tensor([[a_token_id, op_token_id, b_token_id, 0]], dtype=torch.long, device=device)

			with torch.no_grad():
				# Inferencia autónoma sin TF (tau=0.1)
				message = model.generate_message(speaker_input, tau=0.1, hard=True, logit_mask=logit_mask)
				# Oyente decodifica
				logits = model(message, logit_mask=logit_mask)

			pred_a_id = torch.argmax(logits[0, 0, :]).item()
			pred_op_id = torch.argmax(logits[0, 1, :]).item()
			pred_b_id = torch.argmax(logits[0, 2, :]).item()
			pred_r_id = torch.argmax(logits[0, 3, :]).item()

			a_ok = (pred_a_id == a_token_id)
			op_ok = (pred_op_id == op_token_id)
			b_ok = (pred_b_id == b_token_id)
			r_ok = (pred_r_id == r_token_id)

			if a_ok: correct_a += 1
			if op_ok: correct_op += 1
			if b_ok: correct_b += 1
			if r_ok: correct_r += 1

			joint_ok = a_ok and op_ok and b_ok and r_ok
			if joint_ok:
				correct_joint += 1

			# Si es el conjunto de test, mostrar detalle de cada ecuación para depuración
			if set_name == "TEST (No Visto)":
				message_tokens = []
				for step in range(4):
					t_idx = torch.argmax(message[0, step, :]).item()
					message_tokens.append(translator.decode([t_idx]))

				sample_a_name = breeder.get_operand_name(a)
				sample_op_name = breeder.get_operator_name(op)
				sample_b_name = breeder.get_operand_name(b)
				sample_r_name = breeder.get_operand_name(r)

				pred_a_name = translator.decode([pred_a_id])
				pred_op_name = translator.decode([pred_op_id])
				pred_b_name = translator.decode([pred_b_id])
				pred_r_name = translator.decode([pred_r_id])

				status_icon = "✅" if joint_ok else "❌"
				print(f"  {status_icon} Target: {sample_a_name} {sample_op_name} {sample_b_name} = {sample_r_name}")
				print(f"    - Canal:  [{' '.join(message_tokens)}]")
				print(f"    - Predic: {pred_a_name} {pred_op_name} {pred_b_name} = {pred_r_name}")

		print("-" * 65)
		print(f"📊 RESULTADOS DE AUTO-COMUNICACIÓN - {set_name.upper()}")
		print("-" * 65)
		print(f"  - Operando A: {correct_a/total*100:.2f}% ({correct_a}/{total})")
		print(f"  - Operador:   {correct_op/total*100:.2f}% ({correct_op}/{total})")
		print(f"  - Operando B: {correct_b/total*100:.2f}% ({correct_b}/{total})")
		print(f"  - Resultado:  {correct_r/total*100:.2f}% ({correct_r}/{total})")
		print(f"  - CONJUNTO:   {correct_joint/total*100:.2f}% ({correct_joint}/{total})")
		print("-" * 65)

	# Ejecutar evaluaciones
	run_eval_set(breeder.train_equations, "TRAIN (Visto)")
	run_eval_set(breeder.test_equations, "TEST (No Visto)")


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description="Evaluador de Generalización Aritmética")
	parser.add_argument("experiment_id", type=str, nargs="?", default="EXP_013", help="ID del experimento")
	args = parser.parse_args()
	eval_generalization(args.experiment_id)
