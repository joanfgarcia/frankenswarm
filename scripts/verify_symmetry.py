import os
import sys

import torch

# Ensure root dir is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bitnet.generalization_breeder import RelationalLogicDatasetBreeder
from src.bitnet.modeling_bitnet import BitNet4LayerModel
from src.bitnet.translator import SovereignTranslator


def load_agent(checkpoint_path, vocab_embeddings):
	model = BitNet4LayerModel(vocab_embeddings=vocab_embeddings, hidden_dim=256, num_layers=3)
	model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
	model.eval()
	return model


def evaluate_symmetry(model, translator, breeder):
	# Test all 363 equations
	total_eqs = []
	for a in range(11):
		for b in range(11):
			# mayor_que (>)
			total_eqs.append((a, 0, b, 0 if a > b else 1))
			# menor_que (<)
			total_eqs.append((a, 1, b, 0 if a < b else 1))
			# igual_a (igual)
			total_eqs.append((a, 2, b, 0 if a == b else 1))

	# Convert targets to tokens
	op_a_token_ids = torch.tensor([breeder.operand_token_ids[x[0]] for x in total_eqs], dtype=torch.long)
	relation_token_ids = torch.tensor([breeder.relation_token_ids[x[1]] for x in total_eqs], dtype=torch.long)
	op_b_token_ids = torch.tensor([breeder.operand_token_ids[x[2]] for x in total_eqs], dtype=torch.long)
	result_token_ids = torch.tensor([breeder.result_token_ids[x[3]] for x in total_eqs], dtype=torch.long)

	# Inputs: [op_a, relation, op_b, 0]
	inputs = torch.zeros((len(total_eqs), 4), dtype=torch.long)
	inputs[:, 0] = op_a_token_ids
	inputs[:, 1] = relation_token_ids
	inputs[:, 2] = op_b_token_ids

	with torch.no_grad():
		logits = model(inputs)
		preds_r = torch.argmax(logits[:, 3, :], dim=-1)

	# Metrics
	token_id_verdad = breeder.result_token_ids[0]
	token_id_falsedad = breeder.result_token_ids[1]

	# Compute total accuracy
	correct = (preds_r == result_token_ids).sum().item()
	total = len(total_eqs)
	acc = correct / total * 100

	# Compute False Positive Rate (predicting verdad when ground truth is falsedad)
	is_falsedad = result_token_ids == token_id_falsedad
	predicted_verdad = preds_r == token_id_verdad
	false_positives = (is_falsedad & predicted_verdad).sum().item()
	total_falsedad = is_falsedad.sum().item()
	fpr = false_positives / total_falsedad * 100

	# Count symmetry errors specifically for mayor_que and menor_que
	# i.e., instances where a op b is false, but model predicts verdad
	rel_fp = 0
	total_rel_false = 0
	for i, (a, op, b, r) in enumerate(total_eqs):
		if op in (0, 1) and r == 1:  # > or < and ground truth is falsedad
			total_rel_false += 1
			if preds_r[i].item() == token_id_verdad:
				rel_fp += 1
	rel_fpr = rel_fp / total_rel_false * 100

	# Specific test case: 2 > 5
	idx_2_gt_5 = next(i for i, x in enumerate(total_eqs) if x[0] == 2 and x[1] == 0 and x[2] == 5)
	pred_2_gt_5 = translator.decode([preds_r[idx_2_gt_5].item()])

	return acc, fpr, rel_fpr, pred_2_gt_5


def main():
	print("==================================================")
	print("🔬 LOGICAL SPECIALIST SYMMETRY AUDIT")
	print("==================================================")

	translator = SovereignTranslator()
	vocab_embeddings = translator.get_concept_embeddings()
	breeder = RelationalLogicDatasetBreeder(translator, split_ratio=0.9, seed=42)

	base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	ckpt_023 = os.path.join(base_dir, "storage", "experiments", "EXP_023", "best_agent.pt")
	ckpt_024 = os.path.join(base_dir, "storage", "experiments", "EXP_024", "best_agent.pt")

	# Evaluate EXP_023 (Baseline)
	if os.path.exists(ckpt_023):
		model_023 = load_agent(ckpt_023, vocab_embeddings)
		acc, fpr, rel_fpr, pred_2_gt_5 = evaluate_symmetry(model_023, translator, breeder)
		print("\n🔴 [EXP_023] Baseline (No penalty):")
		print(f"   - Total Accuracy: {acc:.2f}%")
		print(f"   - False Positive Rate (FPR): {fpr:.2f}%")
		print(f"   - Asymmetric (>/<) FPR: {rel_fpr:.2f}%")
		print(f"   - Test case '2 > 5' -> Predicted: '{pred_2_gt_5}' (Expected: 'falsedad')")
	else:
		print("⚠️ EXP_023 checkpoint not found.")

	# Evaluate EXP_024 (Mitigation)
	if os.path.exists(ckpt_024):
		model_024 = load_agent(ckpt_024, vocab_embeddings)
		acc, fpr, rel_fpr, pred_2_gt_5 = evaluate_symmetry(model_024, translator, breeder)
		print("\n🟢 [EXP_024] Mitigation (Asymmetric Penalty = 2.5):")
		print(f"   - Total Accuracy: {acc:.2f}%")
		print(f"   - False Positive Rate (FPR): {fpr:.2f}%")
		print(f"   - Asymmetric (>/<) FPR: {rel_fpr:.2f}%")
		print(f"   - Test case '2 > 5' -> Predicted: '{pred_2_gt_5}' (Expected: 'falsedad')")
	else:
		print("⚠️ EXP_024 checkpoint not found. Training might still be in progress.")


if __name__ == "__main__":
	main()
