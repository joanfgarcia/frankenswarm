from src.bitnet.generalization_breeder import CompositionalMathDatasetBreeder
from src.bitnet.translator import SovereignTranslator


def test_compositional_breeder_splits():
	translator = SovereignTranslator()
	breeder = CompositionalMathDatasetBreeder(translator, split_ratio=0.8, seed=42)

	# 1. Verificar tamaño de splits
	total_equations = len(breeder.train_equations) + len(breeder.test_equations)
	assert total_equations == 132
	assert len(breeder.train_equations) == int(132 * 0.8)

	# 2. Verificar que son conjuntos disjuntos (intersección vacía)
	train_set = set(breeder.train_equations)
	test_set = set(breeder.test_equations)
	intersection = train_set.intersection(test_set)
	assert len(intersection) == 0

	# 3. Verificar cobertura completa en el conjunto de entrenamiento
	seen_operands = set()
	seen_operators = set()
	for a, op, b, r in breeder.train_equations:
		seen_operands.update([a, b, r])
		seen_operators.add(op)

	assert len(seen_operands) == 11  # [0..10]
	assert len(seen_operators) == 2  # [suma, resta]

	# 4. Validar generación de lotes en ambos conjuntos
	batch_size = 50

	# Test conjunto de train
	op_a_targets, op_a_token_ids, operator_targets, operator_token_ids, op_b_targets, op_b_token_ids, result_targets, result_token_ids = (
		breeder.generate_batch(batch_size, mode="train")
	)
	assert len(op_a_targets) == batch_size
	for i in range(batch_size):
		a, op, b, r = op_a_targets[i], operator_targets[i], op_b_targets[i], result_targets[i]
		assert (a, op, b, r) in train_set
		if op == 0:
			assert a + b == r
		else:
			assert a - b == r

	# Test conjunto de test
	op_a_targets, op_a_token_ids, operator_targets, operator_token_ids, op_b_targets, op_b_token_ids, result_targets, result_token_ids = (
		breeder.generate_batch(batch_size, mode="test")
	)
	assert len(op_a_targets) == batch_size
	for i in range(batch_size):
		a, op, b, r = op_a_targets[i], operator_targets[i], op_b_targets[i], result_targets[i]
		assert (a, op, b, r) in test_set
		if op == 0:
			assert a + b == r
		else:
			assert a - b == r
