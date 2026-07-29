from src.bitnet.data.dataset_breeder import MathDatasetBreeder
from src.bitnet.translation.translator import SovereignTranslator


def test_math_breeder_equations():
	translator = SovereignTranslator()
	breeder = MathDatasetBreeder(translator)

	# Generar un lote de prueba
	batch_size = 100
	op_a_targets, op_a_token_ids, operator_targets, operator_token_ids, op_b_targets, op_b_token_ids, result_targets, result_token_ids = (
		breeder.generate_batch(batch_size)
	)

	assert len(op_a_targets) == batch_size
	assert len(op_b_targets) == batch_size
	assert len(operator_targets) == batch_size
	assert len(result_targets) == batch_size

	for i in range(batch_size):
		a = op_a_targets[i]
		b = op_b_targets[i]
		op = operator_targets[i]
		r = result_targets[i]

		assert 0 <= a <= 10
		assert 0 <= b <= 10
		assert 0 <= r <= 10

		if op == 0:  # Suma
			assert a + b == r
			assert a + b <= 10
		else:  # Resta
			assert a - b == r
			assert a - b >= 0

		# Verificar decodificación de tokens
		name_a = breeder.get_operand_name(a)
		name_b = breeder.get_operand_name(b)
		name_op = breeder.get_operator_name(op)
		name_r = breeder.get_operand_name(r)

		decoded_a = translator.decode([op_a_token_ids[i]])
		decoded_b = translator.decode([op_b_token_ids[i]])
		decoded_op = translator.decode([operator_token_ids[i]])
		decoded_r = translator.decode([result_token_ids[i]])

		assert name_a == decoded_a
		assert name_b == decoded_b
		assert name_op == decoded_op
		assert name_r == decoded_r
