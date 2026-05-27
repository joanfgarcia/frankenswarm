import os
import sys
import torch
import numpy as np

# Ensure root dir is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.router.swi_prolog_router import NodeTarget, SiliconTarget, route, route_semantic
from src.bitnet.modeling_bitnet import BitNet4LayerModel
from src.bitnet.translator import SovereignTranslator

def run_math_inference(model, translator, op_a_word, operator_word, op_b_word):
	# Encode operands and operator
	tids_a = translator.encode(op_a_word)
	tids_b = translator.encode(op_b_word)
	tids_op = translator.encode(operator_word)

	tid_a = tids_a[0] if tids_a else 0
	tid_b = tids_b[0] if tids_b else 0
	tid_op = tids_op[0] if tids_op else 0

	# Input: [A, op, B, 0]
	input_tensor = torch.tensor([[tid_a, tid_op, tid_b, 0]], dtype=torch.long)

	with torch.no_grad():
		logits = model(input_tensor)
		# Prediction of the result is at sequence index 3
		result_logits = logits[0, 3, :]
		pred_tid = torch.argmax(result_logits).item()

	result_word = translator.decode([pred_tid])
	return result_word, pred_tid

def run_vocab_inference(model, translator, concept_word, emotion_word, homeo_word):
	# Encode concept, emotion, and homeostasis
	tids_c = translator.encode(concept_word)
	tids_e = translator.encode(emotion_word)
	tids_h = translator.encode(homeo_word)

	tid_c = tids_c[0] if tids_c else 0
	tid_e = tids_e[0] if tids_e else 0
	tid_h = tids_h[0] if tids_h else 0

	# Input: [concept, emotion, homeostasis, 0]
	input_tensor = torch.tensor([[tid_c, tid_e, tid_h, 0]], dtype=torch.long)

	with torch.no_grad():
		# Speak: Generate message
		message = model.generate_message(input_tensor, tau=0.3, hard=True)
		# Listen: Decode message
		logits = model(message)
		
		# Speaker message decode for logging
		msg_token_ids = torch.argmax(message[0], dim=-1).tolist()
		msg_decoded = translator.decode(msg_token_ids)

		# Listener predictions
		pred_c_tid = torch.argmax(logits[0, 1, :]).item()
		pred_e_tid = torch.argmax(logits[0, 2, :]).item()
		pred_h_tid = torch.argmax(logits[0, 3, :]).item()

	pred_concept = translator.decode([pred_c_tid])
	pred_emotion = translator.decode([pred_e_tid])
	pred_homeo = translator.decode([pred_h_tid])

	return msg_decoded, pred_concept, pred_emotion, pred_homeo

def main():
	print("==================================================")
	print("🔬 FRANKENSWARM: PROOF OF CONCEPT (POC) PIPELINE")
	print("   SWI-Prolog Router + Ternary BitNet MoE Nodes")
	print("==================================================")

	# 1. Initialize Translator
	translator = SovereignTranslator()
	vocab_embeddings = translator.get_concept_embeddings()

	# 2. Setup models
	base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	math_ckpt = os.path.join(base_dir, "storage", "experiments", "EXP_020", "best_agent.pt")
	vocab_ckpt = os.path.join(base_dir, "storage", "experiments", "EXP_022", "best_agent.pt")

	if not os.path.exists(math_ckpt):
		print(f"❌ Math checkpoint not found at: {math_ckpt}")
		return
	if not os.path.exists(vocab_ckpt):
		print(f"❌ Vocab checkpoint not found at: {vocab_ckpt}")
		return

	# Load Math Model (EXP_020)
	model_math = BitNet4LayerModel(
		vocab_embeddings=vocab_embeddings,
		hidden_dim=256,
		num_layers=3
	)
	model_math.load_state_dict(torch.load(math_ckpt, map_location="cpu"))
	model_math.eval()
	print("✅ Loaded EXP_020 (Math Core Specialist)")

	# Load Vocab Model (EXP_022)
	model_vocab = BitNet4LayerModel(
		vocab_embeddings=vocab_embeddings,
		hidden_dim=256,
		num_layers=3
	)
	model_vocab.load_state_dict(torch.load(vocab_ckpt, map_location="cpu"))
	model_vocab.eval()
	print("✅ Loaded EXP_022 (Vocabulary Specialist)")

	# 3. Test queries
	test_queries = [
		"explain cinco resta dos",
		"hola búnker gato fuego peligro",
	]

	concepts_list = ["gato", "perro", "casa", "árbol", "agua", "fuego", "tierra", "aire", "sol", "luna", "peligro", "seguridad", "búnker", "agente", "código"]
	emotions_list = ["miedo", "alegría", "ira", "tristeza", "dolor", "hambre"]
	homeostasis_list = ["neutral", "dolor", "hambre", "urgencia", "seguridad"]

	for query in test_queries:
		print("\n--------------------------------------------------")
		print(f"💬 Input Query: '{query}'")

		# Route task
		expert = route(query)
		domain = "general"
		if expert == NodeTarget.REASON:
			domain = "logic_math"
		elif expert == NodeTarget.CODE:
			domain = "code_python"
		elif expert == NodeTarget.SYNTH:
			domain = "synth"

		_, silicon = route_semantic(domain, len(query))
		print(f"⚡ [Prolog Route] Node: {expert} | Hardware: {silicon}")

		words = query.lower().replace(",", " ").replace(".", " ").replace("?", " ").split()

		if expert == NodeTarget.REASON:
			# Arithmetic specialization execution
			op_a_word = next((w for w in words if w in ["cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez"]), "cinco")
			op_b_word = next((w for w in reversed(words) if w in ["cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez"]), "dos")
			operator_word = next((w for w in words if w in ["suma", "resta", "multiplica"]), "resta")

			print(f"🧠 [MoE Exec] Routing to EXP_020 | Task: {op_a_word} {operator_word} {op_b_word}...")
			res_word, res_tid = run_math_inference(model_math, translator, op_a_word, operator_word, op_b_word)
			print(f"✨ [Result] Output Token ID: {res_tid} | Decoded: '{res_word}'")

		elif expert == NodeTarget.DEFAULT:
			# General/Vocabulary execution
			concept_word = next((w for w in words if w in concepts_list), "fuego")
			emotion_word = next((w for w in words if w in emotions_list), "miedo")
			homeo_word = next((w for w in words if w in homeostasis_list), "urgencia")

			print(f"🧠 [MoE Exec] Routing to EXP_022 | Task: [Concept: {concept_word}, Emotion: {emotion_word}, Homeo: {homeo_word}]...")
			msg_dec, pred_c, pred_e, pred_h = run_vocab_inference(model_vocab, translator, concept_word, emotion_word, homeo_word)
			print(f"✨ [Result] Emergent Message: '{msg_dec}'")
			print(f"✨ [Decoded] Concept: '{pred_c}' | Emotion: '{pred_e}' | Homeostasis: '{pred_h}'")

if __name__ == "__main__":
	main()
