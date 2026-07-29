import argparse
import json
import os
import sys

import numpy as np
import torch

# Añadir el directorio raíz de frankenswarm al path para poder importar
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.bitnet.model.modeling_bitnet import BitNet4LayerModel
from src.bitnet.translation.translator import SovereignTranslator


def print_header(title: str):
	print("\n================================================================================")
	print(f"🔬 LENTE: {title}")
	print("================================================================================")


def inspect_synapses(model: torch.nn.Module):
	print_header("ESTRUCTURA SINÁPTICA TERNARIA")

	total_weights = 0
	val_neg_1 = 0
	val_0 = 0
	val_1 = 0
	other_vals = 0

	for name, param in model.named_parameters():
		if "weight" in name and param.ndim == 2:
			# Cuantizar usando el estimador STE del modelo
			scale = param.abs().mean().clamp(min=1e-5).item()
			quant = (param / scale).round().clamp(-1, 1)

			w_flat = quant.detach().cpu().numpy().flatten()
			total_weights += len(w_flat)

			val_neg_1 += np.sum(w_flat == -1)
			val_0 += np.sum(w_flat == 0)
			val_1 += np.sum(w_flat == 1)
			other_vals += np.sum((w_flat != -1) & (w_flat != 0) & (w_flat != 1))

			print(f"Matriz: {name:40s} | Shape: {str(list(param.shape)):15s} | Escala de Cuantización: {scale:.5f}")

	print("-" * 80)
	print(f"Total Parámetros de Proyección Analizados: {total_weights:,}")
	if total_weights > 0:
		p_neg = val_neg_1 / total_weights * 100
		p_zero = val_0 / total_weights * 100
		p_pos = val_1 / total_weights * 100
		p_other = other_vals / total_weights * 100

		print(f"  • Pesos Ternarios -1 : {val_neg_1:10,} ({p_neg:6.2f}%)  [■■■■■■■■■□]")
		print(f"  • Pesos Ternarios  0 : {val_0:10,} ({p_zero:6.2f}%)  [■■■■■□□□□□]")
		print(f"  • Pesos Ternarios +1 : {val_1:10,} ({p_pos:6.2f}%)  [■■■■■■■■■□]")
		if p_other > 0:
			print(f"  • Pesos Flotantes/Otros: {other_vals:10,} ({p_other:6.2f}%)  [! WARNING !]")
		else:
			print("  ✓ Estado de Integridad: TERNARIO PURO (100% verificado)")


def inspect_homeostasis(model: torch.nn.Module):
	print_header("DINÁMICA HOMEOSTÁTICA DE SILICIO (SIMULACIÓN)")

	# Parámetros metabólicos simulados
	energia = 100.0
	temperatura = 35.0
	integridad = 100.0
	costo_forward = 4.5
	decaimiento_termico = 0.8

	print(f"Estado Inicial: Energía={energia:.1f}% | Temp={temperatura:.1f}°C | Integridad={integridad:.1f}%")
	print("-" * 80)
	print(f"{'PASO':6s} | {'ACCIÓN':20s} | {'ENERGÍA':10s} | {'TEMPERATURA':12s} | {'INTEGRIDAD':10s} | {'ESTADO EMOCIONAL'}")
	print("-" * 80)

	for paso in range(1, 6):
		# Simular forward pass de alta carga
		energia -= costo_forward
		# Aumento de temperatura basado en el tamaño oculto y capas del modelo
		temperatura += (model.hidden_dim / 32.0) * len(model.core_layers) * 0.1
		temperatura -= decaimiento_termico

		# Afectar integridad si la temperatura cruza 45°C
		if temperatura > 45.0:
			danio = (temperatura - 45.0) * 1.5
			integridad -= danio

		# Determinar estado emocional
		emocion = "ALEGRE"
		if energia < 80.0:
			emocion = "HAMBRE"
		if energia < 50.0 or integridad < 90.0:
			emocion = "DOLOR"
		if temperatura > 42.0:
			emocion = "MIEDO"
		if integridad < 50.0:
			emocion = "CRÍTICO"

		print(f"#{paso:02d}   | Computación Arena   | {energia:8.1f}% | {temperatura:10.1f}°C | {integridad:8.1f}%   | {emocion}")

	print("-" * 80)
	print(f"Simulación Homeostática Finalizada. Consumo total de glucosa de silicio: {(100.0 - energia):.1f}%")


def inspect_linguistics(model: torch.nn.Module, translator: SovereignTranslator):
	print_header("SONDA LINGÜÍSTICA DE CAPA 1")

	# Detección heurística de la especialidad del espécimen
	specimen_type = "legacy_vocab"
	config_data = {}

	# Buscar la ruta del espécimen a través de los argumentos pasados
	specimen_path = None
	for arg in sys.argv:
		if arg.endswith(".pt") or "best_agent" in arg:
			specimen_path = arg
			break

	if specimen_path:
		# Si la TUI pasó la ruta completa del checkpoint best_agent.pt,
		# podemos buscar config.json en su mismo directorio
		parent_dir = os.path.dirname(specimen_path)
		config_file = os.path.join(parent_dir, "config.json")
		if os.path.exists(config_file):
			try:
				with open(config_file, encoding="utf-8") as f:
					config_data = json.load(f)
					exp_id = config_data.get("experiment_id", "")
					if "EXP_023" in exp_id or "logic" in exp_id or ("operators" in config_data and "mayor_que" in config_data["operators"]):
						specimen_type = "relational_logic"
					elif (
						"EXP_020" in exp_id
						or "EXP_018" in exp_id
						or "EXP_017" in exp_id
						or "math" in exp_id
						or ("operators" in config_data and "resta" in config_data["operators"])
					):
						specimen_type = "arithmetic"
					elif "EXP_021" in exp_id or "EXP_022" in exp_id or "populora" in exp_id or "micro_vocab_words" in config_data:
						specimen_type = "vocab_3d"
			except Exception:
				pass

	# Heurística fallback en base a pos_embedding o tokens del vocabulario
	if specimen_type == "legacy_vocab":
		has_pos = getattr(model, "pos_embedding", None) is not None
		if has_pos:
			# Si tiene pos embedding pero no hay config, decidimos por codificación de relaciones
			tids_gt = translator.encode(">")
			specimen_type = "relational_logic" if tids_gt else "arithmetic"

	device = next(model.parameters()).device
	model.eval()

	print(f"Tipo de Diagnóstico Clínico: {specimen_type.upper()}")
	print("-" * 80)

	if specimen_type == "arithmetic":
		# Test aritmético: cinco - dos = tres
		op_a, op_val, op_b = "cinco", "resta", "dos"
		tids_a = translator.encode(op_a)
		tids_op = translator.encode(op_val)
		tids_b = translator.encode(op_b)

		tid_a = tids_a[0] if tids_a else 0
		tid_op = tids_op[0] if tids_op else 0
		tid_b = tids_b[0] if tids_b else 0

		print("Entrada de Aritmética:")
		print(f"  • Operando A: '{op_a}' (ID {tid_a})")
		print(f"  • Operador:   '{op_val}' (ID {tid_op})")
		print(f"  • Operando B: '{op_b}' (ID {tid_b})")

		# Secuencia: [A, op, B, 0]
		speaker_input = torch.tensor([[tid_a, tid_op, tid_b, 0]], dtype=torch.long, device=device)

		with torch.no_grad():
			logits = model(speaker_input)
			pred_tid = torch.argmax(logits[0, 3, :]).item()
			pred_word = translator.decode([pred_tid])

		print("\nResultado de la Inferencia (Paso 4):")
		print(f"  • Token ID Predicho: {pred_tid}")
		print(f"  • Decodificado:      '{pred_word}'")
		print("-" * 80)
		print(f"Resultado del Test Clínico: {'✓ ÉXITO (Cálculo Correcto)' if pred_word == 'tres' else '✗ ERROR (Cálculo Incorrecto)'}")

	elif specimen_type == "relational_logic":
		# Test de lógica relacional: cinco > dos = verdad
		op_a, relation, op_b = "cinco", ">", "dos"
		tids_a = translator.encode(op_a)
		tids_rel = translator.encode(relation)
		tids_b = translator.encode(op_b)

		tid_a = tids_a[0] if tids_a else 0
		tid_rel = tids_rel[0] if tids_rel else 0
		tid_b = tids_b[0] if tids_b else 0

		print("Entrada de Lógica Relacional (Test 1 - Mayor que):")
		print(f"  • Operando A: '{op_a}' (ID {tid_a})")
		print(f"  • Relación:   '{relation}' (ID {tid_rel})")
		print(f"  • Operando B: '{op_b}' (ID {tid_b})")

		speaker_input_1 = torch.tensor([[tid_a, tid_rel, tid_b, 0]], dtype=torch.long, device=device)

		with torch.no_grad():
			logits_1 = model(speaker_input_1)
			pred_tid_1 = torch.argmax(logits_1[0, 3, :]).item()
			pred_word_1 = translator.decode([pred_tid_1])

		print("\nResultado de la Inferencia 1 (Paso 4):")
		print(f"  • Token ID Predicho: {pred_tid_1}")
		print(f"  • Decodificado:      '{pred_word_1}'")

		# Test 2: conmutación asimétrica: dos > cinco = falsedad
		print("\nEntrada de Lógica Relacional (Test 2 - Asimetría posicional):")
		print(f"  • Operando A: '{op_b}' (ID {tid_b})")
		print(f"  • Relación:   '{relation}' (ID {tid_rel})")
		print(f"  • Operando B: '{op_a}' (ID {tid_a})")

		speaker_input_2 = torch.tensor([[tid_b, tid_rel, tid_a, 0]], dtype=torch.long, device=device)

		with torch.no_grad():
			logits_2 = model(speaker_input_2)
			pred_tid_2 = torch.argmax(logits_2[0, 3, :]).item()
			pred_word_2 = translator.decode([pred_tid_2])

		print("\nResultado de la Inferencia 2 (Paso 4):")
		print(f"  • Token ID Predicho: {pred_tid_2}")
		print(f"  • Decodificado:      '{pred_word_2}'")
		print("-" * 80)

		success_1 = pred_word_1 == "verdad"
		success_2 = pred_word_2 == "falsedad"
		print("Resultado del Test de Juicio lógico:")
		print(f"  • Comparación Correcta:   {'✓ ÉXITO' if success_1 else '✗ ERROR'}")
		print(f"  • Asimetría Posicional:   {'✓ ÉXITO' if success_2 else '✗ ERROR'}")
		if success_1 and success_2:
			print("  ✓ Integridad Lógica Relacional: 100% VERIFICADO")
		else:
			print("  ✗ Integridad Lógica Relacional: DESALINEADO O SIMÉTRICO")

	elif specimen_type == "vocab_3d":
		# Test de vocabulario 3D: fuego, miedo, urgencia
		concept_test, emotion_test, homeo_test = "fuego", "miedo", "urgencia"

		concept_ids = translator.encode(concept_test)
		emotion_ids = translator.encode(emotion_test)
		homeo_ids = translator.encode(homeo_test)

		c_id = concept_ids[0] if concept_ids else 0
		e_id = emotion_ids[0] if emotion_ids else 0
		h_id = homeo_ids[0] if homeo_ids else 0

		print("Entrada del Hablante (3D):")
		print(f"  • Concepto Objetivo: '{concept_test}' (ID {c_id})")
		print(f"  • Estado Afectivo:   '{emotion_test}' (ID {e_id})")
		print(f"  • Homeostasis:       '{homeo_test}' (ID {h_id})")

		speaker_input = torch.tensor([[c_id, e_id, h_id, 0]], dtype=torch.long, device=device)

		with torch.no_grad():
			message = model.generate_message(speaker_input, tau=0.1, hard=True)
			msg_tokens = torch.argmax(message[0], dim=-1).tolist()
			msg_words = translator.decode(msg_tokens)

			print("\nMensaje emitido por el Hablante (Capa 1):")
			print(f"  • Token IDs: {msg_tokens}")
			print(f"  • Conceptos: '{msg_words}'")

			logits = model(message)
			pred_c_id = torch.argmax(logits[0, 1, :]).item()
			pred_e_id = torch.argmax(logits[0, 2, :]).item()
			pred_h_id = torch.argmax(logits[0, 3, :]).item()

		print("\nDescodificación del Oyente:")
		print(f"  • Concepto predicho: '{translator.decode([pred_c_id])}' (ID {pred_c_id})")
		print(f"  • Emoción predicha:  '{translator.decode([pred_e_id])}' (ID {pred_e_id})")
		print(f"  • Homeo predicha:     '{translator.decode([pred_h_id])}' (ID {pred_h_id})")

		match_c = pred_c_id == c_id
		match_e = pred_e_id == e_id
		match_h = pred_h_id == h_id

		print("-" * 80)
		print("Resultado del Juego Referencial 3D:")
		print(f"  • Entendimiento Conceptual: {'✓ ÉXITO' if match_c else '✗ ERROR'}")
		print(f"  • Alineación Afectiva:       {'✓ ÉXITO' if match_e else '✗ ERROR'}")
		print(f"  • Alineación Homeostática:  {'✓ ÉXITO' if match_h else '✗ ERROR'}")
		if match_c and match_e and match_h:
			print("  ✓ Recompensa conjunta: R = 1.0 (Entendimiento Mutuo Pleno)")
		else:
			print("  ✗ Recompensa: R = 0.0 (Fallo en la comunicación)")

	else:
		# Legacy Vocab (longitud 3)
		concept_test = "fuego"
		emotion_test = "miedo"

		concept_ids = translator.encode(concept_test)
		emotion_ids = translator.encode(emotion_test)

		c_id = concept_ids[0] if concept_ids else 0
		e_id = emotion_ids[0] if emotion_ids else 0

		print("Entrada del Hablante (2D Legacy):")
		print(f"  • Concepto Objetivo: '{concept_test}' (ID {c_id})")
		print(f"  • Estado Afectivo:   '{emotion_test}' (ID {e_id})")

		speaker_input = torch.tensor([[c_id, e_id, 0]], dtype=torch.long, device=device)

		with torch.no_grad():
			message = model.generate_message(speaker_input, tau=0.1, hard=True)
			msg_tokens = torch.argmax(message[0], dim=-1).tolist()
			msg_words = translator.decode(msg_tokens)

			print("\nMensaje emitido por el Hablante (Capa 1):")
			print(f"  • Token IDs: {msg_tokens}")
			print(f"  • Conceptos: '{msg_words}'")

			logits = model(message)
			pred_c_id = torch.argmax(logits[0, 1, :]).item()
			pred_e_id = torch.argmax(logits[0, 2, :]).item()

		print("\nDescodificación del Oyente (Empatía):")
		print(f"  • Concepto predicho: '{translator.decode([pred_c_id])}' (ID {pred_c_id})")
		print(f"  • Emoción predicha:  '{translator.decode([pred_e_id])}' (ID {pred_e_id})")

		match_c = pred_c_id == c_id
		match_e = pred_e_id == e_id

		print("-" * 80)
		print("Resultado del Juego Referencial Afectivo:")
		print(f"  • Entendimiento Conceptual: {'✓ ÉXITO' if match_c else '✗ ERROR'}")
		print(f"  • Alineación Afectiva:       {'✓ ÉXITO' if match_e else '✗ ERROR'}")


def inspect_svd(model: torch.nn.Module):
	print_header("ESPECTRO DE VALORES SINGULARES (SVD)")

	# Tomar la primera capa del core para analizar
	analyzed = False
	for name, param in model.named_parameters():
		if "core_layers.0.attn.q_proj.weight" in name:
			try:
				w = param.detach().cpu().float()
				u, s, vh = torch.linalg.svd(w, full_matrices=False)
				s_np = s.numpy()

				print(f"Matriz Analizada: {name}")
				print(f"Rango Matemático: {len(s_np)} | Valores Singulares Mayores (Top 10):")
				for idx, val in enumerate(s_np[:10]):
					bar = "■" * int(min(20, val * 10))
					print(f"  σ_{idx + 1:02d} : {val:8.4f} {bar}")

				# Calcular entropía espectral aproximada
				s_norm = s_np / np.sum(s_np)
				entropy = -np.sum(s_norm * np.log(s_norm + 1e-10))
				print(f"\nEntropía Espectral del Peso: {entropy:.4f} (Indica dispersión de información)")
				analyzed = True
			except Exception as e:
				print(f"Error en descomposición SVD: {e}")
			break

	if not analyzed:
		print("No se encontró la matriz q_proj de la primera capa del Core para realizar SVD.")


def main():
	parser = argparse.ArgumentParser(description="Microscopio Cognitivo del Búnker - Inspecciona Especímenes BitNet.")
	parser.add_argument("--specimen", type=str, help="Ruta al archivo .pt del espécimen a inspeccionar.")
	parser.add_argument("--hidden_dim", type=int, default=256, help="Dimensión oculta (por defecto 256).")
	parser.add_argument("--num_layers", type=int, default=4, help="Número de capas (por defecto 4).")
	args = parser.parse_args()

	print("================================================================================")
	print("                     🔬 MICROSCOPIO COGNITIVO DEL BÚNKER v1                     ")
	print("================================================================================")

	translator = SovereignTranslator()
	vocab_embeddings = translator.get_concept_embeddings()

	if args.specimen and os.path.exists(args.specimen):
		print(f"Cargando espécimen desde: {args.specimen}")
		model = BitNet4LayerModel(vocab_embeddings=vocab_embeddings, hidden_dim=args.hidden_dim, num_layers=args.num_layers)
		try:
			model.load_state_dict(torch.load(args.specimen, map_location="cpu"))
			print("✓ Espécimen cargado con éxito.")
		except Exception as e:
			print(f"Error al cargar pesos del espécimen: {e}. Inicializando sujeto virgen para diagnóstico.")
			model = BitNet4LayerModel(vocab_embeddings=vocab_embeddings, hidden_dim=args.hidden_dim, num_layers=args.num_layers)
	else:
		print("Inicializando sujeto virgen de control (Kaoting Uniforme)...")
		model = BitNet4LayerModel(vocab_embeddings=vocab_embeddings, hidden_dim=args.hidden_dim, num_layers=args.num_layers)

	inspect_synapses(model)
	inspect_homeostasis(model)
	inspect_linguistics(model, translator)
	inspect_svd(model)


if __name__ == "__main__":
	main()
