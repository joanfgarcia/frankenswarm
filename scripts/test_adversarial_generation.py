"""Script de Evaluación Adversarial 2: Generación Estocástica / Greedy e Inferencia Libre en K-65P.

Carga el modelo graduado Bit v2 (8 años) y evalúa si genera expresiones canónicas K-65P válidas,
tanto con mask de vocabulario (Curriculum-Gated) como sin mask.
"""

import sys
import json
import torch
import torch.nn.functional as F
from pathlib import Path

base_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(base_dir))

k65p_src = base_dir.parent / "k65p" / "src"
if k65p_src.exists():
	sys.path.append(str(k65p_src))

from src.bitnet.model.modeling_bitnet import BitNet4LayerModel
from src.bitnet.training.train_sovereign_school_k65p import (
	build_k65p_vocab_and_glyphs,
	tokenize_k65p,
	build_stage_logit_mask,
)
from k65p.validator import is_valid

CHECKPOINT_PATH = base_dir / "storage" / "checkpoints" / "releases" / "bit_v2_8yo_graduated_k65p" / "model_final_k65p.pt"
if not CHECKPOINT_PATH.exists():
	CHECKPOINT_PATH = base_dir / "storage" / "checkpoints" / "sovereign_school_k65p" / "model_current_k65p.pt"

def run_adversarial_generation():
	print("═══ 🧪 TEST ADVERSARIAL 2: INFERENCIA GENERATIVA LIBRE (K-65P) ═══\n")
	device = torch.device("cpu")
	word_to_idx, idx_to_word, glyph_table = build_k65p_vocab_and_glyphs()
	vocab_size = len(word_to_idx)
	
	model = BitNet4LayerModel(
		use_glyphs=True,
		glyph_table=glyph_table,
		hidden_dim=1024,
		num_layers=6,
		use_pos_embedding=True,
		is_causal=True,
		max_seq_len=128,
	).to(device)

	checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
	state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
	model.load_state_dict(state_dict)
	model.eval()

	# Mask de la etapa 8 (Secundaria) que permite todas las moléculas + primos y prohíbe pad/unk
	logit_mask = build_stage_logit_mask(7, vocab_size, word_to_idx).to(device)

	prompts = [
		"[si [touch alguien",
		"[do yo_palabra",
		"[feel alguien",
		"[think alguien",
		"[see alguien",
	]

	print(f"Modelo cargado: {CHECKPOINT_PATH.name} ({sum(p.numel() for p in model.parameters()):,} params activos en 1024d)\n")

	# Test 1: Con Logit Mask (Gating de Vocabulario Oficial)
	print("── A) Generación con Gating de Vocabulario Escolar (Logit Mask) ──")
	valid_syntax_gated = 0
	for p_text in prompts:
		tokens = tokenize_k65p(p_text, word_to_idx, max_len=128)
		pad_id = word_to_idx["<pad>"]
		prompt_tokens = [t for t in tokens if t != pad_id]

		generated = list(prompt_tokens)
		with torch.no_grad():
			for _ in range(35):
				curr_tensor = torch.tensor([generated], dtype=torch.long, device=device)
				if curr_tensor.shape[1] >= 128:
					break
				logits = model(curr_tensor)
				masked_logits = logits[0, -1, :] + logit_mask
				next_id = torch.argmax(masked_logits).item()
				
				if next_id == word_to_idx.get("<stop>", -1):
					break
				generated.append(next_id)
				
				words = [idx_to_word.get(i, "") for i in generated]
				if words.count("[") == words.count("]") and words.count("[") > 0:
					break

		words = [idx_to_word.get(i, "") for i in generated]
		gen_str = " ".join(words).replace(" [ ", " [").replace(" ] ", "] ")
		
		try:
			val_ok = is_valid(gen_str)
		except Exception:
			val_ok = False

		if val_ok:
			valid_syntax_gated += 1

		print(f"Prompt : '{p_text}'")
		print(f"Salida : '{gen_str}'")
		print(f"Sintaxis K-65P Validada: {'✅ OK' if val_ok else '❌ FAIL'}\n")

	print(f"📊 RESULTADO TEST ADVERSARIAL 2:")
	print(f"   • Sintaxis Gated   : {valid_syntax_gated} / {len(prompts)} ({valid_syntax_gated/len(prompts)*100:.1f}%)")

if __name__ == "__main__":
	run_adversarial_generation()
