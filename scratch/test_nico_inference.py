import json
import os
import re
import numpy as np
import torch
from src.bitnet.dictionary_tool import SovereignDictionary
from src.bitnet.modeling_bitnet import BitNet4LayerModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_dir = "/home/joan/Documents/IA/frankenswarm"
expanded_glyphs_path = os.path.join(base_dir, "configs", "expanded_glyphs.json")
model_path = os.path.join(base_dir, "storage", "checkpoints", "sovereign_school", "model_final.pt")
state_path = os.path.join(base_dir, "storage", "checkpoints", "sovereign_school", "school_state.json")

with open(expanded_glyphs_path, encoding="utf-8") as f:
    vocab_data = json.load(f)
    words = vocab_data["words"]
    glyphs = np.array(vocab_data["glyphs"], dtype=np.float32)

word_to_idx = {w: i for i, w in enumerate(words)}
idx_to_word = dict(enumerate(words))

with open(state_path, encoding="utf-8") as sf:
    state_data = json.load(sf)
    hidden_dim = state_data.get("hidden_dim", 256)
    num_layers = state_data.get("num_layers", 6)

model = BitNet4LayerModel(
    use_glyphs=True, glyph_table=glyphs, hidden_dim=hidden_dim, num_layers=num_layers, use_pos_embedding=True, is_causal=True, max_seq_len=128
).to(device)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()

dictionary = SovereignDictionary(expanded_glyphs_path)

prompts = [
    "Hola, ¿cómo estás?",
    "¿Quién eres?",
    "¿Qué es una manzana?",
    "El perro corre rápido porque"
]

print("═" * 40)
print(f"Inferencia de Nico (Milestone: 5 años, Hidden Dim: {hidden_dim}, Layers: {num_layers})")
print("═" * 40)

for prompt in prompts:
    raw_words = re.findall(r"[a-zA-ZáéíóüñÁÉÍÓÚÜÑ_]+", prompt.lower())
    mapped_words = [dictionary.map_to_base_word(w) for w in raw_words]
    cleaned_input = " ".join(mapped_words)
    context = [word_to_idx.get(w, 1) for w in mapped_words]
    
    generated_tokens = []
    with torch.no_grad():
        for _ in range(15):
            input_len = len(context)
            padded_input = list(context)
            padded_input = padded_input + [0] * (128 - len(padded_input)) if len(padded_input) < 128 else padded_input[-128:]
            x = torch.tensor([padded_input], dtype=torch.long, device=device)
            logits = model(x)
            last_token_idx = input_len - 1
            next_logits = logits[0, last_token_idx].clone()
            
            # Simple greedy decoding
            next_token = next_logits.argmax(dim=-1).item()
            if next_token <= 1 or idx_to_word.get(next_token) in ["<pad>", "<unk>", "[eos]"]:
                break
            generated_tokens.append(next_token)
            context.append(next_token)
            
    response = " ".join([idx_to_word.get(t, "?") for t in generated_tokens])
    print(f"Pregunta: {prompt}")
    print(f"Mapeado:  {cleaned_input}")
    print(f"Nico:     {response}\n")
