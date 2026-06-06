import json
import os
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# Añadir base dir al path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from src.bitnet.glyph_vocabulary import N_EMOTIONS
from src.bitnet.modeling_bitnet import BitNet4LayerModel


class RecurrentBitNetModel(BitNet4LayerModel):
	"""
	Modelo con Resonancia Temporal acoplada a través de ticks.
	"""
	def __init__(self, *args, recurrent_decay=0.8, **kwargs):
		super().__init__(*args, **kwargs)
		self.recurrent_decay = recurrent_decay

	def forward_recurrent(
		self,
		x: torch.Tensor,
		h_prev: torch.Tensor = None,
		n_steps: int = 2,
		pos_mode: str = "clock",
	) -> tuple[torch.Tensor, torch.Tensor]:
		# Entrada: Capas 1→2
		h = self._embed_input(x)

		# Inyección del estado latente anterior (Resonancia Temporal)
		if h_prev is not None:
			h = h + self.recurrent_decay * h_prev

		# Bucle Latente (Resonancia de Profundidad)
		for step in range(n_steps):
			if pos_mode == "clock" and getattr(self, "resonance_clock", None) is not None:
				h = h + self.resonance_clock[:, step, :].unsqueeze(1)

			for layer in self.core_layers:
				h = layer(h)
			h = self.norm(h)

		logits = self._decode_hidden(h)
		return logits, h.detach()

def run_single_experiment(model_type, hidden_dim, delay_ticks, max_epochs=80):
	"""
	Entrena y evalúa un modelo en una tarea de Delayed Memory.
	Retorna (accuracy_final, epochs_needed, final_loss).
	"""
	# Configurar semilla para reproducibilidad en la inicialización
	torch.manual_seed(42)
	random.seed(42)
	
	if model_type == "recurrent":
		model = RecurrentBitNetModel(
			use_glyphs=True,
			hidden_dim=hidden_dim,
			num_layers=2,
			use_pos_embedding=True,
			max_resonance_steps=3,
			n_emotions=N_EMOTIONS,
			emotion_dim=16,
			emotion_mode="first_only",
			max_seq_len=6,
			recurrent_decay=0.8
		)
	else:
		model = BitNet4LayerModel(
			use_glyphs=True,
			hidden_dim=hidden_dim,
			num_layers=2,
			use_pos_embedding=True,
			max_resonance_steps=3,
			n_emotions=N_EMOTIONS,
			emotion_dim=16,
			emotion_mode="first_only",
			max_seq_len=6
		)

	action_head = nn.Linear(hidden_dim, 2)
	optimizer = torch.optim.AdamW(list(model.parameters()) + list(action_head.parameters()), lr=1e-3)
	
	consecutive_perfect = 0
	epoch_converged = max_epochs
	last_acc = 0.0
	last_loss = 9.9
	
	for epoch in range(max_epochs):
		optimizer.zero_grad()
		loss = 0.0
		correct = 0
		total = 32  # batch size
		
		for _ in range(total):
			trial_type = random.choice([0, 1])
			init_token = 1 if trial_type == 0 else 2
			target = torch.tensor([trial_type])
			
			h_state = None
			final_hidden = None
			
			# T=0 recibe el token; T=1..delay_ticks reciben silencio (0); T=delay_ticks+1 evalúa
			total_ticks = delay_ticks + 1
			for tick in range(total_ticks):
				token_id = init_token if tick == 0 else 0
				x = torch.zeros((1, 6), dtype=torch.long)
				x[0, 0] = token_id
				
				if model_type == "recurrent":
					logits, h_state = model.forward_recurrent(x, h_prev=h_state, n_steps=2, pos_mode="clock")
					if tick == delay_ticks:
						final_hidden = h_state
				else:
					_, meta = model.forward_resonance(x, n_steps=2, pos_mode="clock")
					if tick == delay_ticks:
						final_hidden = meta["final_hidden"]
			
			act_logits = action_head(final_hidden[:, 0, :])
			loss_item = F.cross_entropy(act_logits, target)
			loss += loss_item
			
			pred = torch.argmax(act_logits, dim=-1)
			if pred.item() == trial_type:
				correct += 1
				
		loss = loss / total
		loss.backward()
		optimizer.step()
		
		acc = correct / total
		last_acc = acc
		last_loss = loss.item()
		
		if acc >= 1.0:
			consecutive_perfect += 1
		else:
			consecutive_perfect = 0
			
		if consecutive_perfect >= 3:
			epoch_converged = epoch - 1  # Época real donde ya era perfecto
			break
			
	return last_acc, epoch_converged, last_loss

def main():
	print("=" * 80)
	print("📊 SWEEP DE CAPACIDAD DE MEMORIA LATENTE Y RESONANCIA TEMPORAL 📊")
	print("=" * 80)
	
	# Parámetros del Sweep
	hidden_dims = [8, 16, 32, 64, 128, 256]
	delay_ticks_list = [1, 2, 3, 4, 5, 6, 8]
	
	results = []
	
	# Crear directorio para almacenar resultados si no existe
	os.makedirs(os.path.join(base_dir, "storage"), exist_ok=True)
	
	print(f"{'Modelo':12} | {'Hidden Dim':10} | {'Delay Ticks':11} | {'Acc Final':9} | {'Ep Converg':10} | {'Pérdida':7}")
	print("-" * 80)
	
	for delay in delay_ticks_list:
		for dim in hidden_dims:
			# Primero el recurrente
			rec_acc, rec_ep, rec_loss = run_single_experiment("recurrent", dim, delay)
			results.append({
				"model_type": "recurrent",
				"hidden_dim": dim,
				"delay_ticks": delay,
				"accuracy": rec_acc,
				"epochs_to_converge": rec_ep,
				"loss": rec_loss
			})
			print(f"{'RECURRENTE':12} | {dim:10} | {delay:11} | {rec_acc*100:8.1f}% | {rec_ep:10} | {rec_loss:.4f}")
			
			# Ocasionalmente ejecutamos el estándar como baseline de control para probar el azar (50%)
			# Solo lo hacemos para unas pocas dimensiones para ahorrar tiempo
			if dim in [16, 64, 256] and delay in [1, 3, 6]:
				std_acc, std_ep, std_loss = run_single_experiment("standard", dim, delay)
				results.append({
					"model_type": "standard",
					"hidden_dim": dim,
					"delay_ticks": delay,
					"accuracy": std_acc,
					"epochs_to_converge": std_ep,
					"loss": std_loss
				})
				print(f"{'ESTÁNDAR':12} | {dim:10} | {delay:11} | {std_acc*100:8.1f}% | {std_ep:10} | {std_loss:.4f}")
		print("-" * 80)
		
	# Guardar resultados en JSON
	out_path = os.path.join(base_dir, "storage", "memory_capacity_sweep.json")
	with open(out_path, "w", encoding="utf-8") as f:
		json.dump(results, f, indent=4)
		
	print(f"\n[ÉXITO] Resultados del barrido guardados en {out_path}")
	
	# Calcular curva teórica de degradación vs capacidad
	# Encontramos la menor dimensión que converge al 100% de precisión en cada delay
	print("\n📈 RESUMEN DE COMPACTACIÓN DE MEMORIA (Análisis de Capacidad) 📈")
	print("-" * 60)
	print(f"{'Delay Ticks':15} | {'Min Dimensión para Memoria Perfecta (100% Acc)':40}")
	print("-" * 60)
	for delay in delay_ticks_list:
		converged_dims = [
			r["hidden_dim"] for r in results 
			if r["model_type"] == "recurrent" and r["delay_ticks"] == delay and r["accuracy"] >= 0.98
		]
		if converged_dims:
			min_dim = min(converged_dims)
			print(f"{delay:15} | {min_dim:40}")
		else:
			print(f"{delay:15} | No converge con las dimensiones probadas (>256)")
	print("-" * 60)

if __name__ == "__main__":
	main()
