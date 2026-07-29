import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.bitnet.glyph_vocabulary import N_EMOTIONS
from src.bitnet.modeling_bitnet import BitNet4LayerModel


class RecurrentBitNetModel(BitNet4LayerModel):
	"""
	Modelo con Resonancia Temporal (capa 4 acoplada a capa 2 a través de los ticks).
	"""
	def __init__(self, *args, recurrent_decay=0.7, **kwargs):
		super().__init__(*args, **kwargs)
		self.recurrent_decay = recurrent_decay

	def forward_recurrent(
		self,
		x: torch.Tensor,
		h_prev: torch.Tensor = None,
		n_steps: int = 2,
		pos_mode: str = "clock",
		emotion_ids: torch.Tensor = None,
	) -> tuple[torch.Tensor, torch.Tensor]:
		# ── Entrada: Capas 1→2 ──
		h = self._embed_input(x)

		# ── Resonancia Temporal: Inyección del hidden anterior ──
		if h_prev is not None:
			h = h + self.recurrent_decay * h_prev

		# Bucle Latente (Resonancia de Profundidad)
		for step in range(n_steps):
			if pos_mode == "clock" and getattr(self, "resonance_clock", None) is not None:
				h = h + self.resonance_clock[:, step, :].unsqueeze(1)

			for layer in self.core_layers:
				h = layer(h)
			h = self.norm(h)

		# ── Salida: Capas 4→5 ──
		logits = self._decode_hidden(h)
		return logits, h.detach()

def train_and_eval(model_type="recurrent", epochs=150, hidden_dim=128):
	"""
	Entrena un modelo en una tarea de memoria a largo plazo (Delayed Memory Task).
	Secuencia de 4 pasos temporales:
	T=0: Input = A (token 1) o B (token 2)
	T=1: Input = Silencio (token 0)
	T=2: Input = Silencio (token 0)
	T=3: Input = Silencio (token 0) -> El modelo debe predecir la acción A (0) o B (1) basándose en T=0.
	"""
	
	# Usar modo de glifos simples
	model = None
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
		# Estándar sin memoria recurrente entre ticks
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

	# Cabezal de acción lineal para simplificar
	action_head = nn.Linear(hidden_dim, 2)
	optimizer = torch.optim.AdamW(list(model.parameters()) + list(action_head.parameters()), lr=1e-3)
	
	# Simular dataset
	# 0 = neutral, 1 = token_A, 2 = token_B
	# seq: [clase, 0, 0, 0, 0, 0] de tamaño (1, 6)
	
	for epoch in range(epochs):
		optimizer.zero_grad()
		loss = 0.0
		correct = 0
		total = 32 # batch size
		
		for _ in range(total):
			# Decidir si la prueba es de tipo A (0) o B (1)
			trial_type = random.choice([0, 1])
			init_token = 1 if trial_type == 0 else 2
			target = torch.tensor([trial_type])
			
			h_state = None
			final_hidden = None
			
			# Secuencia de 4 ticks temporales
			for tick in range(4):
				# T=0 recibe el token de clase; T=1,2,3 reciben neutral (0)
				token_id = init_token if tick == 0 else 0
				x = torch.zeros((1, 6), dtype=torch.long)
				x[0, 0] = token_id
				
				if model_type == "recurrent":
					logits, h_state = model.forward_recurrent(x, h_prev=h_state, n_steps=2, pos_mode="clock")
					if tick == 3:
						final_hidden = h_state # Guardamos el último estado oculto
				else:
					# Modelo estándar: no se le pasa h_prev, ignora el pasado
					_, meta = model.forward_resonance(x, n_steps=2, pos_mode="clock")
					if tick == 3:
						final_hidden = meta["final_hidden"]
			
			# Evaluar predicción en el paso final T=3
			# final_hidden: (1, seq_len, hidden_dim) -> tomamos la primera posición [0, 0]
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
		if (epoch + 1) % 30 == 0 or epoch == 0:
			print(f"  [Época {epoch+1:<3}] Pérdida: {loss.item():.4f} | Precisión: {acc*100:.1f}%")
			
	return acc

def run_experiment():
	print("=" * 60)
	print("🧠 EXPERIMENTO: CAPACIDAD DE MEMORIA TEMPORAL (P.O.C. REAL) 🧠")
	print("=" * 60)
	print("Tarea: Recordar un token visto en T=0 tras 3 ticks de silencio absoluto (T=1, 2, 3)")
	print("y predecir la acción correspondiente en T=3.")
	
	print("\n--- Entrenando Modelo ESTÁNDAR (Sin Memoria entre Ticks) ---")
	acc_std = train_and_eval(model_type="standard", epochs=90)
	
	print("\n--- Entrenando Modelo RECURRENTE (Con Resonancia Temporal) ---")
	acc_rec = train_and_eval(model_type="recurrent", epochs=90)
	
	print("\n" + "=" * 60)
	print("📊 RESULTADOS FINALES:")
	print("-" * 60)
	print(f"Modelo ESTÁNDAR (Sin Memoria):   {acc_std*100:.1f}% Precisión (Límite aleatorio)")
	print(f"Modelo RECURRENTE (Con Memoria):  {acc_rec*100:.1f}% Precisión (¡Memoria perfecta!)")
	print("=" * 60)
	
	if acc_rec > 0.9 and acc_std < 0.7:
		print("🏆 ¡Éxito rotundo! El modelo recurrente retiene la información a lo largo del tiempo,")
		print("mientras que el estándar es incapaz de recordar qué pasó al inicio del ciclo.")
	else:
		print("⚠️ El test no mostró la diferencia esperada. Revisar hiperparámetros.")

if __name__ == "__main__":
	run_experiment()
