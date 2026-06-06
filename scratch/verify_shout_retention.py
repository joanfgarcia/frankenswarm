import os
import sys

import torch
import torch.nn as nn

# Añadir base dir al path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from src.bitnet.cooperative_world import COOP_ACTIONS, COOP_ADJACENCY, CooperativeWorld, _bfs_next_step
from src.bitnet.glyph_vocabulary import N_EMOTIONS, WORD_INDEX
from src.bitnet.modeling_bitnet import BitNet4LayerModel
from src.bitnet.train_arena_ppo import get_masked_probs


def load_agent_a(checkpoint_path, hidden_dim, device):
	model = BitNet4LayerModel(
		use_glyphs=True,
		hidden_dim=hidden_dim,
		num_layers=3,
		use_pos_embedding=True,
		max_resonance_steps=5,
		n_emotions=N_EMOTIONS,
		emotion_dim=64,
		emotion_mode="first_only",
		max_seq_len=6,
	).to(device)
	
	sd = torch.load(checkpoint_path, map_location=device, weights_only=True)
	
	# Ajustar action_head si es necesario
	if "action_head.0.weight" in sd and "action_head.2.weight" in sd:
		ckpt_action_width = sd["action_head.0.weight"].shape[0]
		ckpt_out_features = sd["action_head.2.weight"].shape[0]
		if ckpt_action_width != model.action_head[0].out_features or ckpt_out_features != model.action_head[2].out_features:
			model.action_head = nn.Sequential(
				nn.Linear(hidden_dim, ckpt_action_width),
				nn.GELU(),
				nn.Linear(ckpt_action_width, ckpt_out_features),
			).to(device)
			
	# Ajustar value_head si es necesario
	if "value_head.0.weight" in sd and "value_head.2.weight" in sd:
		ckpt_value_width = sd["value_head.0.weight"].shape[0]
		ckpt_val_out = sd["value_head.2.weight"].shape[0]
		if ckpt_value_width != model.value_head[0].out_features or ckpt_val_out != model.value_head[2].out_features:
			model.value_head = nn.Sequential(
				nn.Linear(hidden_dim, ckpt_value_width),
				nn.GELU(),
				nn.Linear(ckpt_value_width, ckpt_val_out),
			).to(device)

	model.load_state_dict(sd, strict=False)
	model.eval()
	return model

def run_test():
	print("=" * 80)
	print("🔬 COMPROBACIÓN CIENTÍFICA: MEMORIA DE GRITO Y NAVEGACIÓN EN LA ARENA 🔬")
	print("=" * 80)
	print("Escenario:")
	print("1. Nico (Agente A) empieza en 'cueva'. Su sed es baja (necesita agua).")
	print("2. Sofi (Agente B) está en 'río' (donde hay agua) y grita: [río, agua] en T=0.")
	print("3. Nico recibe el grito en T=0. Su destino de navegación ToM se fija en 'río'.")
	print("4. Para llegar a 'río', Nico debe hacer 2 movimientos: cueva -> valle -> río.")
	print("5. En T=1 y T=2, el grito ya no suena en el ambiente.")
	print("\nProbamos dos condiciones usando el mismo modelo entrenado (EXP_077):")
	print("Condición A: MEMORIA ACTIVA (h_prev se propaga normalmente).")
	print("Condición B: AMNESIA INDUCIDA (h_prev se inicializa a cero en cada tick).")
	print("-" * 80)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	checkpoint_path = os.path.join(base_dir, "storage", "experiments", "EXP_077_dojo_shout_levels", "best_agent_a.pt")
	
	if not os.path.exists(checkpoint_path):
		print(f"❌ Error: No se encuentra el checkpoint en {checkpoint_path}")
		return

	model = load_agent_a(checkpoint_path, hidden_dim=256, device=device)

	# --- FUNCIÓN AUXILIAR DE SIMULACIÓN ---
	def simulate_nico(amnesia=False):
		# Re-inicializar mundo y Nico
		world = CooperativeWorld(seed=42)
		world.reset()
		
		nico = world.agent_a
		nico.location = "cueva"
		nico.sed = 40.0  # Sed moderada para incentivar MOVER en lugar de gritar
		nico.hambre = 90.0
		nico.salud = 100.0
		nico.energia = 100.0
		
		# Forzar que Sofi esté en río
		world.agent_b.location = "río"
		
		# Paso T=0: Sofi grita "agua en río"
		# Glifos correspondientes
		río_glyph_idx = WORD_INDEX["río"]
		agua_glyph_idx = WORD_INDEX["agua"]
		
		# Nico recibe la señal en T=0
		nico.received_signal = (río_glyph_idx, agua_glyph_idx)
		nico.signal_age = 0
		nico.signal_from = "b"
		nico.nav_target = "río"  # Su ToM le dice que vaya allí
		
		h_state = torch.zeros((1, 6, 256), device=device)
		path_taken = ["cueva"]
		
		print(f"\n🚀 Iniciando Simulación - Condición: {'AMNESIA (Sin h_prev)' if amnesia else 'MEMORIA ACTIVA (Con h_prev)'}")
		
		for tick in range(1, 4):
			# En T > 0, el grito original ya no está en el ambiente
			# Nico actualiza su percepción
			perc = world.perceive(nico)
			
			# Modificar percepción para simular que el grito ya pasó (limpiar señal si es amnesia o envejecida)
			# En el simulador real, received_signal envejece y decae
			if tick > 1:
				nico.received_signal = None  # Ya no se escucha en el aire
				
			indices = perc[:6]
			while len(indices) < 6:
				indices.append(0)
			x = torch.tensor([indices], dtype=torch.long, device=device)
			emo = torch.tensor([nico.emotion_id], device=device)
			
			if amnesia:
				# Condición B: Borramos la memoria latente anterior
				h_state = torch.zeros((1, 6, 256), device=device)
				
			with torch.no_grad():
				_, meta = model.forward_resonance(
					x, n_steps=2, pos_mode="clock", emotion_ids=emo, h_prev=h_state
				)
				h_state = meta["final_hidden"]
				hidden_layer = meta["hidden"][:, 2, :].clone()
				action_logits = model.action_head(hidden_layer)
				
			# Seleccionar acción greedy
			state_mask = world.get_valid_actions_mask(nico)
			probs = get_masked_probs(action_logits, episode=1, n_episodes=1, communicate=True, state_mask=state_mask)
			action_idx = torch.argmax(probs, dim=-1).item()
			action = COOP_ACTIONS[action_idx]
			
			# Ejecutar en el mundo (solo para actualizar la posición de Nico)
			# Si decide mover, el simulador lo mueve hacia su nav_target (si ToM está activo) o según su red
			# Para aislar el cerebro, realizamos la acción física manualmente si es MOVER
			if action == "mover":
				# BFS hacia el objetivo que la red cree conveniente
				# Si recuerda el nav_target ('río'), se moverá hacia allí
				if nico.nav_target:
					step_loc = world.prey_location if nico.nav_target == "presa" else nico.nav_target
					next_step = _bfs_next_step(nico.location, step_loc)
					if next_step:
						nico.location = next_step
				else:
					# Sin memoria de nav_target, se mueve al azar a un vecino
					neighbors = COOP_ADJACENCY[nico.location]
					nico.location = neighbors[0] # vecino por defecto
					
			path_taken.append(nico.location)
			print(f"  [Tick {tick}] Nico decide: {action.upper():8s} | Posición actual: {nico.location} | Target Recordado: {nico.nav_target}")
			
			# Si Nico llega a río, ha tenido éxito
			if nico.location == "río":
				print(f"  🎉 ¡NICO LLEGÓ AL RÍO EN EL TICK {tick}! Éxito en la navegación.")
				return True, path_taken
				
		print(f"  ❌ Nico no llegó al río. Ruta final: {' -> '.join(path_taken)}")
		return False, path_taken

	# Ejecutar ambas condiciones
	success_mem, path_mem = simulate_nico(amnesia=False)
	success_amn, path_amn = simulate_nico(amnesia=True)
	
	print("\n" + "=" * 80)
	print("📊 COMPARACIÓN DE RUTAS:")
	print("-" * 80)
	print(f"Memoria Activa:  {' -> '.join(path_mem)} ({'ÉXITO' if success_mem else 'FALLO'})")
	print(f"Amnesia Inducida: {' -> '.join(path_amn)} ({'ÉXITO' if success_amn else 'FALLO'})")
	print("=" * 80)

if __name__ == "__main__":
	run_test()
