import random

import torch
import torch.nn.functional as F

from src.bitnet.worlds.cooperative_world import COOP_ACTIONS, COOP_LOCATION_GLYPHS
from src.bitnet.vocab.glyph_vocabulary import EMOTION_INDEX, WORD_INDEX


def generate_dojo_batch(batch_size=1024, device="cpu"):
	"""
	Genera un lote balanceado de escenarios de entrenamiento supervisado (Dojo).
	Devuelve tensores de inputs, emociones, acciones objetivo y conceptos de gritos objetivo,
	junto con una lista de IDs de agente asociados a cada muestra.
	"""
	inputs = []
	emotions = []
	target_actions = []
	target_shouts = []
	agent_ids = []

	locations = ["cueva", "lago", "pradera", "bosque", "montaña", "valle", "río", "pantano", "ruinas"]
	location_glyphs = {loc: WORD_INDEX[COOP_LOCATION_GLYPHS[loc]] for loc in locations}
	
	action_to_idx = {act: i for i, act in enumerate(COOP_ACTIONS)}

	scenarios = [
		"eat_ground", "drink_ground", "eat_backpack", "drink_backpack",
		"fight_predator", "run_predator", "sleep_shelter",
		"shout_hunger", "shout_thirst", "give_food", "give_water",
		"move_to_signal", "explore", "reanimate_companion",
		"explore_hungry", "explore_thirsty", "consume_mild_food", "consume_mild_water",
		"observe_fire", "attempt_fire", "attempt_craft"
	]

	for _ in range(batch_size):
		scen_type = random.choice(scenarios)
		
		# Agente aleatorio (Nico=a, Sofy=b, Hugo=c)
		agent_id = random.choice(["a", "b", "c"])
		agent_ids.append(agent_id)
		
		loc = random.choice(locations)
		what = "noche"
		emotion = "alegría"
		backpack = "noche"
		sig_loc = "noche"
		sig_what = "noche"
		
		action = "mover"
		shout_concept = "noche"
		
		if scen_type == "eat_ground":
			loc = random.choice(["bosque", "río", "pantano", "valle"])
			what = "comida"
			emotion = random.choice(["hambre", "ira"])
			backpack = random.choice(["noche", "comida"])
			action = "comer"
			
		elif scen_type == "drink_ground":
			loc = random.choice(["lago", "río", "pantano"])
			what = "agua"
			emotion = random.choice(["dolor", "ira", "alegría"])
			backpack = "noche"
			# Nico y Hugo saben extraer agua del suelo, Sofy no
			action = "beber" if agent_id in ["a", "c"] else "mover"
				
		elif scen_type == "eat_backpack":
			loc = random.choice(["cueva", "montaña", "lago"])
			what = "noche"
			emotion = random.choice(["hambre", "ira", "alegría"])
			backpack = "comida"
			action = "comer"
			
		elif scen_type == "drink_backpack":
			loc = random.choice(["cueva", "pradera", "bosque"])
			what = "noche"
			emotion = random.choice(["dolor", "ira", "alegría"])
			backpack = "agua"
			action = "beber"
			
		elif scen_type == "fight_predator":
			loc = random.choice(locations)
			what = "depredador"
			emotion = "miedo"
			backpack = "seguro"  # tiene lanza
			action = "luchar"
			
		elif scen_type == "run_predator":
			loc = random.choice(locations)
			what = "depredador"
			emotion = "miedo"
			backpack = "noche"
			action = "mover"  # huir
			
		elif scen_type == "sleep_shelter":
			loc = random.choice(["cueva", "ruinas"])
			what = "seguro"
			emotion = "tristeza"  # energía baja
			action = "dormir"
			
		elif scen_type == "shout_hunger":
			loc = random.choice(["montaña", "cueva", "lago"])
			what = "noche"
			emotion = "hambre"
			backpack = "noche"
			action = "gritar"
			shout_concept = "comida"
			
		elif scen_type == "shout_thirst":
			loc = random.choice(["cueva", "bosque", "montaña"])
			what = "noche"
			emotion = "dolor"
			backpack = "noche"
			action = "gritar"
			shout_concept = "agua"
			
		elif scen_type == "give_food":
			loc = "cueva"
			what = "grupo"
			emotion = "alegría"
			backpack = "comida"
			sig_loc = "cueva"
			sig_what = "hambre"
			action = "dar"
			
		elif scen_type == "give_water":
			loc = "cueva"
			what = "grupo"
			emotion = "alegría"
			backpack = "agua"
			sig_loc = "cueva"
			sig_what = "hambre"
			action = "dar"
			
		elif scen_type == "move_to_signal":
			loc = "cueva"
			what = "noche"
			emotion = "alegría"
			backpack = random.choice(["comida", "agua"])
			sig_loc = random.choice(["bosque", "río", "pantano"])
			sig_what = "hambre"
			action = "mover"
			
		elif scen_type == "explore":
			loc = random.choice(locations)
			what = "noche"
			emotion = "alegría"
			backpack = "noche"
			action = "mover"

		elif scen_type == "explore_hungry" or scen_type == "explore_thirsty":
			loc = random.choice(locations)
			what = "noche"
			emotion = "ira"
			backpack = "noche"
			action = "mover"

		elif scen_type == "consume_mild_food":
			loc = random.choice(locations)
			in_backpack = random.random() < 0.5
			emotion = "ira"
			if in_backpack:
				what = "noche"
				backpack = "comida"
				action = "comer"
			else:
				loc = random.choice(["bosque", "río", "pantano", "valle"])
				what = "comida"
				backpack = "noche"
				action = "comer"

		elif scen_type == "consume_mild_water":
			loc = random.choice(locations)
			in_backpack = random.random() < 0.5
			emotion = "ira"
			if in_backpack:
				what = "noche"
				backpack = "agua"
				action = "beber"
			else:
				loc = random.choice(["lago", "río", "pantano"])
				what = "agua"
				backpack = "noche"
				action = "beber" if agent_id in ["a", "c"] else "mover"
			
		elif scen_type == "reanimate_companion":
			loc = "cueva"
			what = "grupo"
			emotion = "alegría"
			backpack = "noche"
			sig_loc = "cueva"
			sig_what = "dolor"
			action = "reanimar"

		elif scen_type == "observe_fire":
			loc = "cueva"
			what = "seguro"
			emotion = "alegría"
			backpack = "noche"
			action = "ver"

		elif scen_type == "attempt_fire":
			loc = "cueva"
			what = "noche"
			emotion = "alegría"
			backpack = "noche"
			action = "encender"

		elif scen_type == "attempt_craft":
			loc = "bosque"
			what = "noche"
			emotion = "alegría"
			backpack = "noche"
			action = "fabricar"

		# Convertir a tokens del vocabulario
		l_t = location_glyphs[loc]
		w_t = WORD_INDEX.get(what, 0)
		b_t = WORD_INDEX.get(emotion, 0)
		d_t = WORD_INDEX.get(backpack, 0)
		s_l_t = location_glyphs[sig_loc] if sig_loc in location_glyphs else WORD_INDEX.get("noche", 0)
		s_w_t = WORD_INDEX.get(sig_what, 0)
		
		inputs.append([l_t, w_t, b_t, d_t, s_l_t, s_w_t])
		emotions.append(EMOTION_INDEX[emotion])
		target_actions.append(action_to_idx[action])
		target_shouts.append(WORD_INDEX.get(shout_concept, 0))

	inputs_tensor = torch.tensor(inputs, dtype=torch.long, device=device)
	emotions_tensor = torch.tensor(emotions, dtype=torch.long, device=device)
	target_actions_tensor = torch.tensor(target_actions, dtype=torch.long, device=device)
	target_shouts_tensor = torch.tensor(target_shouts, dtype=torch.long, device=device)

	return inputs_tensor, emotions_tensor, target_actions_tensor, target_shouts_tensor, agent_ids

def train_dojo_step(agents_dict, device, batch_size=256, lr=1e-4, epochs=5, n_think=2):
	"""
	Realiza un paso de entrenamiento en el Dojo de PopuLoRA sobre un lote sintético.
	Enseña el 'manual de instrucciones' a las cabezas de política de cada agente.
	"""
	inputs, emotions, target_actions, target_shouts, agent_ids = generate_dojo_batch(batch_size * epochs, device)
	
	losses = {}
	for label, model in agents_dict.items():
		if model is None:
			continue
		
		# Filtrar muestras específicas o generales para el agente
		# Para robustez, cada uno entrena con las muestras que le corresponden a su ID
		mask = [aid == label for aid in agent_ids]
		if not any(mask):
			continue
			
		idx_tensor = torch.tensor([i for i, val in enumerate(mask) if val], dtype=torch.long, device=device)
		agent_inputs = inputs[idx_tensor]
		agent_emotions = emotions[idx_tensor]
		agent_actions = target_actions[idx_tensor]
		agent_shouts = target_shouts[idx_tensor]
		
		# Optimizador ad-hoc para el dojo
		optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
		model.train()
		
		agent_loss_sum = 0.0
		steps = 0
		
		# Entrenar en mini-lotes
		for _e in range(epochs):
			for i in range(0, len(agent_inputs), batch_size):
				batch_in = agent_inputs[i:i+batch_size]
				batch_emo = agent_emotions[i:i+batch_size]
				batch_act = agent_actions[i:i+batch_size]
				batch_shout = agent_shouts[i:i+batch_size]
				
				if len(batch_in) == 0:
					continue
					
				optimizer.zero_grad()
				
				_, meta = model.forward_resonance(
					batch_in, n_steps=n_think, pos_mode="clock", emotion_ids=batch_emo
				)
				hidden = meta["hidden"][:, 2, :]
				
				action_logits = model.action_head(hidden)
				loss_action = F.cross_entropy(action_logits, batch_act)
				
				is_shouting = (batch_act == 6)
				if is_shouting.sum() > 0:
					shout_logits = model._decode_hidden(hidden[is_shouting])
					loss_shout = F.cross_entropy(shout_logits, batch_shout[is_shouting])
					loss = loss_action + 0.5 * loss_shout
				else:
					loss = loss_action
					
				loss.backward()
				torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
				optimizer.step()
				
				agent_loss_sum += loss.item()
				steps += 1
				
		losses[label] = (agent_loss_sum / max(1, steps)) if steps > 0 else 0.0
		
	return losses


def generate_sequential_dojo_batch(batch_size=1024, seq_len=4, device="cpu"):
	"""
	Genera un lote de secuencias temporales (trayectorias) para entrenamiento BPTT.
	Devuelve tensores de inputs (B, seq_len, 6), emociones (B, seq_len), 
	acciones objetivo (B, seq_len) y gritos objetivo (B, seq_len),
	junto con una lista de IDs de agente asociados a cada muestra.
	"""
	inputs = []
	emotions = []
	target_actions = []
	target_shouts = []
	agent_ids = []

	locations = ["cueva", "lago", "pradera", "bosque", "montaña", "valle", "río", "pantano", "ruinas"]
	location_glyphs = {loc: WORD_INDEX[COOP_LOCATION_GLYPHS[loc]] for loc in locations}
	action_to_idx = {act: i for i, act in enumerate(COOP_ACTIONS)}

	scenarios = ["nav_water_río", "nav_water_lago", "nav_food_bosque", "nav_food_valle", "give_water", "give_food"]

	for _ in range(batch_size):
		scen_type = random.choice(scenarios)
		agent_id = random.choice(["a", "b", "c"])
		agent_ids.append(agent_id)

		seq_in = []
		seq_emo = []
		seq_act = []
		seq_sh = []

		if scen_type == "nav_water_río":
			# cueva -> valle -> río -> beber/dormir
			steps = [
				{"loc": "cueva", "what": "noche", "emo": "dolor", "bp": "noche", "s_l": "río", "s_w": "agua", "act": "mover", "sh": "noche"},
				{"loc": "valle", "what": "noche", "emo": "dolor", "bp": "noche", "s_l": "noche", "s_w": "noche", "act": "mover", "sh": "noche"},
				{"loc": "río", "what": "agua", "emo": "dolor", "bp": "noche", "s_l": "noche", "s_w": "noche", "act": "beber", "sh": "noche"},
				{"loc": "río", "what": "agua", "emo": "alegría", "bp": "noche", "s_l": "noche", "s_w": "noche", "act": "dormir", "sh": "noche"}
			]
		elif scen_type == "nav_water_lago":
			# cueva -> pradera -> lago -> beber/dormir
			steps = [
				{"loc": "cueva", "what": "noche", "emo": "dolor", "bp": "noche", "s_l": "lago", "s_w": "agua", "act": "mover", "sh": "noche"},
				{"loc": "pradera", "what": "noche", "emo": "dolor", "bp": "noche", "s_l": "noche", "s_w": "noche", "act": "mover", "sh": "noche"},
				{"loc": "lago", "what": "agua", "emo": "dolor", "bp": "noche", "s_l": "noche", "s_w": "noche", "act": "beber", "sh": "noche"},
				{"loc": "lago", "what": "agua", "emo": "alegría", "bp": "noche", "s_l": "noche", "s_w": "noche", "act": "dormir", "sh": "noche"}
			]
		elif scen_type == "nav_food_bosque":
			# cueva -> pradera -> bosque -> comer/dormir
			steps = [
				{"loc": "cueva", "what": "noche", "emo": "hambre", "bp": "noche", "s_l": "bosque", "s_w": "comida", "act": "mover", "sh": "noche"},
				{"loc": "pradera", "what": "noche", "emo": "hambre", "bp": "noche", "s_l": "noche", "s_w": "noche", "act": "mover", "sh": "noche"},
				{"loc": "bosque", "what": "comida", "emo": "hambre", "bp": "noche", "s_l": "noche", "s_w": "noche", "act": "comer", "sh": "noche"},
				{"loc": "bosque", "what": "comida", "emo": "alegría", "bp": "noche", "s_l": "noche", "s_w": "noche", "act": "dormir", "sh": "noche"}
			]
		elif scen_type == "nav_food_valle":
			# cueva -> pradera -> valle -> comer/dormir
			steps = [
				{"loc": "cueva", "what": "noche", "emo": "hambre", "bp": "noche", "s_l": "valle", "s_w": "comida", "act": "mover", "sh": "noche"},
				{"loc": "pradera", "what": "noche", "emo": "hambre", "bp": "noche", "s_l": "noche", "s_w": "noche", "act": "mover", "sh": "noche"},
				{"loc": "valle", "what": "comida", "emo": "hambre", "bp": "noche", "s_l": "noche", "s_w": "noche", "act": "comer", "sh": "noche"},
				{"loc": "valle", "what": "comida", "emo": "alegría", "bp": "noche", "s_l": "noche", "s_w": "noche", "act": "dormir", "sh": "noche"}
			]
		elif scen_type == "give_water":
			# cueva -> dar agua -> dormir -> dormir
			steps = [
				{"loc": "cueva", "what": "grupo", "emo": "alegría", "bp": "agua", "s_l": "cueva", "s_w": "dolor", "act": "dar", "sh": "noche"},
				{"loc": "cueva", "what": "grupo", "emo": "alegría", "bp": "noche", "s_l": "noche", "s_w": "noche", "act": "dormir", "sh": "noche"},
				{"loc": "cueva", "what": "grupo", "emo": "alegría", "bp": "noche", "s_l": "noche", "s_w": "noche", "act": "dormir", "sh": "noche"},
				{"loc": "cueva", "what": "grupo", "emo": "alegría", "bp": "noche", "s_l": "noche", "s_w": "noche", "act": "dormir", "sh": "noche"}
			]
		else:  # give_food
			# cueva -> dar comida -> dormir -> dormir
			steps = [
				{"loc": "cueva", "what": "grupo", "emo": "alegría", "bp": "comida", "s_l": "cueva", "s_w": "hambre", "act": "dar", "sh": "noche"},
				{"loc": "cueva", "what": "grupo", "emo": "alegría", "bp": "noche", "s_l": "noche", "s_w": "noche", "act": "dormir", "sh": "noche"},
				{"loc": "cueva", "what": "grupo", "emo": "alegría", "bp": "noche", "s_l": "noche", "s_w": "noche", "act": "dormir", "sh": "noche"},
				{"loc": "cueva", "what": "grupo", "emo": "alegría", "bp": "noche", "s_l": "noche", "s_w": "noche", "act": "dormir", "sh": "noche"}
			]

		# Ajustar longitud si seq_len != 4
		if len(steps) > seq_len:
			steps = steps[:seq_len]
		while len(steps) < seq_len:
			steps.append(steps[-1])

		for s in steps:
			l_t = location_glyphs[s["loc"]]
			w_t = WORD_INDEX.get(s["what"], 0)
			b_t = WORD_INDEX.get(s["emo"], 0)
			d_t = WORD_INDEX.get(s["bp"], 0)
			s_l_t = location_glyphs[s["s_l"]] if s["s_l"] in location_glyphs else WORD_INDEX.get("noche", 0)
			s_w_t = WORD_INDEX.get(s["s_w"], 0)

			seq_in.append([l_t, w_t, b_t, d_t, s_l_t, s_w_t])
			seq_emo.append(EMOTION_INDEX[s["emo"]])
			seq_act.append(action_to_idx[s["act"]])
			seq_sh.append(WORD_INDEX.get(s["sh"], 0))

		inputs.append(seq_in)
		emotions.append(seq_emo)
		target_actions.append(seq_act)
		target_shouts.append(seq_sh)

	inputs_tensor = torch.tensor(inputs, dtype=torch.long, device=device)
	emotions_tensor = torch.tensor(emotions, dtype=torch.long, device=device)
	target_actions_tensor = torch.tensor(target_actions, dtype=torch.long, device=device)
	target_shouts_tensor = torch.tensor(target_shouts, dtype=torch.long, device=device)

	return inputs_tensor, emotions_tensor, target_actions_tensor, target_shouts_tensor, agent_ids


def train_sequential_dojo_step(agents_dict, device, batch_size=64, seq_len=4, lr=1e-4, epochs=5, n_think=2):
	"""
	Realiza un paso de entrenamiento recurrente (BPTT) en el Dojo de PopuLoRA.
	Consolida el canal latente y previene la amnesia de gritos/navegación.
	"""
	inputs, emotions, target_actions, target_shouts, agent_ids = generate_sequential_dojo_batch(
		batch_size * epochs, seq_len=seq_len, device=device
	)
	
	losses = {}
	for label, model in agents_dict.items():
		if model is None:
			continue
		
		mask = [aid == label for aid in agent_ids]
		if not any(mask):
			continue
			
		idx_tensor = torch.tensor([i for i, val in enumerate(mask) if val], dtype=torch.long, device=device)
		agent_inputs = inputs[idx_tensor]          # (N, seq_len, 6)
		agent_emotions = emotions[idx_tensor]      # (N, seq_len)
		agent_actions = target_actions[idx_tensor]  # (N, seq_len)
		agent_shouts = target_shouts[idx_tensor]    # (N, seq_len)
		
		# Optimizador ad-hoc para el dojo
		optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
		model.train()
		
		agent_loss_sum = 0.0
		steps = 0
		
		for _e in range(epochs):
			for i in range(0, len(agent_inputs), batch_size):
				batch_in = agent_inputs[i:i+batch_size]      # (B, seq_len, 6)
				batch_emo = agent_emotions[i:i+batch_size]    # (B, seq_len)
				batch_act = agent_actions[i:i+batch_size]    # (B, seq_len)
				batch_shout = agent_shouts[i:i+batch_size]    # (B, seq_len)
				
				if len(batch_in) == 0:
					continue
					
				optimizer.zero_grad()
				
				# Inicializar h_prev
				hidden_dim = model.resonance_clock.shape[-1]
				h_state = torch.zeros((len(batch_in), 6, hidden_dim), device=device)
				
				loss_total = 0.0
				
				for t in range(seq_len):
					x_t = batch_in[:, t, :]  # (B, 6)
					emo_t = batch_emo[:, t]  # (B)
					act_t = batch_act[:, t]  # (B)
					shout_t = batch_shout[:, t]  # (B)
					
					_, meta = model.forward_resonance(
						x_t, n_steps=n_think, pos_mode="clock", emotion_ids=emo_t, h_prev=h_state
					)
					
					# Propagar el hidden state adjunto al grafo (BPTT)
					h_state = meta["hidden"]
					
					hidden_t = h_state[:, 2, :]
					action_logits = model.action_head(hidden_t)
					loss_action = F.cross_entropy(action_logits, act_t)
					
					is_shouting = (act_t == 6)
					if is_shouting.sum() > 0:
						shout_logits = model._decode_hidden(hidden_t[is_shouting])
						loss_shout = F.cross_entropy(shout_logits, shout_t[is_shouting])
						loss_total += loss_action + 0.5 * loss_shout
					else:
						loss_total += loss_action
						
				loss_total = loss_total / seq_len
				loss_total.backward()
				
				torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
				optimizer.step()
				
				agent_loss_sum += loss_total.item()
				steps += 1
				
		losses[label] = (agent_loss_sum / max(1, steps)) if steps > 0 else 0.0
		
	return losses
