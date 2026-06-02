"""
run_arena_simulation.py — EXP_073_v5: Arena 3-Agent Simulation with Samantha Translation.

Carga los modelos Nico, Sofi y Hugo (PPO), ejecuta la simulación en la Arena,
y traduce las comunicaciones a español usando Samantha en tiempo real.

Origen: Aleth & Joan, 2026-06-02
"""

import argparse
import json
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Append sharing src path for Samantha
sys.path.append('/home/joan/Documents/IA/sharing/src')
try:
	from red_pill.inference import samantha_on_demand
	samantha_available = True
except ImportError:
	samantha_available = False

from src.bitnet.cooperative_world import (
	COOP_ACTIONS,
	COOP_N_ACTIONS,
	SILENCE_GLYPH,
	COOP_LOCATION_GLYPHS,
	CooperativeWorld,
)
from src.bitnet.glyph_vocabulary import (
	N_EMOTIONS,
	WORD_INDEX,
	WORD_NAMES,
)
from src.bitnet.modeling_bitnet import BitNet4LayerModel


def perception_to_input_coop(perception: list[int], device: torch.device) -> torch.Tensor:
	indices = perception[:6]
	while len(indices) < 6:
		indices.append(SILENCE_GLYPH)
	return torch.tensor([indices], dtype=torch.long, device=device)


def translate_shout(agent_name: str, location_glyph: str, what_glyph: str) -> str:
	if not samantha_available:
		return f"[{location_glyph}, {what_glyph}]"

	prompt = f"{agent_name} grita: [{location_glyph}, {what_glyph}]"
	system_prompt = (
		"Eres Samantha, la traductora del meta-lenguaje de los agentes Bit a español natural. "
		"El meta-lenguaje consiste en gritos de supervivencia con el formato '[Localización, Objeto/Suceso]'. "
		"Traduce este grito a una frase muy corta, expresiva y natural en español, como la diría un niño de 5 a 8 años que intenta cooperar con su amigo para sobrevivir en un mundo hostil. "
		"Devuelve únicamente la traducción, sin explicaciones ni comillas."
	)
	try:
		translation = samantha_on_demand.invoke(prompt, system_prompt=system_prompt, max_tokens=100, temperature=0.7)
		if translation:
			return translation.strip()
	except Exception as e:
		pass
	return f"[{location_glyph}, {what_glyph}]"


def get_glyph_name(idx: int) -> str:
	if 0 <= idx < len(WORD_NAMES):
		return WORD_NAMES[idx]
	return "desconocido"


def load_agent(checkpoint_path: str, model_cfg: dict, emotion_cfg: dict, max_resonance_steps: int, device: torch.device) -> nn.Module:
	m = BitNet4LayerModel(
		use_glyphs=True,
		hidden_dim=model_cfg.get("hidden_dim", 256),
		num_layers=model_cfg.get("num_layers", 3),
		use_pos_embedding=model_cfg.get("use_pos_embedding", True),
		max_resonance_steps=max_resonance_steps,
		n_emotions=N_EMOTIONS,
		emotion_dim=emotion_cfg.get("dim", 64),
		emotion_mode=emotion_cfg.get("mode", "first_only"),
		max_seq_len=6,
	).to(device)

	if os.path.exists(checkpoint_path):
		sd = torch.load(checkpoint_path, map_location=device, weights_only=True)

		# Redimensionar action_head si el checkpoint tiene dimensiones diferentes (ancho oculto o número de acciones)
		if "action_head.0.weight" in sd and "action_head.2.weight" in sd:
			ckpt_action_width = sd["action_head.0.weight"].shape[0]
			ckpt_out_features = sd["action_head.2.weight"].shape[0]
			if ckpt_action_width != m.action_head[0].out_features or ckpt_out_features != m.action_head[2].out_features:
				m.action_head = nn.Sequential(
					nn.Linear(model_cfg.get("hidden_dim", 256), ckpt_action_width),
					nn.GELU(),
					nn.Linear(ckpt_action_width, ckpt_out_features),
				).to(device)

		# Redimensionar value_head si el checkpoint tiene dimensiones diferentes
		if "value_head.0.weight" in sd and "value_head.2.weight" in sd:
			ckpt_value_width = sd["value_head.0.weight"].shape[0]
			ckpt_val_out = sd["value_head.2.weight"].shape[0]
			if ckpt_value_width != m.value_head[0].out_features or ckpt_val_out != m.value_head[2].out_features:
				m.value_head = nn.Sequential(
					nn.Linear(model_cfg.get("hidden_dim", 256), ckpt_value_width),
					nn.GELU(),
					nn.Linear(ckpt_value_width, ckpt_val_out),
				).to(device)

		m.load_state_dict(sd, strict=False)
		print(f"Loaded weights from {checkpoint_path}")
	else:
		print(f"Warning: Checkpoint not found at {checkpoint_path}. Using random initialization.")

	# Asegurar que el action_head tenga salida de COOP_N_ACTIONS preservando pesos pre-entrenados
	if m.action_head[2].out_features != COOP_N_ACTIONS:
		old_first = m.action_head[0]
		old_gelu = m.action_head[1]
		old_linear = m.action_head[2]
		old_width = old_first.out_features
		old_out = old_linear.out_features
		new_linear = nn.Linear(old_width, COOP_N_ACTIONS)
		with torch.no_grad():
			copy_limit = min(old_out, COOP_N_ACTIONS)
			new_linear.weight[:copy_limit] = old_linear.weight[:copy_limit].clone()
			new_linear.bias[:copy_limit] = old_linear.bias[:copy_limit].clone()
			if COOP_N_ACTIONS > old_out:
				new_linear.weight[old_out:] = torch.randn(COOP_N_ACTIONS - old_out, old_width) * 0.01
				new_linear.bias[old_out:] = 0.0
		m.action_head = nn.Sequential(
			old_first,
			old_gelu,
			new_linear,
		).to(device)
		print(f"Resize action head (PRESERVING WEIGHTS): {old_out} -> {COOP_N_ACTIONS} actions (width={old_width})")

	return m


def run_simulation():
	parser = argparse.ArgumentParser(description="Run Cooperative Arena Simulation")
	parser.add_argument("--config", type=str, default="configs/experiments/EXP_050_arena_listener.json")
	parser.add_argument("--ticks", type=int, default=100)
	parser.add_argument("--greedy", action="store_true", help="Use argmax action selection instead of sampling")
	parser.add_argument("--seed", type=int, default=1337)
	args = parser.parse_args()

	base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	config_path = args.config if os.path.isabs(args.config) else os.path.join(base_dir, args.config)

	with open(config_path, encoding="utf-8") as f:
		config = json.load(f)

	experiment_id = config.get("experiment_id", "EXP_050_arena_listener")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model_cfg = config.get("model", {})
	emotion_cfg = config.get("emotion", {})
	max_res_steps = config.get("resonance", {}).get("max_resonance_steps", 5)
	n_think = config.get("resonance", {}).get("n_think", 2)

	# Ubicación de los checkpoints
	exp_dir = os.path.join(base_dir, "storage", "experiments", experiment_id)
	path_a = os.path.join(exp_dir, "best_agent_a.pt")
	path_b = os.path.join(exp_dir, "best_agent_b.pt")
	path_c = os.path.join(exp_dir, "best_agent_c.pt")

	# Hugo loads Nico's (A) weights if best_agent_c.pt doesn't exist
	if not os.path.exists(path_c) and os.path.exists(path_a):
		path_c = path_a

	# Cargar agentes
	print(f"Loading 3 agents for experiment '{experiment_id}'...")
	agent_a = load_agent(path_a, model_cfg, emotion_cfg, max_res_steps, device)
	agent_b = load_agent(path_b, model_cfg, emotion_cfg, max_res_steps, device)
	agent_c = load_agent(path_c, model_cfg, emotion_cfg, max_res_steps, device)

	agent_a.eval()
	agent_b.eval()
	agent_c.eval()

	# Configurar mundo
	world_cfg = config.get("world", {})
	food_interval = world_cfg.get("food_interval", 10)
	food_duration = world_cfg.get("food_duration", 5)
	hunger_rate = world_cfg.get("hunger_rate", 3.5)
	resource_capacity = world_cfg.get("resource_capacity", 5.0)
	resource_recovery = world_cfg.get("resource_recovery", 0.2)
	predator_chance_multiplier = world_cfg.get("predator_chance_multiplier", 1.0)
	predator_damage_multiplier = world_cfg.get("predator_damage_multiplier", 1.0)
	storm_chance_multiplier = world_cfg.get("storm_chance_multiplier", 1.0)
	storm_damage_multiplier = world_cfg.get("storm_damage_multiplier", 1.0)
	prey_spawn_interval = world_cfg.get("prey_spawn_interval", 15)

	world = CooperativeWorld(
		seed=args.seed,
		food_interval=food_interval,
		food_duration=food_duration,
		hunger_rate=hunger_rate,
		resource_capacity=resource_capacity,
		resource_recovery=resource_recovery,
		predator_chance_multiplier=predator_chance_multiplier,
		predator_damage_multiplier=predator_damage_multiplier,
		storm_chance_multiplier=storm_chance_multiplier,
		storm_damage_multiplier=storm_damage_multiplier,
		prey_spawn_interval=prey_spawn_interval,
	)

	state_a, state_b, state_c = world.reset()

	print("\n" + "═"*70)
	print("🏟️  SIMULACIÓN EN LA ARENA COOPERATIVA (3 AGENTES)")
	print(f"  Nico (Agente A) en {state_a.location}")
	print(f"  Sofi (Agente B) en {state_b.location}")
	print(f"  Hugo (Agente C) en {state_c.location}")
	print(f"  Configuración: {args.config}")
	print(f"  Metabolismo (Hambre): {hunger_rate:.2f}/tick")
	print(f"  Capacidad de recursos: {resource_capacity:.1f} | Recuperación: +{resource_recovery:.2f}/tick")
	print(f"  Multiplicadores Depredador: chance={predator_chance_multiplier:.2f}, daño={predator_damage_multiplier:.2f}")
	print(f"  Multiplicadores Clima: chance={storm_chance_multiplier:.2f}, daño={storm_damage_multiplier:.2f}")
	print(f"  Frecuencia Presa: cada {prey_spawn_interval} ticks")
	print(f"  Samantha disponible: {samantha_available}")
	print("═"*70)

	for tick in range(1, args.ticks + 1):
		if not state_a.alive or not state_b.alive or not state_c.alive:
			print(f"\n☠️  La simulación ha terminado en el tick {tick} porque uno de los agentes ha muerto.")
			break

		print(f"\n[Tick {tick:03d}] " + "─"*55)
		# Mostrar estado del mundo y capacidades de recursos y presa
		prey_str = f"🎯 Presa en {world.prey_location} ({world.prey_timer}t)" if world.prey_location else "🎯 Sin presa"
		print(f"  Mundo: Bosque: [🍖: {world.resource_capacities['bosque']['food']:.1f}/{world.resource_capacity:.1f}] | Río: [🍖: {world.resource_capacities['río']['food']:.1f}/{world.resource_capacity:.1f}, 💧: {world.resource_capacities['río']['water']:.1f}/{world.resource_capacity:.1f}] | Lago: [💧: {world.resource_capacities['lago']['water']:.1f}/{world.resource_capacity:.1f}] | {prey_str}")

		# Mostrar estado de Nico (A) y su ToM
		print(f"  Nico (A): Loc: {state_a.location:8s} | Hambre: {state_a.hambre:5.1f} | Sed: {state_a.sed:5.1f} | Salud: {state_a.salud:5.1f} | Energía: {state_a.energia:5.1f} | Emo: {state_a.emotion_name} | Mochila: [🍖: {state_a.mochila_comida}/1, 💧: {state_a.mochila_agua}/1]")
		print(f"            Estima a Sofi (B): [Loc: {state_a.companion_model.get('b', {}).get('location', 'cueva'):8s} | Hambre: {state_a.companion_model.get('b', {}).get('hambre', 60.0):5.1f} | Sed: {state_a.companion_model.get('b', {}).get('sed', 80.0):5.1f} | Salud: {state_a.companion_model.get('b', {}).get('salud', 100.0):5.1f} | Emo: {state_a.companion_model.get('b', {}).get('emotion_name', 'alegría')}]")
		print(f"            Estima a Hugo (c): [Loc: {state_a.companion_model.get('c', {}).get('location', 'cueva'):8s} | Hambre: {state_a.companion_model.get('c', {}).get('hambre', 60.0):5.1f} | Sed: {state_a.companion_model.get('c', {}).get('sed', 80.0):5.1f} | Salud: {state_a.companion_model.get('c', {}).get('salud', 100.0):5.1f} | Emo: {state_a.companion_model.get('c', {}).get('emotion_name', 'alegría')}]")

		# Mostrar estado de Sofi (B) y su ToM
		print(f"  Sofi (B): Loc: {state_b.location:8s} | Hambre: {state_b.hambre:5.1f} | Sed: {state_b.sed:5.1f} | Salud: {state_b.salud:5.1f} | Energía: {state_b.energia:5.1f} | Emo: {state_b.emotion_name} | Mochila: [🍖: {state_b.mochila_comida}/1, 💧: {state_b.mochila_agua}/1]")
		print(f"            Estima a Nico (A): [Loc: {state_b.companion_model.get('a', {}).get('location', 'cueva'):8s} | Hambre: {state_b.companion_model.get('a', {}).get('hambre', 60.0):5.1f} | Sed: {state_b.companion_model.get('a', {}).get('sed', 80.0):5.1f} | Salud: {state_b.companion_model.get('a', {}).get('salud', 100.0):5.1f} | Emo: {state_b.companion_model.get('a', {}).get('emotion_name', 'alegría')}]")
		print(f"            Estima a Hugo (C): [Loc: {state_b.companion_model.get('c', {}).get('location', 'cueva'):8s} | Hambre: {state_b.companion_model.get('c', {}).get('hambre', 60.0):5.1f} | Sed: {state_b.companion_model.get('c', {}).get('sed', 80.0):5.1f} | Salud: {state_b.companion_model.get('c', {}).get('salud', 100.0):5.1f} | Emo: {state_b.companion_model.get('c', {}).get('emotion_name', 'alegría')}]")

		# Mostrar estado de Hugo (C) y su ToM
		print(f"  Hugo (C): Loc: {state_c.location:8s} | Hambre: {state_c.hambre:5.1f} | Sed: {state_c.sed:5.1f} | Salud: {state_c.salud:5.1f} | Energía: {state_c.energia:5.1f} | Emo: {state_c.emotion_name} | Mochila: [🍖: {state_c.mochila_comida}/1, 💧: {state_c.mochila_agua}/1]")
		print(f"            Estima a Nico (A): [Loc: {state_c.companion_model.get('a', {}).get('location', 'cueva'):8s} | Hambre: {state_c.companion_model.get('a', {}).get('hambre', 60.0):5.1f} | Sed: {state_c.companion_model.get('a', {}).get('sed', 80.0):5.1f} | Salud: {state_c.companion_model.get('a', {}).get('salud', 100.0):5.1f} | Emo: {state_c.companion_model.get('a', {}).get('emotion_name', 'alegría')}]")
		print(f"            Estima a Sofi (B): [Loc: {state_c.companion_model.get('b', {}).get('location', 'cueva'):8s} | Hambre: {state_c.companion_model.get('b', {}).get('hambre', 60.0):5.1f} | Sed: {state_c.companion_model.get('b', {}).get('sed', 80.0):5.1f} | Salud: {state_c.companion_model.get('b', {}).get('salud', 100.0):5.1f} | Emo: {state_c.companion_model.get('b', {}).get('emotion_name', 'alegría')}]")

		# Perceive
		perc_a = world.perceive(state_a)
		perc_b = world.perceive(state_b)
		perc_c = world.perceive(state_c)

		# Mostrar percepción de los agentes
		print(f"  Percepción Nico: [{get_glyph_name(perc_a[0])}, {get_glyph_name(perc_a[1])}, {get_glyph_name(perc_a[2])}, {get_glyph_name(perc_a[3])}, RcvLoc: {get_glyph_name(perc_a[4])}, RcvWhat: {get_glyph_name(perc_a[5])}]")
		print(f"  Percepción Sofi: [{get_glyph_name(perc_b[0])}, {get_glyph_name(perc_b[1])}, {get_glyph_name(perc_b[2])}, {get_glyph_name(perc_b[3])}, RcvLoc: {get_glyph_name(perc_b[4])}, RcvWhat: {get_glyph_name(perc_b[5])}]")
		print(f"  Percepción Hugo: [{get_glyph_name(perc_c[0])}, {get_glyph_name(perc_c[1])}, {get_glyph_name(perc_c[2])}, {get_glyph_name(perc_c[3])}, RcvLoc: {get_glyph_name(perc_c[4])}, RcvWhat: {get_glyph_name(perc_c[5])}]")

		input_a = perception_to_input_coop(perc_a, device)
		input_b = perception_to_input_coop(perc_b, device)
		input_c = perception_to_input_coop(perc_c, device)

		emo_a = torch.tensor([state_a.emotion_id], device=device)
		emo_b = torch.tensor([state_b.emotion_id], device=device)
		emo_c = torch.tensor([state_c.emotion_id], device=device)

		# Think
		with torch.no_grad():
			_, meta_a = agent_a.forward_resonance(
				input_a, n_steps=n_think, pos_mode="clock", emotion_ids=emo_a
			)
			_, meta_b = agent_b.forward_resonance(
				input_b, n_steps=n_think, pos_mode="clock", emotion_ids=emo_b
			)
			_, meta_c = agent_c.forward_resonance(
				input_c, n_steps=n_think, pos_mode="clock", emotion_ids=emo_c
			)

			hidden_a = meta_a["hidden"][:, 2, :].clone()
			hidden_b = meta_b["hidden"][:, 2, :].clone()
			hidden_c = meta_c["hidden"][:, 2, :].clone()

			action_logits_a = agent_a.action_head(hidden_a)
			action_logits_b = agent_b.action_head(hidden_b)
			action_logits_c = agent_c.action_head(hidden_c)

			shout_logits_a = agent_a._decode_hidden(hidden_a)
			shout_logits_b = agent_b._decode_hidden(hidden_b)
			shout_logits_c = agent_c._decode_hidden(hidden_c)

		# Elegir acción y concepto a gritar
		if args.greedy:
			idx_a = torch.argmax(action_logits_a, dim=-1).item()
			shout_concept_a_val = torch.argmax(shout_logits_a, dim=-1).item()
			idx_b = torch.argmax(action_logits_b, dim=-1).item()
			shout_concept_b_val = torch.argmax(shout_logits_b, dim=-1).item()
			idx_c = torch.argmax(action_logits_c, dim=-1).item()
			shout_concept_c_val = torch.argmax(shout_logits_c, dim=-1).item()
		else:
			probs_a = F.softmax(action_logits_a, dim=-1)
			shout_probs_a = F.softmax(shout_logits_a, dim=-1)
			dist_a = torch.distributions.Categorical(probs_a)
			dist_shout_a = torch.distributions.Categorical(shout_probs_a)
			idx_a = dist_a.sample().item()
			shout_concept_a_val = dist_shout_a.sample().item()

			probs_b = F.softmax(action_logits_b, dim=-1)
			shout_probs_b = F.softmax(shout_logits_b, dim=-1)
			dist_b = torch.distributions.Categorical(probs_b)
			dist_shout_b = torch.distributions.Categorical(shout_probs_b)
			idx_b = dist_b.sample().item()
			shout_concept_b_val = dist_shout_b.sample().item()

			probs_c = F.softmax(action_logits_c, dim=-1)
			shout_probs_c = F.softmax(shout_logits_c, dim=-1)
			dist_c = torch.distributions.Categorical(probs_c)
			dist_shout_c = torch.distributions.Categorical(shout_probs_c)
			idx_c = dist_c.sample().item()
			shout_concept_c_val = dist_shout_c.sample().item()

		action_a = COOP_ACTIONS[idx_a]
		action_b = COOP_ACTIONS[idx_b]
		action_c = COOP_ACTIONS[idx_c]

		shout_concept_a = shout_concept_a_val if idx_a == 6 else None
		shout_concept_b = shout_concept_b_val if idx_b == 6 else None
		shout_concept_c = shout_concept_c_val if idx_c == 6 else None

		# Ejecutar paso
		res_a, res_b, res_c, world_info = world.step(
			action_a, action_b, action_c,
			shout_concept_a=shout_concept_a, shout_concept_b=shout_concept_b, shout_concept_c=shout_concept_c
		)

		# Imprimir acciones y sucesos (Nico)
		print(f"  👉 Nico (A) decide {action_a.upper()}: {res_a['event']}")
		if res_a.get("shouted"):
			loc_name = get_glyph_name(res_a["shout_content"][0])
			what_name = get_glyph_name(res_a["shout_content"][1])
			translation = translate_shout("Nico", loc_name, what_name)
			print(f"     📢 Nico grita: [{loc_name}, {what_name}]")
			print(f"     💬 Traducción Samantha: \"{translation}\"")
		if action_a == "dar" and res_a.get("success"):
			print(f"     🎁 Nico comparte {res_a['shared_resource']} con {res_a['shared_with'].upper()}!")

		# Imprimir acciones y sucesos (Sofi)
		print(f"  👉 Sofi (B) decide {action_b.upper()}: {res_b['event']}")
		if res_b.get("shouted"):
			loc_name = get_glyph_name(res_b["shout_content"][0])
			what_name = get_glyph_name(res_b["shout_content"][1])
			translation = translate_shout("Sofi", loc_name, what_name)
			print(f"     📢 Sofi grita: [{loc_name}, {what_name}]")
			print(f"     💬 Traducción Samantha: \"{translation}\"")
		if action_b == "dar" and res_b.get("success"):
			print(f"     🎁 Sofi comparte {res_b['shared_resource']} con {res_b['shared_with'].upper()}!")

		# Imprimir acciones y sucesos (Hugo)
		print(f"  👉 Hugo (C) decide {action_c.upper()}: {res_c['event']}")
		if res_c.get("shouted"):
			loc_name = get_glyph_name(res_c["shout_content"][0])
			what_name = get_glyph_name(res_c["shout_content"][1])
			translation = translate_shout("Hugo", loc_name, what_name)
			print(f"     📢 Hugo grita: [{loc_name}, {what_name}]")
			print(f"     💬 Traducción Samantha: \"{translation}\"")
		if action_c == "dar" and res_c.get("success"):
			print(f"     🎁 Hugo comparte {res_c['shared_resource']} con {res_c['shared_with'].upper()}!")

		# Bonos de cooperación
		if world_info.get("coop_bonus_a", 0.0) > 0:
			print(f"     🤝 Nico recibe coop_bonus: {world_info['coop_bonus_a']:.3f}")
		if world_info.get("coop_bonus_b", 0.0) > 0:
			print(f"     🤝 Sofi recibe coop_bonus: {world_info['coop_bonus_b']:.3f}")
		if world_info.get("coop_bonus_c", 0.0) > 0:
			print(f"     🤝 Hugo recibe coop_bonus: {world_info['coop_bonus_c']:.3f}")

	print("\n" + "═"*70)
	print("📊 RESULTADOS FINALES DE LA SIMULACIÓN")
	print(f"  Duración: {world.world_tick} ticks")
	print(f"  Nico vivo: {state_a.alive} (Salud: {state_a.salud:.1f})")
	print(f"  Sofi viva: {state_b.alive} (Salud: {state_b.salud:.1f})")
	print(f"  Hugo vivo: {state_c.alive} (Salud: {state_c.salud:.1f})")
	print("═"*70)


if __name__ == "__main__":
	run_simulation()
