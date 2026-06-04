import sys
import os
import json
import random
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bitnet.cooperative_world import CooperativeWorld, COOP_ACTIONS

def run_debug():
	with open("configs/experiments/EXP_073_easy_train.json") as f:
		config = json.load(f)

	world_cfg = config.get("world", {})
	world = CooperativeWorld(
		seed=42,
		food_interval=world_cfg.get("food_interval", 12),
		food_duration=world_cfg.get("food_duration", 8),
		hunger_rate=world_cfg.get("hunger_rate", 1.5),
		resource_capacity=world_cfg.get("resource_capacity", 10.0),
		resource_recovery=world_cfg.get("resource_recovery", 0.5),
		predator_chance_multiplier=world_cfg.get("predator_chance_multiplier", 0.1),
		predator_damage_multiplier=world_cfg.get("predator_damage_multiplier", 0.1),
		storm_chance_multiplier=world_cfg.get("storm_chance_multiplier", 0.1),
		storm_damage_multiplier=world_cfg.get("storm_damage_multiplier", 0.1),
		prey_spawn_interval=world_cfg.get("prey_spawn_interval", 15)
	)

	state_a, state_b, state_c = world.reset(seed=42)

	consecutive_healthy_ticks = 0
	success_mastery = False

	ko_count_a = 0
	ko_count_b = 0
	ko_count_c = 0

	for tick in range(1, 501):
		# Let's choose random valid actions or just random actions
		mask_a = world.get_valid_actions_mask(state_a)
		mask_b = world.get_valid_actions_mask(state_b)
		mask_c = world.get_valid_actions_mask(state_c)

		allowed_a = [i for i, m in enumerate(mask_a) if m > 0]
		allowed_b = [i for i, m in enumerate(mask_b) if m > 0]
		allowed_c = [i for i, m in enumerate(mask_c) if m > 0]

		idx_a = random.choice(allowed_a)
		idx_b = random.choice(allowed_b)
		idx_c = random.choice(allowed_c)

		action_a = COOP_ACTIONS[idx_a]
		action_b = COOP_ACTIONS[idx_b]
		action_c = COOP_ACTIONS[idx_c]

		# Track if anyone was already KO or fell KO this tick
		ko_before_a = getattr(state_a, 'debuff_inconsciente_ticks', 0) > 0
		ko_before_b = getattr(state_b, 'debuff_inconsciente_ticks', 0) > 0
		ko_before_c = getattr(state_c, 'debuff_inconsciente_ticks', 0) > 0

		res_a, res_b, res_c, info = world.step(action_a, action_b, action_c)

		if getattr(state_a, 'debuff_inconsciente_ticks', 0) > 0 and not ko_before_a:
			ko_count_a += 1
			print(f"[KO EVENT] Agent A (Nico) went KO at tick {tick}! Vitals: H: {state_a.hambre:.1f}, S: {state_a.sed:.1f}")
		if getattr(state_b, 'debuff_inconsciente_ticks', 0) > 0 and not ko_before_b:
			ko_count_b += 1
			print(f"[KO EVENT] Agent B (Sofy) went KO at tick {tick}! Vitals: H: {state_b.hambre:.1f}, S: {state_b.sed:.1f}")
		if getattr(state_c, 'debuff_inconsciente_ticks', 0) > 0 and not ko_before_c:
			ko_count_c += 1
			print(f"[KO EVENT] Agent C (Hugo) went KO at tick {tick}! Vitals: H: {state_c.hambre:.1f}, S: {state_c.sed:.1f}")

		is_any_ko = (
			getattr(state_a, "debuff_inconsciente_ticks", 0) > 0 or
			getattr(state_b, "debuff_inconsciente_ticks", 0) > 0 or
			getattr(state_c, "debuff_inconsciente_ticks", 0) > 0
		)

		if is_any_ko:
			consecutive_healthy_ticks = 0
		else:
			consecutive_healthy_ticks += 1
			if consecutive_healthy_ticks >= 100:
				success_mastery = True

	print(f"\nSimulation finished.")
	print(f"Total KOs: A (Nico)={ko_count_a}, B (Sofy)={ko_count_b}, C (Hugo)={ko_count_c}")
	print(f"Final skills: A: {state_a.learned_skills} | B: {state_b.learned_skills} | C: {state_c.learned_skills}")
	print(f"Mastery success: {success_mastery}")


if __name__ == "__main__":
	run_debug()
