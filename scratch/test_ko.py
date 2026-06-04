import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bitnet.cooperative_world import CooperativeWorld, COOP_ACTIONS

world = CooperativeWorld(seed=42)
state_a, state_b, state_c = world.reset(seed=42)

print("Starting simulation test...")
for tick in range(1, 150):
	# Choose random actions
	action_a = "ver"
	action_b = "ver"
	action_c = "ver"
	
	res_a, res_b, res_c, info = world.step(action_a, action_b, action_c)
	
	print(f"Tick {tick:03d} | A - H: {state_a.hambre:.1f}, S: {state_a.sed:.1f}, KO: {state_a.debuff_inconsciente_ticks} | B - H: {state_b.hambre:.1f}, S: {state_b.sed:.1f}, KO: {state_b.debuff_inconsciente_ticks} | C - H: {state_c.hambre:.1f}, S: {state_c.sed:.1f}, KO: {state_c.debuff_inconsciente_ticks}")
	
	if state_a.debuff_inconsciente_ticks > 0 or state_b.debuff_inconsciente_ticks > 0 or state_c.debuff_inconsciente_ticks > 0:
		print("  --> Alert! Someone went unconscious!")
