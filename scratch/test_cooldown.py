import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.bitnet.cooperative_world import CooperativeWorld


def test_cooldown_and_penalty():
	print("--- Running Cooldown & Penalty Verification Test ---")
	
	# 1. Initialize world with 5 ticks replenishment cooldown
	world = CooperativeWorld(seed=42, replenish_cooldown_ticks=5, resource_capacity=10.0, resource_recovery=1.0)
	
	# Reset world
	world.reset()
	
	# Verify initial resources
	print(f"Initial río water capacity: {world.resource_capacities['río']['water']}")
	assert world.resource_capacities['río']['water'] == 10.0
	
	# 2. Trigger depletion cooldown by setting resource to 0
	# We manually simulate drinking or collecting until empty
	world.resource_capacities['río']['water'] = 0.0
	# Trigger the cooldown manually (simulating the depletion trigger in act)
	world.resource_cooldowns['río']['water'] = world.replenish_cooldown_ticks
	
	print(f"Depleted río water. Current cooldown: {world.resource_cooldowns['río']['water']}")
	assert world.resource_cooldowns['río']['water'] == 5
	
	# 3. Tick the world and check that it does not recover
	for t in range(1, 6):
		world._tick_world()
		current_val = world.resource_capacities['río']['water']
		current_cd = world.resource_cooldowns['río']['water']
		print(f"After tick {t}: río water = {current_val:.1f}, cooldown = {current_cd}")
		if t < 5:
			assert current_val == 0.0
			assert current_cd == 5 - t
		else:
			# Cooldown has run down to 0, so next tick it will start replenishing
			assert current_cd == 0
			assert current_val == 0.0  # remains 0 at the moment it reaches 0, but on next tick it will add recovery
			
	# One more tick: recovery should now apply
	world._tick_world()
	print(f"After tick 6 (cooldown ended): río water = {world.resource_capacities['río']['water']:.1f}")
	assert world.resource_capacities['río']['water'] == 1.0
	
	# 4. Verify the depletion penalty in act and get_reward
	# Reset world again
	world.reset()
	# Empty the río water
	world.resource_capacities['río']['water'] = 0.0
	world.resource_cooldowns['río']['water'] = world.replenish_cooldown_ticks
	
	# Force agent a to río
	agent = world.agent_a
	agent.location = 'río'
	agent.mochila_agua = 0  # no water in backpack
	agent.sed = 50.0
	agent.learned_skills = ['agua'] # knows water
	
	# Attempt to drink from exhausted río
	result = world.act(agent, "beber")
	print(f"Drink action success: {result['success']}")
	print(f"Drink action event: {result['event']}")
	assert not result['success']
	assert "agotado (penalización)" in result['event']
	
	# Compute reward and verify penalty
	reward = world.get_reward(agent, result)
	print(f"Reward after drinking from dry río: {reward}")
	# Standard base: 1.0 (alive) - 0.3 (failed action) - 2.0 (depletion penalty) = -1.3
	# Let's verify it is significantly negative and includes the -2.0 penalty
	assert reward < -1.0
	print("--- Verification Successful! ---")

if __name__ == "__main__":
	test_cooldown_and_penalty()
