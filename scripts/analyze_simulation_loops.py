import os
import re
import sys


def analyze_log(file_path):
	if not os.path.exists(file_path):
		print(f"Error: File {file_path} does not exist.")
		sys.exit(1)

	print(f"📊 ANALIZANDO PATRONES ANÓMALOS EN: {file_path}\n" + "═"*60)

	# Regexes
	tick_re = re.compile(r"\[Tick (\d+)\]")
	state_re = re.compile(r"\s+([A-Za-z]+)\s*\(([A-Z])\):\s*Loc:\s*(\w+)")
	action_re = re.compile(r"\s+👉\s+([A-Za-z]+)\s*\(([A-Z])\)\s+decide\s+(\w+):?\s*(.*)")
	share_re = re.compile(r"\s+🎁\s+([A-Za-z]+)\s+comparte\s+(\w+)\s+con\s+([A-Za-z]+)")

	current_tick = 0
	agent_locations = {}  # agent -> loc
	actions_history = []  # list of dict: {tick, agent, action, loc, detail}
	shares_history = []   # list of dict: {tick, sender, resource, receiver}

	with open(file_path, encoding="utf-8") as f:
		for line in f:
			# Detect tick
			m_tick = tick_re.search(line)
			if m_tick:
				current_tick = int(m_tick.group(1))
				continue

			# Detect agent location from state line
			m_state = state_re.match(line)
			if m_state:
				name, letter, loc = m_state.groups()
				agent_locations[letter.lower()] = loc
				continue

			# Detect decided action
			m_action = action_re.match(line)
			if m_action:
				name, letter, action, detail = m_action.groups()
				loc = agent_locations.get(letter.lower(), "desconocida")
				actions_history.append({
					"tick": current_tick,
					"agent": letter.upper(),
					"name": name,
					"action": action.upper(),
					"loc": loc,
					"detail": detail
				})
				continue

			# Detect sharing
			m_share = share_re.search(line)
			if m_share:
				sender, resource, receiver = m_share.groups()
				shares_history.append({
					"tick": current_tick,
					"sender": sender,
					"resource": resource,
					"receiver": receiver
				})
				continue

	# 1. Analizar loops de acciones consecutivas
	print("\n🔄 LOOPS DE ACCIONES CONSECUTIVAS EN LA MISMA LOCALIZACIÓN:")
	consecutive_counts = {}  # agent -> (action, loc, count, start_tick)
	loops_detected = 0

	for act in actions_history:
		agent = act["agent"]
		action = act["action"]
		loc = act["loc"]
		tick = act["tick"]

		if agent not in consecutive_counts:
			consecutive_counts[agent] = (action, loc, 1, tick)
		else:
			last_action, last_loc, count, start_tick = consecutive_counts[agent]
			if last_action == action and last_loc == loc:
				consecutive_counts[agent] = (action, loc, count + 1, start_tick)
			else:
				if count >= 4:
					print(f"  ⚠️  {act['name']} ({agent}) repitió {last_action} en {last_loc} durante {count} ticks (Ticks {start_tick}-{tick-1})")
					loops_detected += 1
				consecutive_counts[agent] = (action, loc, 1, tick)

	# Flush last entries
	for agent, (last_action, last_loc, count, start_tick) in consecutive_counts.items():
		if count >= 4:
			print(f"  ⚠️  Agente {agent} repitió {last_action} en {last_loc} durante {count} ticks (Ticks {start_tick}-{actions_history[-1]['tick']})")
			loops_detected += 1

	if loops_detected == 0:
		print("  ✅ No se han detectado loops estáticos repetitivos significativos.")

	# 2. Analizar loops de intercambio (Trading loops)
	print("\n🎁 LOOPS DE TRADING Y TRANSACCIONES INNECESARIAS:")
	trading_loops = 0
	window = 5  # ticks de ventana para detectar ida y vuelta

	def normalize_name(name):
		name = name.lower()
		if name in ["nico", "a"]:
			return "a"
		if name in ["sofy", "sofi", "b"]:
			return "b"
		if name in ["hugo", "c"]:
			return "c"
		if name in ["domi", "d"]:
			return "d"
		return name

	for i, share in enumerate(shares_history):
		s1, r1, recv1 = share["sender"], share["resource"], share["receiver"]
		t1 = share["tick"]

		# Buscar si el receptor le devuelve algo en los siguientes ticks
		for j in range(i + 1, len(shares_history)):
			share2 = shares_history[j]
			if share2["tick"] - t1 > window:
				break

			s2, r2, recv2 = share2["sender"], share2["resource"], share2["receiver"]
			s1_norm = normalize_name(s1)
			recv1_norm = normalize_name(recv1)
			s2_norm = normalize_name(s2)
			recv2_norm = normalize_name(recv2)
			if s2_norm == recv1_norm and recv2_norm == s1_norm:
				print(f"  ⚠️  Bucle de Trading: {s1} le da {r1} a {recv1} en Tick {t1} y {s2} le devuelve {r2} en Tick {share2['tick']}")
				trading_loops += 1

	if trading_loops == 0:
		print("  ✅ No se han detectado bucles de intercambio (trading).")

	# 3. Analizar estancamiento espacial
	print("\n📌 TIEMPO DE ESTANCAMIENTO POR LOCALIZACIÓN (Ticks continuos en el mismo sitio):")
	stagnant_locations = {} # agent -> (loc, count, start_tick)
	stagnancy_alerts = 0

	for act in actions_history:
		agent = act["agent"]
		loc = act["loc"]
		tick = act["tick"]

		if agent not in stagnant_locations:
			stagnant_locations[agent] = (loc, 1, tick)
		else:
			last_loc, count, start_tick = stagnant_locations[agent]
			if last_loc == loc:
				stagnant_locations[agent] = (loc, count + 1, start_tick)
			else:
				if count >= 15:
					print(f"  ⚠️  {act['name']} ({agent}) se quedó estancado en {last_loc} durante {count} ticks (Ticks {start_tick}-{tick-1})")
					stagnancy_alerts += 1
				stagnant_locations[agent] = (loc, 1, tick)

	# Flush
	for agent, (last_loc, count, start_tick) in stagnant_locations.items():
		if count >= 15:
			print(f"  ⚠️  Agente {agent} se quedó estancado en {last_loc} durante {count} ticks (Ticks {start_tick}-{actions_history[-1]['tick']})")
			stagnancy_alerts += 1

	if stagnancy_alerts == 0:
		print("  ✅ Movilidad correcta. Ningún agente se estancó en una localización más de 15 ticks.")

	print("\n" + "═"*60)

if __name__ == "__main__":
	target = "scratch/simulation_output_200.txt"
	if len(sys.argv) > 1:
		target = sys.argv[1]
	analyze_log(target)
