# Walkthrough: Sietch Laboratory Command Center & Telemetry Integration

We have successfully expanded the laboratory UX, resolved the swarm chat translation dilemma, and completed the cooperative PPO training of the 3-agent tribe. In the latest phase, we implemented the cooperative hunting of the Gran Herbívoro (Stag Hunt) and surrounding ecological mechanics.

## 🎛️ Unified Control Center (`scripts/microscope_tui.py`)
We transformed the TUI [microscope_tui.py](file:///home/joan/Documents/IA/frankenswarm/scripts/microscope_tui.py) into a unified Command Center. 

New operational modules added:
* **Option 4: 🏟️ Simulador de la Arena**: Lets you select any preset ecological config (Easy, Medium, Hard, Hell), prompts for ticks and random seed, and launches [run_arena_simulation.py](file:///home/joan/Documents/IA/frankenswarm/src/bitnet/run_arena_simulation.py) interactively in the terminal with real-time translation by Samantha.
* **Option 5: 💬 Swarm Chat Launcher**: Prompts for temperature, turns, and rep-penalty, and runs [run_swarm_chat.py](file:///home/joan/Documents/IA/frankenswarm/src/bitnet/run_swarm_chat.py) in-place.

## 💬 Swarm Chat Parameterization & Translation
* We parameterised [run_swarm_chat.py](file:///home/joan/Documents/IA/frankenswarm/src/bitnet/run_swarm_chat.py) to accept Sampler Temperature (`--temp`) and Repetition Penalty (`--penalty`) parameters from the CLI.
* We shortened Nico and Sofi's identity stories to in-vocabulary tokens (`"yo tengo perro"`, `"yo tengo gato"`) in [configs/story_bit_a.txt](file:///home/joan/Documents/IA/frankenswarm/configs/story_bit_a.txt) and [configs/story_bit_b.txt](file:///home/joan/Documents/IA/frankenswarm/configs/story_bit_b.txt) to prevent OOD syntax confusion on the 1.58-bit model.
* We created a translation utility [translate_chat.py](file:///home/joan/Documents/IA/frankenswarm/scratch/translate_chat.py) to map their dialogue into natural Spanish childlike statements via Samantha (local LLM), yielding a funny conversation about survival, Alice, and candy.

## 📈 3-Agent PPO Cooperative Training & Policy Freeze Fix (`EXP_073_easy_train`)

* **Resolved the Policy Freeze**: We identified and fixed the root cause of the agents getting stuck in local minima (redundant sleeping/drinking loops and map blindness):
  * **Map Anchors**: Initialized `map_knowledge` with static resource nodes (e.g. Lago, Río, Bosque = 5.0 resources) to provide a non-zero Potential-Based Reward Shaping (PBRS) gradient from tick 0.
  * **Inaction Penalty**: Penalized standing still (inaction) when hunger/thirst < 50 by `-0.5` (unless successfully eating/drinking), forcing agents to explore.
  * **Useless Sleep Penalty**: Penalized sleeping when energy > 70 by `-1.0` (and disabled the `+0.2` progress reward), preventing energy-clamping exploits.
  * **Failed Action Penalty**: Penalized failed eating/drinking attempts (when resources are absent) by `-0.4` to discourage spamming.
  * **Progressive Action Masking**: Constrained action space to basic survival actions (indices 0 to 7) during the first 35% of training, enabling fabricar/encender at 35%, and fully unmasking at 60% (1200 episodes).
  * **Frozen Backbone Optimization**: Changed `"unfreeze_backbone"` to `false` and learning rate to `0.001` in [EXP_073_easy_train.json](file:///home/joan/Documents/IA/frankenswarm/configs/experiments/EXP_073_easy_train.json). This locked the representation layer, focusing training on the heads and speeding up convergence dramatically.

## 📊 Verification Results
* **Training Convergence**: The training loop successfully terminated early at **Episode 201** because the agents reached the mastery goal.
* **Mastery Criteria Met**: During a separate deterministic evaluation run (greedy actions, seed 999), the agents successfully survived for **150+ consecutive ticks without a single K.O. event**.
* **Detailed Verification Run**: Running the trained models with `scratch/verify_pretrained_ko.py` on seed 999 for 500 ticks yielded:
  - **Max consecutive healthy ticks**: **162** (surpassing the 150-tick goal).
  - **Total K.O.s**: Nico=7, Sofy=3, Hugo=0.
  - The first K.O. event did not occur until tick 163 (Sofy went K.O. due to hunger/thirst depletion).
* **Weights Consolidation**: Saved to `storage/experiments/EXP_073_easy_train/best_agent_{a,b,c}.pt` and `final_agent_{a,b,c}.pt`.

## 🧠 Live Neural Monitoring Dashboard ([dashboard.html](file:///home/joan/Documents/IA/frankenswarm/docs/dashboard.html))
We designed and implemented a self-contained, responsive neural telemetry dashboard.
* **Log Parser**: Reads PPO training log files dynamically (e.g. `task-1562.log`) and extracts episodes, agent survival ticks, training losses, and communication events.
* **Milestone Timeline**: Automatically extracts and visualizes cognitive milestones like **Neurogenesis events** (Net2WiderNet expansions) and **Altruistic teaching/sleep distillation graduations**.
* **Visual Interface**: Utilizes a futuristic cyberpunk theme with glassmorphism panels, Orbitron typography, and interactive charts built on Chart.js.
* **Bug Fix (Spanish Accents & Anchors)**: Fixed a key parser issue where simulation logs (e.g. `task-1608.log`) failed to parse agent telemetry correctly when visiting accented Spanish locations (`montaña`, `río`) due to an ASCII `\w+` match constraint. Also anchored the regex to `^\s*` to prevent false matches on Theory of Mind (estimate) lines, and made simulation type detection robust across the entire log.
* **Robust Log Detection & Log Scale Fix**: Enhanced simulation log signature check (`SIMULACIÓN EN LA ARENA`) to prevent browser file-type detection warnings. Also clamped loss values $\le 0$ to `0.01` to prevent logarithmic scale rendering crashes on inactive/zero-loss elements.
* **Agent Selector UI**: Added clickable filters (`Todos`, `Nico`, `Sofy`, `Hugo`) to toggle datasets in the survival chart, clarifying agent lines even when values overlap.

## 💔 Decoupled Survivor Logic & Sadness Debuff
We implemented decoupled agent survival and the "Sadness Debuff" (*debuff de tristeza*) inside [cooperative_world.py](file:///home/joan/Documents/IA/frankenswarm/src/bitnet/cooperative_world.py), [train_arena_ppo.py](file:///home/joan/Documents/IA/frankenswarm/src/bitnet/train_arena_ppo.py), and [run_arena_simulation.py](file:///home/joan/Documents/IA/frankenswarm/src/bitnet/run_arena_simulation.py).
* **Survival Model**: The simulation runs until all active founder agents are dead. Dead agents output dead dummy states and return terminal reward/GAE values.
* **Sadness Debuff**: When a companion agent dies, all survivors get a 72-tick (3-day) sadness debuff:
  * **Emotion Override**: Emotion is forced to `tristeza`.
  * **Reward Penalty**: Subtracts `-0.5` per step.
  * **Metabolic Toll**: Sleep energy recovery is cut by 50%; hunger/thirst decay rates are increased by 20% (applied only to negative metabolism decays).
* **Validation**: Verified in `task-1990` log where Hugo starved on Tick 18, triggering the sadness debuff on Nico and Sofy. Nico died on Tick 28, refreshing Sofy's debuff cycle to 72t. Sofy died on Tick 72, and the simulation successfully terminated on Tick 73. All parsed cleanly in the updated telemetry dashboard.

## 🦣 Caza del Gran Herbívoro (Stag Hunt) & Ecological Expansion
We designed and implemented a set of cooperative hunting and survival mechanics in [cooperative_world.py](file:///home/joan/Documents/IA/frankenswarm/src/bitnet/cooperative_world.py) and [run_arena_simulation.py](file:///home/joan/Documents/IA/frankenswarm/src/bitnet/run_arena_simulation.py).
* **The Megafauna (Gran Herbívoro)**: Spawns randomly in `pradera` or `valle` every 40 ticks, has a 25-tick lifetime, and a 20% chance to migrate adjacent each tick.
* **Perception Override**: Seen as `"árbol"` (massive living presence) in the second perception token.
* **Combat Resolution**:
  * **Epic Success (3+ ready hunters)**: Kills the beast, breaks hunters' spears, drains `-15` energy, adds `+40.0` coordination bonuses, and spawns **9.0 units of raw food** on the ground.
  * **Retaliation (1 or 2 attackers)**: Attacks fail, beast retaliates. Attackers suffer `-45.0` health, `-25` energy, break their spears, and the beast has a 30% chance to flee.
* **Hunting Call Routing**: Hearing the `"árbol"` (Megafauna) signal automatically routes any receiver who has a spear and the `caza` skill to the sender's location (`nav_target`), coordinating the group.
* **Raw Food Rotting & Fire Curing**: Raw food on the ground decays by `-0.5` units per tick unless a campfire is active in the location, acting as smoke/curing preservation.
* **Enhanced TUI Display**: The simulation TUI now outputs the Gran Herbívoro's active location and timer, alongside detailed agent inventory (spears, branches, stones).

## 🍂 Resource Depletion Cooldown & Exhausted Source Penalties (Simulating Seasons)
We implemented a depletion cooldown and associated behavioral penalties to force the agents to move and prevent them from camping indefinitely:
* **Depletion Cooldown**: When a resource (food, water, branches, stones) on the ground is completely depleted (`0.0` or below), it enters a cooldown state (configured as `replenish_cooldown_ticks: 60` in [EXP_075_loop_fix.json](file:///home/joan/Documents/IA/frankenswarm/configs/experiments/EXP_075_loop_fix.json)). The resource does not regenerate until the cooldown expires.
* **Exhausted Source Penalty**: To prevent agents from spamming eating/drinking actions at dried up nodes, we introduce a severe reward penalty of **`-2.0`** if an agent attempts `"comer"` or `"beber"` when the ground resource at their location is exhausted (`< 1.0` capacity) and they do not have the resource in their backpack.
* **TUI Simulation Telemetry**: The TUI display was updated to print active cooldowns of resources at each location (e.g. `Bosque: [🍖: 0.0 (cd:45)]`), and dry check events are logged as `"(penalización)"` for clear debugging.

## 🎓 The Dojo of PopuLoRA (Supervised Curriculum & Bootstrapping)
We designed and implemented a hybrid training model that combines PPO reinforcement learning with supervised curriculum learning (Behavioral Cloning) in a "Dojo" environment:
* **Supervised Curriculum Generator ([dojo_populora.py](file:///home/joan/Documents/IA/frankenswarm/src/bitnet/dojo_populora.py))**: Generates balanced batches of synthetic state-action-shout scenarios matching expert behavioral rules (eating/drinking when in need + resources present, retrieving from backpack, sleeping in shelters, shouting for food/water at critical needs, giving resources to needy companions, moving towards signaling teammates).
* **Bootstrapping (Pre-training)**: Before entering the PPO reinforcement loop, the agents undergo 50 steps of supervised Dojo training to align their policy heads with the expert manual of instructions, preventing initial random deaths.
* **Sleep Consolidation Dojo Integration**: During the sleep phase (after every episode), the agents visit the Dojo of PopuLoRA with a teacher for 5 epochs of supervised training, preventing policy drift and catastrophic forgetting.
* **PPO Representation Stabilization**: Freezes backbone parameters during active daytime PPO training loop updates (updating only policy and value heads), and unfreezes the backbone during sleep consolidation (Dojo + latent resonance), preventing representational collapse and gradient explosions (`nan` value errors).
* **First Aid & Reanimation Dojo Scenario**: Added a `reanimate_companion` template to the Dojo synthetic dataset to instruct agents to use the `reanimar` action when a teammate falls unconscious in their current location.


