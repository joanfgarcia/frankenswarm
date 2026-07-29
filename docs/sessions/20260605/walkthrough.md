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

## 🗺️ FrankenSwarm Construct: Map Editor & Visualizer (`playground/map_editor.html`)
We designed and implemented a standalone visual tool for building maps and watching simulations:
* **Interactive Canvas Map Editor**:
  * Double-click to create new locations.
  * Drag-and-drop to position them geometrically in 2D space.
  * Shift-drag to create bidirectionally linked paths (graph edges).
  * Panel Inspector: Enables setting Name, Description, resources (Food, Water, Branches, Stones, Fish), Storm Shelter status, and Predator Chance.
  * Export Graph: Exports the configuration as a standard JSON map, or compiles it directly to Python dictionaries (`COOP_LOCATIONS` / `COOP_ADJACENCY`) to copy/paste.
* **Simulation Player**:
  * Supports drag-and-drop loading of simulation trajectory JSON files.
  * Smooth coordinate interpolation: Animates Nico, Sofi, and Hugo moving along connection edges, resolving spatial positions.
  * Comic-style shout bubbles: Renders conceptual agent screams alongside Samantha-translated Spanish natural language phrases.
  * Side Panel Telemetry: Updates health, hunger, thirst, energy, active actions, and backpack contents in real-time.

## ⚙️ Zero-Copy Dynamic Map Loading Pipeline
We decoupled the map layout from `cooperative_world.py` to allow dynamic map loading:
* **Dynamic Environment Loading**: Modified `CooperativeWorld`'s constructor to accept a `map_path: str | None` parameter, clearing and updating module global parameters (`COOP_LOCATIONS`, `COOP_ADJACENCY`, etc.) on initialization.
* **Heuristic Substring Glyph Matcher**: Automatically resolves custom location names to the model's vocabulary. Substrings like "rio" -> "río", "selva/bosque" -> "bosque", "refugio/cueva" -> "cueva" ensure the model receives in-vocabulary inputs.
* **Simulation CLI integration**: Added `--map` and `--json-out` parameters to `run_arena_simulation.py` to run dynamic simulations and output play logs.
* **Training Config integration**: Added `map_path` support to the `"world"` block in training JSON configs.

## 🛡️ Policy Underflow Numerical Protection (NaNs Fix)
* **Underflow Crash Resolution**: Fixed a training crash in PPO updates where policy probabilities underflowed to exactly `0.0` for all allowed actions (resulting in `nan` values when normalized by PyTorch's `Categorical` distribution).
* **Self-Healing Fallback**: Implemented a numerical guard in `get_masked_probs`. When the sum of masked probabilities drops below `1e-7`, it automatically falls back to a uniform distribution over the valid actions (`mask / mask_sum`), preventing crashes and stabilizing training during policy shifts.

## 🎨 ANSI Terminal Simulation Colorizer
We color-coded the agent printouts in the terminal simulation visualizer to match the HTML map player:
* **Color Schemes**: Nico is colored **Cyan** (`\033[96m`), Sofi is colored **Magenta/Pink** (`\033[95m`), and Hugo is colored **Yellow** (`\033[93m`).
* **Visual Clarity**: Colored their names, states, decisions, shouts, resource-sharing events, and final health outcomes in [run_arena_simulation.py](file:///home/joan/Documents/IA/frankenswarm/src/bitnet/run_arena_simulation.py).
* **ToM Integration**: Colored the companion identity tags within each agent's mental model display (Theory of Mind), making it instantly clear whose perspective is being represented.

## 🗺️ Visualizer Canvas Color Bugfix (`playground/map_editor.html`)
* **The Canvas CSS Variables Bug**: Identified that the HTML5 Canvas 2D rendering context does not support CSS custom property lookup (`var(--accent-green)`, `var(--color-nico)`, etc.) natively in `ctx.fillStyle` and `ctx.strokeStyle`, causing the browser to fall back to the previous style (rendering all resource indicators and agent dots as solid white circles).
* **Hex Resolution**: Created a static JavaScript `colors` mapping object inside the script to resolve these color tokens to their exact hexadecimal codes (Nico=Cyan `#00f0ff`, Sofi=Magenta `#ff007f`, Hugo=Yellow `#ffea00`, plus resource indicators). All agents and node pips now render with their proper distinct colors in the simulation visualizer.

## 🗣️ Shout Cooldown & Give Action Restrictions (`cooperative_world.py`)
To prevent the agents from getting stuck in static shouting loops (staying still and shouting in a half-duplex deaf loop) and regulate resource giving:
* **Shout Cooldown**: Added `shout_cooldown_ticks` to `CoopAgentState`. When an agent shouts, this cooldown is set to `3` ticks, preventing them from shouting in consecutive ticks. In `get_valid_actions_mask`, action 6 (`"gritar"`) is masked out while the cooldown is active.
* **Requested-only Giving**: Constrained action 7 (`"dar"`) in `get_valid_actions_mask` to prevent agents from giving resources unless a teammate in the same location has actively requested that resource (via a shout) in the last 15 ticks.

## 🎓 Dojo Homeostasis Refactoring (`dojo_populora.py`)
Refactored the Dojo synthetic scenario generator to support homeostasis-based training rules:
* **Severe Need Scenarios**: Refactored `shout_hunger` to use `emotion = "hambre"` and `shout_thirst` to use `emotion = "dolor"` to represent severe homeostasis deficits (< 20 hunger/thirst/health). Shouting target action is `"gritar"`.
* **Mild Need Scenarios**: Added `explore_hungry`/`explore_thirsty` and `consume_mild_food`/`consume_mild_water` using `emotion = "ira"` to represent mild homeostasis deficits (20 to 50 hunger/thirst). This trains the agents to explore/move if resources are absent, or eat/drink if resources are present, rather than shouting.
* **Training Verification**: Verified using a test script that the Dojo batch generation executes cleanly and correctly maps emotions and actions.

## 🚀 Temporal Resonance Implementation & Launch of `EXP_077_dojo_shout_levels`
We have successfully implemented the full temporal recurrence loop across all modules:
* **Core Model (`modeling_bitnet.py`)**: Added an optional `h_prev` parameter to `forward_resonance` and `forward_resonance_training`, blending the previous tick's latent state with the current sensory input at Layer 2.
* **Rollout Simulation (`run_arena_simulation.py` & `train_arena_ppo.py`)**: Modified the simulation and episode rollout loops to track and carry the hidden state `h_state_x` of each agent step-by-step across ticks, storing it as a detached input feature (`h_prevs`) in the PPO trajectory buffers.
* **Batched PPO Updates (`train_arena_ppo.py`)**: Stacked the stored recurrent hidden states during optimization epochs and passed them as a batch tensor `h_prevs_tensor` to `forward_resonance` to reconstruct policy probabilities accurately.
* **Empirical Validation**:
  * **Memory Toy Task (`scratch/test_temporal_memory_task.py`)**: A delayed memory experiment demonstrated that a standard model fails to remember inputs across ticks (56.2% accuracy), whereas the recurrent model achieves **100% accuracy** within 30 epochs.
  * **Full Simulator Compilation**: Verified that the updated `run_arena_simulation.py` runs and prints step-by-step execution without shape or traceback errors.
  * **PPO Training Launch**: Cancelled the old non-recurrent run and started `EXP_077_dojo_shout_levels` with active temporal recurrence using the cgroup limit of 10G:
    ```bash
    systemd-run --user --scope -p MemoryMax=10G bash -c "PYTHONPATH=. .venv/bin/python src/bitnet/train_arena_ppo.py --config configs/experiments/EXP_077_dojo_shout_levels.json"
    ```
  * Verified that the training loop has entered active PPO episodes, executing the recurrent updates cleanly.
* **Child Agent Initialization Bug Fix**: Fixed a critical size mismatch crash at episode 1701 where a newborn child agent (Domi) had its recurrent hidden state `h_state_d` initialized using `child_width` (288, the action head intermediate width) instead of `hidden_dim` (256, the backbone dimension), resulting in a `RuntimeError` due to tensor shape mismatch.
* **Relaunch, Resume & Completion**: Decoupled the config to load from the latest checkpoints in `storage/experiments/EXP_077_dojo_shout_levels` and successfully resumed the PPO Dojo training. The run completed all **2000 episodes** cleanly without any further crashes.
  * **Final Training Results**:
    - **Best combined survival**: **324 ticks** (a new historical record).
    - **Average (last 100 episodes)**: **71.2 ticks**.
    - **Total shouts**: Nico (A)=9088, Sofy (B)=10849, Hugo (C)=10443, Domi (D)=26.
    - **Final head width**: A=648, B=648, C=648, D=648 (max capacity reached!).
    - **Successful Child Birth**: Domi (D) was successfully born in multiple episodes (e.g. Ep 1965, 1978, 1980, 1990) inheriting parent skills and a head width of 648, running PPO evaluations and optimization steps stably.
* **Mathematical Correction of Net2Net Attention Scaling (`net2net.py`)**: Fixed the multi-head attention weight mapping inside `net2wider_model` to mathematically preserve attention scores when the backbone hidden dimension grows. Queries are scaled by $\sqrt{\frac{d'}{d}} \cdot \frac{1}{\sqrt{\text{copy\_count}}}$, and keys are scaled by $\frac{1}{\sqrt{\text{copy\_count}}}$, keeping attention dot products perfectly invariant.

## 🧠 Dynamic Tech Tree & Prolog Skill Discovery Integration

We have successfully completed the implementation of the Prolog-guided technology tree and dynamic skill discovery pipeline:

* **Prolog Tech Tree & Evaluation Rules (`cooperative_rules.pl`)**:
	* Appended tech tree prerequisites, listing that `fuego` requires `artesanía`, and `cocina` requires `comida`, `agua`, and `fuego`.
	* Implemented `que_observar/14` to dynamically identify which skills an agent can observe and discover based on their location, active elements (like campfires), and current knowledge slots.
	* Implemented `evaluar_intento/16` to categorize skill attempts into `intento_valido` (valid, proceeds with learning experience), `error_requisito` (missing prerequisites), or `error_fisico` (missing physical materials/conditions).
* **Python World Integration (`cooperative_world.py`)**:
	* Added world parameters `food_decay_rate` (controlling food rotting over time) and `permanent_fire_location` (allowing permanent fire placements like campfires in the `cueva`).
	* Opened up action attempts in `get_valid_actions_mask` by removing strict skill checks, permitting agents to attempt any action that is physically possible (e.g. attempting to start a fire or craft a tool).
	* Integrated Prolog queries directly inside `act()` to evaluate action attempts:
		* If an attempt is valid but the agent lacks the skill, they consume the resources and gain experience scaled by their emotion multiplier.
		* Added observation discovery inside the `ver` (look) action: looking at fire or resources allows agents to discover and learn new skills over time.
	* Tuned rewards and penalties inside `get_reward()` based on Prolog's feedback, including a `-5.0` penalty for false shouts to keep communication reliable.
* **Dojo Scenarios (`dojo_populora.py`)**:
	* Added new training templates: `"observe_fire"`, `"attempt_fire"`, and `"attempt_craft"` to teach the neural policy heads correct discovery and attempt actions.
* **Unit Tests & Validation (`tests/test_prolog_discovery.py`)**:
	* Created comprehensive tests covering Prolog tech tree logic, dynamic experience gain under emotional states (comfort thresholds), and map curiosity resets.
	* Validated all tests passing successfully.
