"""
EXP_046/047: Arena Training — Dos agentes en el mundo cooperativo.

Los agentes entrenados en el Dojo (MinimalWorld) se gradúan a la Arena
(CooperativeWorld) donde la comida es escasa, hay fog of war, y
la comunicación emerge de la necesidad de sobrevivir.

EXP_046: Silencio (gritar = no-op) → baseline cooperativo
EXP_047: Comunicación (gritar = broadcast) → ¿mejora la supervivencia?

Origen: Joan Garcia — "los mudos mueren" — 2026-06-01
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.bitnet.cooperative_world import (
    COOP_ACTIONS,
    COOP_N_ACTIONS,
    SILENCE_GLYPH,
    CooperativeWorld,
)
from src.bitnet.glyph_vocabulary import (
    N_EMOTIONS,
)
from src.bitnet.modeling_bitnet import BitNet4LayerModel
from src.bitnet.net2net import grow_action_head


def perception_to_input_coop(perception: list[int], device: torch.device) -> torch.Tensor:
    """
    Convertir percepción del mundo cooperativo a tensor.
    La percepción ya viene como lista de glyph indices del CooperativeWorld.
    Input: 6 tokens [loc, what, emotion, detail, sig_loc, sig_what]
    """
    indices = perception[:6]
    while len(indices) < 6:
        indices.append(SILENCE_GLYPH)
    return torch.tensor([indices], dtype=torch.long, device=device)


def run_arena_training():
    parser = argparse.ArgumentParser(description="Arena Training")
    parser.add_argument("--config", type=str, required=True)
    args, _ = parser.parse_known_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = args.config if os.path.isabs(args.config) else os.path.join(base_dir, args.config)
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    experiment_id = config.get("experiment_id", "EXP_046_arena")
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_dir = os.path.join(base_dir, "storage", "experiments", experiment_id)
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    model_cfg = config.get("model", {})
    emotion_cfg = config.get("emotion", {})

    # ═══ Build agents ═══
    def make_agent():
        m = BitNet4LayerModel(
            use_glyphs=True,
            hidden_dim=model_cfg.get("hidden_dim", 256),
            num_layers=model_cfg.get("num_layers", 3),
            use_pos_embedding=model_cfg.get("use_pos_embedding", True),
            max_resonance_steps=config.get("resonance", {}).get("max_resonance_steps", 5),
            n_emotions=N_EMOTIONS,
            emotion_dim=emotion_cfg.get("dim", 64),
            emotion_mode=emotion_cfg.get("mode", "first_only"),
        ).to(device)

        # Load pretrained (from dojo)
        pretrained_from = config.get("pretrained_from")
        if pretrained_from:
            p = pretrained_from if os.path.isabs(pretrained_from) else os.path.join(base_dir, pretrained_from)
            if os.path.exists(p):
                sd = torch.load(p, map_location=device, weights_only=True)
                if "action_head.0.weight" in sd:
                    ckpt_width = sd["action_head.0.weight"].shape[0]
                    if ckpt_width != m.action_head[0].out_features:
                        m.action_head = nn.Sequential(
                            nn.Linear(model_cfg.get("hidden_dim", 256), ckpt_width),
                            nn.GELU(),
                            nn.Linear(ckpt_width, 6),
                        ).to(device)
                m.load_state_dict(sd, strict=False)
                print(f"🧒→🧑 Loaded from dojo: {p}")

        # Resize action head for 7 actions (6 original + GRITAR)
        old_width = m.action_head[0].out_features
        old_out = m.action_head[2].out_features
        if old_out != COOP_N_ACTIONS:
            # Preserve first layer weights, only change output layer
            old_first = m.action_head[0]
            old_gelu = m.action_head[1]
            m.action_head = nn.Sequential(
                old_first,           # KEEP: learned representations from dojo
                old_gelu,
                nn.Linear(old_width, COOP_N_ACTIONS),  # NEW: 7 actions
            ).to(device)
            print(f"🔧 Action head resized: {old_out} → {COOP_N_ACTIONS} actions (width={old_width})")

        # Unfreeze backbone for plasticity
        unfreeze = config.get("unfreeze_backbone", True)  # default: unfrozen in arena
        if not unfreeze:
            for name, param in m.named_parameters():
                if "action_head" not in name:
                    param.requires_grad = False
            print("🔒 Backbone frozen")
        else:
            print("🔓 Backbone unfrozen — full plasticity")

        return m

    agent_a = make_agent()
    agent_b = make_agent()

    communicate = config.get("communication", {}).get("enabled", True)
    comm_str = "📡 COMUNICACIÓN ON" if communicate else "🔇 SILENCIO"

    print(f"═══ 🏟️ Arena — {experiment_id} ═══")
    print(f"  {comm_str}")
    head_w = agent_a.action_head[0].out_features
    head_p = sum(p.numel() for p in agent_a.action_head.parameters())
    print(f"🧠 Agent A: width={head_w}, params={head_p:,}")
    print(f"🧠 Agent B: width={head_w}, params={head_p:,}")

    # ═══ Training config ═══
    train_cfg = config.get("training", {})
    lr = train_cfg.get("lr", 5e-4)
    gamma = train_cfg.get("gamma", 0.97)
    entropy_bonus = train_cfg.get("entropy_bonus", 0.05)
    grad_clip = train_cfg.get("grad_clip", 1.0)
    update_interval = train_cfg.get("update_interval", 16)
    n_episodes = train_cfg.get("episodes", 500)
    max_ticks = config.get("world", {}).get("max_ticks", 200)
    n_think = config.get("resonance", {}).get("n_think", 2)
    n_verify = config.get("resonance", {}).get("n_verify", 2)

    # World config
    world_cfg = config.get("world", {})
    food_interval = world_cfg.get("food_interval", 10)
    food_duration = world_cfg.get("food_duration", 5)
    hunger_rate = world_cfg.get("hunger_rate", 5.0)

    # Growth config
    growth_cfg = config.get("growth", {})
    growth_factor = growth_cfg.get("factor", 1.5)
    max_width = growth_cfg.get("max_width", 648)
    growth_patience = growth_cfg.get("patience", 50)

    opt_a = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, agent_a.parameters()),
        lr=lr, weight_decay=0.01
    )
    opt_b = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, agent_b.parameters()),
        lr=lr, weight_decay=0.01
    )

    # ═══ Tracking ═══
    best_combined = 0.0
    survival_hist = []
    total_growths_a = 0
    total_growths_b = 0
    total_shouts_a = 0
    total_shouts_b = 0

    print(f"🌍 Arena: food_interval={food_interval}, food_duration={food_duration}, hunger_rate={hunger_rate}")
    print(f"📈 Episodes: {n_episodes}, max_ticks: {max_ticks}")

    for episode in range(n_episodes):
        world = CooperativeWorld(
            seed=seed + episode,
            food_interval=food_interval,
            food_duration=food_duration,
            hunger_rate=hunger_rate,
        )
        state_a, state_b = world.reset()

        buf_a = {"log_probs": [], "rewards": [], "entropies": []}
        buf_b = {"log_probs": [], "rewards": [], "entropies": []}

        agent_a.train()
        agent_b.train()

        ep_shouts_a = 0
        ep_shouts_b = 0

        for tick in range(max_ticks):
            if not state_a.alive or not state_b.alive:
                break

            # ── PERCIBIR (fog of war) ──
            perc_a = world.perceive(state_a)
            perc_b = world.perceive(state_b)

            # Si comunicación deshabilitada, silenciar canal
            if not communicate:
                perc_a[4] = SILENCE_GLYPH
                perc_a[5] = SILENCE_GLYPH
                perc_b[4] = SILENCE_GLYPH
                perc_b[5] = SILENCE_GLYPH

            input_a = perception_to_input_coop(perc_a, device)
            input_b = perception_to_input_coop(perc_b, device)

            emo_a = torch.tensor([state_a.emotion_id], device=device)
            emo_b = torch.tensor([state_b.emotion_id], device=device)

            # ── PENSAR ──
            with torch.no_grad():
                logits_a, meta_a = agent_a.forward_deep_think(
                    input_a, n_think=n_think, n_verify=n_verify,
                    pos_mode="clock", emotion_ids=emo_a,
                )
                logits_b, meta_b = agent_b.forward_deep_think(
                    input_b, n_think=n_think, n_verify=n_verify,
                    pos_mode="clock", emotion_ids=emo_b,
                )

            # ── DECIDIR ──
            hidden_a = meta_a["hidden"][:, 2, :]
            hidden_b = meta_b["hidden"][:, 2, :]
            action_logits_a = agent_a.action_head(hidden_a)
            action_logits_b = agent_b.action_head(hidden_b)
            probs_a = F.softmax(action_logits_a, dim=-1)
            probs_b = F.softmax(action_logits_b, dim=-1)

            # Graduated exploration: cachorro→juvenil→adulto
            # Agents arrive from dojo with basic skills — no need for 100% random
            phase_1 = int(n_episodes * 0.20)  # 0-20%: juvenil, probing new world
            phase_2 = int(n_episodes * 0.60)  # 20-60%: refining
            if episode < phase_1:
                explore_rate = 0.20  # mostly exploit dojo skills, probe new actions
            elif episode < phase_2:
                explore_rate = 0.10  # refine what works
            else:
                explore_rate = 0.05  # near-pure exploitation

            # If communication disabled, mask out GRITAR (action 6)
            if not communicate:
                probs_a = probs_a.clone()
                probs_a[0, 6] = 0.0
                probs_a = probs_a / probs_a.sum()
                probs_b = probs_b.clone()
                probs_b[0, 6] = 0.0
                probs_b = probs_b / probs_b.sum()

            # Agent A
            if torch.rand(1).item() < explore_rate:
                n_act = COOP_N_ACTIONS if communicate else COOP_N_ACTIONS - 1
                idx_a = torch.randint(0, n_act, (1,)).item()
                lp_a = torch.log(probs_a[0, idx_a] + 1e-8).squeeze()
            else:
                dist_a = torch.distributions.Categorical(probs_a)
                idx_a = dist_a.sample().item()
                lp_a = dist_a.log_prob(torch.tensor(idx_a, device=device))

            # Agent B
            if torch.rand(1).item() < explore_rate:
                n_act = COOP_N_ACTIONS if communicate else COOP_N_ACTIONS - 1
                idx_b = torch.randint(0, n_act, (1,)).item()
                lp_b = torch.log(probs_b[0, idx_b] + 1e-8).squeeze()
            else:
                dist_b = torch.distributions.Categorical(probs_b)
                idx_b = dist_b.sample().item()
                lp_b = dist_b.log_prob(torch.tensor(idx_b, device=device))

            ent_a = -(probs_a * (probs_a + 1e-8).log()).sum()
            ent_b = -(probs_b * (probs_b + 1e-8).log()).sum()

            # ── ACTUAR ──
            action_a = COOP_ACTIONS[idx_a]
            action_b = COOP_ACTIONS[idx_b]

            result_a, result_b, world_info = world.step(action_a, action_b)

            if result_a.get("shouted"):
                ep_shouts_a += 1
            if result_b.get("shouted"):
                ep_shouts_b += 1

            reward_a = world.get_reward(state_a, result_a)
            reward_b = world.get_reward(state_b, result_b)

            # Reward cooperativo: cuando la comunicación FUNCIONA, ambos se benefician.
            # Gritador: "mi información salvó a mi compañero"
            # Receptor: "escuché, seguí la señal, comí"
            # Bonus alto (5.0) para crear señal fuerte. Shared fate hace el resto.
            if communicate:
                # B comió siguiendo señal de A → AMBOS reciben bonus
                if world_info.get("coop_bonus_a", 0.0) > 0:
                    reward_a += 5.0  # A: "mi grito sirvió"
                    reward_b += 5.0  # B: "escuchar me salvó"
                # A comió siguiendo señal de B → AMBOS reciben bonus
                if world_info.get("coop_bonus_b", 0.0) > 0:
                    reward_b += 5.0  # B: "mi grito sirvió"
                    reward_a += 5.0  # A: "escuchar me salvó"

            buf_a["log_probs"].append(lp_a)
            buf_a["rewards"].append(reward_a)
            buf_a["entropies"].append(ent_a)
            buf_b["log_probs"].append(lp_b)
            buf_b["rewards"].append(reward_b)
            buf_b["entropies"].append(ent_b)

            # ── Online update ──
            if tick % update_interval == 0 and tick > 0:
                for buf, opt, model in [(buf_a, opt_a, agent_a), (buf_b, opt_b, agent_b)]:
                    if len(buf["log_probs"]) > 1:
                        returns = []
                        G = 0
                        for r in reversed(buf["rewards"]):
                            G = r + gamma * G
                            returns.insert(0, G)
                        returns = torch.tensor(returns, dtype=torch.float32, device=device)
                        if len(returns) > 1:
                            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
                        loss = torch.tensor(0.0, device=device)
                        for i in range(len(buf["log_probs"])):
                            lp = buf["log_probs"][i].squeeze()
                            ret = returns[i].squeeze()
                            loss -= lp * ret
                            loss -= entropy_bonus * buf["entropies"][i]
                        loss = loss / len(buf["log_probs"])
                        opt.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            filter(lambda p: p.requires_grad, model.parameters()),
                            max_norm=grad_clip
                        )
                        opt.step()
                    buf["log_probs"].clear()
                    buf["rewards"].clear()
                    buf["entropies"].clear()

        # ── Metrics ──
        combined = min(state_a.tick, state_b.tick)
        survival_hist.append(combined)
        total_shouts_a += ep_shouts_a
        total_shouts_b += ep_shouts_b

        if combined > best_combined:
            best_combined = combined
            torch.save(agent_a.state_dict(), os.path.join(exp_dir, "best_agent_a.pt"))
            torch.save(agent_b.state_dict(), os.path.join(exp_dir, "best_agent_b.pt"))

        # ── Post-episode growth ──
        for label, model, opt_ref in [("A", agent_a, "opt_a"), ("B", agent_b, "opt_b")]:
            curr_w = model.action_head[0].out_features
            if len(survival_hist) >= 10 and curr_w < max_width:
                avg_s = np.mean(survival_hist[-10:])
                if avg_s < max_ticks * 0.10:  # lower threshold for arena
                    gi = grow_action_head(model, growth_factor=growth_factor)
                    if label == "A":
                        total_growths_a += 1
                        opt_a = torch.optim.AdamW(
                            filter(lambda p: p.requires_grad, model.parameters()),
                            lr=lr, weight_decay=0.01
                        )
                    else:
                        total_growths_b += 1
                        opt_b = torch.optim.AdamW(
                            filter(lambda p: p.requires_grad, model.parameters()),
                            lr=lr, weight_decay=0.01
                        )
                    print(f"  🧬 {label} NEUROGENÉSIS | width: {gi['old_width']}→{gi['new_width']}")

        # ── Log ──
        if episode % 10 == 0 or episode == n_episodes - 1:
            avg = np.mean(survival_hist[-50:]) if survival_hist else 0
            w_a = agent_a.action_head[0].out_features
            w_b = agent_b.action_head[0].out_features
            food_str = f"🍖{world_info['food_location'] or 'none'}"
            print(
                f"Ep {episode+1:4d}/{n_episodes} | "
                f"A:{state_a.tick:3d}t@{state_a.location:8s} B:{state_b.tick:3d}t@{state_b.location:8s} | "
                f"Best:{best_combined:.0f} Avg50:{avg:.1f} | "
                f"{'📡' if communicate else '🔇'} shouts:{ep_shouts_a+ep_shouts_b} | "
                f"G_A:{total_growths_a} G_B:{total_growths_b} | {food_str}"
            )

    # ═══ Final ═══
    torch.save(agent_a.state_dict(), os.path.join(exp_dir, "final_agent_a.pt"))
    torch.save(agent_b.state_dict(), os.path.join(exp_dir, "final_agent_b.pt"))

    avg_final = np.mean(survival_hist[-100:])
    print(f"\n{'═'*60}")
    print(f"📊 Arena Results — {experiment_id}")
    print(f"   {'📡 Communication ON' if communicate else '🔇 Silent'}")
    print(f"   Best combined: {best_combined:.0f} ticks")
    print(f"   Avg (last 100): {avg_final:.1f} ticks")
    print(f"   Total shouts: A={total_shouts_a}, B={total_shouts_b}")
    print(f"   Final width: A={agent_a.action_head[0].out_features}, B={agent_b.action_head[0].out_features}")
    print(f"   Growths: A={total_growths_a}, B={total_growths_b}")
    print(f"{'═'*60}")


if __name__ == "__main__":
    run_arena_training()
