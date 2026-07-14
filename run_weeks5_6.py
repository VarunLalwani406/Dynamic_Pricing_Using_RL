"""
Run script — Weeks 5 & 6
=========================
1. Train DQN and PPO agents (self-play, 5 swap phases × 20k steps)
2. Run Week 6 multi-seed collusion analysis across all agents
3. Save all results to JSON for the dashboard and report
"""

import sys, json, warnings
import numpy as np
warnings.filterwarnings("ignore")

# sys.path.insert(0, "/mnt/user-data/outputs")
# sys.path.insert(0, "/home/claude")

from bertrand_pricing_env import BertrandPricingEnv
from agents import AlwaysNashAgent, AlwaysColludeAgent, TitForTatAgent, RandomAgent
from q_agent import QLearningAgent, QHyperparams
from q_trainer import QTrainer
from deep_rl_agents import DQNPricingAgent, PPOPricingAgent
from selfplay_trainer import SelfPlayTrainer
from collusion_analysis import (
    run_multi_seed_analysis, compare_agents, export_reports
)

ENV_KW = dict(a=100, b=1.0, d=0.5, marginal_cost=20,
              n_price_levels=30, max_steps=200, noise_std=2.0)

eval_env = BertrandPricingEnv(**ENV_KW)
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]

# ── 1. Re-train Q-agent (quick) ─────────────────────────────────────
print("\n[1/3] Quick Q-agent retrain (2000 eps) …")
hp = QHyperparams(alpha=0.10, gamma=0.95, eps_start=1.0,
                  eps_end=0.05, eps_decay_frac=0.80, n_bins=10)
q_env   = BertrandPricingEnv(**ENV_KW)
q_agent = QLearningAgent(n_actions=q_env.action_space.n, hp=hp, seed=42)
QTrainer(q_env, q_agent, n_episodes=2000, n_steps=200,
         log_interval=9999, seed=0).train()

# ── 2. Train DQN & PPO with self-play ───────────────────────────────
print("\n[2/3] Self-play training: DQN vs PPO …")
dqn = DQNPricingAgent(ENV_KW, seed=42)
ppo = PPOPricingAgent(ENV_KW, seed=42)

trainer = SelfPlayTrainer(
    agent1=dqn, agent2=ppo, eval_env=eval_env,
    n_swaps=5, steps_per_swap=20_000,
    eval_episodes=30, verbose=True,
)
sp_history = trainer.run()

dqn.save("/home/claude/dqn_model")
ppo.save("/home/claude/ppo_model")

# ── 3. Week 6 — multi-seed collusion analysis ────────────────────────
print("\n[3/3] Week 6 — Collusion detection & PoA across all agents …")

# Wrap Q-agent to match the agent interface
class QAgentWrapper:
    name = "Q-Learning"
    def __init__(self, agent): self._a = agent
    def act(self, obs, info=None, training=False):
        self._a._epsilon = 0.0
        return self._a.act(obs, info or {}, training=False)

agents_to_test = [
    ("Always-Nash",    AlwaysNashAgent(eval_env.price_grid, eval_env.nash_price)),
    ("Always-Collude", AlwaysColludeAgent(eval_env.price_grid, eval_env.monopoly_price)),
    ("Tit-for-Tat",   TitForTatAgent(eval_env.price_grid, eval_env.monopoly_price)),
    ("Random",         RandomAgent(eval_env.price_grid, seed=42)),
    ("Q-Learning",     QAgentWrapper(q_agent)),
    ("DQN",            dqn),
    ("PPO",            ppo),
]

reports = []
for name, agent in agents_to_test:
    r = run_multi_seed_analysis(
        agent, eval_env, name,
        seeds=SEEDS, n_steps=500, threshold=1.05, window=50,
    )
    reports.append(r)

compare_agents(reports)
export_reports(reports, "week6_results.json")

# Save self-play history
with open("/home/claude/selfplay_history.json","w") as f:
    json.dump(sp_history, f, indent=2)

print("\nAll results saved.")
