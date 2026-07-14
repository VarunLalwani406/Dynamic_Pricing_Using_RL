"""
Week 5 — Self-Play Trainer
============================
Alternating frozen-opponent self-play loop for DQN and PPO agents.

Algorithm
---------
1.  Initialise agent1 and agent2 with no frozen policy (both start
    against the best-response baseline opponent).
2.  For each swap phase:
      a. Freeze agent2's current policy.
      b. Train agent1 for `steps_per_swap` steps against frozen agent2.
      c. Freeze agent1's updated policy.
      d. Train agent2 for `steps_per_swap` steps against frozen agent1.
3.  After all phases, evaluate both agents and report results.

Why alternating and not simultaneous?
--------------------------------------
Simultaneous updates make the environment non-stationary for both agents.
From agent1's perspective, its "environment" (which includes agent2's policy)
is changing every step — violating the stationary MDP assumption that SB3's
algorithms rely on. Alternating frozen updates restore stationarity within
each training phase while still allowing both agents to improve.

Reference: Heinrich & Silver (2016), Section 2.
"""

import sys, time, json
import numpy as np

# sys.path.insert(0, "/mnt/user-data/outputs")
# sys.path.insert(0, "/home/claude")

from bertrand_pricing_env import BertrandPricingEnv


class SelfPlayTrainer:
    """
    Runs alternating self-play between two agents of any type
    (DQNPricingAgent, PPOPricingAgent, or any object with
    .train(), .get_policy_fn(), .update_frozen_policy(), .evaluate()).

    Parameters
    ----------
    agent1          : first agent (e.g. DQNPricingAgent)
    agent2          : second agent (e.g. PPOPricingAgent)
    eval_env        : BertrandPricingEnv for evaluation
    n_swaps         : number of alternating swap phases
    steps_per_swap  : SB3 timesteps per training phase per agent
    eval_episodes   : episodes per evaluation call
    verbose         : print progress
    """

    def __init__(
        self,
        agent1,
        agent2,
        eval_env        : BertrandPricingEnv,
        n_swaps         : int = 5,
        steps_per_swap  : int = 20_000,
        eval_episodes   : int = 50,
        verbose         : bool = True,
    ):
        self.agent1         = agent1
        self.agent2         = agent2
        self.eval_env       = eval_env
        self.n_swaps        = n_swaps
        self.steps_per_swap = steps_per_swap
        self.eval_episodes  = eval_episodes
        self.verbose        = verbose

        self.history = []   # list of dicts, one per swap phase

    def run(self) -> list:
        total_steps = self.n_swaps * self.steps_per_swap * 2
        print(f"\n{'═'*62}")
        print(f"  SELF-PLAY TRAINING")
        print(f"  {self.n_swaps} swap phases × {self.steps_per_swap:,} steps/phase × 2 agents")
        print(f"  Total timesteps: {total_steps:,}")
        print(f"  Agent1: {self.agent1.name}   Agent2: {self.agent2.name}")
        print(f"{'═'*62}")

        t0 = time.time()

        for swap in range(self.n_swaps):
            if self.verbose:
                print(f"\n  ── Swap {swap+1}/{self.n_swaps} ──────────────────────────────────")

            # Phase A: train agent1, agent2 frozen
            frozen2 = self.agent2.get_policy_fn()
            self.agent1.update_frozen_policy(frozen2)
            self.agent1.train(self.steps_per_swap)

            # Phase B: train agent2, agent1 frozen
            frozen1 = self.agent1.get_policy_fn()
            self.agent2.update_frozen_policy(frozen1)
            self.agent2.train(self.steps_per_swap)

            # Evaluate both
            r1 = self.agent1.evaluate(self.eval_env, self.eval_episodes)
            r2 = self.agent2.evaluate(self.eval_env, self.eval_episodes)

            record = {
                "swap"              : swap + 1,
                "agent1_profit"     : r1["mean_profit"],
                "agent2_profit"     : r2["mean_profit"],
                "agent1_price"      : r1["mean_price"],
                "agent2_price"      : r2["mean_price"],
                "agent1_ci"         : r1["collusion_index"],
                "agent2_ci"         : r2["collusion_index"],
                "elapsed_s"         : round(time.time() - t0, 1),
            }
            self.history.append(record)

            if self.verbose:
                print(
                    f"    {self.agent1.name}: π={r1['mean_profit']:.0f}  "
                    f"P̄={r1['mean_price']:.2f}  CI={r1['collusion_index']:.3f}  |  "
                    f"{self.agent2.name}: π={r2['mean_profit']:.0f}  "
                    f"P̄={r2['mean_price']:.2f}  CI={r2['collusion_index']:.3f}"
                )

        elapsed = time.time() - t0
        print(f"\n  Self-play complete in {elapsed:.0f}s")
        return self.history

    def save_history(self, path: str):
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"  History saved → {path}")
