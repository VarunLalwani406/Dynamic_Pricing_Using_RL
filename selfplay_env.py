"""
Week 5 — Self-Play Environment Wrapper
========================================
Wraps BertrandPricingEnv for two simultaneous SB3 agents.

Key problem with naive self-play
---------------------------------
When both agents update simultaneously, the environment is non-stationary
from each agent's perspective: agent 1's "environment" keeps changing because
agent 2's policy keeps changing, and vice versa. The resource sheet warns this
causes instability.

Solution used here (§Week5 pitfall fix)
-----------------------------------------
We run agents in alternating frozen-opponent mode:
  - Train agent 1 for N episodes while agent 2's policy is frozen.
  - Swap: train agent 2 while agent 1 is frozen.
  - Repeat until convergence.

This makes each agent's environment stationary for the duration of its
training phase, while still allowing both agents to improve over the
full training run.

Two env classes
---------------
SelfPlayEnv1  — firm 1 learns; firm 2 plays a frozen policy passed in.
SelfPlayEnv2  — firm 2 learns; firm 1 plays a frozen policy passed in.
Both are standard gym.Env subclasses — wrap with DummyVecEnv for SB3.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Callable
import sys

# sys.path.insert(0, "/mnt/user-data/outputs")
from bertrand_pricing_env import BertrandPricingEnv


class SelfPlayEnv(gym.Env):
    """
    Single-agent view of the Bertrand duopoly for self-play training.

    The learning agent is always 'firm 1' from this env's perspective.
    The opponent (firm 2) uses a frozen_policy callable that maps
    (obs) → action_index.

    Parameters
    ----------
    env_kwargs     : dict — passed to BertrandPricingEnv.__init__
    frozen_policy  : callable(obs) → int, or None (falls back to best-response)
    agent_id       : int (1 or 2) — which firm this env is training
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        env_kwargs    : dict,
        frozen_policy : Optional[Callable] = None,
        agent_id      : int = 1,
    ):
        super().__init__()
        self._base     = BertrandPricingEnv(**env_kwargs)
        self.frozen_policy = frozen_policy  # updated externally between swap phases
        self.agent_id  = agent_id

        # Mirror the base env's spaces
        self.action_space      = self._base.action_space
        self.observation_space = self._base.observation_space

        # Expose benchmarks
        self.nash_price     = self._base.nash_price
        self.monopoly_price = self._base.monopoly_price
        self.price_grid     = self._base.price_grid

    def reset(self, *, seed=None, options=None):
        obs, info = self._base.reset(seed=seed, options=options)
        return obs, info

    def step(self, action: int):
        """
        The learning agent takes `action` as firm 1.
        The frozen opponent responds as firm 2.
        """
        # Get opponent action — use frozen policy or fall back to best-response
        if self.frozen_policy is not None:
            obs_for_opp = self._base._get_obs()
            opp_action  = int(self.frozen_policy(obs_for_opp))
        else:
            # No frozen policy yet: opponent plays best-response (same as base env)
            opp_action = None   # base env handles it internally

        if opp_action is not None:
            # Override firm 2's price using the frozen policy's chosen action
            p2_new = float(self._base.price_grid[opp_action])
            # Step firm 1 with action, manually override firm 2
            p1_new = float(self._base.price_grid[action])
            shock1 = self._base.np_random.normal(0, self._base.noise_std) if self._base.noise_std > 0 else 0.0
            shock2 = self._base.np_random.normal(0, self._base.noise_std) if self._base.noise_std > 0 else 0.0
            pi1, pi2 = self._base._compute_profits(p1_new, p2_new, shock1, shock2)
            self._base._p1, self._base._p2 = p1_new, p2_new
            self._base._pi1, self._base._pi2 = pi1, pi2
            self._base._step += 1
            max_p = self._base._max_possible_profit()
            reward = float(pi1 / max_p) if max_p > 0 else float(pi1)
            truncated  = self._base._step >= self._base.max_steps
            terminated = False
            obs  = self._base._get_obs()
            info = self._base._get_info()
        else:
            obs, reward, terminated, truncated, info = self._base.step(action)

        return obs, reward, terminated, truncated, info

    def update_frozen_policy(self, policy_fn: Callable):
        """Called by SelfPlayTrainer to swap in updated opponent policy."""
        self.frozen_policy = policy_fn

    def render(self):
        return self._base.render()

    def close(self):
        return self._base.close()
