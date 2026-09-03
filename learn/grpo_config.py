# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GRPO configuration dataclass.

Centralises every hyperparameter for the three GRPO variants (sampled,
exhaustive, phased curriculum) so that callers only need a single import.
"""

import dataclasses


@dataclasses.dataclass
class GRPOConfig:
  """Configuration for the GRPO training algorithm.

  Attributes:
    num_generations: Number of completions to sample per prompt in GRPO (group
      size K). Only used when ``exhaustive_groups`` is False.
    collect_episodes: Number of episodes to play for collecting game state
      prompts before each GRPO training pass. Only used when
      ``exhaustive_groups`` is False.
    train_epochs: Number of training epochs per GRPO pass.
    passes: Number of collect-then-train passes for GRPO.
    max_completion_length: Maximum completion length for GRPO generation.
    lr: Learning rate for the optimizer.
    kl_coeff: KL penalty coefficient.
    max_grad_norm: Maximum gradient norm for clipping.
    gradient_accumulation_steps: Number of groups to accumulate gradients over
      before taking an optimizer step. Defaults to 1 (step after every group).
    temperature: Sampling temperature for LLM action selection during data
      collection.
    num_eval_episodes: Number of episodes per evaluation round.
    per_player_updates: If True, group collected prompts by player_id and run a
      separate GRPO training step for each player group. This gives cleaner
      gradient signal in cooperative games where players have different
      information (e.g. Tiny Hanabi). Only used when ``exhaustive_groups`` is
      False.
    temperature_anneal_end: If set, linearly anneal the sampling temperature
      from ``temperature`` to this value over the course of training. Defaults
      to 0.7 (annealing from 1.2 to 0.7) for high early exploration with
      gradual convergence. Set to ``None`` for constant temperature.
    reward_variance_penalty: Coefficient for penalizing high-variance reward
      outcomes. The effective reward becomes ``mean_reward - penalty *
      std_reward`` when ``reward_num_simulations > 1``. Encourages convergence
      to consistent strategies. Only used when ``exhaustive_groups`` is False.
    reward_num_simulations: Number of independent game simulations to run per
      completion for estimating reward mean and variance. Only used when
      ``reward_variance_penalty > 0``.
    exhaustive_groups: If True, enumerate all possible game states and form
      GRPO groups where only the target player's action varies while all other
      variables (chance outcomes, other players' actions) are fixed. This
      produces deterministic, zero-variance advantage estimates by reading
      rewards directly from the game's payoff structure. For small games like
      Tiny Hanabi this covers the full game tree; for larger games, set this
      to False and use sampled rollouts instead.
    optimistic_reward_alpha: Blending weight for optimistic (max-over-partner)
      rewards in exhaustive-group GRPO.  For the first-acting player, rewards
      are computed as::

        reward = α * max_over_partner_actions(r) + (1 - α) * simulated(r)

      When α=1.0 (fully optimistic), the reward assumes the best possible
      partner cooperation, breaking the chicken-and-egg bootstrapping problem
      in signaling games.  When α=0.0, rewards reflect the partner's current
      policy.  The value is linearly annealed from ``optimistic_reward_alpha``
      toward ``optimistic_reward_alpha_min`` over training passes.
      Only used when ``exhaustive_groups`` is True.  Set to 0.0 to disable.
    optimistic_reward_alpha_min: Minimum floor for optimistic reward alpha
      annealing.  Maintains a baseline optimistic drive to prevent premature
      collapse into non-cooperative equilibria (such as the 8.0 fallback in
      Tiny Hanabi).  Defaults to 0.2.
  """

  num_generations: int = 8
  collect_episodes: int = 50
  train_epochs: int = 1
  passes: int = 25
  max_completion_length: int = 16
  lr: float = 3e-5
  kl_coeff: float = 0.05
  max_grad_norm: float = 1.0
  gradient_accumulation_steps: int = 1
  temperature: float = 1.2
  num_eval_episodes: int = 10
  per_player_updates: bool = True
  temperature_anneal_end: float | None = 0.7
  reward_variance_penalty: float = 0.0
  reward_num_simulations: int = 5
  exhaustive_groups: bool = False
  optimistic_reward_alpha: float = 1.0
  optimistic_reward_alpha_min: float = 0.2
  signal_entropy_coeff: float = 0.0
  """Coefficient for the cross-state signal entropy bonus.

  In signaling games, the first-acting player (P0) may converge to the
  same action for all dealt cards, collapsing into a non-signaling
  equilibrium (e.g. the 8.0 plateau in Tiny Hanabi).

  When ``signal_entropy_coeff > 0``, an additional loss term is added
  after each GRPO pass that maximizes the entropy of P0's *marginal*
  action distribution across all game states::

    π_marginal(a) = (1/N) Σ_s π(a | s)
    L_entropy = -signal_entropy_coeff * H(π_marginal)

  This encourages P0 to use different actions for different states,
  breaking symmetry without prescribing a specific signaling convention.

  Only used when ``exhaustive_groups`` is True.  A value of 0.1–0.5 is
  recommended.  The bonus applies only to player 0 (the signaler).
  """
  phased_training: bool = False
  """Enable phased (curriculum) training with per-player LoRA adapters.

  When True, training proceeds in phases:
    - Phase 1: Train P1 only against oracle-best P0 actions.
    - Phase 2: Freeze P1 and train P0 against P1's learned policy.
    - Phase 3 (optional): Joint fine-tuning with both adapters.

  Each player gets its own LoRA adapter so gradients don't interfere.
  Only used when ``exhaustive_groups`` is True.
  """
  phase1_max_passes: int = 50
  """Maximum passes for Phase 1 (P1 training against oracle P0)."""
  phase2_max_passes: int = 50
  """Maximum passes for Phase 2 (P0 training against frozen P1)."""
  phase3_max_passes: int = 10
  """Maximum passes for Phase 3 (joint fine-tuning). Set to 0 to skip."""
  convergence_patience: int = 5
  """Stop a phase early if eval reward hasn't improved by more than
  ``convergence_min_delta`` for this many consecutive passes."""
  convergence_min_delta: float = 0.1
  """Minimum reward improvement to reset the patience counter."""

  # ── Multi-turn episode settings ──────────────────────────────────────
  # These parameters control how GRPO handles longer, multi-turn games
  # where exhaustive enumeration is infeasible.  They are only used by
  # the sampled GRPO path (``exhaustive_groups = False``).

  pivot_decisions_per_episode: int = 5
  """Maximum number of decision points to resample per collected episode.

  In multi-turn games, each episode contains many decision points.
  Resampling all of them is computationally expensive (each requires
  simulating the game to completion K times).  Instead, select a subset
  of high-information "pivot" decisions per episode for GRPO groups.

  Set to 0 or a very large value to resample all decision points
  (original behaviour).  Only used when ``exhaustive_groups`` is False.
  """

  decision_priority_sampling: bool = True
  """Weight pivot-point selection toward high-information decisions.

  When True, decision points with more diverse legal actions (e.g. hint
  decisions in Hanabi) are more likely to be selected as pivot points.
  When False, pivot points are selected uniformly at random.

  Only used when ``pivot_decisions_per_episode > 0``.
  """

  truncated_rollout_horizon: int | None = None
  """If set, simulate only this many turns ahead (instead of to terminal
  state) when computing rewards for alternative actions.

  Reduces the cost of reward simulation in long games.  The reward for
  a truncated rollout is the intermediate score at the truncation point.
  ``None`` means simulate to the terminal state (original behaviour).
  """

  reward_simulation_mode: str = 'rollout'
  """How to compute rewards for GRPO training.

  Options:
    'rollout'     -- Random playout for ``truncated_rollout_horizon``
                    turns (default 6), then evaluate the resulting state
                    with a game-specific heuristic (e.g. Hanabi score +
                    discounted potential).  Fast (~1 ms) with decent
                    signal.  Best default for multi-turn games.
    'random'      -- Random legal actions all the way to terminal state.
                    Very fast (~1 ms) but noisier.  Works well with
                    ``reward_num_simulations > 1`` to reduce variance.
    'llm'         -- Use the model (with frozen LoRA) for all remaining
                    turns.  Most accurate but very slow (~18 sec per eval
                    for 12B model).
    'heuristic'   -- Use a rule-based player (Hanabi-only).  Moderate
                    speed and accuracy.  Falls back to 'random' if no
                    heuristic is available for the current game.
    'dense'       -- Dense per-action reward shaping (Hanabi-only).  No
                    simulation at all: evaluates the immediate quality
                    of the chosen action (play success, discard safety,
                    hint informativeness).  Zero variance, ~100x faster
                    than rollout.  Best for initial training phases.
    'dense_chain' -- Dense rewards over a short heuristic continuation
                    (controlled by ``truncated_rollout_horizon``,
                    default 4 turns).  Addresses myopic play concerns
                    by including discounted rewards for subsequent
                    heuristic actions.  Good balance of density and
                    strategic context.
  """

  dense_chain_discount: float = 0.9
  """Discount factor for chained dense reward evaluation.

  When ``reward_simulation_mode='dense_chain'``, future action
  rewards are discounted by this factor per turn:
  ``r_0 + gamma*r_1 + gamma^2*r_2 + ...``  Higher values (closer to 1.0)
  weight future actions more equally; lower values focus on the
  immediate action.
  """

