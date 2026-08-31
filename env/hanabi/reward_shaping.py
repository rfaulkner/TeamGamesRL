# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Intermediate reward shaping for full Hanabi games.

In full Hanabi the only reward signal is the final game score (0-25), which
makes credit assignment across a multi-turn episode difficult.  This module
provides lightweight per-turn shaping rewards derived from observable game
state transitions (firework progress, life/info token changes) so that an RL
agent can learn more efficiently.
"""

import dataclasses
import re


@dataclasses.dataclass(frozen=True)
class RewardShapingConfig:
    """Configuration for per-turn reward shaping signals.

    All shaping terms are additive.  Set ``enabled=False`` to disable shaping
    entirely (``compute_shaped_reward`` will return 0.0).
    """

    # Whether reward shaping is active.
    enabled: bool = False

    # Reward granted each time a firework pile advances by one card.
    firework_progress_reward: float = 0.25

    # Penalty incurred when a life token is lost (should be negative).
    life_loss_penalty: float = -0.5

    # Reward for recovering an information token (e.g. via a successful play).
    info_token_recovery_reward: float = 0.1

    # Penalty for discarding a card that can no longer be played (should be
    # negative).  See ``compute_shaped_reward`` for why this is not yet used.
    critical_discard_penalty: float = -0.3

    # Bonus awarded when a firework pile is completed (reaches 5).
    perfect_play_bonus: float = 1.0


def parse_fireworks(fireworks_str: str) -> dict[str, int]:
    """Parses a fireworks string into a colour -> height mapping.

    Args:
        fireworks_str: Space-separated tokens such as ``'R0 Y2 G1 B0 W0'``.

    Returns:
        A dict mapping each colour letter to its current pile height, e.g.
        ``{'R': 0, 'Y': 2, 'G': 1, 'B': 0, 'W': 0}``.
    """
    result: dict[str, int] = {}
    for token in fireworks_str.strip().split():
        colour = token[0]
        height = int(token[1:])
        result[colour] = height
    return result


def parse_observation_tokens(obs_str: str) -> dict[str, object]:
    """Extracts life tokens, info tokens, and fireworks from an observation.

    The observation string is expected to contain lines of the form::

        Life tokens: 3
        Info tokens: 6
        Fireworks: R0 Y0 G0 B0 W0

    Args:
        obs_str: A multi-line observation string.

    Returns:
        A dict with keys ``'life_tokens'`` (int), ``'info_tokens'`` (int),
        and ``'fireworks'`` (dict mapping colour str to int height).

    Raises:
        ValueError: If any of the expected fields cannot be found.
    """
    life_match = re.search(r"Life tokens:\s*(\d+)", obs_str)
    if life_match is None:
        raise ValueError(
            "Could not find 'Life tokens: <int>' in observation string."
        )

    info_match = re.search(r"Info tokens:\s*(\d+)", obs_str)
    if info_match is None:
        raise ValueError(
            "Could not find 'Info tokens: <int>' in observation string."
        )

    fw_match = re.search(r"Fireworks:\s*(.+)", obs_str)
    if fw_match is None:
        raise ValueError(
            "Could not find 'Fireworks: ...' in observation string."
        )

    return {
        "life_tokens": int(life_match.group(1)),
        "info_tokens": int(info_match.group(1)),
        "fireworks": parse_fireworks(fw_match.group(1)),
    }


def sum_fireworks(fireworks: dict[str, int]) -> int:
    """Returns the sum of all firework pile heights (i.e. the current score).

    Args:
        fireworks: Mapping from colour letter to pile height.

    Returns:
        The total score contribution from all firework piles.
    """
    return sum(fireworks.values())


def compute_shaped_reward(
    obs_before: str,
    obs_after: str,
    config: RewardShapingConfig,
) -> float:
    """Computes the intermediate shaped reward for a single action transition.

    The reward is the sum of several terms derived from the change in
    observable game state between ``obs_before`` and ``obs_after``:

    * **Firework progress**: ``config.firework_progress_reward`` for each
      colour whose pile height increased.
    * **Perfect play bonus**: ``config.perfect_play_bonus`` (in addition to
      the progress reward) for each colour that reached height 5.
    * **Life loss penalty**: ``config.life_loss_penalty`` if a life token
      was lost.
    * **Info token recovery**: ``config.info_token_recovery_reward`` if an
      information token was gained.

    Note:
        ``config.critical_discard_penalty`` is **not** applied here because
        detecting critical discards requires card identity information that is
        not available from the observation string alone.

    .. TODO:: Add critical-discard detection when the training loop passes
       additional card-identity context alongside observations.

    Args:
        obs_before: Observation string *before* the action was taken.
        obs_after: Observation string *after* the action was taken.
        config: Reward shaping configuration.

    Returns:
        The shaped reward for this transition.  Returns ``0.0`` when
        ``config.enabled`` is ``False``.
    """
    if not config.enabled:
        return 0.0

    state_before = parse_observation_tokens(obs_before)
    state_after = parse_observation_tokens(obs_after)

    fw_before: dict[str, int] = state_before["fireworks"]  # type: ignore[assignment]
    fw_after: dict[str, int] = state_after["fireworks"]  # type: ignore[assignment]
    life_before: int = state_before["life_tokens"]  # type: ignore[assignment]
    life_after: int = state_after["life_tokens"]  # type: ignore[assignment]
    info_before: int = state_before["info_tokens"]  # type: ignore[assignment]
    info_after: int = state_after["info_tokens"]  # type: ignore[assignment]

    reward = 0.0

    # Firework progress & perfect-play bonus.
    for colour in fw_before:
        if fw_after.get(colour, 0) > fw_before[colour]:
            reward += config.firework_progress_reward
            if fw_after[colour] == 5 and fw_before[colour] < 5:
                reward += config.perfect_play_bonus

    # Life token loss.
    if life_after < life_before:
        reward += config.life_loss_penalty

    # Information token recovery.
    if info_after > info_before:
        reward += config.info_token_recovery_reward

    return reward


def compute_episode_shaped_rewards(
    observation_sequence: list[str],
    config: RewardShapingConfig,
) -> list[float]:
    """Computes shaped rewards for every transition in an episode.

    This is a convenience wrapper around ``compute_shaped_reward`` that
    iterates over consecutive observation pairs.

    Args:
        observation_sequence: Observation strings in chronological order, one
            per environment step.  Must contain at least two entries for any
            rewards to be computed.
        config: Reward shaping configuration.

    Returns:
        A list of shaped rewards of length
        ``len(observation_sequence) - 1``.  Each element ``i`` is the shaped
        reward for the transition from step ``i`` to step ``i + 1``.
    """
    return [
        compute_shaped_reward(observation_sequence[i], observation_sequence[i + 1], config)
        for i in range(len(observation_sequence) - 1)
    ]
