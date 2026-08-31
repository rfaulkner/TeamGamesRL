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

"""Hanabi-specific strategic competence metrics.

Provides evaluation metrics that go beyond raw game score to measure *how*
an agent plays Hanabi. Metrics cover action distributions, play success/risk,
hint efficiency, discard safety, and an aggregate strategic competence profile.
"""

import collections
import math
import re

# Precompiled patterns for OpenSpiel Hanabi action strings.
_PLAY_RE = re.compile(r'^\(Play \d+\)$')
_DISCARD_RE = re.compile(r'^\(Discard \d+\)$')
_HINT_COLOR_RE = re.compile(r'^\(Reveal player \+\d+ color [A-Z]\)$')
_HINT_RANK_RE = re.compile(r'^\(Reveal player \+\d+ rank \d+\)$')

_ACTION_TYPES = ('play', 'discard', 'hint_color', 'hint_rank')


def classify_action(action_str: str) -> str:
    """Classifies an OpenSpiel Hanabi action string into a category.

    Args:
        action_str: An OpenSpiel-formatted action string, e.g.
            ``(Play 0)``, ``(Discard 2)``,
            ``(Reveal player +1 color R)``, ``(Reveal player +1 rank 3)``.

    Returns:
        One of ``'play'``, ``'discard'``, ``'hint_color'``, or ``'hint_rank'``.

    Raises:
        ValueError: If ``action_str`` does not match any recognized format.
    """
    if _PLAY_RE.match(action_str):
        return 'play'
    if _DISCARD_RE.match(action_str):
        return 'discard'
    if _HINT_COLOR_RE.match(action_str):
        return 'hint_color'
    if _HINT_RANK_RE.match(action_str):
        return 'hint_rank'
    raise ValueError(f'Unrecognized Hanabi action string: {action_str!r}')


def compute_action_distribution(
    action_strings: list[str],
) -> dict[str, float]:
    """Computes the normalized action-type frequency distribution and entropy.

    Args:
        action_strings: Sequence of OpenSpiel Hanabi action strings from one
            or more episodes.

    Returns:
        A dict with keys ``'play'``, ``'discard'``, ``'hint_color'``,
        ``'hint_rank'`` mapping to their normalized frequencies (summing to
        1.0), plus an ``'action_type_entropy'`` key giving the Shannon
        entropy in bits (log base 2) over the four categories.  If
        ``action_strings`` is empty, all frequencies are 0.0 and entropy
        is 0.0.
    """
    counts: dict[str, int] = collections.Counter(
        classify_action(a) for a in action_strings
    )
    total = len(action_strings)

    distribution: dict[str, float] = {}
    for action_type in _ACTION_TYPES:
        distribution[action_type] = counts.get(action_type, 0) / total if total else 0.0

    entropy = 0.0
    if total:
        for action_type in _ACTION_TYPES:
            p = distribution[action_type]
            if p > 0.0:
                entropy -= p * math.log2(p)
    distribution['action_type_entropy'] = entropy

    return distribution


def compute_play_metrics(
    episode_data: list[dict],
) -> dict[str, float]:
    """Computes play-action metrics from a single episode.

    Args:
        episode_data: List of step dicts, each containing at minimum:
            - ``action_str`` (str): The OpenSpiel action string.
            - ``life_tokens_before`` (int): Life tokens before the step.
            - ``life_tokens_after`` (int): Life tokens after the step.
            - ``fireworks_before`` (dict[str, int]): Fireworks pile heights
              before the step, keyed by color letter.
            - ``fireworks_after`` (dict[str, int]): Fireworks pile heights
              after the step, keyed by color letter.

    Returns:
        A dict with:
            - ``play/rate``: Fraction of all steps that are plays.
            - ``play/success_rate``: Fraction of plays that advanced a
              firework (i.e. any color's pile height increased).
            - ``play/life_loss_rate``: Fraction of plays that lost a life
              token.
    """
    total_steps = len(episode_data)
    play_count = 0
    success_count = 0
    life_loss_count = 0

    for step in episode_data:
        if classify_action(step['action_str']) != 'play':
            continue
        play_count += 1

        fireworks_before = step['fireworks_before']
        fireworks_after = step['fireworks_after']
        if any(
            fireworks_after.get(c, 0) > fireworks_before.get(c, 0)
            for c in fireworks_after
        ):
            success_count += 1

        if step['life_tokens_after'] < step['life_tokens_before']:
            life_loss_count += 1

    return {
        'play/rate': play_count / total_steps if total_steps else 0.0,
        'play/success_rate': success_count / play_count if play_count else 0.0,
        'play/life_loss_rate': life_loss_count / play_count if play_count else 0.0,
    }


def compute_hint_metrics(
    episode_data: list[dict],
) -> dict[str, float]:
    """Computes hint-action metrics from a single episode.

    Args:
        episode_data: List of step dicts, each containing at minimum:
            - ``action_str`` (str): The OpenSpiel action string.
            - ``subsequent_partner_actions`` (list[str]): The next 2 action
              strings performed by the hinted player after receiving the hint.

    Returns:
        A dict with:
            - ``hint/rate``: Fraction of all steps that are hints (color or
              rank).
            - ``hint/efficiency``: Fraction of hints where at least one of
              the ``subsequent_partner_actions`` is classified as a play.
    """
    total_steps = len(episode_data)
    hint_count = 0
    effective_hint_count = 0

    for step in episode_data:
        action_type = classify_action(step['action_str'])
        if action_type not in ('hint_color', 'hint_rank'):
            continue
        hint_count += 1

        subsequent = step.get('subsequent_partner_actions', [])
        if any(classify_action(a) == 'play' for a in subsequent):
            effective_hint_count += 1

    return {
        'hint/rate': hint_count / total_steps if total_steps else 0.0,
        'hint/efficiency': (
            effective_hint_count / hint_count if hint_count else 0.0
        ),
    }


def compute_discard_metrics(
    episode_data: list[dict],
) -> dict[str, float]:
    """Computes discard-action metrics from a single episode.

    Args:
        episode_data: List of step dicts, each containing at minimum:
            - ``action_str`` (str): The OpenSpiel action string.
            - ``discarded_card`` (str): Card notation like ``'R3'`` where the
              first character is the color and the rest is the rank.
            - ``fireworks_state`` (dict[str, int]): Current fireworks pile
              heights at the time of the discard, keyed by color letter.
            - ``remaining_copies`` (int): Number of copies of the discarded
              card remaining in the deck/hands *after* the discard (0 means
              no copies left).

    Returns:
        A dict with:
            - ``discard/rate``: Fraction of all steps that are discards.
            - ``discard/critical_rate``: Among discards, fraction where
              ``remaining_copies == 0`` AND the card is still needed
              (fireworks pile for that color is below the card's rank).
            - ``discard/safe_rate``: Among discards, ``1 - critical_rate``.
    """
    total_steps = len(episode_data)
    discard_count = 0
    critical_count = 0

    for step in episode_data:
        if classify_action(step['action_str']) != 'discard':
            continue
        discard_count += 1

        card = step['discarded_card']
        color = card[0]
        rank = int(card[1:])
        fireworks_height = step['fireworks_state'].get(color, 0)

        if step['remaining_copies'] == 0 and fireworks_height < rank:
            critical_count += 1

    critical_rate = critical_count / discard_count if discard_count else 0.0
    return {
        'discard/rate': discard_count / total_steps if total_steps else 0.0,
        'discard/critical_rate': critical_rate,
        'discard/safe_rate': 1.0 - critical_rate,
    }


def compute_strategic_competence(
    episode_data: list[dict],
    final_score: int,
    random_baseline_score: float = 2.5,
) -> dict[str, float]:
    """Computes a comprehensive strategic-competence profile for one episode.

    Aggregates all per-category metrics (play, hint, discard, action
    distribution) and adds score-level metrics.  Gracefully skips metric
    categories when the required keys are missing from the step dicts.

    Args:
        episode_data: List of step dicts.  Each dict must contain at least
            ``action_str``.  Additional keys enable further metric categories
            (see individual ``compute_*`` functions for required keys).
        final_score: The final Hanabi game score (0–25).
        random_baseline_score: Expected score of a uniformly random agent,
            used to compute ``score/cooperation``.  Defaults to 2.5.

    Returns:
        A flat dict mapping metric names (e.g. ``'play/success_rate'``,
        ``'score/final'``) to float values.
    """
    metrics: dict[str, float] = {}

    # Score metrics (always available).
    metrics['score/final'] = float(final_score)
    metrics['score/cooperation'] = float(final_score) - random_baseline_score

    # Action distribution (requires only 'action_str').
    action_strings = [
        step['action_str'] for step in episode_data if 'action_str' in step
    ]
    if action_strings:
        metrics.update(compute_action_distribution(action_strings))

    # Play metrics.
    _required_play_keys = {
        'action_str',
        'life_tokens_before',
        'life_tokens_after',
        'fireworks_before',
        'fireworks_after',
    }
    if episode_data and _required_play_keys.issubset(episode_data[0].keys()):
        metrics.update(compute_play_metrics(episode_data))

    # Hint metrics.
    _required_hint_keys = {'action_str', 'subsequent_partner_actions'}
    if episode_data and _required_hint_keys.issubset(episode_data[0].keys()):
        metrics.update(compute_hint_metrics(episode_data))

    # Discard metrics.
    _required_discard_keys = {
        'action_str',
        'discarded_card',
        'fireworks_state',
        'remaining_copies',
    }
    if episode_data and _required_discard_keys.issubset(episode_data[0].keys()):
        metrics.update(compute_discard_metrics(episode_data))

    return metrics


def summarize_competence_report(metrics: dict[str, float]) -> str:
    """Formats a strategic-competence metrics dict as a human-readable report.

    Metrics are grouped by their category prefix (e.g. ``score/``, ``play/``,
    ``hint/``, ``discard/``).  Metrics without a ``/`` separator are grouped
    under a ``general`` category.

    Args:
        metrics: A flat dict of metric names to float values, as returned by
            :func:`compute_strategic_competence`.

    Returns:
        A formatted multi-line string suitable for logging.
    """
    # Group metrics by category prefix.
    groups: dict[str, list[tuple[str, float]]] = collections.defaultdict(list)
    for key, value in sorted(metrics.items()):
        if '/' in key:
            prefix, suffix = key.split('/', 1)
            groups[prefix].append((suffix, value))
        else:
            groups['general'].append((key, value))

    # Desired display order; unlisted categories follow alphabetically.
    category_order = ['score', 'play', 'hint', 'discard', 'general']
    ordered_categories = [c for c in category_order if c in groups]
    for cat in sorted(groups):
        if cat not in ordered_categories:
            ordered_categories.append(cat)

    lines: list[str] = []
    lines.append('=== Hanabi Strategic Competence Report ===')
    for category in ordered_categories:
        lines.append('')
        lines.append(f'[{category}]')
        for name, value in groups[category]:
            lines.append(f'  {name:30s} {value:>10.4f}')

    return '\n'.join(lines)
