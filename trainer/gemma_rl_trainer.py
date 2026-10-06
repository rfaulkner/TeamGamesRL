"""Gemma 2B RL trainer for multi-agent OpenSpiel team games.

This is the CLI entry point that wires together:
  - The Gemma LLM backend (backend/gemma_backend.py).
  - The RL training algorithms (learn/reinforce.py, learn/grpo.py).
  - The model-agnostic training orchestrator (trainer.py).
  - The OpenSpiel game environments (env/).

Usage:
  python gemma_rl_trainer.py --game=tiny_hanabi --num_episodes=500
  python gemma_rl_trainer.py --game=hanabi --lora_rank=32 --lr=5e-5
  python gemma_rl_trainer.py --rl_algorithm=grpo --game=tiny_hanabi
"""

import dataclasses
import os
import sys
from typing import Any

# Prevent PyTorch CUDA memory fragmentation on large allocations
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

# Ensure project root is in sys.path when running as a package or binary
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
  sys.path.insert(0, _REPO_ROOT)

# In Google3, open_spiel is located under third_party.open_spiel.
# Use sys.modules aliasing instead of adding third_party to sys.path,
# which would corrupt package discovery for non-Python directories in third_party (e.g. brotli).
if 'open_spiel' not in sys.modules:
  try:
    import open_spiel  # pylint: disable=unused-import
  except ImportError:
    try:
      import third_party.open_spiel as _os
      sys.modules['open_spiel'] = _os
      import third_party.open_spiel.python as _osp
      sys.modules['open_spiel.python'] = _osp
    except ImportError:
      try:
        import google3.third_party.open_spiel as _os
        sys.modules['open_spiel'] = _os
        import google3.third_party.open_spiel.python as _osp
        sys.modules['open_spiel.python'] = _osp
      except ImportError:
        pass

from absl import app
from absl import flags
from absl import logging
from env.game_config import AVAILABLE_GAMES
import numpy as np
import torch

FLAGS = flags.FLAGS

# ============================================================================
# Flags
# ============================================================================

flags.DEFINE_enum(
    'rl_algorithm',
    'grpo',
    ['reinforce', 'grpo'],
    'RL algorithm to use. "reinforce" uses the hand-rolled REINFORCE '
    'with baseline + KL penalty. "grpo" uses TRL\'s GRPOTrainer.',
)
flags.DEFINE_enum(
    'game',
    'tiny_hanabi',
    AVAILABLE_GAMES,
    'Name of the OpenSpiel game to train on.',
)
flags.DEFINE_integer(
    'num_episodes', 500, 'Total number of training episodes to run.'
)
flags.DEFINE_integer(
    'eval_every', 50, 'Run evaluation every this many episodes.'
)
flags.DEFINE_integer(
    'num_eval_episodes', 10, 'Number of episodes per evaluation round.'
)
flags.DEFINE_integer(
    'eval_batch_size',
    4,
    'Batch size for parallel evaluation episodes. Running multiple evaluation '
    'games in lockstep batches GPU generations for ~4x faster evaluation.',
)
flags.DEFINE_integer(
    'eval_full_episodes',
    32,
    'Episodes of full self-play evaluation (LLM plays every turn, no bot '
    'hand-off) run after every GRPO pass, logged as eval_full/* (see '
    '--eval_mode). Also sets the size of the one-off pass-0 baselines '
    '(initial adapter full self-play + all-bot). 0 disables.',
)
flags.DEFINE_enum(
    'eval_mode',
    'full',
    ['full', 'bot_guided', 'both'],
    'Which per-pass evaluation(s) to run after each GRPO pass. "full" '
    '(default): only the fixed-deal full self-play eval (eval_full/*, '
    '--eval_full_episodes games). "bot_guided": only the curriculum eval '
    '(eval/*, --num_eval_episodes games; LLM plays to the curriculum horizon, '
    'the heuristic bot finishes the game). "both": run both (previous '
    'behaviour). The pass-0 baselines and eval/collection_mean_reward are '
    'unaffected.',
)
flags.DEFINE_integer(
    'eval_seed',
    10000,
    'Base seed for fixed-deal evaluation: eval episode i is dealt with seed '
    'eval_seed + i so every pass is scored on the same deals (paired '
    'comparison). Negative disables seeding (fresh random deals per pass).',
)
flags.DEFINE_boolean(
    'eval_bot_baseline',
    True,
    'Before pass 1, also evaluate the heuristic bot in both seats for every '
    'turn on the fixed eval deals (eval_bot/*, logged at episode 0).',
)
flags.DEFINE_float(
    'temperature', 0.8, 'Sampling temperature for LLM action selection.'
)
flags.DEFINE_float(
    'temperature_anneal_end',
    None,
    'If set, linearly anneal temperature to this value over training. '
    'E.g. --temperature=1.2 --temperature_anneal_end=0.7 anneals from 1.2 '
    'to 0.7 over the course of GRPO passes.',
)
flags.DEFINE_float(
    'temperature_floor',
    0.5,
    'Minimum floor for annealed temperature to prevent generation collapse.',
)
flags.DEFINE_float('lr', 3e-5, 'Learning rate for the LoRA adapter.')
flags.DEFINE_integer('lora_rank', 16, 'LoRA adapter rank.')
flags.DEFINE_integer('lora_alpha', 32, 'LoRA scaling alpha.')
flags.DEFINE_float('lora_dropout', 0.05, 'LoRA dropout probability.')
flags.DEFINE_string(
    'model_name', 'google/gemma-2-2b', 'HuggingFace model ID for Gemma 2B.'
)
flags.DEFINE_bool(
    'use_4bit', True, 'Use 4-bit NF4 quantization for the base model.'
)
flags.DEFINE_string(
    'initial_lora_checkpoint',
    None,
    'Path to a pre-trained LoRA adapter to load for warm-start initialization.',
)
flags.DEFINE_string(
    'output_dir',
    '/tmp/teamgamesrl',
    'Directory for checkpoints, logs, and metrics.',
)
flags.DEFINE_integer(
    'log_every', 10, 'Log training metrics every this many episodes.'
)
flags.DEFINE_integer(
    'checkpoint_every', 100, 'Save a LoRA checkpoint every this many episodes.'
)
flags.DEFINE_integer('seed', 42, 'Random seed for reproducibility.')
flags.DEFINE_integer(
    'max_seq_len', 2048, 'Maximum sequence length for the model.'
)
flags.DEFINE_float('max_grad_norm', 1.0, 'Maximum gradient norm for clipping.')
flags.DEFINE_bool('use_wandb', False, 'Enable Weights & Biases logging.')
flags.DEFINE_string('wandb_project', 'TeamGamesRL', 'Wandb project name.')
flags.DEFINE_integer(
    'log_episodes_every',
    10,
    'Log full episode transcripts (game state + LLM responses) every this '
    'many episodes. Set to 0 to disable.',
)
flags.DEFINE_float(
    'kl_coeff',
    0.05,
    'KL penalty coefficient against the reference (pre-trained) model. '
    'Prevents mode collapse and language degradation.',
)
flags.DEFINE_integer(
    'gradient_accumulation_steps',
    8,
    'Number of episodes to accumulate gradients over before '
    'updating the model. Reduces REINFORCE variance by ~sqrt(N).',
)
flags.DEFINE_integer(
    'baseline_window_size',
    50,
    'Number of recent episodes to use for the reward baseline '
    '(sliding window mean). Replaces the EMA baseline.',
)

# GRPO-specific flags.
flags.DEFINE_integer(
    'grpo_num_generations',
    8,
    'Number of completions to sample per prompt in GRPO (group size K).',
)
flags.DEFINE_integer(
    'grpo_collect_episodes',
    50,
    'Number of episodes to play for collecting game state prompts '
    'before each GRPO training pass.',
)
flags.DEFINE_integer(
    'grpo_collect_batch_size',
    16,
    'Batch size for parallel game collection episodes during GRPO.',
)
flags.DEFINE_integer(
    'grpo_prompt_batch_size',
    40,
    'Batch size for parallel LLM prompt generation during GRPO training.',
)
flags.DEFINE_integer(
    'grpo_train_batch_size',
    4,
    'Maximum per-device batch size for TRL GRPOTrainer backward passes.',
)
flags.DEFINE_integer(
    'grpo_train_epochs', 1, 'Number of training epochs per GRPO pass.'
)
flags.DEFINE_integer(
    'grpo_passes', 25, 'Number of collect-then-train passes for GRPO.'
)
flags.DEFINE_integer(
    'grpo_initial_pass',
    0,
    'Pass to start from (1-based). 0 = auto-detect from existing checkpoints.',
)
flags.DEFINE_integer(
    'grpo_max_completion_length',
    16,
    'Maximum completion length for GRPO generation.',
)
flags.DEFINE_bool(
    'grpo_exhaustive_groups',
    False,
    'If True, enumerate all possible game states and form GRPO groups '
    'where only the target player\'s action varies. Produces deterministic '
    'advantage estimates. Best for small games (e.g. tiny_hanabi).',
)
flags.DEFINE_float(
    'grpo_optimistic_alpha',
    1.0,
    'Blending weight for optimistic (max-over-partner) rewards. '
    'When 1.0, P0 rewards assume best possible partner cooperation. '
    'Linearly annealed toward alpha_min over training. Only used with '
    '--grpo_exhaustive_groups.',
)
flags.DEFINE_float(
    'grpo_optimistic_alpha_min',
    0.2,
    'Minimum floor for optimistic reward alpha annealing. '
    'Prevents premature collapse into uncoordinated equilibria (e.g. 8.0). '
    'Only used with --grpo_exhaustive_groups.',
)
flags.DEFINE_float(
    'grpo_signal_entropy_coeff',
    0.0,
    'Coefficient for the cross-state signal entropy bonus on P0. '
    'Encourages P0 to use different actions for different cards by '
    'maximizing the entropy of the marginal action distribution across '
    'game states. 0.1-0.5 recommended for Tiny Hanabi. '
    'Only used with --grpo_exhaustive_groups.',
)
flags.DEFINE_bool(
    'grpo_phased_training',
    False,
    'Enable phased (curriculum) training with per-player LoRA adapters. '
    'Phase 1: train P1 against oracle-best P0. Phase 2: freeze P1, train P0. '
    'Phase 3: optional joint fine-tuning. '
    'Only used with --grpo_exhaustive_groups.',
)
flags.DEFINE_integer(
    'grpo_phase1_passes',
    50,
    'Maximum passes for Phase 1 (P1 training against oracle P0).',
)
flags.DEFINE_integer(
    'grpo_phase2_passes',
    50,
    'Maximum passes for Phase 2 (P0 training against frozen P1).',
)
flags.DEFINE_integer(
    'grpo_phase3_passes',
    10,
    'Maximum passes for Phase 3 (joint fine-tuning). Set to 0 to skip.',
)
flags.DEFINE_integer(
    'grpo_convergence_patience',
    5,
    'Stop a phase early if eval reward has not improved for this many '
    'consecutive passes.',
)
flags.DEFINE_float(
    'grpo_convergence_min_delta',
    0.1,
    'Minimum improvement in eval reward to reset the convergence patience '
    'counter.',
)
# Multi-turn episode flags.
flags.DEFINE_integer(
    'grpo_pivot_decisions_per_episode',
    5,
    'Maximum number of decision points to resample per episode. '
    'Reduces simulation cost in long multi-turn games. '
    'Set to 0 to resample all decision points (original behaviour).',
)
flags.DEFINE_bool(
    'grpo_decision_priority_sampling',
    True,
    'Weight pivot-point selection toward high-information decisions '
    '(e.g. hint decisions in Hanabi) rather than uniform random.',
)
flags.DEFINE_integer(
    'grpo_truncated_rollout_horizon',
    None,
    'If set, simulate only this many turns ahead when computing rewards '
    'for alternative actions. None means simulate to terminal state.',
)
flags.DEFINE_string(
    'reward_simulation_mode',
    'rollout',
    'How to compute rewards for GRPO training. '
    "'rollout' = random playout for k turns + heuristic eval (default, fast + good signal), "
    "'random' = random legal actions to terminal (~1ms/eval), "
    "'llm' = use model (accurate, ~18s/eval), "
    "'heuristic' = rule-based player (Hanabi-only), "
    "'dense' = per-action reward shaping (no simulation, ~100x faster), "
    "'dense_chain' = dense rewards over a short heuristic continuation.",
)
flags.DEFINE_integer(
    'max_history_turns',
    20,
    'Maximum number of recent moves to include in the Hanabi prompt. '
    'Set to 0 to show all moves (no truncation). '
    'Default 20 covers the last ~10 turns per player in 2-player Hanabi.',
)
flags.DEFINE_float(
    'dense_chain_discount',
    0.9,
    'Discount factor for chained dense reward evaluation. '
    'Only used with --reward_simulation_mode=dense_chain. '
    'Higher values (closer to 1.0) weight future actions more equally.',
)
flags.DEFINE_float(
    'reward_blend_weight',
    0.0,
    'Weight for blending game-outcome reward with primary reward signal. '
    'Effective reward = (1-w)*primary + w*game_score/25. '
    'Set > 0 to anchor training to actual game outcomes and prevent '
    'proxy reward hacking. Recommended: 0.2-0.5. Default 0.0 (disabled). '
    'At w=1.0 the primary (dense-chain) term is skipped entirely, which is '
    'both cheaper and a clean pure-rollout objective.',
)
flags.DEFINE_integer(
    'reward_rollout_samples',
    1,
    'Number of heuristic rollouts to average when computing the blended '
    'game-outcome reward. The heuristic partner is stochastic, so a single '
    'rollout is a noisy value estimate; averaging N reduces that noise by '
    'sqrt(N) at N x rollout cost (rollouts are milliseconds against a '
    '~16-18 s LLM generation step). Default 1 (no averaging).',
)
flags.DEFINE_bool(
    'reward_rollout_common_seed',
    True,
    'Use common random numbers across a GRPO group when rolling out. Every '
    'completion in a group shares one prompt, so all candidate actions are '
    'scored against the same heuristic-partner RNG stream, removing '
    'partner randomness from the within-group comparison that GRPO '
    'actually differentiates. Free variance reduction; default True.',
)
flags.DEFINE_float(
    'reward_survival_exponent',
    0.0,
    'Convex penalty on spent life tokens: the rollout score is multiplied '
    'by (lives_after / max_life_tokens) ** exponent. SafePlayPlayer never '
    'bombs, so a rollout is provably invariant to lives remaining -- the '
    'reward is blind to the first two bombs and puts the whole penalty on '
    'the third. This restores the missing gradient. 2.0 gives multipliers '
    '1.00 / 0.44 / 0.11 for 3 / 2 / 1 lives. Default 0.0 (disabled).',
)
flags.DEFINE_float(
    'reward_turn_discount',
    1.0,
    'Per-turn discount gamma on the heuristic rollout score: the reward '
    'becomes gamma ** turns_to_terminal * score. Undiscounted, the reward '
    'is deal-dominated and nearly flat across a group, and the only action '
    'class with real downside is playing (only a play can bomb) -- so the '
    'argmax is "never play" and runs collapse into a hint/discard loop that '
    'runs to deck exhaustion at score 0. Discounting makes a successful '
    'play win twice: higher score AND fewer turns left. Must be '
    'multiplicative, not an additive per-turn cost, or bombing out early '
    'would outscore a slow positive result. Try 0.95-0.99. '
    'Default 1.0 (disabled).',
)
flags.DEFINE_integer(
    'reward_policy_turns',
    1,
    'Number of turns played by the policy at the head of a reward rollout '
    '(m). The candidate action is turn 1; turns 2..m are sampled from the '
    'frozen policy, alternating players, before SafePlayPlayer finishes the '
    'game. Odd m ends the policy segment on the acting player, so it has '
    'responded to one partner move -- the shortest rollout in which a hint '
    'can pay off. m=1 is the previous behaviour, m=2 equals the old '
    '--llm_partner_response. Cost is ~linear in m: K=8 at m=3 is 24 '
    'generations per group versus 8. Default 1.',
)
flags.DEFINE_string(
    'grpo_scale_rewards',
    'batch',
    "How TRL turns rewards into advantages: 'group' (divide by the "
    "within-group std, TRL's default), 'batch' (divide by the batch-wide "
    "std), or 'none' (no division). 'group' rescales a group of "
    'near-identical rewards -- pure rollout noise -- to the same +/-1 '
    'advantages as a group with real spread, so it hands noise '
    "full-magnitude gradients. Default 'batch'. Older TRL releases type "
    'this as a bool; the trainer probes and falls back automatically.',
)
flags.DEFINE_bool(
    'constrained_action_types',
    False,
    'Force action-type diversity in GRPO groups via constrained decoding. '
    'A fraction of K completions will have their first token forced to '
    'Play/Discard/Hint to ensure all action types are represented. '
    'Prevents group collapse to a single action type.',
)
flags.DEFINE_bool(
    'strategic_action_selection',
    False,
    'Use game-state-aware strategic action injection in GRPO groups. '
    'Replaces naive first-token forcing with a strategic selector that '
    'forces specific complete actions (safe plays, risky plays, smart '
    'discards, diverse hints) based on the full game state. '
    'Takes precedence over --constrained_action_types when both are True.',
)
flags.DEFINE_enum(
    'strategic_action_mode',
    'substitute',
    ['substitute', 'logits'],
    'How strategic actions enter the GRPO group. "substitute" (default) '
    'samples all K completions freely, de-duplicates them by parsed '
    'action, then overwrites the duplicate and unparseable slots with '
    'strategic actions not already in the group. "logits" is the original '
    'behaviour: force actions into fixed slots during generation, before '
    'seeing what the policy would have sampled.',
)
flags.DEFINE_float(
    'strategic_action_forced_ratio',
    1.0,
    'Fraction of K completions strategic actions may occupy (0.0 to 1.0). '
    'Default 1.0. Under mode=logits this many slots are forced '
    'unconditionally; under mode=substitute it is only a cap on how many '
    'duplicate slots may be rewritten.',
)
flags.DEFINE_bool(
    'llm_partner_response',
    False,
    'Sample one LLM partner response (frozen weights) before heuristic '
    'rollout in dense_chain reward computation. Closes the train-eval gap '
    'from using SafePlayPlayer vs LLM as partner. Adds ~10-15%% overhead.',
)
flags.DEFINE_bool(
    'bot_partner',
    False,
    'Partner with a bot during collection and evaluation. '
    'Collection alternates roles (odd: P0=LLM, P1=Bot; even: P0=Bot, P1=LLM) '
    'and evaluation splits into LLM+Bot, Bot+LLM, and LLM+LLM sets.',
)
flags.DEFINE_string(
    'bot_type',
    'belief_lookahead',
    'Type of bot partner to use: "belief_lookahead" (SafeBeliefLookaheadPlayer) '
    'or "safe_play" (SafePlayPlayer).',
)
flags.DEFINE_float(
    'checkpoint_interval_minutes',
    10.0,
    'Interval in minutes for saving interim checkpoints and decision-point eval caches.',
)
flags.DEFINE_bool(
    'reasoning',
    False,
    'Prompt the LLM to think step-by-step inside <think>...</think> before '
    'selecting an action (Chain of Thought single-pass).',
)
flags.DEFINE_integer(
    'curriculum_window_size',
    4,
    'Window size in turns for sliding-window curriculum. Set to 0 to disable.',
)
flags.DEFINE_integer(
    'curriculum_passes_per_phase',
    2,
    'Number of GRPO passes per curriculum phase before advancing horizon.',
)
flags.DEFINE_integer(
    'curriculum_max_horizon',
    0,
    'Maximum turn horizon for curriculum training (0 = uncapped, train to end of game).',
)
flags.DEFINE_integer(
    'curriculum_max_lookback',
    -1,
    'Max turn lookback from active window for replay sampling '
    '(-1 = unlimited, 0 = no replay, N = last N turns).',
)
flags.DEFINE_float(
    'curriculum_replay_ratio',
    0.30,
    'Fraction of decision points sampled from earlier curriculum phases.',
)
# ============================================================================
# Entry point & Configuration Logging
# ============================================================================


def _get_experiment_flags() -> tuple[dict[str, Any], dict[str, Any]]:
  """Extracts all module flags and explicitly set CLI flags.

  Returns:
    (all_flags, explicit_flags) dictionaries mapping flag name to value.
  """
  all_flags = {}
  explicit_flags = {}
  for mod, flag_list in FLAGS.flags_by_module_dict().items():
    if mod in (__name__, '__main__', sys.argv[0]) or 'gemma_rl_trainer' in str(mod):
      for f in flag_list:
        all_flags[f.name] = f.value
        if f.present:
          explicit_flags[f.name] = f.value

  # Fallback: if empty for any reason, collect all non-internal absl flags
  if not all_flags:
    for name in FLAGS:
      if not name.startswith(
          ('log', 'run_with', 'pdb', 'test', 'xml', 'help', 'undefok', 'profile_file')
      ):
        f = FLAGS[name]
        all_flags[name] = f.value
        if f.present:
          explicit_flags[name] = f.value

  return all_flags, explicit_flags


def _print_experiment_configuration(
    all_flags: dict[str, Any], explicit_flags: dict[str, Any]
) -> None:
  """Prints structured flags and parameters to stdout (captured in .out log)."""
  print('=' * 80, flush=True)
  print(' TeamGamesRL — Run Configuration & Hyperparameters', flush=True)
  print('=' * 80, flush=True)

  if explicit_flags:
    print(' [CLI Overrides / Explicitly Set Flags]', flush=True)
    for name in sorted(explicit_flags.keys()):
      print(f'   --{name}={explicit_flags[name]}', flush=True)
    print('', flush=True)

  categories = [
      ('Game & Model', [
          'game',
          'model_name',
          'use_4bit',
          'max_seq_len',
          'lora_rank',
          'lora_alpha',
          'lora_dropout',
          'initial_lora_checkpoint',
          'seed',
      ]),
      ('Training & Schedule', [
          'rl_algorithm',
          'lr',
          'num_episodes',
          'max_grad_norm',
          'kl_coeff',
          'eval_every',
          'num_eval_episodes',
          'eval_batch_size',
          'eval_full_episodes',
          'eval_mode',
          'eval_seed',
          'eval_bot_baseline',
          'checkpoint_every',
          'log_every',
          'log_episodes_every',
          'max_history_turns',
          'output_dir',
          'use_wandb',
          'wandb_project',
      ]),
      ('Sampling & Exploration', [
          'temperature',
          'temperature_anneal_end',
          'temperature_floor',
      ]),
      ('GRPO Configuration', [
          'grpo_passes',
          'grpo_collect_episodes',
          'grpo_num_generations',
          'grpo_train_epochs',
          'grpo_max_completion_length',
          'grpo_scale_rewards',
          'grpo_exhaustive_groups',
          'grpo_optimistic_alpha',
          'grpo_optimistic_alpha_min',
          'grpo_signal_entropy_coeff',
          'grpo_phased_training',
          'grpo_phase1_passes',
          'grpo_phase2_passes',
          'grpo_phase3_passes',
          'grpo_convergence_patience',
          'grpo_convergence_min_delta',
          'grpo_pivot_decisions_per_episode',
          'grpo_decision_priority_sampling',
          'grpo_truncated_rollout_horizon',
      ]),
      ('Reward Simulation', [
          'reward_simulation_mode',
          'dense_chain_discount',
          'reward_blend_weight',
          'reward_rollout_samples',
          'reward_rollout_common_seed',
          'reward_survival_exponent',
          'reward_turn_discount',
          'reward_policy_turns',
      ]),
      ('Strategy, Diversity & Partner', [
          'bot_partner',
          'bot_type',
          'llm_partner_response',
          'reasoning',
          'constrained_action_types',
          'strategic_action_selection',
          'strategic_action_mode',
          'strategic_action_forced_ratio',
      ]),
      ('Curriculum', [
          'curriculum_window_size',
          'curriculum_passes_per_phase',
          'curriculum_max_horizon',
          'curriculum_max_lookback',
          'curriculum_replay_ratio',
      ]),
      ('REINFORCE (if active)', [
          'gradient_accumulation_steps',
          'baseline_window_size',
      ]),
  ]

  printed_flags = set()
  for cat_name, flag_names in categories:
    cat_items = [(f, all_flags[f]) for f in flag_names if f in all_flags]
    if cat_items:
      print(f' [{cat_name}]', flush=True)
      for fname, fval in cat_items:
        printed_flags.add(fname)
        present_mark = ' (explicit)' if fname in explicit_flags else ''
        print(f'   {fname:<36} = {fval}{present_mark}', flush=True)
      print('', flush=True)

  remaining = [f for f in sorted(all_flags.keys()) if f not in printed_flags]
  if remaining:
    print(' [Other Flags]', flush=True)
    for fname in remaining:
      present_mark = ' (explicit)' if fname in explicit_flags else ''
      print(f'   {fname:<36} = {all_flags[fname]}{present_mark}', flush=True)
    print('', flush=True)

  print('=' * 80, flush=True)


def _find_latest_checkpoint(output_dir: str) -> tuple[str | None, int]:
  """Scans output_dir (CNS or local) for checkpoint_ep* and checkpoint_interim directories.

  Returns:
    (latest_checkpoint_path, latest_episode) or (None, 0).
  """
  if not output_dir:
    return None, 0

  candidates: list[tuple[int, str]] = []
  interim_path: str | None = None
  gfile_mod = None

  # Check CNS output_dir
  if output_dir.startswith('/cns/'):
    try:
      from pyglib import gfile as _gfile
      gfile_mod = _gfile
    except ImportError:
      try:
        from tensorflow.io import gfile as _gfile
        gfile_mod = _gfile
      except ImportError:
        gfile_mod = None

    if gfile_mod is not None:
      exists_fn = getattr(gfile_mod, 'Exists', getattr(gfile_mod, 'exists', None))
      listdir_fn = getattr(gfile_mod, 'ListDirectory', getattr(gfile_mod, 'listdir', None))
      if exists_fn and exists_fn(output_dir) and listdir_fn:
        try:
          entries = listdir_fn(output_dir)
          for entry in entries:
            clean_entry = entry.strip('/')
            if clean_entry == 'checkpoint_interim':
              interim_path = os.path.join(output_dir, clean_entry)
            elif 'checkpoint_ep' in clean_entry:
              part = clean_entry.split('checkpoint_ep')[-1]
              try:
                ep = int(part)
                candidates.append((ep, os.path.join(output_dir, clean_entry)))
              except ValueError:
                pass
        except Exception as e:
          logging.warning('Error listing CNS output directory %s: %s', output_dir, e)

  # Check local output_dir
  if os.path.isdir(output_dir):
    try:
      for entry in os.listdir(output_dir):
        if entry == 'checkpoint_interim':
          interim_path = os.path.join(output_dir, entry)
        elif 'checkpoint_ep' in entry:
          part = entry.split('checkpoint_ep')[-1]
          try:
            ep = int(part)
            candidates.append((ep, os.path.join(output_dir, entry)))
          except ValueError:
            pass
    except Exception as e:
      logging.warning('Error listing local output directory %s: %s', output_dir, e)

  def _is_valid_checkpoint(ckpt_path: str) -> bool:
    """Verifies that adapter_model.safetensors exists and has a valid header."""
    try:
      from backend.gemma_backend import _stage_checkpoint_if_cns  # pylint: disable=g-import-not-at-top
      from safetensors import safe_open  # pylint: disable=g-import-not-at-top

      staged = _stage_checkpoint_if_cns(ckpt_path)
      sf_path = os.path.join(staged, 'adapter_model.safetensors')
      if not os.path.exists(sf_path):
        logging.warning('Checkpoint %s is missing adapter_model.safetensors', ckpt_path)
        return False
      with safe_open(sf_path, framework='pt', device='cpu') as f:
        if not f.keys():
          return False
      return True
    except Exception as e:
      logging.warning(
          'Skipping corrupted/incomplete checkpoint %s: %s', ckpt_path, e
      )
      return False

  # Newest first; stop at the first valid checkpoint. Validating a CNS
  # checkpoint stages it to /tmp (~1 min each), so checking every one would
  # cost O(passes) minutes per restart.
  candidates.sort(key=lambda x: x[0], reverse=True)
  latest_ep, latest_ckpt = 0, None
  for ep, path in candidates:
    if _is_valid_checkpoint(path):
      latest_ep, latest_ckpt = ep, path
      break

  # Check if checkpoint_interim is active for the current pass (newer than latest_ep)
  if interim_path is not None:
    meta_file = os.path.join(interim_path, 'checkpoint_metadata.json')
    try:
      import json as _json
      meta_data = None
      if meta_file.startswith('/cns/') and gfile_mod is not None:
        open_fn = getattr(gfile_mod, 'GFile', getattr(gfile_mod, 'Open', None))
        exists_fn = getattr(gfile_mod, 'Exists', getattr(gfile_mod, 'exists', None))
        if open_fn and exists_fn and exists_fn(meta_file):
          with open_fn(meta_file, 'r') as f:
            meta_data = _json.load(f)
      elif os.path.exists(meta_file):
        with open(meta_file, 'r') as f:
          meta_data = _json.load(f)

      if (
          isinstance(meta_data, dict)
          and meta_data.get('status') == 'in_progress'
          and int(meta_data.get('total_episodes', -1)) >= latest_ep
          and _is_valid_checkpoint(interim_path)
      ):
        logging.info(
            'Found active mid-pass interim checkpoint %s (pass %s, base_ep=%d)',
            interim_path,
            meta_data.get('pass_idx'),
            latest_ep,
        )
        return interim_path, latest_ep
    except Exception as e:
      logging.warning('Failed to inspect interim checkpoint metadata %s: %s', meta_file, e)

  return latest_ckpt, latest_ep


def main(argv: list[str]) -> None:
  """Main entry point for Gemma RL training."""
  del argv

  np.random.seed(FLAGS.seed)
  torch.manual_seed(FLAGS.seed)

  all_flags, explicit_flags = _get_experiment_flags()
  _print_experiment_configuration(all_flags, explicit_flags)

  logging.info('=== TeamGamesRL — Gemma 2B RL Training ===')
  logging.info('Game: %s', FLAGS.game)
  logging.info('Model: %s (4-bit=%s)', FLAGS.model_name, FLAGS.use_4bit)
  logging.info(
      'LoRA: rank=%d, alpha=%d, dropout=%.2f',
      FLAGS.lora_rank,
      FLAGS.lora_alpha,
      FLAGS.lora_dropout,
  )
  logging.info(
      'Training: episodes=%d, lr=%g, temp=%.2f',
      FLAGS.num_episodes,
      FLAGS.lr,
      FLAGS.temperature,
  )

  # ── Auto-detect existing checkpoint to prevent losing progress on Borg restarts ──
  auto_resume_checkpoint, latest_ep = _find_latest_checkpoint(FLAGS.output_dir)
  initial_pass = 1
  lora_checkpoint_to_load = None

  if FLAGS.grpo_initial_pass > 0:
    initial_pass = FLAGS.grpo_initial_pass
    logging.info('Explicit initial pass configured via flag: %d', initial_pass)

  if auto_resume_checkpoint is not None:
    collect_ep = FLAGS.grpo_collect_episodes if FLAGS.grpo_collect_episodes > 0 else 50
    completed_passes = latest_ep // collect_ep
    if FLAGS.grpo_initial_pass <= 0:
      initial_pass = completed_passes + 1
    lora_checkpoint_to_load = auto_resume_checkpoint
    logging.info(
        '=== [AUTO-RESUME] Found existing checkpoint in output_dir: %s (episode %d). '
        'Resuming from Pass %d ===',
        auto_resume_checkpoint,
        latest_ep,
        initial_pass,
    )
    print(
        f'=== [AUTO-RESUME] Found existing checkpoint {auto_resume_checkpoint} (episode {latest_ep})! '
        f'Resuming from Pass {initial_pass} ===',
        flush=True,
    )

  # ── Load model ──
  from backend.gemma_backend import GemmaLLMBackend  # pylint: disable=g-import-not-at-top

  backend = GemmaLLMBackend(
      model_name=FLAGS.model_name,
      lora_rank=FLAGS.lora_rank,
      lora_alpha=FLAGS.lora_alpha,
      lora_dropout=FLAGS.lora_dropout,
      use_4bit=FLAGS.use_4bit,
      max_seq_len=FLAGS.max_seq_len,
      lora_checkpoint=lora_checkpoint_to_load,
      base_lora_checkpoint=FLAGS.initial_lora_checkpoint,
  )

  # ── Build full experiment config for reproducibility ──
  experiment_config = dict(all_flags)

  # ── Flatboard / S2 eval sink (no-op outside XManager) ──
  from trainer.s2_eval_logger import S2EvalLogger  # pylint: disable=g-import-not-at-top

  # Explicit CLI flags cover every swept hyperparameter (the launcher passes
  # them per work unit), so Flatboard can group curves by S2_METADATA_<flag>.
  s2_metadata = dict(explicit_flags)
  for key in (
      'game',
      'model_name',
      'lr',
      'reasoning',
      'curriculum_window_size',
      'curriculum_max_lookback',
      'reward_policy_turns',
      'initial_lora_checkpoint',
      'eval_mode',
  ):
    s2_metadata.setdefault(key, all_flags.get(key))
  eval_sink = S2EvalLogger.maybe_create(
      dataframe='eval',
      episodes_per_step=(
          FLAGS.grpo_collect_episodes if FLAGS.rl_algorithm == 'grpo' else 1
      ),
      metadata=s2_metadata,
  )

  # ── Build trainer ──
  from trainer.rl_trainer import RLTrainer  # pylint: disable=g-import-not-at-top

  trainer = RLTrainer(
      game_name=FLAGS.game,
      backend=backend,
      num_episodes=FLAGS.num_episodes,
      eval_every=FLAGS.eval_every,
      num_eval_episodes=FLAGS.num_eval_episodes,
      lr=FLAGS.lr,
      max_grad_norm=FLAGS.max_grad_norm,
      temperature=FLAGS.temperature,
      output_dir=FLAGS.output_dir,
      log_every=FLAGS.log_every,
      checkpoint_every=FLAGS.checkpoint_every,
      log_episodes_every=FLAGS.log_episodes_every,
      use_wandb=FLAGS.use_wandb,
      wandb_project=FLAGS.wandb_project,
      wandb_config=experiment_config,
      max_history_turns=FLAGS.max_history_turns or None,
      experiment_config=experiment_config,
      bot_partner=FLAGS.bot_partner,
      bot_type=FLAGS.bot_type,
      reasoning=FLAGS.reasoning,
      eval_batch_size=FLAGS.eval_batch_size,
      eval_sink=eval_sink,
      # Evaluate with at least the training completion budget: expert-CoT
      # completions reach ~350 tokens, and truncated ones are parsed as random
      # fallback actions.
      eval_max_tokens=max(
          FLAGS.grpo_max_completion_length, 256 if FLAGS.reasoning else 64
      ),
  )

  # ── Train ──
  if FLAGS.rl_algorithm == 'grpo':
    from learn.grpo import GRPOConfig  # pylint: disable=g-import-not-at-top

    logging.info('Using GRPO (TRL) training.')
    grpo_config = GRPOConfig(
        num_generations=FLAGS.grpo_num_generations,
        collect_episodes=FLAGS.grpo_collect_episodes,
        collect_batch_size=FLAGS.grpo_collect_batch_size,
        prompt_batch_size=FLAGS.grpo_prompt_batch_size,
        train_batch_size=FLAGS.grpo_train_batch_size,
        train_epochs=FLAGS.grpo_train_epochs,
        gradient_accumulation_steps=FLAGS.gradient_accumulation_steps,
        passes=FLAGS.grpo_passes,
        initial_pass=initial_pass,
        max_seq_len=FLAGS.max_seq_len,
        max_completion_length=FLAGS.grpo_max_completion_length,
        lr=FLAGS.lr,
        kl_coeff=FLAGS.kl_coeff,
        max_grad_norm=FLAGS.max_grad_norm,
        temperature=FLAGS.temperature,
        num_eval_episodes=FLAGS.num_eval_episodes,
        eval_full_episodes=FLAGS.eval_full_episodes,
        eval_mode=FLAGS.eval_mode,
        eval_seed=FLAGS.eval_seed,
        eval_bot_baseline=FLAGS.eval_bot_baseline,
        exhaustive_groups=FLAGS.grpo_exhaustive_groups,
        optimistic_reward_alpha=FLAGS.grpo_optimistic_alpha,
        optimistic_reward_alpha_min=FLAGS.grpo_optimistic_alpha_min,
        signal_entropy_coeff=FLAGS.grpo_signal_entropy_coeff,
        phased_training=FLAGS.grpo_phased_training,
        phase1_max_passes=FLAGS.grpo_phase1_passes,
        phase2_max_passes=FLAGS.grpo_phase2_passes,
        phase3_max_passes=FLAGS.grpo_phase3_passes,
        convergence_patience=FLAGS.grpo_convergence_patience,
        convergence_min_delta=FLAGS.grpo_convergence_min_delta,
        pivot_decisions_per_episode=FLAGS.grpo_pivot_decisions_per_episode,
        decision_priority_sampling=FLAGS.grpo_decision_priority_sampling,
        truncated_rollout_horizon=FLAGS.grpo_truncated_rollout_horizon,
        reward_simulation_mode=FLAGS.reward_simulation_mode,
        dense_chain_discount=FLAGS.dense_chain_discount,
        temperature_anneal_end=FLAGS.temperature_anneal_end,
        temperature_floor=FLAGS.temperature_floor,
        reward_blend_weight=FLAGS.reward_blend_weight,
        reward_rollout_samples=FLAGS.reward_rollout_samples,
        reward_rollout_common_seed=FLAGS.reward_rollout_common_seed,
        reward_survival_exponent=FLAGS.reward_survival_exponent,
        reward_turn_discount=FLAGS.reward_turn_discount,
        reward_policy_turns=FLAGS.reward_policy_turns,
        grpo_scale_rewards=FLAGS.grpo_scale_rewards,
        constrained_action_types=FLAGS.constrained_action_types,
        strategic_action_selection=FLAGS.strategic_action_selection,
        strategic_action_mode=FLAGS.strategic_action_mode,
        strategic_action_forced_ratio=FLAGS.strategic_action_forced_ratio,
        llm_partner_response=FLAGS.llm_partner_response,
        bot_partner=FLAGS.bot_partner,
        bot_type=FLAGS.bot_type,
        checkpoint_interval_minutes=FLAGS.checkpoint_interval_minutes,
        reasoning=FLAGS.reasoning,
        curriculum_window_size=FLAGS.curriculum_window_size,
        curriculum_passes_per_phase=FLAGS.curriculum_passes_per_phase,
        curriculum_max_horizon=FLAGS.curriculum_max_horizon,
        curriculum_max_lookback=FLAGS.curriculum_max_lookback,
        curriculum_replay_ratio=FLAGS.curriculum_replay_ratio,
    )
    # ── Tiny Hanabi-specific tuning ──
    # For tiny_hanabi, enable exhaustive-group GRPO by default.  This
    # enumerates all game states and forms groups where only the target
    # player's action varies, producing zero-variance advantage estimates.
    if FLAGS.game == 'tiny_hanabi':
      tiny_hanabi_passes = (
          FLAGS.grpo_passes if FLAGS['grpo_passes'].present else 100
      )
      use_exhaustive = (
          FLAGS.grpo_exhaustive_groups
          if FLAGS['grpo_exhaustive_groups'].present
          else True
      )
      # Anneal optimistic alpha from 1.0 → 0.0 for tiny_hanabi.
      # Early passes use fully optimistic rewards (α=1.0) to bootstrap
      # P0 signaling.  Later passes decay toward P1's actual policy so
      # that P0's convention reflects what P1 can really decode.
      use_alpha_min = (
          FLAGS.grpo_optimistic_alpha_min
          if FLAGS['grpo_optimistic_alpha_min'].present
          else 0.0
      )
      # Signal entropy bonus is disabled by default.  The optimistic
      # rewards and KL regularization already break the R=8 plateau
      # without risking gradient conflicts from a second optimizer step.
      # Use a small value (0.01–0.05) only if the model still collapses
      # to a single action for all cards.
      use_entropy = (
          FLAGS.grpo_signal_entropy_coeff
          if FLAGS['grpo_signal_entropy_coeff'].present
          else 0.0
      )
      grpo_config = dataclasses.replace(
          grpo_config,
          passes=tiny_hanabi_passes,
          exhaustive_groups=use_exhaustive,
          optimistic_reward_alpha_min=use_alpha_min,
          signal_entropy_coeff=use_entropy,
      )
      override_msg = (
          'Applied Tiny Hanabi-specific GRPO overrides: passes=%d, '
          'exhaustive_groups=%s, alpha=[%.2f -> %.2f], '
          'signal_entropy_coeff=%.3f, phased_training=%s'
          % (
              grpo_config.passes,
              grpo_config.exhaustive_groups,
              grpo_config.optimistic_reward_alpha,
              grpo_config.optimistic_reward_alpha_min,
              grpo_config.signal_entropy_coeff,
              grpo_config.phased_training,
          )
      )
      logging.info(override_msg)
      print(f'[CONFIG OVERRIDE] {override_msg}', flush=True)
      if grpo_config.phased_training:
        logging.info(
            '  Phased training: phase1=%d, phase2=%d, phase3=%d, '
            'patience=%d, min_delta=%.2f',
            grpo_config.phase1_max_passes,
            grpo_config.phase2_max_passes,
            grpo_config.phase3_max_passes,
            grpo_config.convergence_patience,
            grpo_config.convergence_min_delta,
        )
    trainer.train_grpo(grpo_config)
  else:
    from learn.reinforce import ReinforceConfig  # pylint: disable=g-import-not-at-top

    logging.info('Using REINFORCE training.')
    reinforce_config = ReinforceConfig(
        kl_coeff=FLAGS.kl_coeff,
        gradient_accumulation_steps=FLAGS.gradient_accumulation_steps,
        baseline_window_size=FLAGS.baseline_window_size,
        max_grad_norm=FLAGS.max_grad_norm,
    )
    trainer.train_reinforce(reinforce_config)


if __name__ == '__main__':
  app.run(main, flags_parser=lambda argv: flags.FLAGS(argv, known_only=True))
