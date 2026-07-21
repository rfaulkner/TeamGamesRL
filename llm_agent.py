# Copyright 2024 Google LLC
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

"""LLM-based agents for playing OpenSpiel games.

This module implements agents that use Large Language Models to play games in
the OpenSpiel framework. The agents convert game states to text prompts using
state renderers, query an LLM for action decisions, and parse the responses
back into valid action IDs.

Classes:
  LLMInterface: Abstract base class for LLM backends.
  MockLLM: A mock LLM that randomly selects actions (for testing).
  GeminiLLM: LLM backend using Google's GenAI SDK.
  LLMAgent: Main agent class compatible with OpenSpiel's RL agent interface.
  RandomAgent: Simple baseline agent that picks uniformly from legal actions.

Example usage:
  >>> import pyspiel
  >>> game = pyspiel.load_game("tic_tac_toe")
  >>> state = game.new_initial_state()
  >>> llm = MockLLM(seed=42)
  >>> renderer = state_renderers.DefaultStateRenderer()
  >>> agent = LLMAgent(
  ...     player_id=0, renderer=renderer, llm=llm, game=game)
"""

import abc
import math
import re
import time
from typing import Optional

from absl import logging
import numpy as np

from open_spiel.python import rl_agent

import state_renderers


# System prompt template for the LLM agent.
_SYSTEM_PROMPT_TEMPLATE = """\
You are an expert game-playing AI agent. You are playing the game: {game_name}.

{game_description}

RULES:
- You must select exactly one action from the list of legal actions provided.
- Respond with ONLY the action number (integer) on a single line.
- Do not include any explanation, commentary, or extra text.
- Think strategically to maximize your chance of winning.

You are Player {player_id}.
"""

# User prompt template for each turn.
_USER_PROMPT_TEMPLATE = """\
Current game state:
{state_text}

Legal actions:
{actions_text}

Select your action (respond with the action number only):"""


class LLMInterface(abc.ABC):
  """Abstract base class for LLM backends.

  Subclasses must implement `generate` and optionally `generate_with_logprobs`
  to provide text generation capabilities for the LLMAgent.
  """

  @abc.abstractmethod
  def generate(
      self,
      prompt: str,
      temperature: float = 0.7,
      max_tokens: int = 256,
  ) -> str:
    """Generate text from a prompt.

    Args:
      prompt: The input prompt string.
      temperature: Sampling temperature. Higher values produce more random
        output. Must be >= 0.
      max_tokens: Maximum number of tokens to generate.

    Returns:
      The generated text string.
    """

  def generate_with_logprobs(
      self,
      prompt: str,
      temperature: float = 0.7,
      max_tokens: int = 256,
  ) -> tuple[str, float]:
    """Generate text and return the associated log probability.

    The default implementation calls `generate` and returns a log probability
    of 0.0 (i.e., probability 1.0). Subclasses should override this to
    provide actual log probabilities when the underlying LLM supports them.

    Args:
      prompt: The input prompt string.
      temperature: Sampling temperature.
      max_tokens: Maximum number of tokens to generate.

    Returns:
      A tuple of (generated_text, log_probability).
    """
    text = self.generate(prompt, temperature=temperature, max_tokens=max_tokens)
    return text, 0.0


class MockLLM(LLMInterface):
  """A mock LLM backend for testing without a real model.

  This implementation parses the legal actions from the prompt text and
  randomly selects one. Useful for integration testing and development.

  Attributes:
    _rng: NumPy random number generator for reproducible action selection.
  """

  def __init__(self, seed: Optional[int] = None):
    """Initializes the MockLLM.

    Args:
      seed: Optional random seed for reproducibility.
    """
    self._rng = np.random.RandomState(seed)

  def generate(
      self,
      prompt: str,
      temperature: float = 0.7,
      max_tokens: int = 256,
  ) -> str:
    """Generate a mock response by randomly selecting a legal action.

    Parses action numbers from the prompt text (looking for lines matching
    the pattern 'Action N:' or just integers) and returns one at random.

    Args:
      prompt: The input prompt containing legal action descriptions.
      temperature: Unused in mock implementation.
      max_tokens: Unused in mock implementation.

    Returns:
      A string containing a single action number.
    """
    del temperature, max_tokens  # Unused in mock.

    # Parse action numbers from the prompt.
    # Expected format: "Action N: description" or just lines with numbers.
    action_numbers = re.findall(r'Action\s+(\d+)', prompt)
    if not action_numbers:
      # Fallback: look for any standalone integers in the legal actions section.
      action_numbers = re.findall(r'^\s*(\d+)\s*[:\-]', prompt, re.MULTILINE)
    if not action_numbers:
      # Last resort: find any integers in the prompt.
      action_numbers = re.findall(r'\b(\d+)\b', prompt)

    if action_numbers:
      chosen = self._rng.choice(action_numbers)
      return str(chosen)
    else:
      logging.warning('MockLLM: Could not parse any action numbers from prompt')
      return '0'

  def generate_with_logprobs(
      self,
      prompt: str,
      temperature: float = 0.7,
      max_tokens: int = 256,
  ) -> tuple[str, float]:
    """Generate a mock response with a synthetic log probability.

    Args:
      prompt: The input prompt.
      temperature: Unused in mock implementation.
      max_tokens: Unused in mock implementation.

    Returns:
      A tuple of (action_string, mock_log_prob). The log probability is
      computed as -log(num_actions), simulating uniform random selection.
    """
    action_numbers = re.findall(r'Action\s+(\d+)', prompt)
    if not action_numbers:
      action_numbers = re.findall(r'^\s*(\d+)\s*[:\-]', prompt, re.MULTILINE)

    num_actions = max(len(action_numbers), 1)
    mock_log_prob = -math.log(num_actions)

    text = self.generate(prompt, temperature=temperature, max_tokens=max_tokens)
    return text, mock_log_prob


class GeminiLLM(LLMInterface):
  """LLM backend using Google's GenAI SDK (Gemini models).

  This class wraps the Google GenAI client to provide text generation for
  game-playing agents. It handles API errors gracefully with exponential
  backoff retries.

  The GenAI SDK is imported conditionally so the module can be loaded even
  when the SDK is not available (falling back to MockLLM for testing).

  Attributes:
    _client: The GenAI Client instance.
    _model_name: Name of the Gemini model to use.
    _max_retries: Maximum number of API call retries on failure.
    _base_delay: Base delay in seconds for exponential backoff.
  """

  def __init__(
      self,
      model_name: str = 'gemini-2.0-flash',
      max_retries: int = 3,
      base_delay: float = 1.0,
      api_key: Optional[str] = None,
  ):
    """Initializes the GeminiLLM.

    Args:
      model_name: The Gemini model to use (e.g., 'gemini-2.0-flash',
        'gemini-2.5-pro').
      max_retries: Maximum number of retries on API errors.
      base_delay: Base delay in seconds for exponential backoff between
        retries.
      api_key: Optional API key. If not provided, the client will attempt
        to use default credentials or environment variables.

    Raises:
      ImportError: If the google.genai package is not available.
    """
    try:
      # pylint: disable=g-import-not-at-top
      from google import genai
      from google.genai import types
      # pylint: enable=g-import-not-at-top
    except ImportError as e:
      raise ImportError(
          'GeminiLLM requires the google-genai package. '
          'Install it or use MockLLM for testing.'
      ) from e

    self._genai = genai
    self._types = types
    self._model_name = model_name
    self._max_retries = max_retries
    self._base_delay = base_delay

    if api_key:
      self._client = genai.Client(api_key=api_key)
    else:
      self._client = genai.Client()

  def generate(
      self,
      prompt: str,
      temperature: float = 0.7,
      max_tokens: int = 256,
  ) -> str:
    """Generate text using the Gemini model.

    Calls the GenAI API with exponential backoff retry logic for transient
    errors.

    Args:
      prompt: The input prompt string.
      temperature: Sampling temperature for generation.
      max_tokens: Maximum number of output tokens.

    Returns:
      The generated text string.

    Raises:
      RuntimeError: If all retry attempts are exhausted.
    """
    config = self._types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
    )

    last_error = None
    for attempt in range(self._max_retries):
      try:
        response = self._client.models.generate_content(
            model=self._model_name,
            contents=prompt,
            config=config,
        )
        if response.text:
          return response.text.strip()
        else:
          logging.warning(
              'GeminiLLM: Empty response on attempt %d/%d',
              attempt + 1,
              self._max_retries,
          )
          return ''
      except Exception as e:  # pylint: disable=broad-except
        last_error = e
        delay = self._base_delay * (2 ** attempt)
        logging.warning(
            'GeminiLLM: API error on attempt %d/%d: %s. '
            'Retrying in %.1f seconds.',
            attempt + 1,
            self._max_retries,
            str(e),
            delay,
        )
        time.sleep(delay)

    raise RuntimeError(
        f'GeminiLLM: All {self._max_retries} retry attempts failed. '
        f'Last error: {last_error}'
    )

  def generate_with_logprobs(
      self,
      prompt: str,
      temperature: float = 0.7,
      max_tokens: int = 256,
  ) -> tuple[str, float]:
    """Generate text with log probability estimation.

    Note: The standard Gemini API does not directly expose per-response
    log probabilities. This implementation returns a placeholder log
    probability of 0.0. For RL training with actual log probabilities,
    a specialized API endpoint or model configuration may be needed.

    Args:
      prompt: The input prompt string.
      temperature: Sampling temperature.
      max_tokens: Maximum number of output tokens.

    Returns:
      A tuple of (generated_text, log_probability). Log probability is
      currently a placeholder (0.0).
    """
    text = self.generate(prompt, temperature=temperature, max_tokens=max_tokens)
    # TODO(rfaulk): Integrate actual log probability extraction when
    # the Gemini API supports it for this use case.
    log_prob = 0.0
    return text, log_prob


class LLMAgent:
  """An agent that uses an LLM to play OpenSpiel games.

  This agent is compatible with OpenSpiel's RL agent interface. It renders
  game states to text, constructs prompts with game instructions and legal
  action descriptions, queries an LLM for a decision, and parses the
  response back to a valid action ID.

  The agent includes retry logic for parsing failures and falls back to
  random action selection if the LLM consistently produces unparseable
  output.

  Attributes:
    player_id: The player index this agent controls.
    _renderer: State renderer for converting game states to text.
    _llm: The LLM backend for text generation.
    _game: The OpenSpiel game instance.
    _temperature: Sampling temperature for LLM generation.
    _max_retries: Maximum retries for LLM response parsing.
    _rng: Random number generator for fallback action selection.
    _system_prompt: Pre-computed system prompt for this agent.
  """

  def __init__(
      self,
      player_id: int,
      renderer: 'state_renderers.StateRenderer',
      llm: LLMInterface,
      env,
      temperature: float = 0.7,
      max_retries: int = 3,
      seed: Optional[int] = None,
  ):
    """Initializes the LLMAgent.

    Args:
      player_id: The player index (0-based) this agent controls.
      renderer: A StateRenderer instance for converting game states to text.
      llm: An LLMInterface implementation for generating responses.
      env: The OpenSpiel rl_environment.Environment object.
      temperature: Sampling temperature for LLM generation. Lower values
        produce more deterministic play.
      max_retries: Maximum number of retries when the LLM response cannot
        be parsed into a valid action.
      seed: Optional random seed for the fallback random action selection.
    """
    self.player_id = player_id
    self._renderer = renderer
    self._llm = llm
    self._env = env
    game = env.game
    self._game = game
    self._temperature = temperature
    self._max_retries = max_retries
    self._rng = np.random.RandomState(seed)

    # Pre-compute the system prompt.
    game_type = game.get_type()
    game_description = (
        f'Game type: {game_type.short_name}\n'
        f'Number of players: {game.num_players()}\n'
        f'Number of distinct actions: {game.num_distinct_actions()}'
    )
    self._system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
        game_name=game_type.short_name,
        game_description=game_description,
        player_id=player_id,
    )

  def _build_prompt(
      self,
      state_text: str,
      legal_actions: list[int],
      action_descriptions: list[str],
  ) -> str:
    """Constructs the full prompt for the LLM.

    Combines the system prompt with the current game state and legal
    action descriptions.

    Args:
      state_text: Text representation of the current game state.
      legal_actions: List of legal action IDs.
      action_descriptions: Human-readable descriptions for each legal action.

    Returns:
      The complete prompt string to send to the LLM.
    """
    actions_lines = []
    for action_id, desc in zip(legal_actions, action_descriptions):
      actions_lines.append(f'  Action {action_id}: {desc}')
    actions_text = '\n'.join(actions_lines)

    user_prompt = _USER_PROMPT_TEMPLATE.format(
        state_text=state_text,
        actions_text=actions_text,
    )

    return f'{self._system_prompt}\n{user_prompt}'

  def _parse_action(
      self,
      response: str,
      legal_actions_with_desc: list[tuple[int, str]],
  ) -> Optional[int]:
    """Parse the LLM response to extract a valid action ID.

    Attempts to extract an integer from the LLM response and validates
    that it corresponds to a legal action.

    Args:
      response: The raw text response from the LLM.
      legal_actions_with_desc: List of currently legal action IDs and descriptions.

    Returns:
      A valid action ID if parsing succeeds, or None if the response
      cannot be parsed into a legal action.
    """
    return self._renderer.parse_action(response, legal_actions_with_desc)

  def step(
      self,
      time_step,
      is_evaluation: bool = False,
  ) -> rl_agent.StepOutput:
    """Returns an action for the current time step.

    Implements the OpenSpiel RL agent interface. Renders the current game
    state, queries the LLM, and parses the response to select an action.

    Args:
      time_step: An rl_environment.TimeStep containing the current
        observations, rewards, and step type.
      is_evaluation: If True, the agent is in evaluation mode. Currently
        unused but reserved for future use (e.g., lower temperature).

    Returns:
      An rl_agent.StepOutput containing the chosen action and a
      probability distribution over actions.
    """
    del is_evaluation  # Reserved for future use.

    # Handle terminal states.
    if time_step.last():
      action_probs = np.zeros(self._game.num_distinct_actions())
      return rl_agent.StepOutput(action=0, probs=action_probs)

    # Extract observations.
    legal_actions = time_step.observations['legal_actions'][self.player_id]
    current_player = time_step.observations['current_player']

    # If it's not our turn, return a no-op.
    if current_player != self.player_id:
      action_probs = np.zeros(self._game.num_distinct_actions())
      return rl_agent.StepOutput(action=0, probs=action_probs)

    # Render state text using the renderer.
    state = self._env._state
    state_text = self._renderer.render_state(state, self.player_id, self._game)

    # Get action descriptions.
    legal_actions_with_desc = self._renderer.render_legal_actions(
        state, self.player_id, self._game
    )
    action_descriptions = [desc for _, desc in legal_actions_with_desc]

    # Build the prompt.
    prompt = self._build_prompt(state_text, legal_actions, action_descriptions)

    # Query the LLM with retries.
    action_id = None
    for attempt in range(self._max_retries):
      try:
        response = self._llm.generate(
            prompt, temperature=self._temperature, max_tokens=64
        )
        action_id = self._parse_action(response, legal_actions_with_desc)
        if action_id is not None:
          break
        logging.warning(
            'LLMAgent: Failed to parse action from LLM response on attempt '
            '%d/%d. Response: %r',
            attempt + 1,
            self._max_retries,
            response,
        )
      except Exception as e:  # pylint: disable=broad-except
        logging.error(
            'LLMAgent: LLM generation error on attempt %d/%d: %s',
            attempt + 1,
            self._max_retries,
            str(e),
        )

    # Fallback to random action if all retries failed.
    if action_id is None:
      action_id = self._rng.choice(legal_actions)
      logging.warning(
          'LLMAgent: All %d retries failed. Falling back to random action: %d',
          self._max_retries,
          action_id,
      )

    # Build action probability distribution.
    # For now, assign probability 1.0 to the chosen action (greedy).
    action_probs = np.zeros(self._game.num_distinct_actions())
    action_probs[action_id] = 1.0

    return rl_agent.StepOutput(action=action_id, probs=action_probs)

  def act(self, time_step) -> tuple[int, float]:
    """Simplified interface returning (action_id, log_prob).

    This method is useful for RL training loops that need both the chosen
    action and its log probability for policy gradient computation.

    Args:
      time_step: An rl_environment.TimeStep containing the current
        observations.

    Returns:
      A tuple of (action_id, log_prob) where action_id is the chosen
      action and log_prob is the log probability of choosing that action
      under the LLM's distribution.
    """
    # Handle terminal states.
    if time_step.last():
      return 0, 0.0

    # Extract observations.
    legal_actions = time_step.observations['legal_actions'][self.player_id]
    current_player = time_step.observations['current_player']

    if current_player != self.player_id:
      return 0, 0.0

    # Render state text.
    if 'info_state' in time_step.observations:
      info_state = time_step.observations['info_state'][self.player_id]
      state_text = self._renderer.render_from_observation(
          info_state, self.player_id
      )
    else:
      state_text = str(time_step.observations)

    # Get action descriptions.
    action_descriptions = []
    for action in legal_actions:
      try:
        desc = self._renderer.action_to_string(action, self.player_id)
      except (AttributeError, NotImplementedError):
        desc = str(action)
      action_descriptions.append(desc)

    # Build prompt and query LLM with log probabilities.
    prompt = self._build_prompt(state_text, legal_actions, action_descriptions)

    action_id = None
    log_prob = 0.0

    for attempt in range(self._max_retries):
      try:
        response, response_log_prob = self._llm.generate_with_logprobs(
            prompt, temperature=self._temperature, max_tokens=64
        )
        parsed_action = self._parse_action(response, legal_actions)
        if parsed_action is not None:
          action_id = parsed_action
          log_prob = response_log_prob
          break
        logging.warning(
            'LLMAgent: Failed to parse action on attempt %d/%d. Response: %r',
            attempt + 1,
            self._max_retries,
            response,
        )
      except Exception as e:  # pylint: disable=broad-except
        logging.error(
            'LLMAgent: Error on attempt %d/%d: %s',
            attempt + 1,
            self._max_retries,
            str(e),
        )

    # Fallback to random action.
    if action_id is None:
      action_id = int(self._rng.choice(legal_actions))
      log_prob = -math.log(len(legal_actions))
      logging.warning(
          'LLMAgent: All retries failed in act(). '
          'Falling back to random action: %d',
          action_id,
      )

    return action_id, log_prob


class RandomAgent:
  """A simple random baseline agent for OpenSpiel games.

  Picks uniformly at random from the set of legal actions at each step.
  Compatible with both the `step()` and `act()` interfaces.

  Attributes:
    player_id: The player index this agent controls.
    _rng: NumPy random number generator.
    _num_actions: Total number of distinct actions in the game.
  """

  def __init__(
      self,
      player_id: int,
      game,
      seed: Optional[int] = None,
  ):
    """Initializes the RandomAgent.

    Args:
      player_id: The player index (0-based) this agent controls.
      game: The OpenSpiel game object.
      seed: Optional random seed for reproducibility.
    """
    self.player_id = player_id
    self._rng = np.random.RandomState(seed)
    self._num_actions = game.num_distinct_actions()

  def step(
      self,
      time_step,
      is_evaluation: bool = False,
  ) -> rl_agent.StepOutput:
    """Returns a random action for the current time step.

    Args:
      time_step: An rl_environment.TimeStep.
      is_evaluation: Unused.

    Returns:
      An rl_agent.StepOutput with a uniformly random legal action and
      uniform probability distribution over legal actions.
    """
    del is_evaluation

    if time_step.last():
      action_probs = np.zeros(self._num_actions)
      return rl_agent.StepOutput(action=0, probs=action_probs)

    legal_actions = time_step.observations['legal_actions'][self.player_id]
    current_player = time_step.observations['current_player']

    if current_player != self.player_id:
      action_probs = np.zeros(self._num_actions)
      return rl_agent.StepOutput(action=0, probs=action_probs)

    action_id = int(self._rng.choice(legal_actions))

    # Uniform probability over legal actions.
    action_probs = np.zeros(self._num_actions)
    for a in legal_actions:
      action_probs[a] = 1.0 / len(legal_actions)

    return rl_agent.StepOutput(action=action_id, probs=action_probs)

  def act(self, time_step) -> tuple[int, float]:
    """Returns a random action with its log probability.

    Args:
      time_step: An rl_environment.TimeStep.

    Returns:
      A tuple of (action_id, log_prob) with uniform random selection.
    """
    if time_step.last():
      return 0, 0.0

    legal_actions = time_step.observations['legal_actions'][self.player_id]
    current_player = time_step.observations['current_player']

    if current_player != self.player_id:
      return 0, 0.0

    action_id = int(self._rng.choice(legal_actions))
    log_prob = -math.log(len(legal_actions))

    return action_id, log_prob
