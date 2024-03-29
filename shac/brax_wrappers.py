""" 
CHANGES FROM ORIGINAL TRAINING.PY FROM BRAX WRAPPERS:

AUTORESETWRAPPER: auto-reset the rewards as well; otherwise nan's can carry over after the reset. 

"""

from typing import Callable, Dict, Optional, Tuple

from brax.base import System
from brax.envs.base import Env, State, Wrapper
from flax import struct
import jax
from jax import numpy as jp


class AppendObsHistory(Wrapper):
  
  def __init__(self, env: Env, **kwargs):
    super().__init__(env)
    # if 'num_history_steps' not in dir(self.env):
    #   raise ValueError("Wrapped environment didn't specify number of history steps!")
    self.h = kwargs.get('h', 3)

    assert len(self.env.observation_size) == 1, "Haven't implemented history yet for multi-D observations."
    
    self.d_obs = self.env.observation_size[0]

  def reset(self, rng:jax.Array)->State:
    """ 
    0. Init history
    1. Append history to obs
    2. Update obs history
    """
    
    state = self.env.reset(rng)
    cur_obs = state.obs
        
    obs_hist = jp.zeros((self.h, self.d_obs))
    
    stacked_obs = self.stack_obs(cur_obs, obs_hist)
    state = state.replace(obs=stacked_obs)
    
    obs_hist = obs_hist.at[0].set(cur_obs)
    state.info['obs_hist'] = obs_hist
    
    return state
  
  def stack_obs(self, obs, obs_history):
    return jp.concatenate([
      obs,
      jp.ravel(obs_history).reshape(
        self.d_obs * self.h)
    ])
    
  def step(self, state: State, action: jax.Array) -> State:
    """ 
    1. Append history to obs
    2. Update obs history
    """
    
    ## Main
    state = self.env.step(state, action)
    cur_obs = state.obs

    stacked_obs = self.stack_obs(cur_obs, state.info['obs_hist'])
    state = state.replace(obs=stacked_obs)

    obs_hist = state.info['obs_hist']
    obs_hist = jp.roll(obs_hist, 1, axis=0)
    obs_hist = obs_hist.at[0].set(cur_obs)
    state.info['obs_hist'] = obs_hist
        
    return state
  
  @property
  def observation_size(self) -> int:
    nom_obs_size = self.env.observation_size
    return (nom_obs_size[0]*(self.h + 1),)

  
class AutoSampleInitQ(Wrapper):
  """Autoreset Wrapper just pulls up the random position sampled from the original reset. This wrapper resets that value.
  TODO: 
  - [ ] Explicitly throw if autoresetwrapper isn't applied. 
  - [ ] Make it faster. For example, all envs must have a fast_reset function that only does the minimum init_qpos + scale * rand(); return (pipeline_state, obs).
  """

  def __init__(self, env: Env):
    super().__init__(env)

  def reset(self, rng:jax.Array)->State:
    state = self.env.reset(rng)
    state.info['initq_key'] = rng
    return state
  
  def step(self, state: State, action: jax.Array) -> State:
    ## Generate new randomness
    key = state.info['initq_key']
    key, key_reset = jax.random.split(key)
    state.info['initq_key'] = key
    new_init = self.env.reset(key_reset)

    ## Update
    state.info['first_obs'] = new_init.obs
    state.info['first_pipeline_state'] = new_init.pipeline_state
    
    ## Main
    state = self.env.step(state, action)
    return state

class AutoResetWrapper(Wrapper):
  """ Automatically resets Brax envs that are done.
  Unmodified; just added comments.
  Tracking the dones is a bit intricate; comments are given.
     """

  def reset(self, rng: jax.Array) -> State:
    state = self.env.reset(rng)
    state.info['first_pipeline_state'] = state.pipeline_state
    state.info['first_obs'] = state.obs
    return state

  def step(self, state: State, action: jax.Array) -> State:
    if 'steps' in state.info:
      steps = state.info['steps']
      steps = jp.where(state.done, jp.zeros_like(steps), steps)
      state.info.update(steps=steps)
    # s_k: reset the "done"
    state = state.replace(done=jp.zeros_like(state.done))
    # s_k+1: get a new done signal
    state = self.env.step(state, action)
    # wrap in the new done signal
    def where_done(x, y):
      done = state.done
      if done.shape:
        done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
      return jp.where(done, x, y)

    # If s_k+1 was done, immediately reset the state.
    pipeline_state = jax.tree_map(
        where_done, state.info['first_pipeline_state'], state.pipeline_state
    )
    obs = where_done(state.info['first_obs'], state.obs)
    # The done signal for s_k+1 exits the func. 
    return state.replace(pipeline_state=pipeline_state, obs=obs)

# def wrap(
#     env: Env,
#     episode_length: int = 1000,
#     action_repeat: int = 1,
#     randomization_fn: Optional[
#         Callable[[System], Tuple[System, System]]
#     ] = None,
# ) -> Wrapper:
#   """Common wrapper pattern for all training agents.

#   Args:
#     env: environment to be wrapped
#     episode_length: length of episode
#     action_repeat: how many repeated actions to take per step
#     randomization_fn: randomization function that produces a vectorized system
#       and in_axes to vmap over

#   Returns:
#     An environment that is wrapped with Episode and AutoReset wrappers.  If the
#     environment did not already have batch dimensions, it is additional Vmap
#     wrapped.
#   """
#   env = EpisodeWrapper(env, episode_length, action_repeat)
#   if randomization_fn is None:
#     env = VmapWrapper(env)
#   else:
#     env = DomainRandomizationVmapWrapper(env, randomization_fn)
#   env = AutoResetWrapper(env)
#   return env


# class VmapWrapper(Wrapper):
#   """Vectorizes Brax env."""

#   def __init__(self, env: Env, batch_size: Optional[int] = None):
#     super().__init__(env)
#     self.batch_size = batch_size

#   def reset(self, rng: jax.Array) -> State:
#     if self.batch_size is not None:
#       rng = jax.random.split(rng, self.batch_size)
#     return jax.vmap(self.env.reset)(rng)

#   def step(self, state: State, action: jax.Array) -> State:
#     return jax.vmap(self.env.step)(state, action)


# class EpisodeWrapper(Wrapper):
#   """Maintains episode step count and sets done at episode end."""

#   def __init__(self, env: Env, episode_length: int, action_repeat: int):
#     super().__init__(env)
#     self.episode_length = episode_length
#     self.action_repeat = action_repeat

#   def reset(self, rng: jax.Array) -> State:
#     state = self.env.reset(rng)
#     state.info['steps'] = jp.zeros(rng.shape[:-1])
#     state.info['truncation'] = jp.zeros(rng.shape[:-1])
#     return state

#   def step(self, state: State, action: jax.Array) -> State:
#     def f(state, _):
#       nstate = self.env.step(state, action)
#       return nstate, nstate.reward

#     state, rewards = jax.lax.scan(f, state, (), self.action_repeat)
#     state = state.replace(reward=jp.sum(rewards, axis=0))
#     steps = state.info['steps'] + self.action_repeat
#     one = jp.ones_like(state.done)
#     zero = jp.zeros_like(state.done)
#     episode_length = jp.array(self.episode_length, dtype=jp.int32)
#     done = jp.where(steps >= episode_length, one, state.done)
#     state.info['truncation'] = jp.where(
#         steps >= episode_length, 1 - state.done, zero
#     )
#     state.info['steps'] = steps
#     return state.replace(done=done)


# class AutoResetWrapper(Wrapper):
#   """Automatically resets Brax envs that are done."""

#   def reset(self, rng: jax.Array) -> State:
#     state = self.env.reset(rng)
#     state.info['first_pipeline_state'] = state.pipeline_state
#     state.info['first_obs'] = state.obs
#     state.info['first_reward'] = state.reward
#     return state

#   def step(self, state: State, action: jax.Array) -> State:
#     if 'steps' in state.info:
#       steps = state.info['steps']
#       steps = jp.where(state.done, jp.zeros_like(steps), steps)
#       state.info.update(steps=steps)
#     state = state.replace(done=jp.zeros_like(state.done))
#     state = self.env.step(state, action)

#     def where_done(x, y):
#       done = state.done
#       if done.shape:
#         done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
#       return jp.where(done, x, y)

#     pipeline_state = jax.tree_map(
#         where_done, state.info['first_pipeline_state'], state.pipeline_state
#     )
#     obs = where_done(state.info['first_obs'], state.obs)
#     reward = where_done(state.info['first_reward'], state.reward)
#     return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward)


# @struct.dataclass
# class EvalMetrics:
#   """Dataclass holding evaluation metrics for Brax.

#   Attributes:
#       episode_metrics: Aggregated episode metrics since the beginning of the
#         episode.
#       active_episodes: Boolean vector tracking which episodes are not done yet.
#       episode_steps: Integer vector tracking the number of steps in the episode.
#   """

#   episode_metrics: Dict[str, jax.Array]
#   active_episodes: jax.Array
#   episode_steps: jax.Array


# class EvalWrapper(Wrapper):
#   """Brax env with eval metrics."""

#   def reset(self, rng: jax.Array) -> State:
#     reset_state = self.env.reset(rng)
#     reset_state.metrics['reward'] = reset_state.reward
#     eval_metrics = EvalMetrics(
#         episode_metrics=jax.tree_util.tree_map(
#             jp.zeros_like, reset_state.metrics
#         ),
#         active_episodes=jp.ones_like(reset_state.reward),
#         episode_steps=jp.zeros_like(reset_state.reward),
#     )
#     reset_state.info['eval_metrics'] = eval_metrics
#     return reset_state

#   def step(self, state: State, action: jax.Array) -> State:
#     state_metrics = state.info['eval_metrics']
#     if not isinstance(state_metrics, EvalMetrics):
#       raise ValueError(
#           f'Incorrect type for state_metrics: {type(state_metrics)}'
#       )
#     del state.info['eval_metrics']
#     nstate = self.env.step(state, action)
#     nstate.metrics['reward'] = nstate.reward
#     episode_steps = jp.where(
#         state_metrics.active_episodes,
#         nstate.info['steps'],
#         state_metrics.episode_steps,
#     )
#     episode_metrics = jax.tree_util.tree_map(
#         lambda a, b: a + b * state_metrics.active_episodes,
#         state_metrics.episode_metrics,
#         nstate.metrics,
#     )
#     active_episodes = state_metrics.active_episodes * (1 - nstate.done)

#     eval_metrics = EvalMetrics(
#         episode_metrics=episode_metrics,
#         active_episodes=active_episodes,
#         episode_steps=episode_steps,
#     )
#     nstate.info['eval_metrics'] = eval_metrics
#     return nstate


# class DomainRandomizationVmapWrapper(Wrapper):
#   """Wrapper for domain randomization."""

#   def __init__(
#       self,
#       env: Env,
#       randomization_fn: Callable[[System], Tuple[System, System]],
#   ):
#     super().__init__(env)
#     self._sys_v, self._in_axes = randomization_fn(self.sys)

#   def _env_fn(self, sys: System) -> Env:
#     env = self.env
#     env.unwrapped.sys = sys
#     return env

#   def reset(self, rng: jax.Array) -> State:
#     def reset(sys, rng):
#       env = self._env_fn(sys=sys)
#       return env.reset(rng)

#     state = jax.vmap(reset, in_axes=[self._in_axes, 0])(self._sys_v, rng)
#     return state

#   def step(self, state: State, action: jax.Array) -> State:
#     def step(sys, s, a):
#       env = self._env_fn(sys=sys)
#       return env.step(s, a)

#     res = jax.vmap(step, in_axes=[self._in_axes, 0, 0])(
#         self._sys_v, state, action
#     )
#     return res
