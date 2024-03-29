# Copyright 2022 The Brax Authors.
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

"""Network definitions."""

import dataclasses
from typing import Any, Callable, Sequence, Tuple
import warnings

from brax.training import types
from brax.training.spectral_norm import SNDense
from flax import linen
import jax
import jax.numpy as jnp
import functools

ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]


@dataclasses.dataclass
class FeedForwardNetwork:
  init: Callable[..., Any]
  apply: Callable[..., Any]

from flax.linen.initializers import constant, orthogonal
import numpy as np

class SmallNet(linen.Module):
  """ Architecture from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_envpool_xla_jax_scan.py """
  layer_sizes: Sequence[int]
  # Unused args; just for API compatibility with MLP. 
  activation: ActivationFn = linen.relu
  kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
  activate_final: bool = False
  bias: bool = True
  dense_init_var: float = 0.01
  layer_norm: bool = False
  output_dim: int = dataclasses.field(init=False)
  
  def setup(self):
    self.output_dim = self.layer_sizes[-1]
  
  @linen.compact
  def __call__(self, x):
      # x = jnp.transpose(x, (0, 2, 3, 1))
      # x = x / (255.0)
      x = linen.Conv(
          32,
          kernel_size=(5, 5),
          strides=(1, 3),
          padding="VALID",
          kernel_init=orthogonal(np.sqrt(2)),
          bias_init=constant(0.0),
      )(x)
      x = linen.elu(x)
      x = linen.LayerNorm()(x)
      x = linen.Conv(
          16,
          kernel_size=(3, 3),
          strides=(1, 2),
          padding="VALID",
          kernel_init=orthogonal(np.sqrt(2)),
          bias_init=constant(0.0),
      )(x)
      x = linen.elu(x)
      x = linen.LayerNorm()(x)
      x = linen.Conv(
          8,
          kernel_size=(3, 3),
          strides=(1, 1),
          padding="VALID",
          kernel_init=orthogonal(np.sqrt(2)),
          bias_init=constant(0.0),
      )(x)
      x = linen.elu(x)
      x = linen.LayerNorm()(x)
      x = x.reshape((x.shape[0], -1))
      x = linen.Dense(128, kernel_init=orthogonal(np.sqrt(2)), 
                      bias_init=constant(0.0))(x)
      x = linen.elu(x)
      x = linen.LayerNorm()(x)
      # Output layer
      x = linen.Dense(self.output_dim, kernel_init=orthogonal(self.dense_init_var), 
                      bias_init=constant(0.0))(x)
      
      return x

# @vPsC: CNN-only Network

class MLP(linen.Module):
  """MLP module."""
  layer_sizes: Sequence[int]
  activation: ActivationFn = linen.relu
  kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
  activate_final: bool = False
  bias: bool = True
  layer_norm: bool = True

  @linen.compact
  def __call__(self, data: jnp.ndarray):
    hidden = data
    for i, hidden_size in enumerate(self.layer_sizes):
      hidden = linen.Dense(
          hidden_size,
          name=f'hidden_{i}',
          kernel_init=self.kernel_init,
          use_bias=self.bias)(
              hidden)
      if i != len(self.layer_sizes) - 1 or self.activate_final:
        hidden = self.activation(hidden)
        if self.layer_norm:
          hidden = linen.LayerNorm()(hidden)
    return hidden


class SNMLP(linen.Module):
  """MLP module with Spectral Normalization."""
  layer_sizes: Sequence[int]
  activation: ActivationFn = linen.relu
  kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
  activate_final: bool = False
  bias: bool = True

  @linen.compact
  def __call__(self, data: jnp.ndarray):
    hidden = data
    for i, hidden_size in enumerate(self.layer_sizes):
      hidden = SNDense(
          hidden_size,
          name=f'hidden_{i}',
          kernel_init=self.kernel_init,
          use_bias=self.bias)(
              hidden)
      if i != len(self.layer_sizes) - 1 or self.activate_final:
        hidden = self.activation(hidden)
    return hidden

# def make_CNN_preprocessing_network( # @ vPsC
#   ...):
#   """ 
#   Tuple: normalizer params, CNN params. 
#   """
#   pass

def make_policy_network(
    param_size: int,
    obs_size: tuple,
    preprocess_observations_fn: types.PreprocessObservationFn = types
    .identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    layer_norm: bool = False,
    vision: bool = False) -> FeedForwardNetwork:
  
  """Creates a policy network."""
  Archi = functools.partial(SmallNet, dense_init_var=0.01) if vision else MLP
  policy_module = Archi(
      layer_sizes=list(hidden_layer_sizes) + [param_size],
      activation=activation,
      layer_norm=layer_norm,
      kernel_init=jax.nn.initializers.lecun_uniform())

  def apply(processor_params, policy_params, obs):
    obs = preprocess_observations_fn(obs, processor_params)
    return policy_module.apply(policy_params, obs)

  dummy_obs = jnp.zeros((1, *obs_size))
  return FeedForwardNetwork(
      init=lambda key: policy_module.init(key, dummy_obs), apply=apply)


def make_value_network(
    obs_size: tuple,
    preprocess_observations_fn: types.PreprocessObservationFn = types
    .identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    layer_norm: bool = False,
    vision: bool = False) -> FeedForwardNetwork:
  
  """Creates a value network."""
  Archi = functools.partial(SmallNet, dense_init_var=1) if vision else MLP
  value_module = Archi(
      layer_sizes=list(hidden_layer_sizes) + [1],
      activation=activation,
      layer_norm=layer_norm,
      kernel_init=jax.nn.initializers.lecun_uniform())

  def apply(processor_params, policy_params, obs):
    obs = preprocess_observations_fn(obs, processor_params)
    return jnp.squeeze(value_module.apply(policy_params, obs), axis=-1)

  dummy_obs = jnp.zeros((1, *obs_size))
  return FeedForwardNetwork(
      init=lambda key: value_module.init(key, dummy_obs), apply=apply)


def make_q_network(
    obs_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types
    .identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    n_critics: int = 2) -> FeedForwardNetwork:
  """Creates a value network."""

  class QModule(linen.Module):
    """Q Module."""
    n_critics: int

    @linen.compact
    def __call__(self, obs: jnp.ndarray, actions: jnp.ndarray):
      hidden = jnp.concatenate([obs, actions], axis=-1)
      res = []
      for _ in range(self.n_critics):
        q = MLP(
            layer_sizes=list(hidden_layer_sizes) + [1],
            activation=activation,
            kernel_init=jax.nn.initializers.lecun_uniform())(
                hidden)
        res.append(q)
      return jnp.concatenate(res, axis=-1)

  q_module = QModule(n_critics=n_critics)

  def apply(processor_params, q_params, obs, actions):
    obs = preprocess_observations_fn(obs, processor_params)
    return q_module.apply(q_params, obs, actions)

  dummy_obs = jnp.zeros((1, obs_size))
  dummy_action = jnp.zeros((1, action_size))
  return FeedForwardNetwork(
      init=lambda key: q_module.init(key, dummy_obs, dummy_action), apply=apply)


def make_model(
    layer_sizes: Sequence[int],
    obs_size: int,
    activation: Callable[[jnp.ndarray], jnp.ndarray] = linen.swish,
    spectral_norm: bool = False,
) -> FeedForwardNetwork:
  """Creates a model.

  Args:
    layer_sizes: layers
    obs_size: size of an observation
    activation: activation
    spectral_norm: whether to use a spectral normalization (default: False).

  Returns:
    a model
  """
  warnings.warn(
      'make_model is deprecated, use make_{policy|q|value}_network instead.')
  dummy_obs = jnp.zeros((1, obs_size))
  if spectral_norm:
    module = SNMLP(layer_sizes=layer_sizes, activation=activation)
    model = FeedForwardNetwork(
        init=lambda rng1, rng2: module.init({
            'params': rng1,
            'sing_vec': rng2
        }, dummy_obs),
        apply=module.apply)
  else:
    module = MLP(layer_sizes=layer_sizes, activation=activation)
    model = FeedForwardNetwork(
        init=lambda rng: module.init(rng, dummy_obs), apply=module.apply)
  return model


def make_models(policy_params_size: int,
                obs_size: int) -> Tuple[FeedForwardNetwork, FeedForwardNetwork]:
  """Creates models for policy and value functions.

  Args:
    policy_params_size: number of params that a policy network should generate
    obs_size: size of an observation

  Returns:
    a model for policy and a model for value function
  """
  warnings.warn(
      'make_models is deprecated, use make_{policy|q|value}_network instead.')
  policy_model = make_model([32, 32, 32, 32, policy_params_size], obs_size)
  value_model = make_model([256, 256, 256, 256, 256, 1], obs_size)
  return policy_model, value_model