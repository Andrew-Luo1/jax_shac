# Copyright 2023 The Brax Authors.
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

# pylint:disable=g-multiple-import
"""An inverted pendulum environment."""
import sys
from pathlib import Path

resources_dir = Path(__file__).parent.parent
sys.path.append(str(resources_dir))

from brax import envs 
import jax
from jax import numpy as jp
import mujoco
from resources.mjx_envs import State, MjxEnv

class InvertedPendulum(MjxEnv):
  def __init__(self, **kwargs):
    assets_path = Path(Path(__file__).parent.parent, Path("assets"))
    xml_path = str(Path(assets_path, Path("inverted_pendulum.xml")))
    mj_model = mujoco.MjModel.from_xml_path(xml_path)

    self.init_noise = kwargs.get("init_noise", 0.01)
    mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    mj_model.opt.iterations = 1
    mj_model.opt.ls_iterations = 4

    self._init_q = jp.array([0, 0])
    
    # Also give the observation history.
    self.num_history_steps = 3

    n_frames = 5 # 100 fps; env default timestep is 0.002. 
    self.model = mj_model
    kwargs['physics_steps_per_control_step'] = kwargs.get('physics_steps_per_control_step', n_frames)
    self.pole_angle_penalty = 1
    self.pole_velocity_penalty = 0.0
    self.cart_pos_penalty = 0.01
    self.cart_vel_penalty = 0.0
    self.action_penalty = 0   

    super().__init__(mj_model=mj_model, **kwargs)

  def reset(self, rng: jax.Array) -> State:
    """Resets the environment to an initial state."""
    rng, rng1, rng2 = jax.random.split(rng, 3)

    q = self._init_q + jax.random.uniform(
        rng1, (self.sys.nq,), minval=-self.init_noise, maxval=self.init_noise
    )
    qd = jax.random.uniform(
        rng2, (self.sys.nv,), minval=-self.init_noise*0.5, maxval=self.init_noise*0.5
    )
    pipeline_state = self.pipeline_init(q, qd)
    reward, done = jp.zeros(2)
    metrics = {}

    d_oh = self.sys.nq + self.sys.nv + self.sys.nu
    obs_history = jp.zeros((self.num_history_steps, d_oh))

    obs = self._get_obs(pipeline_state, obs_history)
    
    state_info = {
        'reward_tuple': {
          "pole_angle" : 0.0,
          "pole_velocity" : 0.0,
          "cart_pos" : 0.0,
          "cart_vel" : 0.0,
          "action" : 0.0
        },
        'obs_hist': obs_history
    }
    
    return jax.lax.stop_gradient(State(pipeline_state, obs, reward, done, metrics, state_info))

  def step(self, state: State, action: jax.Array) -> State:
    """Run one timestep of the environment's dynamics."""
    raw_action = jp.clip(action, -1, 1)
    action = raw_action * self.action_strength
    pipeline_state = self.pipeline_step(state.pipeline_state, action)
    obs = self._get_obs(pipeline_state, state.info['obs_hist'])
        
    reward_tuple = {
      "pole_angle" :   - 1*jp.square(obs[1]) * self.pole_angle_penalty,
      "pole_velocity" :- 1*jp.square(obs[3]) * self.pole_velocity_penalty, 
      "cart_pos" : - 1*jp.square(obs[0]) * self.cart_pos_penalty,
      "cart_vel" : - 1*jp.square(obs[2]) * self.cart_vel_penalty,
      "action" : - 1*jp.square(action.reshape()) * self.action_penalty
    }

    reward = sum(reward_tuple.values())
    state.info['reward_tuple'] = reward_tuple

    obs_hist = state.info['obs_hist']
    obs_hist = jp.roll(obs_hist, 1, axis=0)
    obs_hist = obs_hist.at[0].set(jp.concatenate(
        [pipeline_state.qpos.reshape(2),
          pipeline_state.qvel.reshape(2),
          raw_action.reshape(1)]))
    
    state.info['obs_hist'] = obs_hist

    # done = jp.where(jp.abs(obs[1]) > 0.2, 1.0, 0.0)
    done = 0.0

    return state.replace(
        pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
    )

  @property
  def action_size(self):
    return 1

  def _get_obs(self, pipeline_state: State, obs_hist: jax.Array) -> jax.Array:
    """Observe cartpole body position and velocities."""
    hist_d = (self.sys.nq + self.sys.nv + self.sys.nu)*self.num_history_steps

    return jp.concatenate([pipeline_state.qpos, # cart position, cart velocity
                           pipeline_state.qvel, # pole position (rad; 0 deg for upright); pole angular velocity (rad/s)
                           jp.ravel(obs_hist).reshape(hist_d)]) # observations for previous timesteps 


envs.register_environment("inverted_pendulum_diffble", InvertedPendulum) # Register as a Brax environment