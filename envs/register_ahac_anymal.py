import sys
from pathlib import Path
import warnings
import functools 

from brax import envs
from mujoco import mjx
import jax
from jax import numpy as jp
import numpy as np
from typing import Any, Dict, Tuple, Union

from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.envs.base import Env, State
from ml_collections import config_dict
import mujoco

from . import anymal_xml
from .mjx_envs import State, MjxEnv
from ..utils.math_utils import axis_angle_to_quaternion

def get_config():
  """Returns reward config for anymal quadruped environment."""

  def get_default_rewards_config():
    default_config = config_dict.ConfigDict(
        dict(
            scales=config_dict.ConfigDict(
                dict(
                    up=0.1,
                    heading=1.0,
                    height=1.0,
                    progress=1.0,
                    action=-0.005,
                    termination=0.0,
                    healthy=0.0,
                    action_rate=0.0,
                    standing=0.0)
            ),
        )
    )
    return default_config

  default_config = config_dict.ConfigDict(
      dict(rewards=get_default_rewards_config(),))

  return default_config


class DiffAnymal(MjxEnv):
  """
  #### Tested Rewards (Walking)
  - Up
  - Termination
  - Standing
  - [X] Progress
  - [X] Height
  - Healthy
  - [X] Heading
  - [ ] action rate
  - [X] Action


  #### SIMULATION STABILITY ####
  - Setting action scale to 80 doesn't blow up nominally, but does upon vmapping env.step and env.reset. 
  - Lowering even down to 40 doesn't result in stable sim. 
  - 

  #### FEATURES ####
  DiffAnymal:
  - Termination Height: 0.25
  - Termination criteria: Termination height
  - dt: 50 fps
  - init: stochastic
    - xyz: -1 to 1 * 0.1
    - angle: -.5 to .5 * pi/12 (15 deg range)
    - axis: -.5 to .5; normalized. Then, set quat to axis_angle_to_quat. 
    - joint q: 0.2 * -.5 to .5
    - joint qd: 0.5 * -.5 to .5
  - obs noise: no
  - reward and obs clipping: no
  - termination reward (penalty): no

  AHAC:
  - Termination Height: 0.25
  - Termination criteria
      - a) simulator blows up
      - b) termination height
  - dt: 60 fps
  - obs noise: 
  - init: stochastic
      - xyz: -1 to 1 * 0.1
      - angle: -.5 to .5 * pi/12 (15 deg range)
      - axis: -.5 to .5; normalized. Then, set quat to axis_angle_to_quat. 
      - joint q: 0.2 * -.5 to .5
      - joint qd: 0.5 * -.5 to .5
  - obs noise: no
  - reward and obs clipping: no
  - termination reward (penalty): no

  Brax:
  - dt: 50 fps
  - init: deterministic
  - obs noise: yes
  - reward and obs clipping: yes
  - termination reward: yes
"""

  def __init__(
      self,
      action_scale: float=80,
      termination_height: float=0.25,
      s_afilt_buf: float=1,
      **kwargs,
  ):
    self.model_variant = kwargs.get('model_variant', 'anymal')

    match self.model_variant:
      case "anymal":
        f_path = anymal_xml
      case _:
        raise ValueError("Invalid model specified!")
    
    self.early_termination = kwargs.get('early_termination', True)

    mj_model = mujoco.MjModel.from_xml_path(f_path)

    self.s_afilt_buf = s_afilt_buf # Number of previous actions to average together.
    if s_afilt_buf > 1:
      warnings.warn("s_afilt_buf > 1 gives undefined observations")
    
    physics_steps_per_control_step = 10
    kwargs['physics_steps_per_control_step'] = kwargs.get(
        'physics_steps_per_control_step', physics_steps_per_control_step)
    super().__init__(mj_model=mj_model, **kwargs)    
    
    self.action_scale = action_scale
    self.termination_height = termination_height
    self.target = jp.array([10000.0, 0.0, 0.0])
    
    self._init_q = mj_model.keyframe('standing').qpos
    self._default_ap_pose = mj_model.keyframe('standing').qpos[7:]
    self.reward_config = get_config()

    # Used for termination.
    self.lowers = self._default_ap_pose - jp.array([0.2, 0.8, 0.8] * 4)
    self.uppers = self._default_ap_pose + jp.array([0.2, 0.8, 0.8] * 4)

  def reset(self, rng: jax.Array) -> State:
    rng, key_xyz, key_ang, key_ax, key_q, key_qd = jax.random.split(rng, 6)

    qpos = jp.array(self._init_q)
    qvel = jp.zeros(18)
    
    
    #### Add Randomness ####
  
    r_xyz = 0.2 * (jax.random.uniform(key_xyz, (3,))-0.5)
    r_angle = (jp.pi/12) * (jax.random.uniform(key_ang, (1,)) - 0.5) # 15 deg range
    r_axis = (jax.random.uniform(key_ax, (3,)) - 0.5)
    r_axis = r_axis / jp.linalg.norm(r_axis)
    r_quat = axis_angle_to_quaternion(r_axis, r_angle)

    r_joint_q = 0.2 * (jax.random.uniform(key_q, (12,)) - 0.5)
    r_joint_qd = 0.5 * (jax.random.uniform(key_qd, (12,)) - 0.5)
  
    qpos = qpos.at[0:3].set(qpos[0:3] + r_xyz)
    qpos = qpos.at[3:7].set(r_quat)
    qpos = qpos.at[7:19].set(qpos[7:19] + r_joint_q)
    qvel = qvel.at[6:18].set(qvel[6:18] + r_joint_qd)
    
    data = self.pipeline_init(qpos, qvel) # Computes kinematics
    state_info = {
        'rng': rng,
        'reward_tuple': {
            'up': 0.0,
            'heading': 0.0,
            'height': 0.0,
            'progress': 0.0,
            'action': 0.0,
            'termination': 0.0,
            'healthy': 0.0,
            'action_rate': 0.0,
            'standing': 0.0
        },
        'last_action': jp.zeros(12), # from MJX tutorial.
        'afilt_buf': jp.zeros((self.s_afilt_buf, 12))
    }

    x, xd = self._pos_vel(data)
    obs = self._get_obs(data.qpos, data.qvel, x, xd, state_info)
    reward, done = jp.zeros(2)
    metrics = {}
    for k in state_info['reward_tuple']:
      metrics[k] = state_info['reward_tuple'][k]
    state = State(data, obs, reward, done, metrics, state_info)
    return state

  def compute_termination(self, x: Any, obs: jax.Array, data: Any):

    # basic termination
    done = 0.0
    done = jp.where(x.pos[0, 2] < self.termination_height, 1.0, done)

    # AHAC env-style terminations. 
    joint_qd = data.qvel[6:]
    joint_angles = data.qpos[7:]

    nonfinite_mask = jp.any(~jp.isfinite(data.qpos))
    nonfinite_mask = jp.any(~jp.isfinite(data.qvel)) | nonfinite_mask
    nonfinite_mask = jp.any(~jp.isfinite(obs)) | nonfinite_mask

    invalid_value_mask = jp.any(jp.abs(joint_angles) > 1e4)
    invalid_value_mask = jp.any(jp.abs(joint_qd) > 1e4) | invalid_value_mask
    invalid_value_mask = jp.any(jp.abs(obs) > 1e4) | invalid_value_mask
    ahac_done = nonfinite_mask | invalid_value_mask

    # Must done to bool or can't use |. 
    done = jp.logical_or(done, ahac_done)
    done = jp.array(done, dtype=jp.float32) # Keep function outputs as float32's.

    # Brax-style terminations
    up = jp.array([0.0, 0.0, 1.0])
    done = jp.where(jp.dot(math.rotate(up, x.rot[0]), up) < 0, 1.0, done)
    
    # oor = jp.where(jp.logical_or(
    #     jp.any(joint_angles < .98 * self.lowers),
    #     jp.any(joint_angles > .98 * self.uppers)), 1.0, 0.0)
    # done = jp.where(jp.logical_and(oor, self.train_standing), 1.0, done)

    # Finally, are we even terminating?
    done = jp.where(self.early_termination, done, 0.0)

    return done
  
  def step(self, state: State, action: jax.Array) -> State:
        
    # Process actions. afilt_buf stores unfiltered actions. last_action is the last filtered action.
    action = jp.clip(action, -1, 1) # Raw action
    action *= self.action_scale
    afilt_buf = state.info['afilt_buf']
    afilt_buf = jp.roll(afilt_buf, shift=1, axis=0)
    afilt_buf = afilt_buf.at[0,:].set(action)
    f_action = jp.mean(afilt_buf, axis=0)
    state.info['afilt_buf'] = afilt_buf

    data = self.pipeline_step(state.pipeline_state, f_action)

    # observation data
    x, xd = self._pos_vel(data)
    obs = self._get_obs(data.qpos, data.qvel, x, xd, state.info)
    done = self.compute_termination(x, obs, data) # Done if nan's. 

    # Now, get rid of nan's so algorithms don't blow up.
    data = jax.tree_util.tree_map(lambda x: jp.nan_to_num(x), data)
    x, xd = self._pos_vel(data)
    obs = self._get_obs(data.qpos, data.qvel, x, xd, state.info)

    # reward
    reward_tuple = {
        'up': (
            self._reward_up(obs)
            * self.reward_config.rewards.scales.up
        ),
        'heading': (
          self._reward_heading(obs)
          * self.reward_config.rewards.scales.heading
        ),
        'height': (
          self._reward_height(obs) 
          * self.reward_config.rewards.scales.height
        ),
        'progress': (
          self._reward_progress(obs)
          * self.reward_config.rewards.scales.progress
        ),
        'action': (
          self._reward_action(action)
          * self.reward_config.rewards.scales.action
        ),
        'termination': (
          self._reward_termination(done)
          * self.reward_config.rewards.scales.termination
        ),
        'healthy': (
          self._reward_healthy(done)
          * self.reward_config.rewards.scales.healthy
        ),
        'action_rate': (
          self._reward_action_rate(action, state.info['last_action'])
          * self.reward_config.rewards.scales.action_rate
        ),
        'standing': (
          self._reward_standing(obs)
          * self.reward_config.rewards.scales.standing
        )
    }
    
    reward = sum(reward_tuple.values())
    # reward = jp.clip(reward * self.dt, 0.0, 10000.0)

    # state management
    state.info['reward_tuple'] = reward_tuple
    state.info['last_action'] = f_action # used for observation. 

    for k in state.info['reward_tuple'].keys():
      state.metrics[k] = state.info['reward_tuple'][k]

    state = state.replace(
        pipeline_state=data, obs=obs, reward=reward,
        done=done)
    return state

  def _get_obs(self, qpos: jax.Array, qvel: jax.Array,
               x: Transform, xd: Motion,
               state_info: Dict[str, Any]) -> jax.Array:
    """ 
    Brax 
    - yaw rate
    - projected gravity
    - motor angles 
    - history
    
    AHAC
    -[X] torso height
    -[X] torso rot -> x.rot[0,:]
    -[X] torso lin vel (world frame) -> xd.vel[0,:]
    -[X] torso ang vel -> qvel[3:6]
    -[X] joint positions
    -[X] joint velocities
    -[X] up vector (1D) -> just use projected_gravity
    -[X] dot prod between heading vector and command (1D)
    -[X] previous action
    
    """
    
    ang_vel = qvel[3:6]
    torso_rot = x.rot[0, :]
    up_vec = math.rotate(jp.array([0.0, 0.0, 1.0]), torso_rot)
    heading_vec = math.rotate(jp.array([1.0, 0.0, 0.0]), torso_rot)
    torso_pos = x.pos[0, :]
    to_target = self.target - torso_pos
    to_target = to_target / jp.linalg.norm(to_target)
    to_target = to_target.at[2].set(0.0) # Only concerned about x, y. 
    dir_dot = jp.dot(heading_vec, to_target)
   
    action = state_info['last_action']
    
    obs_list = jp.concatenate([
      x.pos[0, 2].reshape(1), # 0:1 Torso height
      torso_rot.reshape(4), # 1:5 Torso rotation
      xd.vel[0,:].reshape(3), # 5:8 Torso lin vel
      ang_vel.reshape(3), # 8:11 Torso angular velocity
      qpos[7:].reshape(12), # 11:23 Joint positions
      qvel[6:].reshape(12), # 23:35 Joint velocities
      up_vec[2].reshape(1), # 35:36
      dir_dot.reshape(1), # 36:37
      (action / self.action_scale).reshape(12) # 37:49
    ])

    return obs_list

  # ------------ reward functions----------------
  def _reward_up(self, obs) -> jax.Array:
    # Penalize z axis base linear velocity
    return obs[35]
  def _reward_heading(self, obs) -> jax.Array:
    return obs[36]
  def _reward_height(self, obs) -> jax.Array:
    return obs[0] - self.termination_height
  def _reward_progress(self, obs) -> jax.Array:
    return obs[5] # forward velocity
  def _reward_action(self, action) -> jax.Array:
    return jp.sqrt(jp.sum(jp.square(action)))
  def _reward_termination(
      self, done: jp.float32) -> jax.Array:
      return jp.where(done, 1.0, 0.0)
  def _reward_action_rate(
      self, act: jax.Array, last_act: jax.Array) -> jax.Array:
    # Penalize jerky motion
    return jp.sqrt(jp.sum(jp.square(act - last_act)))
  def _reward_healthy(
      self, done: jp.float32) -> jax.Array:
    return jp.where(jp.logical_not(done), 1.0, 0.0)
  def _reward_standing(self, obs) -> jax.Array:
    qpos = obs[11:23]
    pos_err = jp.sum(jp.square(qpos - self._default_ap_pose))
    # vel_err = jp.sum(jp.square(qvel))
    # qvel = obs[23:35]
    rew = jp.clip(pos_err, 0, 10)
    return rew
envs.register_environment('ahac_anymal', DiffAnymal)