from brax.base import Base, Motion, Transform
from brax.envs.base import Env
import jax
from jax import numpy as jp
from flax import struct
from typing import Any, Dict, Tuple
import mujoco
from mujoco import mjx
import numpy as np

@struct.dataclass
class State(Base):
  """Environment state for training and inference with brax.

  Args:
    pipeline_state: the physics state, mjx.Data
    obs: environment observations
    reward: environment reward
    done: boolean, True if the current episode has terminated
    metrics: metrics that get tracked per environment step
    info: environment variables defined and updated by the environment reset
      and step functions
  """

  pipeline_state: mjx.Data
  obs: jax.Array
  reward: jax.Array
  done: jax.Array
  metrics: Dict[str, jax.Array] = struct.field(default_factory=dict)
  info: Dict[str, Any] = struct.field(default_factory=dict)


class MjxEnv(Env):
  """API for driving an MJX system for training and inference in brax."""

  def __init__(
      self,
      mj_model: mujoco.MjModel,
      physics_steps_per_control_step: int = 1,
      **kwargs
  ):
    """Initializes MjxEnv.

    Args:
      mj_model: mujoco.MjModel
      physics_steps_per_control_step: the number of times to step the physics
        pipeline for each environment step
    """
    self.model = mj_model
    self.data = mujoco.MjData(mj_model)
    self.sys = mjx.device_put(mj_model)
    self._physics_steps_per_control_step = physics_steps_per_control_step

  def pipeline_init(
      self, qpos: jax.Array, qvel: jax.Array
  ) -> mjx.Data:
    """Initializes the physics state."""
    data = mjx.device_put(self.data)
    data = data.replace(qpos=qpos, qvel=qvel, ctrl=jp.zeros(self.sys.nu))
    data = mjx.forward(self.sys, data)
    return data

  def pipeline_step(
      self, data: mjx.Data, ctrl: jax.Array
  ) -> mjx.Data:
    """Takes a physics step using the physics pipeline."""
    def f(data, _):
      data = data.replace(ctrl=ctrl)
      return (
          mjx.step(self.sys, data),
          None,
      )
    data, _ = jax.lax.scan(f, data, (), self._physics_steps_per_control_step)
    return data

  @property
  def dt(self) -> jax.Array:
    """The timestep used for each env step."""
    return self.sys.opt.timestep * self._physics_steps_per_control_step

  @property
  def observation_size(self) -> int:
    rng = jax.random.PRNGKey(0)
    reset_state = self.unwrapped.reset(rng)
    return reset_state.obs.shape

  @property
  def action_size(self) -> int:
    return self.sys.nu

  @property
  def backend(self) -> str:
    return 'mjx'

  def _pos_vel(
      self, data: mjx.Data
      ) -> Tuple[Transform, Motion]:
    """Returns 6d spatial transform and 6d velocity for all bodies."""
    x = Transform(pos=data.xpos[1:, :], rot=data.xquat[1:, :])
    cvel = Motion(vel=data.cvel[1:, 3:], ang=data.cvel[1:, :3])
    offset = data.xpos[1:, :] - data.subtree_com[
        self.model.body_rootid[np.arange(1, self.model.nbody)]]
    xd = Transform.create(pos=offset).vmap().do(cvel)
    return x, xd