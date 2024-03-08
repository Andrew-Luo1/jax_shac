""" 
A collection of functions imported to the SHAC trainer, for debugging / visualization purposes. 
"""
import jax.numpy as jnp
from typing import Callable, Optional, Tuple, Union
import pickle
from brax.v1 import envs as envs_v1
from brax import envs
from brax.training.types import PRNGKey
import jax
from brax.training import types
from pathlib import Path
import functools
import mujoco
import mujoco.mjx as mjx
import numpy as np

# Jacobian-able
def fjac_env_step(self, diffwrt, env_state, actions):
    sys = self.env.sys
    ps = env_state.pipeline_state
    ps = ps.tree_replace({
        'qpos': diffwrt[:sys.nq],
        'qvel': diffwrt[sys.nq:sys.nq+sys.nv],
        'ctrl': diffwrt[sys.nq+sys.nv:]
    })
    env_state = env_state.replace(pipeline_state=ps)
    
    nstate = self.env.step(env_state, actions)
    diffwrt_out = jnp.squeeze(jnp.concatenate(
                [jnp.expand_dims(nstate.pipeline_state.qpos, 1), 
                    jnp.expand_dims(nstate.pipeline_state.qvel, 1)],
                    axis=0))
    
    return diffwrt_out, nstate

def fscannable_jac_env_step(self,
    carry: Tuple[Union[envs.State, envs_v1.State], PRNGKey],
    _step_index: int,
    policy: types.Policy):
    env_state, key = carry
    key, key_sample = jax.random.split(key)
    actions = policy(env_state.obs, key_sample)[0]
    
    mjx_data = env_state.pipeline_state
    x_i = jnp.squeeze(
            jnp.concatenate(
                [ mjx_data.qpos.reshape(self.env.sys.nq), 
                    mjx_data.qvel.reshape(self.env.sys.nv),
                    actions.reshape(self.env.sys.nu)], 
                axis=0))
    
    cur_jac, nstate = self.jac_env_step(x_i, env_state, actions)
    extras = {"action": actions}
    return (nstate, key), (cur_jac, env_state, extras)
    
def fjac_rollout(self, policy_params, normalizer_params, state, key, unroll_length):
    key, key_unroll = jax.random.split(key)
    # As done in the paper to prevent gradient exposion.
    state = jax.lax.stop_gradient(state)

    # From Brax APG
    f = functools.partial(
        self.scannable_jac_env_step, policy=self.make_policy((normalizer_params, policy_params)))
    
    (state_h, _), (jacs, states, extras) = jax.lax.scan(f, (state, key_unroll), (jnp.array(range(unroll_length))))
    
    return jacs, states, state_h, extras

def fload_checkpoint(self, it):
    """ 
    Return the algo state for the specified iteration, saved under the checkpoints directory.
    """

    file_name = f'checkpoint_{it}.pkl'
    file_path = str(Path(Path(self.__file__).parent,
                    Path('checkpoints'),
                    Path(file_name)))
    return pickle.load(open(file_path, "rb"))

def fget_image(self, mj_data, camera = None) -> np.ndarray:
  """Renders the environment state."""
  mujoco.mj_forward(self.env.model, mj_data)
  # use the mjData object to update the renderer
  if camera is not None:
    self.renderer.update_scene(mj_data, camera=camera)
  else:
    self.renderer.update_scene(mj_data)
    
  return self.renderer.render()

def norm(l, axes):
    ret = jnp.square(l)
    ret = jnp.mean(ret, axis=axes)
    ret = jnp.sqrt(ret)
    return ret

def frender_states(self, states, jacs, thresh, camera=None):
    """ 
    If on step k, dx_{k+1}/dx_k is large, render the whole body as red.
    
    states: mjx state; batched across time.
    
    - [X] Detect if jacs too large
    - [ ] Copy the env model, to use for rendering
    - [X] Set geoms to red if that's the case
    - [X] Set them back to nominal otherwise
    """
    if "renderer" not in dir(self): # Only init once.
        self.renderer = mujoco.Renderer(self.env.model)
    
    # Ensure that it's not batched across environments.
    assert len(jacs.shape) == 3, "wrong jacobians dimension; have you unbatched environnments?"
    jacs = np.array(jacs)
    jac_mask = norm(jacs, axes=(1, 2))  > thresh
    
    # mj_data0 = mjx.get_data(self.env.model, s0.pipeline_state)
    # Index i of mj_datas corresponds to i+1 of the rollout. 
    mj_datas = mjx.get_data(self.env.model, states.pipeline_state)
    # mj_datas = [mj_data0] + mj_datas # Combine the lists
    
    im_buf = []
    rgba_red = [1, 0, 0 ,1]
    rgba_grey = [0.5, 0.5, 0.5, 1]

    for i, mj_data in enumerate(mj_datas):
        if jac_mask[i]:
            self.env.model.geom_rgba = np.repeat(np.reshape(rgba_red, (1,4)), self.env.model.ngeom, axis=0)
        else:
            # Reset cube color to grey
            self.env.model.geom_rgba = np.repeat(np.reshape(rgba_grey, (1,4)), self.env.model.ngeom, axis=0)
        
        im_buf.append(self.get_image(mj_data, camera))
    
    return im_buf