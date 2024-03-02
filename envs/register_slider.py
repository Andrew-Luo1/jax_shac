import sys
from pathlib import Path

from brax import envs
from jax_shac.envs.mjx_envs import State, MjxEnv
from jax_shac.envs import slider_motor_xml, slider_position_xml
import mujoco
import jax.numpy as jp
import jax

class Slider(MjxEnv):
    
    def __init__(self, **kwargs):
        self.motor = kwargs.get("motor", True)
        self.ref_pos = kwargs.get("ref_pos", 2.5)
        
        if self.motor:
            xml_path = slider_motor_xml
            self.action_strength = 1
            self.action_penalty = 0
        else:
            xml_path = slider_position_xml
            self.action_strength = 4
            self.action_penalty = 0

        self.position_penalty = 1.0

        mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.init_noise = kwargs.get("init_noise", 1)
        
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        mj_model.opt.iterations = 1
        mj_model.opt.ls_iterations = 4
        self._init_q = jp.array(0).reshape((1,)) # Ensure not a scalar.
        n_frames = 10 # 50 fps
        self.model = mj_model
        kwargs['physics_steps_per_control_step'] = kwargs.get('physics_steps_per_control_step', n_frames)
        
        super().__init__(mj_model=mj_model, **kwargs)
                
    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        q = self._init_q + 1*jax.random.uniform(
            rng1, (self.sys.nq,), minval=-self.init_noise/2, maxval=self.init_noise/2
        )
        qd = 0.5*jax.random.uniform(
            rng2, (self.sys.nv,), minval=-self.init_noise/2, maxval=self.init_noise/2
        )
        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)
        reward, done = jp.zeros(2)
        metrics = {}

        return jax.lax.stop_gradient(State(pipeline_state, obs, reward, done, metrics))

    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        action = jp.clip(action, -1, 1)
        action *= self.action_strength
        pipeline_state = self.pipeline_step(state.pipeline_state, action)
        obs = self._get_obs(pipeline_state)

        reward = - jp.square(obs[0] - self.ref_pos) * self.position_penalty \
                 - jp.square(action.reshape()) * self.action_penalty

        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=0.0
        )

    def _get_obs(self, pipeline_state: State) -> jax.Array:
        """Observe cartpole body position and velocities."""
        return jp.concatenate([pipeline_state.qpos, pipeline_state.qvel])

envs.register_environment("slider", Slider)