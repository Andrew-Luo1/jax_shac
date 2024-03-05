import sys
from pathlib import Path

resources_dir = Path(__file__).parent.parent
sys.path.append(str(resources_dir))

from brax import envs
from jax_shac.envs.mjx_envs import State, MjxEnv
import mujoco
import jax.numpy as jp
import jax


class FramedHopper(MjxEnv):
    
    def __init__(self, **kwargs):
        """ 
        For sim stability, the ctrl is limited to 0, 1. 
        """
        self.action_strength = 0.5
        self.action_offset = 0.5

        self.velocity_reward = 1.0
        self.action_reward = -0.2 # Make it negative.

        xml_path = Path(Path(__file__).parent, Path("assets"), 
                        Path("framed_hopper.xml"))
        
        mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.init_noise = kwargs.get("init_noise", 0.1) # diameter of the ball
        
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        mj_model.opt.iterations = 1
        mj_model.opt.ls_iterations = 4
        self._init_q = jp.zeros(3) # Ensure not a scalar.
        n_frames = 10 # 50 fps
        self.model = mj_model
        kwargs['physics_steps_per_control_step'] = kwargs.get('physics_steps_per_control_step', n_frames)
        
        # Also give the observation history.
        self.num_history_steps = 3 # 1 step doesn't work; i.e. merely providing acceleration.

        super().__init__(mj_model=mj_model, **kwargs)
        assert self.sys.nq == 3, "Wrong model!"
                
    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        q = self._init_q + 1*jax.random.uniform(
            rng1, (self.sys.nq,), minval=-self.init_noise/2, maxval=self.init_noise/2
        )
        q = q.at[2].set(jp.clip(q[2], 0, 0.1)) # Range is 0 to 0.15
        
        qd = 0.5*jax.random.uniform(
            rng2, (self.sys.nv,), minval=-self.init_noise/2, maxval=self.init_noise/2
        )
        pipeline_state = self.pipeline_init(q, qd)

        # Tell the neural net about previous states and actions.
        d_oh = self.sys.nq + self.sys.nv + self.sys.nu
        obs_history = jp.zeros((self.num_history_steps, d_oh))

        obs = self._get_obs(pipeline_state, obs_history)
        reward, done = jp.zeros(2)
        metrics = {}

        state_info = {
            'reward_tuple': {
                'action': 0.0,
                'velocity': 0.0
            },
            'obs_hist': obs_history
        }

        return jax.lax.stop_gradient(State(pipeline_state, obs, reward, done, metrics, state_info))

    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        """A bit intricate: 
        - Raw Action: History buffer
        - Processed Action: Happens to correspond 1 to 1 to torque
        """
        raw_action = jp.clip(action, -1, 1)
        action = self.action_offset + raw_action*self.action_strength
        torques = action
        pipeline_state = self.pipeline_step(state.pipeline_state, action)
        obs = self._get_obs(pipeline_state, state.info['obs_hist'])

        reward_tuple = {
            'velocity': obs[3] * self.velocity_reward,
            'action': (jp.linalg.norm(torques.reshape()) 
                       * self.action_reward)
        }
        reward = sum(reward_tuple.values())
        state.info['reward_tuple'] = reward_tuple
        
        obs_hist = state.info['obs_hist']
        obs_hist = jp.roll(obs_hist, 1, axis=0)
        obs_hist = obs_hist.at[0].set(jp.concatenate(
            [pipeline_state.qpos.reshape(3),
             pipeline_state.qvel.reshape(3),
             raw_action.reshape(1)]))
        
        state.info['obs_hist'] = obs_hist

        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=0.0
        )

    def _get_obs(self, pipeline_state: State, obs_hist: jax.Array) -> jax.Array:
        """Observe cartpole body position and velocities."""
        hist_d = (self.sys.nq + self.sys.nv + self.sys.nu)*self.num_history_steps
        return jp.concatenate([pipeline_state.qpos, # 0:3
                               pipeline_state.qvel, # 3:6
                               jp.ravel(obs_hist).reshape(hist_d)]) # Plus a lot 

envs.register_environment("framed_hopper", FramedHopper)