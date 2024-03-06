"""
LIMITATIONS:
- doesn't support multiple processes. 
"""


"""Short-Horizon Actor Critic.

See: https://arxiv.org/pdf/2204.07137.pdf
and  https://github.com/NVlabs/DiffRL/blob/main/algorithms/shac.py
"""

import functools
import time
from typing import Callable, Optional, Tuple, Union
import pickle
import os
from pathlib import Path

from absl import logging
from brax import envs
from brax.envs import training as orig_wraps
from brax.envs import wrappers
from brax.training import acting
from brax.training import gradients
from brax.training import types
from brax.v1 import envs as envs_v1
from brax.training.acme import running_statistics
from brax.training.acme import specs
from tensorboardX import SummaryWriter
from brax.training.types import Params
from brax.training.types import PRNGKey
from brax.training.types import Transition
import flax
import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import optax

# Safe Coding
import warnings
from jax.experimental import checkify
from jaxtyping import Array, Shaped, jaxtyped, Shaped
from typeguard import typechecked as typechecker

from . import losses as shac_losses # relative intra-package imports.
from . import networks as shac_networks
from . import brax_wrappers as brax_wrappers

InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]
Metrics = types.Metrics

@flax.struct.dataclass
class TrainingState:
  """Contains training state for the learner."""
  policy_optimizer_state: optax.OptState
  policy_params: Params
  value_optimizer_state: optax.OptState
  value_params: Params
  target_value_params: Params
  normalizer_params: running_statistics.RunningStatisticsState
  env_steps: jnp.ndarray

extra_fields=('truncation',) # for def env_step

def unvmap(x, ind):
    return jax.tree_util.tree_map(lambda x: x[ind], x)

def to_float64(v): # Used for numerical differencing. 
    return jax.tree_util.tree_map(lambda x: jnp.array(x, dtype=jnp.float64), v)

class SHAC:
    """ 
    Class version allows greater flexibility at cost of additional overhead.

    """
    def __init__(self,
                 environment: envs.Env,
                 num_timesteps: int,
                 episode_length: int,
                 num_envs: int = 1,
                 num_eval_envs: int = 128,
                 actor_learning_rate: float = 1e-2, # default for xu, cartpole
                 critic_learning_rate: float = 1e-3, # default for xu, cartpole
                 adam_b: Array = [0.7, 0.95],
                 entropy_cost: float = 1e-4,
                 discounting: float = 0.9,
                 seed: int = 0,
                 unroll_length: int = 10,
                 critic_batch_size: int = 5,
                 critic_epochs: int = 16,
                 num_evals: int = 1, # number of full evaluations; only done if num_tbx_evals = None. 
                 use_tbx = False, # number of tensorboardx evaluations; lighter than abv.
                 tbx_experiment_name = "",
                 tbx_logdir = "log",
                 scramble_initial_times = False,
                 normalize_observations: bool = False,
                 reward_scaling: float = 1.,
                 target_critic_alpha: float = 0.2,  # from xu: cartpole_swing_up
                 lambda_: float = .95,
                 deterministic_eval: bool = False,
                 network_factory: types.NetworkFactory[
                     shac_networks.SHACNetworks] = shac_networks.make_shac_networks,
                 log_sigma = None,
                 resample_init = False,
                 num_grad_checks = None, # number of random parameters to numerically differentiate every eval. 
                 policy_init_params = None,
                 normalizer_init_params = None,
                 save_all_checkpoints = False,
                 save_all_policy_gradients = False,
                 value_burn_in = 0,
                 progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
                 eval_env: Optional[envs.Env] = None,
                 polgrad_thresh = 1e3):
        
        # Save to self
        self.num_evals = num_evals
        self.progress_fn = progress_fn
        self.eval_env = eval_env
        self.deterministic_eval = deterministic_eval

        self.num_envs = num_envs
        self.num_eval_envs = num_eval_envs
        self.seed = seed
        self.critic_epochs = critic_epochs
        self.target_critic_alpha = target_critic_alpha

        self.use_tbx = use_tbx
        self.tbx_logdir = tbx_logdir
        self.tbx_experiment_name = tbx_experiment_name
        self.save_all_checkpoints = save_all_checkpoints
        self.save_all_policy_gradients = save_all_policy_gradients
        
        self.episode_length = episode_length
        self.num_grad_checks = num_grad_checks
        self.unroll_length = unroll_length
        self.discounting = discounting
        self.reward_scaling = reward_scaling
        self.lambda_ = lambda_
        self.log_sigma = log_sigma
        
        # Tricks
        self.scramble_initial_times = scramble_initial_times
        self.polgrad_thresh = polgrad_thresh
        self.value_burn_in = value_burn_in
        self.policy_init_params = policy_init_params
        self.normalizer_init_params = normalizer_init_params
        
        #### misc
        self.training_walltime = 0
        ####
        self.env_step_per_training_step = (
            num_envs * unroll_length)

        print("Env steps per training step: {}".format(self.env_step_per_training_step))
        self.num_evals_after_init = max(num_evals - 1, 1)

        # The number of training_step calls per training_epoch call.
        # equals to ceil(num_timesteps / (num_evals * env_step_per_training_step))

        self.num_training_steps_per_epoch = -(
            -num_timesteps // (self.num_evals_after_init * self.env_step_per_training_step))

        print("Training steps per epoch: {}".format(self.num_training_steps_per_epoch))

        # Verify grads only designed for specific scenario. 
        if num_grad_checks is not None:
            assert self.num_training_steps_per_epoch == 1, "Gradient verification not supported for multi training steps per epoch"
            assert log_sigma == None, "Gradient verification not supported for scalar variance"
            
            if num_grad_checks == 0: 
                num_grad_checks = None

            from jax import config
            config.update("jax_enable_x64", True)
            test_arr = jnp.zeros(1)
            if test_arr.dtype != jnp.float64:
                raise ValueError("Failed to set default Jax dtype to float64!")

        # Convert everything to ints. 
        self.critic_batch_size = int(critic_batch_size)
        num_envs = int(num_envs)
        num_timesteps = int(num_timesteps)
        unroll_length = int(unroll_length)

        # Used for critic epochs.
        assert (num_envs*unroll_length) % self.critic_batch_size == 0
        self.num_critic_minibatches = int((num_envs*unroll_length) / self.critic_batch_size)

        print("Critic minibatches per critic epoch: {}".format(self.num_critic_minibatches))
        # assert num_envs % device_count == 0

        env = environment
        env = orig_wraps.AutoResetWrapper(orig_wraps.EpisodeWrapper(env, episode_length, action_repeat=1))
        if resample_init:
            env = brax_wrappers.AutoSampleInitQ(env)
        
        self.env = env
        
        self.reset_fn = jax.jit(jax.vmap(env.reset))

        normalize = lambda x, y: x
        if normalize_observations:
            normalize = running_statistics.normalize
        
        self.shac_network = network_factory(
            env.observation_size,
            env.action_size,
            preprocess_observations_fn=normalize)
        
        if log_sigma is not None:
            self.make_policy = shac_networks.make_inference_fn_log_sigma(self.shac_network)
        else:
            self.make_policy = shac_networks.make_inference_fn(self.shac_network)

        # betas, default learning rate, clipping agree with Xu. 
        self.policy_optimizer = optax.chain(
            optax.clip(1.0),
            optax.adam(learning_rate=actor_learning_rate, b1=adam_b[0], b2=adam_b[1])
        )
        self.value_optimizer = optax.chain(
            optax.clip(1.0),
            optax.adam(learning_rate=critic_learning_rate, b1=adam_b[0], b2=adam_b[1])
        )
        value_loss_fn = functools.partial(
            shac_losses.compute_shac_critic_loss,
            shac_network=self.shac_network)

        self.compute_td_lambda_vals = jax.jit(shac_losses.compute_td_lambda_vals)

        self.value_gradient_update_fn = gradients.gradient_update_fn(
            value_loss_fn, self.value_optimizer, 
            has_aux=True, pmap_axis_name=None)

        self.policy_loss_fn = functools.partial(
            shac_losses.compute_shac_policy_loss,
            shac_network=self.shac_network,
            discounting=discounting,
            reward_scaling=reward_scaling)
        
        if log_sigma is not None:
            entropy_loss_fn = shac_losses.compute_entropy_lnsig_loss
        else:
            entropy_loss_fn = shac_losses.compute_entropy_loss
        
        self.policy_entropy_loss_fn = functools.partial(
            entropy_loss_fn,
            shac_network=self.shac_network,
            entropy_cost=entropy_cost)
        
        in_axes = (None,)*3 + (0,)*2
        g_fn = jax.value_and_grad(self.rollout_loss_fn, has_aux=True)
        self.vg_fn = jax.vmap(g_fn, in_axes=in_axes) # Expected output: b x 1
        
        self.policy_gradient_update_fn = jax.jit(
            self.policy_gradient_update_fn)
        
        self.critic_epoch = jax.jit(self.critic_epoch)    
        
        self.training_step = checkify.checkify(self.training_step)
    
        self.training_epoch = jax.jit(self.training_epoch)
            
    def scramble_times(self, state, key):
        import numpy as np
        import copy
        
        """ 
        Scramble the time in environment. Helps improve stability when all envs would otherwise simultaneously truncate at the same time, for example when there's no reset criteria except episode length. 
        """

        key = jax.random.PRNGKey(0)
        key, key_steps = jax.random.split(key)

        steps = jnp.round(jax.random.uniform(key_steps, (self.num_envs,), maxval=self.episode_length))
        steps = jnp.array(steps, dtype=jnp.int32)

        info = copy.deepcopy(state.info)
        info['steps'] = steps
        
        return state.replace(info=info)
    
    def fd_gradient_checks(self,
        policy_params, value_params,
        normalizer_params, state, ind_key,
        unroll_key, ad_grads):

        policy_params = to_float64(policy_params)

        # Make sure the inputs are already unbatched! 
        @jaxtyped(typechecker=typechecker)
        def check_ins(state_check: Shaped[Array, "dim1"], 
                       key_check: Shaped[Array, "dim2"]):
            pass
        check_ins(state.pipeline_state.qpos, unroll_key)
        
        eps = 1e-7
        flattened_polp, polp_tree_fn = ravel_pytree(policy_params)
        flattened_ad_grads, _ = ravel_pytree(ad_grads)
        rand_inds = jax.random.choice(ind_key, flattened_polp.shape[0], shape=(self.num_grad_checks,), replace=False)
        
        def fd_gradient_check(_, x):
            ind2eval = x
            orig_val = flattened_polp[ind2eval]
            
            p_pert_pars = polp_tree_fn(flattened_polp.at[ind2eval].set(orig_val + eps))
            m_pert_pars = polp_tree_fn(flattened_polp.at[ind2eval].set(orig_val - eps))

            p_pert_loss, _ = self.rollout_loss_fn(p_pert_pars, value_params, normalizer_params, state, unroll_key)
            m_pert_loss, _ = self.rollout_loss_fn(m_pert_pars, value_params, normalizer_params, state, unroll_key)
            
            fd_grad = (p_pert_loss - m_pert_loss) / (2*eps)
            
            return (), fd_grad

        _, fd_grads = jax.lax.scan(fd_gradient_check, init=(), xs=rand_inds)
        ad_grads = flattened_ad_grads[rand_inds]
        
        return jnp.sqrt(jnp.mean(jnp.square(ad_grads - fd_grads)))
    
    fd_gradient_checks = jax.jit(checkify.checkify(fd_gradient_checks))
    
    def env_step(self,
        carry: Tuple[Union[envs.State, envs_v1.State], PRNGKey],
        _step_index: int,
        policy: types.Policy):
        """ 
        From brax apg.
        """
        env_state, key = carry
        key, key_sample = jax.random.split(key)
        actions = policy(env_state.obs, key_sample)[0]
        nstate = self.env.step(env_state, actions)
        state_extras = {x: nstate.info[x] for x in extra_fields}
        return (nstate, key), Transition(
            observation=env_state.obs,
            action=actions,
            reward=nstate.reward,
            discount=1 - nstate.done,
            next_observation=nstate.obs,
            extras={'state_extras': state_extras})
    
    def rollout_loss_fn(self,
        policy_params, value_params, 
        normalizer_params, state, 
        key):
        
        # Ensure non-batched
        @jaxtyped(typechecker=typechecker)
        def ensure_non_batched(obs: Shaped[Array, "obs_size"]):
            pass
        ensure_non_batched(state.obs)

        key, key_unroll = jax.random.split(key)

        # As done in the paper to prevent gradient exposion.
        state = jax.lax.stop_gradient(state)

        # From Brax APG
        f = functools.partial(
            self.env_step, policy=self.make_policy((normalizer_params, policy_params)))
        
        (state_h, key), data = jax.lax.scan(f, (state, key_unroll), (jnp.array(range(self.unroll_length))))
        
        key, key_entropy = jax.random.split(key)
        policy_loss, metrics = self.policy_loss_fn(
            value_params=value_params,
            normalizer_params=normalizer_params,
            data=data,
            last_state=state_h
        )
        
        entropy_loss = self.policy_entropy_loss_fn(
            policy_params=policy_params,
            normalizer_params=normalizer_params,
            data=data,
            rng=key_entropy
        )
        metrics['entropy_loss'] = entropy_loss
        
        loss = policy_loss+entropy_loss

        return loss.reshape(), (state_h, data, metrics)
            
    def policy_gradient_update_fn(self,
        policy_params, value_params, 
        normalizer_params, state, key,
        optimizer_state, env_steps):

        # Ensure that state and key are batched. 
        @jaxtyped(typechecker=typechecker)
        def ensure_batched(obs: Shaped[Array, "b obs"], key: Shaped[Array, "b 2"]):
            pass
        ensure_batched(state.obs, key)

        (bvalue, baux), bgrad = self.vg_fn(
            policy_params, value_params, normalizer_params,
            state, key)
        
        # Nans can occur upon hitting joint limits.
        policy_grad = jax.tree_util.tree_map(lambda x: jnp.nanmean(x, axis=0), bgrad)

        flattened_vals, _ = ravel_pytree(policy_grad)
        agg = jnp.sqrt(jnp.mean(jnp.square(flattened_vals)))
        # jax.debug.print("Gradient norm = {x}", x=agg)
        check1 = jnp.where(agg > 1e-6, 0, 1) # Default python comparators don't work; traced bool error. 
        check2 = jnp.where(agg < self.polgrad_thresh, 0, 1)
        check3 = jnp.isnan(agg)

        checkify.check(check1 == 0,
                       "Gradient ~ 0!",
                       check1=check1)
        checkify.check(check2 == 0,
                       "Gradient too large! Policy not updated", check2=check2)
        checkify.check(check3 == 0,
                       "Gradient is nan!",
                       check3=check3)
        
        # Optimizer step
        params_update, noptimizer_state = self.policy_optimizer.update(
            policy_grad, optimizer_state)
        npolicy_params = optax.apply_updates(
            policy_params, params_update)

        policy_loss = bvalue
        state, data, policy_metrics = baux

        policy_metrics['policy_gradient'] = policy_grad

        # What percentage of all values were nan?
        flattened_bgrad, _ = ravel_pytree(bgrad)
        policy_metrics['p_nan_grads'] = jnp.sum(jnp.isnan(flattened_bgrad)) / flattened_bgrad.shape[0]

        if self.num_grad_checks is not None:
            policy_metrics['b_policy_gradient'] = bgrad # For gradient checking. Don't store usually; can easily get to 60+ mb. 
        policy_metrics['unroll_keys'] = key
        
        # Value function burn-in: we don't update the policy. Update_policy must be int. 
        n_train_step = env_steps / self.env_step_per_training_step
        update_policy = jnp.where(jnp.logical_and(agg < self.polgrad_thresh , (n_train_step >= self.value_burn_in)), 1, 0)
        
        policy_params = jax.tree_util.tree_map(
            lambda x, y: x * update_policy + y * (1-update_policy), 
            npolicy_params, policy_params)
        
        optimizer_state = jax.tree_util.tree_map(
            lambda x, y: x * update_policy + y * (1-update_policy),
            noptimizer_state, optimizer_state)
        
        return (policy_loss, (state, data, policy_metrics)
                ), policy_params, optimizer_state

    def critic_sgd_step(self,
        carry, x: Transition,
        normalizer_params: running_statistics.RunningStatisticsState):
        
        obs, target_vals = x
        optimizer_state, params, key = carry
        key, key_loss = jax.random.split(key)
        (_, metrics), params, optimizer_state = self.value_gradient_update_fn(
            params,
            normalizer_params,
            obs,
            target_vals,
            optimizer_state=optimizer_state)

        return (optimizer_state, params, key), metrics
    
    def critic_epoch(self,
                     carry, unused_t, target_vals, obs: jnp.ndarray,
                normalizer_params: running_statistics.RunningStatisticsState):
        optimizer_state, params, key = carry

        key, key_perm, key_grad = jax.random.split(key, 3)

        def convert_data(x: jnp.ndarray):
            x = jnp.reshape(x, (self.num_critic_minibatches, self.critic_batch_size, -1))
            x = jax.random.permutation(key_perm, x)

            return x

        # CHECK target_vals
        shuffled_target_vals = convert_data(target_vals)
        shuffled_obs = convert_data(obs)
        # shuffled_data = jax.tree_util.tree_map(convert_data, data) # same key so shuffling is aligned.

        # @jaxtyped(typechecker=typechecker)
        # def check_tv(tvs: Shaped[Array, "num_critic_minibatches critic_batch_size 1"], 
        #              obs: Shaped[Array, "num_critic_minibatches critic_batch_size obs_d"]):
        #     pass
        # check_tv(shuffled_target_vals, shuffled_data.observation)
        
        (optimizer_state, params, _), metrics = jax.lax.scan(
            functools.partial(self.critic_sgd_step, normalizer_params=normalizer_params),
            (optimizer_state, params, key_grad),
            (shuffled_obs, shuffled_target_vals),
            length=self.num_critic_minibatches)
        return (optimizer_state, params, key), metrics

    def training_step(self,
        carry: Tuple[TrainingState, envs.State, PRNGKey],
        unused_t) -> Tuple[Tuple[TrainingState, envs.State, PRNGKey], Metrics]:
        
        training_state, state, key = carry
        
        # Check it's batched
        @jaxtyped(typechecker=typechecker)
        def ensure_batched(obs: Shaped[Array, "bdim obs_size"]):
            pass
        ensure_batched(state.obs)

        key = key.reshape(2) # TODO: Remove
        key_sgd, key_generate_unroll, new_key = jax.random.split(key, 3)
        key_generate_unroll = jax.random.split(key_generate_unroll, self.num_envs)
        (_, (state, data, policy_metrics)
            ), policy_params, policy_optimizer_state = self.policy_gradient_update_fn(
            training_state.policy_params, training_state.target_value_params,
            training_state.normalizer_params, state, key_generate_unroll,
            optimizer_state=training_state.policy_optimizer_state, 
            env_steps=training_state.env_steps)
        
        # Ensure outputs of next_values are batched.
        next_values = policy_metrics['next_values']
        @jaxtyped(typechecker=typechecker)
        def check_ins(dis: Shaped[Array, "unroll_length num_envs"], 
                       nv: Shaped[Array, "unroll_length num_envs"]):
            pass
        check_ins(data.discount, next_values)
        
        # Update normalization params and normalize observations.
        n_normalizer_params = running_statistics.update(
            training_state.normalizer_params,
            data.observation)
        
        # A) Value function burn-in: we don't update the normalizer.
        n_train_step = training_state.env_steps / self.env_step_per_training_step
        # B) Don't update normalizer if policy grad blew up. 
        policy_grad = policy_metrics['policy_gradient']
        flattened_vals, _ = ravel_pytree(policy_grad)
        agg = jnp.sqrt(jnp.mean(jnp.square(flattened_vals)))

        update_policy = jnp.where(jnp.logical_and(agg < self.polgrad_thresh, n_train_step >= self.value_burn_in), 1, 0)

        normalizer_params = jax.tree_util.tree_map(
            lambda x, y: x * update_policy + y * (1-update_policy),
            n_normalizer_params, training_state.normalizer_params)

        target_vals = self.compute_td_lambda_vals(data,
                                             next_values,
                                             self.discounting,
                                             self.reward_scaling,
                                             self.lambda_)
        
        # CHECK target_vals
        @jaxtyped(typechecker=typechecker)
        def check_tv(tvs: Shaped[Array, "unroll_length num_envs"]):
            pass
        check_tv(target_vals)
        
        # Partial is used to pass constants through scans. 
        # Re-initialize the critic for fresh fits every time. Has been observed to improve critic stability.
        key_sgd, key_init_value = jax.random.split(key_sgd)
        value_params = self.shac_network.value_network.init(key_init_value)
        value_optimizer_state = self.value_optimizer.init(value_params)

        pcritic_epoch = functools.partial(self.critic_epoch, obs=data.observation, normalizer_params=normalizer_params, target_vals=target_vals)

        (value_optimizer_state, value_params, _), metrics = jax.lax.scan(pcritic_epoch, (value_optimizer_state, value_params, key_sgd), (), length=self.critic_epochs)

        # Alpha usually < 0.5, so you mostly keep the new critic.
        target_value_params = jax.tree_util.tree_map(
            lambda x, y: x * self.target_critic_alpha + y * (1-self.target_critic_alpha), training_state.target_value_params,
            value_params)

        metrics.update(policy_metrics)
        #### DEBUG
        agg = jnp.sqrt(jnp.mean(jnp.square(target_vals)))
        metrics['target_value_params_size'] = agg
        agg = jnp.sqrt(jnp.mean(jnp.square(next_values)))
        metrics['target_critic_output_size'] = agg
        agg = jnp.sqrt(jnp.mean(jnp.square(data.action)))
        metrics['action_size'] = agg

        new_training_state = TrainingState(
            policy_optimizer_state=policy_optimizer_state,
            policy_params=policy_params,
            value_optimizer_state=value_optimizer_state,
            value_params=value_params,
            target_value_params=target_value_params,
            normalizer_params=normalizer_params,
            env_steps=training_state.env_steps + self.env_step_per_training_step)
        return (new_training_state, state, new_key), metrics
    
    def training_step_wrapper(self,
        carry: Tuple[TrainingState, envs.State, PRNGKey],
        unused_t) -> Tuple[Tuple[TrainingState, envs.State, PRNGKey], Metrics]:
        """ 
        If not for this, only 1 checkify error per epoch can get returned. Now it's 1 error per training step.
        """
        err, (c, x) = self.training_step(carry, unused_t)
        return c, (x, err)
    
    def training_epoch(self, training_state: TrainingState, state: envs.State,
                        key: PRNGKey) -> Tuple[TrainingState, envs.State, Metrics]:
        """ 
        Run a bunch of training steps (policy updates).
        """
        (training_state, state, _), (loss_metrics, err) = jax.lax.scan(
            self.training_step_wrapper, (training_state, state, key), (),
            length=self.num_training_steps_per_epoch)
            # loss_metrics = jax.tree_util.tree_map(jnp.mean, loss_metrics)
        return training_state, state, loss_metrics, err

    def training_epoch_with_timing(self,
        training_state: TrainingState, env_state: envs.State,
        key: PRNGKey) -> Tuple[TrainingState, envs.State, Metrics]:
        
        """ 
        Wrapper around training_epoch to time it.
        """
        t = time.time()
        training_state, env_state, metrics, err = self.training_epoch(
            training_state, env_state, key)
        
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

        #### CHECKIFYING
        # err.throw()
        err = err.get()
        if err:
            print(err)

        #### TIMING ####
        epoch_training_time = time.time() - t
        self.training_walltime += epoch_training_time
        sps = (self.num_training_steps_per_epoch *
                self.env_step_per_training_step) / epoch_training_time
        metrics = {
            'training/sps': sps,
            'training/walltime': self.training_walltime,
            **{f'training/{name}': value for name, value in metrics.items()}
        }

        return training_state, env_state, metrics
    
    def init_training_state(self, key):
        """ 
        Uses the original key from __init__. 
        """
        
        key_policy, key_value, key = jax.random.split(key, 3)

        if self.policy_init_params is None:
            policy_init_params = self.shac_network.policy_network.init(key_policy)
        else:
            policy_init_params = self.policy_init_params
        if self.normalizer_init_params is None:
            if self.num_grad_checks is not None:
                dtype = jnp.float64
            else:
                dtype = jnp.float32

            normalizer_init_params = running_statistics.init_state(
                specs.Array(self.env.observation_size, dtype))
        else:
            normalizer_init_params = self.normalizer_init_params

        value_init_params = self.shac_network.value_network.init(key_value)
        if self.log_sigma is not None:
            policy_init_params = {
                "policy_network_params": policy_init_params,
                "policy_lnsig_params": jax.numpy.ones(self.env.action_size) * self.log_sigma
            }
        
        training_state = TrainingState(
            policy_optimizer_state=self.policy_optimizer.init(policy_init_params),
            policy_params=policy_init_params,
            value_optimizer_state=self.value_optimizer.init(value_init_params),
            value_params=value_init_params,
            target_value_params=value_init_params,
            normalizer_params=normalizer_init_params,
            env_steps=0)

        return training_state, key

    def train(self):
        """ 
        1. Init batched environment states
        2. Init training state
        3. Run initial eval
        4. Delete old checkpoints
        5. Train while logging
        """
        xt = time.time()
        key = jax.random.PRNGKey(self.seed)
        key_env, key = jax.random.split(key)
        
        if self.use_tbx:
            algo_dir = Path(__file__).parent
            log_dir = Path(algo_dir, Path("tensorboards"), Path(self.tbx_logdir), Path(self.tbx_experiment_name))
            writer = SummaryWriter(str(log_dir))
        
        key_envs = jax.random.split(key_env, self.num_envs)
        env_state = self.reset_fn(key_envs)
        p_shst = env_state.info['steps'].shape

        if self.scramble_initial_times:
            key, key_scramble = jax.random.split(key)
            env_state = self.scramble_times(env_state, key_scramble)

        assert env_state.info['steps'].shape == p_shst
        
        if not self.eval_env:
            eval_env = self.env
        else:
            # Not backpropping through, so no problem to just use vmapped env. 
            eval_env = orig_wraps.wrap(
                self.eval_env, episode_length=self.episode_length
            )

        training_state, key = self.init_training_state(key)
        key_eval, key = jax.random.split(key)
        evaluator = acting.Evaluator(
            eval_env,
            functools.partial(self.make_policy, deterministic=self.deterministic_eval),
            num_eval_envs=self.num_eval_envs,
            episode_length=self.episode_length,
            action_repeat=1,
            key=key_eval)

        # # Run initial eval
        if self.num_evals > 1:
            t0 = time.time()
            metrics = evaluator.run_evaluation(
                (training_state.normalizer_params, training_state.policy_params),
                training_metrics={})
            logging.info(metrics)
            print("Initial eval time: {:.4f} s".format(time.time() - t0))
            self.progress_fn(0, metrics)

        self.training_walltime = 0
        current_step = 0
        
        # Delete previous checkpoints
        print("Deleting old checkpoints!")
        top = Path(Path(__file__).parent,
                Path('checkpoints'))
        for constituent in top.iterdir():
            constituent.unlink()
        
        local_key, key = jax.random.split(key)
        for it in range(self.num_evals_after_init):
            logging.info('starting iteration %s %s', it, time.time() - xt)

            # optimization
            epoch_key, local_key = jax.random.split(local_key)

            # DEBUG: save state
            algo_state = {
                "epoch_key": epoch_key,
                "local_key": local_key,
                "training_state": training_state,
                "env_state": env_state
            }

            file_name = f'checkpoint_{it}.pkl' if self.save_all_checkpoints else 'checkpoint.pkl'
            save_to = str(Path(Path(__file__).parent,
                            Path('checkpoints'),
                            Path(file_name)))
                                
            pickle.dump(algo_state, open(save_to, "wb"))
            
            print("Checkpointed for epoch {}".format(it))
            epoch_start_env_time = env_state.pipeline_state.time[0]
            epoch_start_state = env_state.pipeline_state.qpos[0, 0]

            (training_state, env_state,
                training_metrics) = self.training_epoch_with_timing(training_state, env_state, epoch_key)
            
            # VERIFY AUTODIFF
            if self.num_grad_checks is not None:
                pre_update = pickle.load(open(save_to, "rb"))
                pts = pre_update['training_state']
                key_ub, key_ind, epoch_key = jax.random.split(epoch_key, 3)
                i_ub = jax.random.choice(key_ub, self.num_envs)
                
                state = unvmap(pre_update['env_state'], i_ub)
                ad_grads_extradim = training_metrics['training/b_policy_gradient']
                ad_grads = unvmap(ad_grads_extradim, 0) # batched across episodes
                ad_grads_ub = unvmap(ad_grads, i_ub)

                unroll_key = unvmap(training_metrics['training/unroll_keys'][0], i_ub)
                
                _checkify_err, grad_err = self.fd_gradient_checks(pts.policy_params, pts.value_params, 
                                            pts.normalizer_params, state, 
                                            key_ind, unroll_key, ad_grads_ub)
                
                writer.add_scalar('policy/autodiff_error', grad_err, it)
                
            current_step = int(training_state.env_steps)

            epoch_end_env_time = env_state.pipeline_state.time[0]
            epoch_end_env_state = env_state.pipeline_state.qpos[0, 0]

            #### Policy Gradient Saving ####
            if self.save_all_policy_gradients:
                file_name = f'bgrad_{it}.pkl'
                save_to = str(Path(Path(__file__).parent,
                                Path('policy_gradients'),
                                Path(file_name)))
                to_save = unvmap(training_metrics['training/b_policy_gradient'], 0) # Empty batch dim.

                pickle.dump(to_save, open(save_to, "wb"))
            
            #### TENSORBOARDING ####
            """ 
            Folders: 
            - env
            - policy
            - critic
            """
            
            # Get rid of batch dim from def training_epoch
            # metrics = jax.tree_util.tree_map(lambda x: x[0], metrics)
            
            if self.use_tbx:
                ## ENV ##
                writer.add_scalar("env/Starting env 0 time", epoch_start_env_time, it)
                writer.add_scalar("env/Ending env 0 time", epoch_end_env_time, it)
                writer.add_scalar("env/Starting env 0 state", epoch_start_state, it)
                writer.add_scalar("env/Ending env 0 state", epoch_end_env_state, it)
                # Additional debug
                writer.add_scalar("env/Env 0 done", env_state.done[0], it)
                writer.add_scalar("env/Env 0 height", env_state.pipeline_state.qpos[0, 2], it)
                writer.add_scalar("env/Env 0 up", env_state.obs[0, 35], it)

                ## POLICY ##
                act_s = training_metrics['training/action_size'][0]
                writer.add_scalar('policy/action size', act_s, it)
                ub_pg = jax.tree_util.tree_map(lambda x: x[0], 
                                            training_metrics['training/policy_gradient'])
                flattened_vals, _ = ravel_pytree(ub_pg)
                policy_grad_size = jnp.sqrt(jnp.mean(jnp.square(flattened_vals)))
                writer.add_scalar('policy/||Policy gradient||', policy_grad_size, it)
                # if jnp.isnan(policy_grad_size):
                #     raise ValueError("Nan policy gradient!")
                avg_policy_loss = training_metrics['training/policy_loss'][0].mean()
                writer.add_scalar('policy/Policy loss', avg_policy_loss, it)
                avg_entropy_loss = training_metrics['training/entropy_loss'][0].mean()
                writer.add_scalar('policy/Entropy loss', avg_entropy_loss, it)
                p_nan_grads = training_metrics['training/p_nan_grads'][0]
                writer.add_scalar('policy/p_nan_grads', p_nan_grads, it)
                ## CRITIC ##
                td_lam = training_metrics['training/target_value_params_size'][0]
                writer.add_scalar('critic/td_lam_value_norm', td_lam, it)
                crit_s = training_metrics['training/target_critic_output_size'][0]
                writer.add_scalar('critic/critic output size', crit_s, it)
                last_critic_loss = (training_metrics['training/v_loss'][0].mean(axis=1))[-1] # 16 x 256
                writer.add_scalar('critic/Last epoch critic loss', last_critic_loss, it)
                
                ## Timing
                writer.add_scalar("sps", training_metrics['training/sps'], it)
                writer.add_scalar('Wall-Clock Time', training_metrics['training/walltime'], it)

            if not self.use_tbx:
                metrics = evaluator.run_evaluation(
                        (training_state.normalizer_params, training_state.policy_params),
                    training_metrics)
                logging.info(metrics)
                print("Training epoch SPS: {}".format(metrics["training/sps"]))
                self.progress_fn(current_step, metrics)

        total_steps = current_step
        assert total_steps >= self.num_timesteps

        logging.info('total steps: %s', total_steps)
        policy_params = (training_state.normalizer_params, training_state.policy_params)
        value_params = (training_state.normalizer_params, training_state.target_value_params)
        return (self.make_policy, policy_params, value_params, metrics)
    