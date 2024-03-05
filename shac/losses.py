from typing import Any, Tuple

from brax.training import types
from jax_shac.shac import networks as shac_networks
from brax.training.types import Params
import flax
import jax
import jax.numpy as jp
from jaxtyping import Array, Shaped, jaxtyped, Shaped
from typeguard import typechecked as typechecker
from jax.experimental import checkify
import numpy as np

@flax.struct.dataclass
class SHACNetworkParams:
  """Contains training state for the learner."""
  policy: Params
  value: Params

def compute_entropy_loss(
    policy_params: Params,
    normalizer_params: Any,
    shac_network: shac_networks.SHACNetworks,
    data: types.Transition,
    rng: jp.ndarray,
    entropy_cost: float = 1e-4,
):
    """ 
    
    """
    parametric_action_distribution = shac_network.parametric_action_distribution
    policy_apply = shac_network.policy_network.apply

    policy_logits = policy_apply(normalizer_params, policy_params,
                                data.observation)
    entropy = jp.mean(parametric_action_distribution.entropy(policy_logits, rng))
    entropy_loss = entropy_cost * -entropy
    return entropy_loss    

def compute_entropy_lnsig_loss(
    policy_params: Params,
    normalizer_params: Any,
    shac_network: shac_networks.SHACNetworks,
    data: types.Transition,
    rng: jp.ndarray,
    entropy_cost: float = 1e-4,
):
    """ 
    Penalize collapsed variance, when using a vector variance.
    Included unused parameters for easy swapping with def compute_entropy_loss.
    """
    action_var = jp.exp(policy_params["policy_lnsig_params"])
    entropy = jp.mean(jp.square(action_var))
    entropy_loss = entropy_cost * -entropy
    return entropy_loss


def compute_shac_policy_loss(
    value_params: Params,
    normalizer_params: Any,
    data: types.Transition,
    last_state,
    shac_network: shac_networks.SHACNetworks,
    discounting: float = 0.9,
    reward_scaling: float = 1.0) -> Tuple[jp.ndarray, types.Metrics]:
    """Computes SHAC critic loss.

    This implements Eq. 5 of 2204.07137.

    Args:
    value_params: Value network parameters,
    normalizer_params: Parameters of the normalizer.
    data: Transition that with leading dimension [B, T]. extra fields required
        are ['state_extras']['truncation'] ['policy_extras']['raw_action']
        ['policy_extras']['log_prob']
    rng: Random key
    shac_network: SHAC networks.
    entropy_cost: entropy cost.
    discounting: discounting,
    reward_scaling: reward multiplier.

    Returns:
    A scalar loss
        
    Key quantities: 
    all_rews: rewards from unroll, times scaling
    next_values: index i corresponds to V(s_{i+1})
    
    """
    value_apply = shac_network.value_network.apply
    
    num_steps = data.discount.shape[0]
    
    self_gamma = discounting

    # INIT
    all_rews = data.reward * reward_scaling 
    all_truncated = data.extras['state_extras']['truncation']
    all_truncated = all_truncated.at[0].set(0) # Don't have access to the previous batch. 
    all_terminated = jp.logical_and((1 - data.discount), (1 - all_truncated))
    all_dones = jp.logical_or(all_truncated, all_terminated)
    
    # TODO: REMOVE FOR NON-VISION VERSION!
    # @jaxtyped(typechecker=typechecker) # Ensure non env-batched.
    # def dimcheck(obs: Shaped[Array, "num_steps w h d"]):
    #     pass
    # dimcheck(data.observation)
    
    ### Process all values
    values = value_apply(normalizer_params, value_params, data.observation) # s0 to sh-1
    
    # @VISION
    terminal_values = value_apply(normalizer_params, value_params, 
                                  jp.expand_dims(last_state.obs, 0))

    terminal_values = jp.squeeze(terminal_values)
    # Ensure everything's row vectors, but still 2D.
    @jaxtyped(typechecker=typechecker)
    def in_check(discount: Shaped[Array, "num_steps"], 
                 vals: Shaped[Array, "num_steps"],
                 rews: Shaped[Array, "num_steps"]) -> Shaped[Array, ""]:
        return terminal_values
    
    in_check(data.discount, values, all_rews)

    # Need an index shift of 1, for xu. 
    next_values = jp.roll(values, -1) # size h.
    next_values.at[-1].set(terminal_values)

    # Process Terminations
    next_values = jp.where(all_terminated, jp.zeros_like(all_terminated), next_values)

    # Process Truncations.
    vals_before_resets = jp.roll(values, 1) # The new values are set on the same timestep as done = True, by the autoresetwrapper.

    next_values = jp.where(all_truncated, vals_before_resets, next_values)
    
    # Perform checks
    # checkify.check(jp.sum(all_truncated[0]) == 0, "policy loss: impossible truncation on rollout step 0!", all_truncated=all_truncated)

    # a_act = jp.abs(data.action)
    # checkify.check(jp.sum(a_act > 1e6) 
    #                + jp.sum(jp.isnan(a_act)) == 0,
    #                "policy loss: action misbehaved",
    #                a_act=a_act)

    # checkify.check(jp.sum(all_truncated[0]) == 0, 
    #                "policy loss: impossible truncation on rollout step 0!", all_truncated=all_truncated)

    # a_obs = jp.abs(data.observation)
    # checkify.check(jp.sum(a_obs > 1e6) 
    #                + jp.sum(jp.isnan(a_obs))
    #                + jp.sum(jp.isinf(a_obs)) == 0,
    #                "Policy loss: observation misbehaved",
    #                a_obs=a_obs)
    # a_vals = jp.abs(next_values)
    # checkify.check(jp.sum(a_vals > 1e6)
    #                + jp.sum(jp.isnan(a_vals))
    #                + jp.sum(jp.isinf(a_vals)) == 0,
    #                "Policy loss: critic misbehaved",
    #                a_vals=a_vals)
    
    def f(carry, x):
        """ 
        Compared to code in original paper: 
        - rew_acc: rew_acc[i+1]
        - l: loss
        - next_val: next_vals[i+1]
        """

        l, rew_acc, gamma = carry
        rew, next_val, done = x # all scalars

        rew_acc += jp.multiply(gamma,rew)
        
        l -=      jp.where(done, rew_acc+self_gamma*jp.multiply(gamma,next_val), 0)
        rew_acc = jp.where(done, 0, rew_acc)
        gamma =   jp.where(done, 1, self_gamma * gamma)
        
        return (l, rew_acc, gamma), 0

    # scan until and including h-2
    (actor_loss, rew_acc_hm1, gamma), _ = jax.lax.scan(f, init=(0.0, 0.0, 1.0), xs = (all_rews[:-1], next_values[:-1], all_dones[:-1]), length = num_steps - 1)

    # Process last index
    rew_acc_h = rew_acc_hm1 + gamma * all_rews[-1]
    actor_loss -= jp.sum((rew_acc_h + self_gamma*gamma*next_values[-1]))

    actor_loss /= (num_steps)
    total_loss = actor_loss

    # loss must be scalar.
    return total_loss, {
        'policy_loss': actor_loss,
        'next_values': next_values
    }

@jax.jit
def compute_td_lambda_vals(
    data,
    next_values,
    discounting,
    reward_scaling,
    lambda_):

    """ 
    Key quantities: 
    1. terminal_value: 
    2. termination: non-truncation but done. horizon_length x n_envs
    3. rewards: horizon_length x n_envs
    4. next_values: calculated during policy update. 
    
    rew_buf (rewards): just rew from env * rew_scale. Must apply. 
    done_mask (termination): from step (data.discount). Last row = 1's.
    """
    self_lam = lambda_
    self_gamma = discounting
    
    done_mask = data.discount
    done_mask = done_mask.at[-1, :].set(1) # end of rollout
    
    rewards = data.reward * reward_scaling
    terminal_value = next_values[-1, :]
    
    # Algo 
    def compute_v_st(carry, target_t):
        Ai, Bi, lam = carry
        reward, vtp1, termination = target_t

        # reward = reward * termination

        lam = lam * self_lam * (1 - termination) + termination
        Ai = (1 - termination) * (self_lam * self_gamma * Ai + self_gamma * vtp1 + (1. - lam) / (1. - self_lam) * reward)

        Bi = self_gamma * (vtp1 * termination + Bi * (1.0 - termination)) + reward
        vs = (1.0 - self_lam) * Ai + lam * Bi
        # jax.debug.print("{x}", x=Ai)

        return (Ai, Bi, lam), (vs)

    Ai =  jp.zeros_like(terminal_value)
    Bi =  jp.zeros_like(terminal_value)
    lam = jp.ones_like(terminal_value)
    (_, _, _), (vs) = jax.lax.scan(compute_v_st, (Ai, Bi, lam),
        (rewards, next_values, done_mask),
        length=int(done_mask.shape[0]),
        reverse=True)
    
    return vs

def compute_shac_critic_loss(
    params: Params,
    normalizer_params: Any,
    obs: types.Transition,
    target_vals,
    shac_network: shac_networks.SHACNetworks) -> Tuple[jp.ndarray, types.Metrics]:
    """Computes SHAC critic loss.

    Args:
    params: Value network parameters,
    normalizer_params: Parameters of the normalizer.
    data: Transition that with leading dimension [B, T]. extra fields required
        are ['state_extras']['truncation'] ['policy_extras']['raw_action']
        ['policy_extras']['log_prob']
    target_vals: The values to fit the critic to, coming from TD-Lambda. 
    shac_network: SHAC networks.

    Returns:
    A tuple (loss, metrics)
    """
    #@VISION
    # def check_ins(obs: Shaped[Array, "b_size w h d"], tv: Shaped[Array, "b_size"]):
    #     pass
    # check_ins(obs, target_vals)

    value_apply = shac_network.value_network.apply

    # DEBUG
    # obs = jp.expand_dims(obs, axis=0)
    values = value_apply(normalizer_params, params, obs)
    # values = jp.squeeze(values)
    target_vals = jax.lax.stop_gradient(target_vals)

    v_loss = jp.mean((target_vals - values) ** 2)

    total_loss = v_loss
    return total_loss, {
        'total_loss': total_loss,
        'v_loss': v_loss,
    }