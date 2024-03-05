from brax.training.acme import running_statistics, specs

import jax.numpy as jp
import jax
import jax_shac.shac.networks as shac_networks

import functools
import mujoco

from mujoco import mjx
import mediapy as media
import pickle

import matplotlib.pyplot as plt 
import numpy as np
from jax.flatten_util import ravel_pytree

def sstep(diffwrt, dobj, sys):
    """
    Diffwrt: [state, ctrl]. Zero-D array.
    dobj: mjxData
    S for separated step. 
    """
    
    dobj = dobj.tree_replace({
        'qpos': diffwrt[:sys.nq],
        'qvel': diffwrt[sys.nq:sys.nq+sys.nv],
        'ctrl': diffwrt[sys.nq+sys.nv:]
    })

    dobj = jmjxstep(sys, dobj)

    # Sorry!
    state = jp.squeeze(jp.concatenate(
                           [jp.expand_dims(dobj.qpos, 1), 
                            jp.expand_dims(dobj.qvel, 1)],
                            axis=0))
    return state

fjacstep = jax.jit(jax.jacfwd(sstep, 0))
rjacstep = jax.jit(jax.jacrev(sstep, 0))
jmjxstep = jax.jit(mjx.step)

def plot_jacobian_norms(state, key, env, jit_inference_fn, 
                        jit_step, episode_length,
                        ret_jacs = False, jacmode='fwd'):
    
    state_dim = env.sys.nq + env.sys.nv
    ctrl_dim = env.sys.nu
    all_jacs = jp.zeros((episode_length,state_dim,state_dim + ctrl_dim))
    # all_jac_norms = jp.zeros(episode_length)

    for f in range(int(episode_length)):    
        key, key_action = jax.random.split(key)
        action, _ = jit_inference_fn(state.obs, key_action)

        state = jit_step(state, action)
        mjx_data = state.pipeline_state

        # Eval jac
        x_i = jp.squeeze(
                jp.concatenate(
                    [ mjx_data.qpos.reshape(env.sys.nq), 
                      mjx_data.qvel.reshape(env.sys.nv),
                      action.reshape(1)], 
                    axis=0))
        assert len(x_i) == state_dim + ctrl_dim

        match jacmode:
            case "fwd":
                cur_jac = fjacstep(x_i, mjx_data, env.sys)
            case "rev":
                cur_jac = rjacstep(x_i, mjx_data, env.sys)
            case _:
                raise ValueError("Invalid jacobian mode!")

        all_jacs = all_jacs.at[f, ...].set(cur_jac)
        # all_jac_norms = all_jac_norms.at[f].set(jp.linalg.norm(cur_jac))
    
    state_jacs = all_jacs[..., :state_dim]
    ctrl_jacs = all_jacs[..., state_dim:state_dim+ctrl_dim]
    
    t_ax = jp.linspace(0, episode_length * env.dt, int(episode_length))

    plt.figure()
    fig, axs = plt.subplots(nrows=2,ncols=1,figsize=(7,7))
    axs[0].plot(t_ax, jp.linalg.norm(state_jacs, axis=(1, 2)))
    axs[0].set_ylabel("Norm of state jacobian")
    axs[0].set_xlabel("Time (s)")
    axs[1].plot(t_ax, jp.linalg.norm(ctrl_jacs, axis=(1, 2)))
    axs[1].set_ylabel("Norm of control jacobian")    
    axs[1].set_xlabel("Time (s)")

    if ret_jacs:
        return all_jacs, state_jacs, ctrl_jacs

def plot_rews(states, ylims=None):
    """ Simple plot of the rewards associated with a set of states.

        Typical call signature: 
        - rews = plot_rews(get_rollout(...), ylims=[-1, 1])
        - rews = plot_rews(get_rollout(...))
        Side effects: generates a matplotlib plot with labels being reward names.     
    """
    reward_keys = states[0].info['reward_tuple'].keys()
    num_rews = len(reward_keys)
    num_steps = len(states)

    rew_tups = [s.info['reward_tuple'] for s in states]

    flat_rews, _ = ravel_pytree(rew_tups)
    rews = np.reshape(flat_rews, (num_steps, num_rews), order='C')
    if ylims is not None:
        plt.ylim(ylims)
    plt.plot(rews)
    plt.legend(reward_keys)
    return rews

def init_SHAC_policy_from_checkpoint(env, checkpoint_path, make_networks_factory, scalar_var):
    """ 
    Assumptions: 
    - Using normalize
    """

    normalize = running_statistics.normalize
    algo_state = pickle.load(open(checkpoint_path, "rb"))
    shac_network = make_networks_factory(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=normalize)
    
    if scalar_var:
        make_policy = shac_networks.make_inference_fn_log_sigma(shac_network)
    else:
        make_policy = shac_networks.make_inference_fn(shac_network)
    ts = algo_state['training_state']

    return make_policy, (ts.normalizer_params, ts.policy_params)


def init_SHAC_policy(env, key):
    make_networks_factory = functools.partial(
        shac_networks.make_shac_networks,
            policy_hidden_layer_sizes=(256, 128),
            value_hidden_layer_sizes =(256, 128),
            scalar_var=False)

    normalize = running_statistics.normalize

    shac_network = make_networks_factory(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=normalize)
    make_policy = shac_networks.make_inference_fn(shac_network)

    key, key_policy = jax.random.split(key)

    normalizer_params = running_statistics.init_state(
                specs.Array((env.observation_size,), jp.float32))
    policy_params = shac_network.policy_network.init(key_policy)

    return make_policy, (normalizer_params, policy_params)

def get_image(state, renderer, env, camera = None):
    """Renders the environment state."""
    d = mujoco.MjData(env.model)
    # write the mjx.Data into an mjData object
    mjx.device_get_into(d, state.pipeline_state)
    mujoco.mj_forward(env.model, d)
    # use the mjData object to update the renderer
    if camera is not None:
        renderer.update_scene(d, camera=camera)
    else:
        renderer.update_scene(d)
    return renderer.render()

def visualize_rollout(state, rng, env, jit_inference_fn, 
                      jit_step, renderer, episode_length=100, 
                      print_nans=False, print_dones=False, 
                      ret_states = False, camera=None, vision=False):

    
    render_every = 2
    images = [get_image(state, renderer, env, camera=camera)]
    
    states = []

    if vision:
        j_update_image = jax.jit(env.update_image)

    for i in range(episode_length):
        act_rng, rng = jax.random.split(rng)
        if not vision:
            ctrl, _ = jit_inference_fn(state.obs, act_rng)
            state = jit_step(state, ctrl)
        else:
            image = env.render(state)
            state = j_update_image(state, image)
            obs = jp.expand_dims(state.obs, 0)
            action, _ = jit_inference_fn(obs, act_rng)
            action = jp.squeeze(action, 0)
            state = jit_step(state, action)

        if print_nans:
            if jp.any(~jp.isfinite(state.obs)):
                print(f"Nan at timestep {i}! ({i*env.dt})s")
        
        if print_dones:
            if state.done:
                print(f"Done at timestep {i}! ({i*env.dt})s")

        if ret_states:
            states.append(state)

        if i % render_every == 0:
            images.append(get_image(state, renderer, env, camera=camera))

    media.show_video(images, fps=1.0 / env.dt / render_every)

    if ret_states:
        return state, states
    else:
        return state