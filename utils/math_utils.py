from mujoco import mjx
from jax import numpy as jp

def axis_angle_to_quaternion(v: jp.ndarray, theta:jp.float32):
    """ 
    axis angle representation: rotation of theta around v. 
    """    
    return jp.concatenate([jp.cos(0.5*theta).reshape(1), jp.sin(0.5*theta)*v.reshape(3)])

def flatten_quaternion(quat):
  """ 
  Extract the angle.
  """
  return 2*jp.arccos(quat[0])

def qpos_freejoint2hinge(qpos):
  qpos_new = jp.concatenate([
    qpos[0].reshape(1),
    qpos[2].reshape(1),
    flatten_quaternion(qpos[3:7]).reshape(1),
    qpos[7:].reshape(12)
  ])
  return qpos_new

def qpos_hinge2freejoint(qpos):
  qpos_new = jp.concatenate([
    qpos[0].reshape(1),
    jp.array(0).reshape(1),
    qpos[1].reshape(1),
    axis_angle_to_quaternion(jp.array([0, 1, 0]), qpos[2]).reshape(4),
    qpos[3:].reshape(12)
  ])
  return qpos_new

def unflatten_angvel(theta, axis):
  return axis * theta

def qvel_freejoint2hinge(qvel):
  qvel_new = jp.concatenate([
    qvel[0].reshape(1),
    qvel[2].reshape(1),
    jp.linalg.norm(qvel[3:6]).reshape(1),
    qvel[6:].reshape(12)
    ])
  return qvel_new

def qvel_hinge2freejoint(qvel):
  qvel_new = jp.concatenate([
    qvel[0].reshape(1), # x
    jp.array(0).reshape(1), # y
    qvel[1].reshape(1), # z
    unflatten_angvel(qvel[2], jp.array([0, 1, 0])).reshape(3),
    qvel[3:].reshape(12) # size of hinge qvel: 3 + 12 = 15. 
    ])
  return qvel_new

def w_planar_pipeline_init(qpos, qvel, pipeline_init, fj_model, t_fjdata):
    """ 
    t_hdata: template hinge data
    t_fjdata: template freejoint data.
    """
    qpos = qpos_freejoint2hinge(qpos)
    qvel = qvel_freejoint2hinge(qvel)
    ps = pipeline_init(qpos, qvel)
    fj_data = t_fjdata.replace(
      qpos = qpos_hinge2freejoint(ps.qpos), 
      qvel = qvel_hinge2freejoint(ps.qvel))
    return mjx.forward(fj_model, fj_data)

def w_planar_step(state, ctrl, pipeline_step, hmodel, fj_model, t_hdata, t_fjdata):
  """ 
  ps = pipeline_step(state.pipeline_state, ctrl)
  - state.pipeline_state: change qpos & qvel, then forward
  - ctrl: ok
  """
  hdata = t_hdata.replace(
    qpos = qpos_freejoint2hinge(state.qpos),
    qvel = qvel_freejoint2hinge(state.qvel)
  )

  hdata = mjx.forward(hmodel, hdata)
  hdata = pipeline_step(hdata, ctrl)

  fj_data = t_fjdata.replace(
    qpos = qpos_hinge2freejoint(hdata.qpos),
    qvel = qvel_hinge2freejoint(hdata.qvel)
  )

  return mjx.forward(fj_model, fj_data)
  