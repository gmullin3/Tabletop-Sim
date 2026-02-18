from tabletop.constants import ALOHA_XML_DIR
import numpy as np

from dm_control import mujoco
from dm_control.utils import inverse_kinematics as ik

_JOINTS = ['left/waist', 'left/shoulder', 'left/elbow', 'left/forearm_roll', 'left/wrist_angle', 'left/wrist_rotate']
_TOL = 1.2e-14
_MAX_STEPS = 5000
_MAX_RESETS = 10
_SITE_NAME = 'left/gripper'

class _ResetArm:
  """
  Helper class to reset the arm to a random configuration within joint limits.
  """
  def __init__(self, seed=None):
    self._rng = np.random.RandomState(seed)
    self._lower = None
    self._upper = None

  def _cache_bounds(self, physics):
    """Cache the joint limits from the physics model."""
    self._lower, self._upper = physics.named.model.jnt_range[_JOINTS].T
    limited = physics.named.model.jnt_limited[_JOINTS].astype(bool)
    # Positions for hinge joints without limits are sampled between 0 and 2pi
    self._lower[~limited] = 0
    self._upper[~limited] = 2 * np.pi

  def __call__(self, physics, curr_qpos=None):
    """Reset the arm to a random or specified configuration."""
    if self._lower is None:
      self._cache_bounds(physics)
    # NB: This won't work for joints with  1 DOF
    if curr_qpos is None:
        new_qpos = self._rng.uniform(self._lower, self._upper)
        physics.named.data.qpos[_JOINTS] = new_qpos
    else:
        physics.named.data.qpos[_JOINTS] = curr_qpos

class AlohaIK:
    """
    Inverse Kinematics solver for the Aloha robot arm.
    """
    def __init__(self):
        ## For Jacobian method
        self.resetter = _ResetArm(seed=0)
        self.ik_physics = mujoco.Physics.from_xml_path(f'{ALOHA_XML_DIR}/aloha_ik.xml')

    def get_joint_pos(self, target_pos, target_quat, curr_qpos=None):
        """
        Calculate joint positions for a given target end-effector pose.
        
        Args:
            target_pos: Target position [x, y, z].
            target_quat: Target quaternion [w, x, y, z].
            curr_qpos: Current joint positions (optional seed).
            
        Returns:
            np.array: Joint positions (6 elements).
        """
        count = 0
        physics2 = self.ik_physics.copy(share_model=True)
        self.resetter(physics2, curr_qpos)
        while True:
            result = ik.qpos_from_site_pose(
                physics=physics2,
                site_name=_SITE_NAME,
                target_pos=target_pos,
                target_quat=target_quat,
                joint_names=_JOINTS,
                tol=_TOL,
                max_steps=_MAX_STEPS,
                inplace=True,
            )
            if result.success:
                break
            elif count < _MAX_RESETS:
                self.resetter(physics2)
                count += 1
            else:
                break
        return result.qpos[:6]