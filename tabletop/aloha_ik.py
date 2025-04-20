from tabletop.constants import ALOHA_XML_DIR
import numpy as np
import collections
import os
import random

from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R

from tabletop.constants import *
from tabletop.wrappers import *
from tabletop.utils import sample_box_pose, sample_insertion_pose
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control import mjcf
# from dm_robotics.moma.models import types
# from dm_robotics.moma.models.robots.robot_arms import robot_arm
# from dm_robotics.moma.effectors import arm_effector, cartesian_6d_velocity_effector
from dm_control.utils import inverse_kinematics as ik

_JOINTS = ['left/waist', 'left/shoulder', 'left/elbow', 'left/forearm_roll', 'left/wrist_angle', 'left/wrist_rotate']
_TOL = 1.2e-14
_MAX_STEPS = 5000
_MAX_RESETS = 10
_SITE_NAME = 'left/gripper'

class _ResetArm:
  def __init__(self, seed=None):
    self._rng = np.random.RandomState(seed)
    self._lower = None
    self._upper = None

  def _cache_bounds(self, physics):
    self._lower, self._upper = physics.named.model.jnt_range[_JOINTS].T
    limited = physics.named.model.jnt_limited[_JOINTS].astype(bool)
    # Positions for hinge joints without limits are sampled between 0 and 2pi
    self._lower[~limited] = 0
    self._upper[~limited] = 2 * np.pi

  def __call__(self, physics, curr_qpos=None):
    if self._lower is None:
      self._cache_bounds(physics)
    # NB: This won't work for joints with  1 DOF
    if curr_qpos is None:
        new_qpos = self._rng.uniform(self._lower, self._upper)
        physics.named.data.qpos[_JOINTS] = new_qpos
    else:
        physics.named.data.qpos[_JOINTS] = curr_qpos

class AlohaIK:
    def __init__(self):
        ## For Jacobian method
        self.resetter = _ResetArm(seed=0)
        self.ik_physics = mujoco.Physics.from_xml_path(f'{ALOHA_XML_DIR}/aloha_ik.xml')

        # For delta method
        # self.relative_max_joint_delta = np.array([0.4, 0.4, 0.4, 0.4, 0.4, 0.4])
        # self.max_joint_delta = self.relative_max_joint_delta.max()
        # self.max_gripper_delta = 0.25
        # self.max_lin_delta = 0.075
        # self.max_rot_delta = 0.3
        # self.control_hz = 20
        # self._arm = AlohaArm()
        # self._physics = mjcf.Physics.from_mjcf_model(self._arm.mjcf_model)
        # self._effector = arm_effector.ArmEffector(arm=self._arm, action_range_override=None, robot_name=self._arm.name)
        # self._effector_model = cartesian_6d_velocity_effector.ModelParams(self._arm.wrist_site, self._arm.joints)

        # self._effector_control = cartesian_6d_velocity_effector.ControlParams(
        #     control_timestep_seconds=1 / self.control_hz,
        #     max_lin_vel=self.max_lin_delta,
        #     max_rot_vel=self.max_rot_delta,
        #     joint_velocity_limits=self.relative_max_joint_delta,
        #     nullspace_joint_position_reference=[0] * 6,
        #     nullspace_gain=0.025,
        #     regularization_weight=1e-2,
        #     enable_joint_position_limits=True,
        #     minimum_distance_from_joint_position_limit=0.3,
        #     joint_position_limit_velocity_scale=1.0,
        #     max_cartesian_velocity_control_iterations=300,
        #     max_nullspace_control_iterations=300,
        # )

        # self._cart_effector_6d = cartesian_6d_velocity_effector.Cartesian6dVelocityEffector(
        #     self._arm.name, self._effector, self._effector_model, self._effector_control
        # )
        # self._cart_effector_6d.after_compile(self._arm.mjcf_model, self._physics)

    def get_joint_pos(self, target_pos, target_quat, joint_vel=False, curr_pos=None, curr_quat=None, curr_qpos=None, curr_qvel=None):
        if not joint_vel:
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
            return result.qpos[:6]
        # else:
        #     assert curr_pos is not None and curr_quat is not None, "Input current state for velocity control"
        #     target_euler = self.quat_to_euler(target_quat)
        #     curr_euler = self.quat_to_euler(curr_quat)
        #     diff = self.pose_diff(np.concatenate([target_pos, target_euler]), np.concatenate([curr_pos, curr_euler]))
            
        #     command = self.cartesian_delta_to_velocity(diff)
        #     command[3:] = 0
        #     joint_velocity = self.cartesian_velocity_to_joint_velocity(command, qpos=curr_qpos, qvel=curr_qvel)
        #     print(joint_velocity)
        #     return joint_velocity * 100.0
        
    ### Inverse Kinematics ###
#     def cartesian_velocity_to_joint_velocity(self, cartesian_velocity, qpos, qvel):
#         cartesian_delta = self.cartesian_velocity_to_delta(cartesian_velocity)  
#         self._arm.update_state(self._physics, qpos, qvel)
#         self._cart_effector_6d.set_control(self._physics, cartesian_delta)
#         joint_delta = self._physics.bind(self._arm.actuators).ctrl.copy()
#         np.any(joint_delta)

#         joint_velocity = self.joint_delta_to_velocity(joint_delta)

#         return joint_velocity

#     def cartesian_delta_to_velocity(self, cartesian_delta):
#         if isinstance(cartesian_delta, list):
#             cartesian_delta = np.array(cartesian_delta)

#         cartesian_velocity = np.zeros_like(cartesian_delta)
#         cartesian_velocity[:3] = cartesian_delta[:3] / self.max_lin_delta
#         cartesian_velocity[3:6] = cartesian_delta[3:6] / self.max_rot_delta

#         return cartesian_velocity

#     def cartesian_velocity_to_delta(self, cartesian_velocity):
#         if isinstance(cartesian_velocity, list):
#             cartesian_velocity = np.array(cartesian_velocity)

#         lin_vel, rot_vel = cartesian_velocity[:3], cartesian_velocity[3:6]

#         lin_vel_norm = np.linalg.norm(lin_vel)
#         rot_vel_norm = np.linalg.norm(rot_vel)

#         if lin_vel_norm > 1:
#             lin_vel = lin_vel / lin_vel_norm
#         if rot_vel_norm > 1:
#             rot_vel = rot_vel / rot_vel_norm

#         lin_delta = lin_vel * self.max_lin_delta
#         rot_delta = rot_vel * self.max_rot_delta

#         return np.concatenate([lin_delta, rot_delta])

#     def joint_delta_to_velocity(self, joint_delta):
#         if isinstance(joint_delta, list):
#             joint_delta = np.array(joint_delta)

#         return joint_delta / self.max_joint_delta

#     def pose_diff(self, target, source, degrees=False):
#         lin_diff = np.array(target[:3]) - np.array(source[:3])
#         rot_diff = self.angle_diff(target[3:6], source[3:6], degrees=degrees)
#         result = np.concatenate([lin_diff, rot_diff])
#         return result

#     def angle_diff(self, target, source, degrees=False):
#         target_rot = R.from_euler("xyz", target, degrees=degrees)
#         source_rot = R.from_euler("xyz", source, degrees=degrees)
#         result = target_rot * source_rot.inv()
#         return result.as_euler("xyz")

#     def quat_to_euler(self, quat, degrees=False):
#         euler = R.from_quat(quat).as_euler("xyz", degrees=degrees)
#         return euler

# class RobotArm(robot_arm.RobotArm):
#     def _build(self, model_file):
#         self._mjcf_root = mjcf.from_path(self._model_file)

#     def _create_body(self):
#         # Find MJCF elements that will be exposed as attributes.
#         self._joints = self._mjcf_root.find_all("joint")
#         self._bodies = self.mjcf_model.find_all("body")
#         self._actuators = self.mjcf_model.find_all("actuator")
#         self._wrist_site = self.mjcf_model.find("site", "wrist_site")
#         self._base_site = self.mjcf_model.find("site", "base_site")

#     def name(self) -> str:
#         return self._name

#     @property
#     def joints(self):
#         """List of joint elements belonging to the arm."""
#         return self._joints

#     @property
#     def actuators(self):
#         """List of actuator elements belonging to the arm."""
#         return self._actuators

#     @property
#     def mjcf_model(self):
#         """Returns the `mjcf.RootElement` object corresponding to this robot."""
#         return self._mjcf_root

#     def update_state(self, physics: mjcf.Physics, qpos: np.ndarray, qvel: np.ndarray) -> None:
#         physics.bind(self._joints).qpos[:] = qpos
#         physics.bind(self._joints).qvel[:] = qvel

#     def set_joint_angles(self, physics: mjcf.Physics, qpos: np.ndarray) -> None:
#         physics.bind(self._joints).qpos[:] = qpos

#     @property
#     def base_site(self) -> types.MjcfElement:
#         return self._base_site

#     @property
#     def wrist_site(self) -> types.MjcfElement:
#         return self._wrist_site

#     def initialize_episode(self, physics: mjcf.Physics, random_state: np.random.RandomState):
#         """Function called at the beginning of every episode."""
#         del random_state  # Unused.
#         return


# class AlohaArm(RobotArm):
#     def _build(self):
#         self._name = "aloha"
#         self._model_file = f'{ALOHA_XML_DIR}/aloha_ik_dm.xml'
#         self._mjcf_root = mjcf.from_path(self._model_file)
#         self._create_body()
