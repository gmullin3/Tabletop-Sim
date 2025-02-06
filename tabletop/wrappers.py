import numpy as np
import os
import collections
import matplotlib.pyplot as plt
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
import random

from pyquaternion import Quaternion
import math
from scipy.spatial.transform import Rotation

def quat_to_rpy(w, x, y, z, mode='abs'):
    try:
        r = Rotation.from_quat([w, x, y, z], scalar_first=True)
    except:
        return np.array([0.0, 0.0, 0.0])
    return np.array(r.as_euler('zyx', degrees=False))

def rpy_to_quat(roll, pitch, yaw):
    try:
        r = Rotation.from_euler('zyx', [roll, pitch, yaw], degrees=False)
    except:
        return np.array([0.0, 0.0, 0.0, 0.0])
    return np.array(r.as_quat(scalar_first=True))

def get_ee_vel_wrapper(target_class):
    class ee_vel_class(target_class):
        def __init__(self, random=False):
            self.prev_action = None
            super().__init__(False)

        def before_step(self, action, physics):
            if self.prev_action is None:
                self.prev_action = self.get_eepos(physics)
            self.prev_action[:7] += action[:7]
            self.prev_action[7] = action[7]
            self.prev_action[8:-1] += action[8:-1]
            self.prev_action[-1] = action[-1]
            super().before_step(self.prev_action, physics)

        def initialize_robots(self, physics):
            self.prev_action = None
            super().initialize_robots(physics)

    return ee_vel_class(False)

def get_ee_rpy_vel_wrapper(target_class):
    class ee_rpy_vel_class(target_class):
        def __init__(self, random=False):
            self.prev_action = None
            super().__init__(False)
        def before_step(self, action, physics):
            if self.prev_action is None:
                self.prev_action = self.get_eepos(physics)
            # print('a', action[7:])
            self.prev_action[:3] += action[:3]
            self.prev_action[7] = action[6]
            self.prev_action[8:11] += action[7:10]
            self.prev_action[-1] = action[-1]

            delta_quat = Quaternion(rpy_to_quat(*action[3:6]))
            curr_quat = Quaternion(self.prev_action[3:7])
            next_quat = (curr_quat * delta_quat).elements
            self.prev_action[3:7] = next_quat

            delta_quat = Quaternion(rpy_to_quat(*action[10:13]))
            curr_quat = Quaternion(self.prev_action[11:15])
            next_quat = (curr_quat * delta_quat).elements
            self.prev_action[11:15] = next_quat
            # print('pa', self.prev_action[8:])
            super().before_step(self.prev_action, physics)

        def initialize_robots(self, physics):
            self.prev_action = None
            super().initialize_robots(physics)
    return ee_rpy_vel_class(False)

def get_ee_rpy_pos_wrapper(target_class):
    class ee_rpy_pos_class(target_class):
        def before_step(self, action, physics):
            curr_ee = self.get_eepos(physics)
            curr_quat_left = curr_ee[3:7]
            ee_pos_raw_left = action[:3]
            ee_rpy_raw_left = action[3:6]
            grp_left = action[6]
            ee_quat_raw_left = rpy_to_quat(ee_rpy_raw_left[0], ee_rpy_raw_left[1], ee_rpy_raw_left[2])

            curr_quat_right = curr_ee[11:15]
            ee_pos_raw_right = action[7:10]
            ee_rpy_raw_right = action[10:13]
            grp_right = action[-1]
            ee_quat_raw_right = rpy_to_quat(ee_rpy_raw_right[0], ee_rpy_raw_right[1], ee_rpy_raw_right[2])
            
            real_action = np.concatenate([ee_pos_raw_left, ee_quat_raw_left, [grp_left], ee_pos_raw_right, ee_quat_raw_right, [grp_right],], axis=0)
            
            super().before_step(real_action, physics)
    return ee_rpy_pos_class(False)

def get_onearm_ee_vel_wrapper(target_class):
    class ee_vel_class(target_class):
        def __init__(self, random=False):
            self.prev_action = None
            super().__init__(False)
        def before_step(self, action, physics):
            if self.prev_action is None:
                self.prev_action = self.get_eepos(physics)
            self.prev_action[:7] += action[:7]
            self.prev_action[7] = action[7]
            super().before_step(self.prev_action, physics)
        def initialize_robots(self, physics):
            self.prev_action = None
            super().initialize_robots(physics)
    return ee_vel_class(False)


def get_onearm_ee_rpy_vel_wrapper(target_class):
    class ee_rpy_vel_class(target_class):
        def __init__(self, random=False):
            self.prev_action = None
            super().__init__(False)
        def before_step(self, action, physics):
            if self.prev_action is None:
                self.prev_action = self.get_eepos(physics)

            self.prev_action[:3] += action[:3]
            self.prev_action[7] = action[6]
            delta_quat = Quaternion(rpy_to_quat(*action[3:6]))
            curr_quat = Quaternion(self.prev_action[3:7])
            next_quat = (curr_quat * delta_quat).elements
            self.prev_action[3:7] = next_quat
            super().before_step(self.prev_action, physics)
        def initialize_robots(self, physics):
            self.prev_action = None
            super().initialize_robots(physics)

    return ee_rpy_vel_class(False)

def get_onearm_ee_rpy_pos_wrapper(target_class):
    class ee_rpy_pos_class(target_class):
        def before_step(self, action, physics):
            curr_ee = self.get_eepos(physics)
            curr_quat = curr_ee[3:7]
            
            ee_pos_raw = action[:3]
            ee_rpy_raw = action[3:6]
            grp = action[6]
            
            ee_quat_raw = rpy_to_quat(ee_rpy_raw[0], ee_rpy_raw[1], ee_rpy_raw[2])
            
            real_action = np.concatenate([ee_pos_raw, ee_quat_raw, [grp]], axis=0)
            
            super().before_step(real_action, physics)
    return ee_rpy_pos_class(False)

def get_joint_vel_wrapper(target_class):
    class joint_vel_class(target_class):
        def __init__(self, random=False):
            self.prev_action = None
            super().__init__(False)
            
        def before_step(self, action, physics):
            if self.prev_action is None:
                self.prev_action = self.get_qpos(physics)
            self.prev_action = self.get_qpos(physics)
            self.prev_action[:6] += action[:6]
            self.prev_action[:7:-1] += action[7:-1]
            self.prev_action[6] = action[6]
            self.prev_action[-1] = action[-1]
            
            super().before_step(self.prev_action, physics)
        def initialize_robots(self, physics):
            self.prev_action = None
            super().initialize_robots(physics)

    return joint_vel_class(False)

def get_onearm_joint_vel_wrapper(target_class):
    class joint_vel_class(target_class):
        def __init__(self, random=False):
            self.prev_action = None
            super().__init__(False)
            
        def before_step(self, action, physics):
            if self.prev_action is None:
                self.prev_action = self.get_qpos(physics)
            self.prev_action[:6] += action[:6]
            self.prev_action[6] = action[6]
            super().before_step(self.prev_action, physics)

        def initialize_robots(self, physics):
            self.prev_action = None
            super().initialize_robots(physics)

    return joint_vel_class(False)
            
            
            