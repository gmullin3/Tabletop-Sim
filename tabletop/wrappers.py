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
import xml.etree.ElementTree as ET
import numpy as np

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

def mat_to_rpy(mat):
    return Rotation.from_matrix(mat).as_euler('zyx', degrees=False)

def mat_to_quat(mat):
    return Rotation.from_matrix(mat).as_quat()

def ltor(pos=None, quat=None, euler=None):
    if pos is not None:
        pos[:2] = -pos[:2]
    if quat is not None:
        quat[1:3] = -quat[1:3]
    if euler is not None:
        euler[:2] = -euler[:2]
    return pos, quat, euler
    
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
            # print(action[3:6], curr_quat, delta_quat, next_quat)
            self.prev_action[3:7] = next_quat

            delta_quat = Quaternion(rpy_to_quat(*action[10:13]))
            curr_quat = Quaternion(self.prev_action[11:15])
            next_quat = (curr_quat * delta_quat).elements
            self.prev_action[11:15] = next_quat
            # print('pa', self.prev_action[8:])
            # print(self.prev_action)
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

class GSOWrapper:
    def __init__(self, name, pos, quat=[1, 0, 0, 0], scale=[1, 1, 1], mass=1.0, id=0):
        self.name = name
        self.pos = pos
        self.quat = quat
        self.scale = scale
        self.mass = mass
        self.gso_dir = f'mujoco_scanned_objects/models/{name}/'
        self.id = id
        
    def generate_xml(self):
        ## ASSET
        asset = ET.Element("asset")
        ET.SubElement(asset, "texture", type="2d", name=f"{self.name}_texture_{self.id}", file=f"{self.gso_dir}texture.png")
        ET.SubElement(asset, "material", name=f"{self.name}_material_0_{self.id}", texture=f"{self.name}_texture_{self.id}")
        ET.SubElement(asset, "mesh", name=f"{self.name}_model_{self.id}", file=f"{self.gso_dir}model.obj", scale="{} {} {}".format(*self.scale))
        for i in range(32):
            ET.SubElement(asset, "mesh", name=f"{self.name}_collision_{i}_{self.id}", file=f"{self.gso_dir}model_collision_{i}.obj", scale="{} {} {}".format(*self.scale))

        ## OBJECT
        body = ET.Element("body", name=f"{self.name}_object_{self.id}", pos="{} {} {}".format(*self.pos), quat="{} {} {} {}".format(*self.quat))
        ET.SubElement(body, "joint", name=f"{self.name}_joint_{self.id}", type="free", frictionloss="0.01")
        ET.SubElement(body, "inertial", pos="0 0 0", mass=str(self.mass), diaginertia="0.002 0.002 0.002")
        ET.SubElement(body, "geom", material=f"{self.name}_material_0_{self.id}", mesh=f"{self.name}_model_{self.id}", type="mesh", contype="0", conaffinity="0", group="2")
        for i in range(32):
            ET.SubElement(body, "geom", name=f"{self.name}_collision_{i}_{self.id}", mesh=f"{self.name}_collision_{i}_{self.id}", type="mesh", group="3")
        return asset, body

    def get_joint_name(self):
        return f'{self.name}_joint_{self.id}'

    def get_geoms(self):
        geoms = []
        for i in range(32):
            geoms.append(f'{self.name}_collision_{i}_{self.id}')
        return geoms

class RewardFunction:
    def __init__(self, max_reward=4):
        self.max_reward = 4
        self.reward = 0
    
    def check_reward(self, physics, reward_condition_list):
        if reward_condition_list[self.reward]:
            self.reward += 1
        return self.reward
    


        


            