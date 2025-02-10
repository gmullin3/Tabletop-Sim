import numpy as np
import collections
import os
import random
from tabletop.constants import *
from tabletop.wrappers import *
from tabletop.utils import sample_box_pose, sample_insertion_pose
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation

class AlohaTask(base.Task):
    def __init__(self, random=None):
        self.obj_dict = {}
        self.obj_id_counter = 0
        self.time_limit = 20
        super().__init__(random=random)

    def add_object(self, nick, name, pos=[0, 0, 0.02], rpy=[0, 0, 0], scale=[1, 1, 1], mass=1.0):
        r = Rotation.from_euler('zyx', rpy, degrees=True)
        quat = np.array(r.as_quat())
        obj = GSOWrapper(name, pos, quat, scale, mass, self.obj_id_counter)        
        self.obj_dict[nick] = obj
        self.obj_id_counter += 1

    def before_step(self, action, physics):
        g_left_ctrl = ALOHA_GRIPPER_UNNORMALIZE_FN(action[6])
        g_right_ctrl = ALOHA_GRIPPER_UNNORMALIZE_FN(action[-1])
        np.copyto(physics.data.ctrl, np.concatenate([action[:6], [g_left_ctrl], action[7:-1], [g_right_ctrl]]))

    def initialize_robots(self, physics):
        np.copyto(physics.data.qpos[:16], np.array([0, -0.96, 1.16, 0, -0.3, 0, 0.0084, 0.0084, 0, -0.96, 1.16, 0, -0.3, 0, 0.0084, 0.0084]))

    def initialize_episode(self, physics):
        self.initialize_robots(physics)
        super().initialize_episode(physics)

    def set_object_pose(self, physics, nick, pos=None, rpy=None):
        obj_id = self.obj_dict[nick].id
        joint_id = 16 + 7 * obj_id
        np.copyto(physics.data.qpos[joint_id:joint_id + 7], np.concatenate([pos, rpy_to_quat(*rpy)]))

    def get_object_pose(self, physics, nick):
        obj_id = self.obj_dict[nick].id
        joint_id = 16 + 7 * obj_id
        pos = physics.data.qpos[joint_id : joint_id + 3].copy()
        quat = physics.data.qpos[joint_id + 3 : joint_id + 7].copy()
        return pos, quat

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        left_qpos_raw = qpos_raw[:8]
        right_qpos_raw = qpos_raw[8:16]
        left_arm_qpos = left_qpos_raw[:6]
        right_arm_qpos = right_qpos_raw[:6]
        left_gripper_qpos = [ALOHA_GRIPPER_NORMALIZE_FN(left_qpos_raw[6])]
        right_gripper_qpos = [ALOHA_GRIPPER_NORMALIZE_FN(right_qpos_raw[6])]
        return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        left_qvel_raw = qvel_raw[:8]
        right_qvel_raw = qvel_raw[8:16]
        left_arm_qvel = left_qvel_raw[:6]
        right_arm_qvel = right_qvel_raw[:6]
        left_gripper_qvel = [ALOHA_GRIPPER_VELOCITY_NORMALIZE_FN(left_qvel_raw[6])]
        right_gripper_qvel = [ALOHA_GRIPPER_VELOCITY_NORMALIZE_FN(right_qvel_raw[6])]
        return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])

    def get_eepos(self, physics):
        site_id_left = physics.model.site('left/gripper').id
        site_id_right = physics.model.site('right/gripper').id

        left_ee_pos_raw = physics.data.site_xpos[site_id_left].copy()
        right_ee_pos_raw = physics.data.site_xpos[site_id_right].copy()

        left_ee_mat_raw = physics.data.site_xmat[site_id_left].copy().reshape(3, 3)
        right_ee_mat_raw = physics.data.site_xmat[site_id_right].copy().reshape(3, 3)

        left_ee_quat_raw = mat_to_quat(left_ee_mat_raw)
        right_ee_quat_raw = mat_to_quat(right_ee_mat_raw)

        right_ee_pos_raw, right_ee_quat_raw, _ = ltor(right_ee_pos_raw, right_ee_quat_raw)

        ## Need to subtract from actuator centor
        site_id_left_center = physics.model.site('left/actuation_center').id
        site_id_right_center = physics.model.site('right/actuation_center').id
        left_ee_pos_center = physics.data.site_xpos[site_id_left_center].copy()
        right_ee_pos_center = physics.data.site_xpos[site_id_right_center].copy()
        left_ee_pos_raw = left_ee_pos_raw - left_ee_pos_center
        right_ee_pos_raw = right_ee_pos_raw - right_ee_pos_center

        qpos_raw = physics.data.qpos.copy()
        left_qpos_raw = qpos_raw[:8]
        right_qpos_raw = qpos_raw[8:16]
        left_gripper_qpos = [ALOHA_GRIPPER_NORMALIZE_FN(left_qpos_raw[6])]
        right_gripper_qpos = [ALOHA_GRIPPER_NORMALIZE_FN(right_qpos_raw[6])]

        return np.concatenate([left_ee_pos_raw, left_ee_quat_raw, left_gripper_qpos, right_ee_pos_raw, right_ee_quat_raw, right_gripper_qpos])
        
    @staticmethod
    def get_eepos_rpy(physics):
        site_id_left = physics.model.site('left/gripper').id
        site_id_right = physics.model.site('right/gripper').id

        left_ee_pos_raw = physics.data.site_xpos[site_id_left].copy()
        right_ee_pos_raw = physics.data.site_xpos[site_id_right].copy()

        left_ee_mat_raw = physics.data.site_xmat[site_id_left].copy().reshape(3, 3)
        right_ee_mat_raw = physics.data.site_xmat[site_id_right].copy().reshape(3, 3)

        left_ee_rpy_raw = mat_to_rpy(left_ee_mat_raw)
        right_ee_rpy_raw = mat_to_rpy(right_ee_mat_raw)

        right_ee_pos_raw, _, right_ee_rpy_raw = ltor(right_ee_pos_raw, None, right_ee_rpy_raw)

        ## Need to subtract from actuator centor
        site_id_left_center = physics.model.site('left/actuation_center').id
        site_id_right_center = physics.model.site('right/actuation_center').id
        left_ee_pos_center = physics.data.site_xpos[site_id_left_center].copy()
        right_ee_pos_center = physics.data.site_xpos[site_id_right_center].copy()
        left_ee_pos_raw = left_ee_pos_raw - left_ee_pos_center
        right_ee_pos_raw = right_ee_pos_raw - right_ee_pos_center
        
        qpos_raw = physics.data.qpos.copy()
        left_qpos_raw = qpos_raw[:8]
        right_qpos_raw = qpos_raw[8:16]
        left_gripper_qpos = [ALOHA_GRIPPER_NORMALIZE_FN(left_qpos_raw[6])]
        right_gripper_qpos = [ALOHA_GRIPPER_NORMALIZE_FN(right_qpos_raw[6])]
        return np.concatenate([left_ee_pos_raw, left_ee_rpy_raw, left_gripper_qpos, right_ee_pos_raw, right_ee_rpy_raw, right_gripper_qpos])

    @staticmethod
    def get_env_state(physics):
        return physics.data.qpos.copy()[16:]

    def get_observation(self, physics):
        # note: it is important to do .copy()
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['ee_pos'] = self.get_eepos(physics)
        obs['ee_rpy_pos'] = self.get_eepos_rpy(physics)
        obs['env_state'] = self.get_env_state(physics)
        obs['images'] = dict()
        obs['images']['back'] = physics.render(height=480, width=640, camera_id='teleoperator_pov')
        obs['images']['front'] = physics.render(height=480, width=640, camera_id='collaborator_pov')
        obs['images']['wrist_left'] = physics.render(height=480, width=640, camera_id='wrist_cam_left')
        obs['images']['wrist_right'] = physics.render(height=480, width=640, camera_id='wrist_cam_right')
        return obs

    def get_reward(self, physics, reward_condition_list = []):
        self.all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            self.all_contact_pairs.append(contact_pair)
        if self.reward < len(reward_condition_list) and reward_condition_list[self.reward]:
            self.reward += 1
        return self.reward

    def get_touch(self, physics, name1, name2):
        name1 = self.get_geoms(physics, name1)
        name2 = self.get_geoms(physics, name2)
        for contact_pair in self.all_contact_pairs:
            if (contact_pair[0] in name1 and contact_pair[1] in name2) or \
            (contact_pair[0] in name2 and contact_pair[1] in name1):
                return True  # Contact detected
        return False  # No contact

    def get_geoms(self, physics, name):
        if name == 'right_arm':
            geoms = ['right/right_g0', 'right/right_g1', 'right/right_g2', 'right/left_g0', 'right/left_g1', 'right/left_g2']
        elif name == 'left_arm':
            geoms = ['left/right_g0', 'left/right_g1', 'left/right_g2', 'left/left_g0', 'left/left_g1', 'left/left_g2']
        elif name == 'table':
            geoms = ['table']
        elif name in self.obj_dict.keys():
            geoms = self.obj_dict[name].get_geoms()
        else:
            geoms = []
        return geoms




class DishDrainer(AlohaTask):
    def __init__(self, random=None):
        super().__init__(random=random) ## always first
        self.add_object('drainer', 'Rubbermaid_Large_Drainer', pos=[-0.1, 0.1, 0.01], rpy=[0, 0, -60], scale=[0.6, 0.6, 0.6])
        self.add_object('plate', 'Threshold_Bistro_Ceramic_Dinner_Plate_Ruby_Ring', pos=[0.1, 0, 0.01], rpy=[0, 0, 0], scale=[0.6, 0.6, 0.6], mass=0.2)
        self.reward = 0

    def initialize_episode(self, physics):
        random_vector = np.random.randn(2)
        plate_pos = np.array([0.1, 0.0, 0.01])
        plate_pos[:2] += random_vector*0.01
        plate_rpy = np.array([0, 0, 0],)
        self.set_object_pose(physics, 'plate', pos=plate_pos, rpy=plate_rpy)
        super().initialize_episode(physics) ## always last

    def get_reward(self, physics):
        reward_condition_list = [
            self.get_touch(physics, 'right_arm', 'plate')
        ]
        return super().get_reward(physics, reward_condition_list) ### always first

ALOHA_TASK_CONFIGS = {
    'aloha_dish_drainer': {
        'task_class': DishDrainer,
        'episode_len': 1200,
        'camera_names': ['front', 'back']
    }
}