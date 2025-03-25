import numpy as np
import random
from tabletop.constants import *
from tabletop.wrappers import *
from tabletop.aloha_env_base import AlohaTask

class DishDrainer(AlohaTask):
    def __init__(self, random=None):
        super().__init__(random=random, single_arm=False) ## always first
        self.add_object('drainer', 'Rubbermaid_Large_Drainer', pos=[-0.1, 0.1, 0.01], rpy=[0, 0, -60], scale=[0.6, 0.6, 0.6])
        self.add_object('plate', 'Threshold_Bistro_Ceramic_Dinner_Plate_Ruby_Ring', pos=[0.1, 0, 0.01], scale=[0.6, 0.6, 0.6], mass=0.2)

    def initialize_episode(self, physics):
        random_vector = np.random.randn(2)
        plate_pos = np.array([0.13, 0.0, 0.01])
        plate_pos[:2] += random_vector * 0.01
        plate_rpy = np.array([0, 0, 0],)
        self.set_object_pose(physics, 'plate', pos=plate_pos, rpy=plate_rpy)
        super().initialize_episode(physics) ## always last

    def get_reward(self, physics):
        ## [condition, counter]
        reward_condition_list = [
            [self.get_touch_condition(physics, 'right_arm', 'plate'), 10],
            [not self.get_touch_condition(physics, 'table', 'plate'), 50],
            [self.get_touch_condition(physics, 'drainer', 'plate'), 100],
        ]
        return super().get_reward(physics, reward_condition_list) ### always first
    
    def get_instruction(self, reward):
        return 'Pick up the dish and put on to the drainer'
    

#####Single Arm Tasks##############################
class UprightMug(AlohaTask):
    def __init__(self, random=None):
        super().__init__(random=random, single_arm=True, single_arm_dir='right') ## always first
        self.add_object('mug', 'Threshold_Porcelain_Coffee_Mug_All_Over_Bead_White', pos=[0.3, 0.1, 0.04], rpy=[90, 0, 20], scale=[0.8, 0.8, 0.8])

    def initialize_episode(self, physics):
        mug_pos = np.array([0.0, 0.0, 0.04])
        mug_pos[:2] += np.random.uniform([0.1, -0.3], [0.25, 0.1], size=2)
        mug_rpy = np.array([90, 0, 0],)
        mug_rpy[-1] = np.random.uniform(-180, 180, 1)
        self.set_object_pose(physics, 'mug', pos=mug_pos, rpy=mug_rpy)
        super().initialize_episode(physics) ## always last

    def get_reward(self, physics):
        ## [condition, counter]
        _, quat = self.get_object_pose(physics, 'mug')
        rpy = quat_to_rpy(quat)
        reward_condition_list = [
            [self.get_touch_condition(physics, 'mug', 'table') and
              self.get_touch_condition(physics, 'mug', 'right_arm') and
              self.get_rpy_condtion(physics, rpy, [0, 0, 0], [1, 1, 0]), 10],
        ]
        return super().get_reward(physics, reward_condition_list) ### always first
    
    def get_instruction(self, reward):
        return 'Upright the white mug'
    
class ToyBasket(AlohaTask):
    def __init__(self, random=None):
        super().__init__(random=random, single_arm=True, single_arm_dir='left') ## always first
        self.add_object('basket', 'Target_Basket_Medium', pos=[0.0, 0.15, 0.00], rpy=[0, 0, 0], scale=[0.6, 0.6, 0.6])
        self.add_object('toy', 'My_First_Wiggle_Crocodile', pos=[-0.2, -0.3, 0.00], rpy=[0, 0, 20], scale=[0.6, 0.6, 0.6])

    def initialize_episode(self, physics):
        toy_pos = np.array([0.0, -0.15, 0.0])
        toy_pos[:2] += np.random.uniform([-0.3, -0.25], [-0.05, -0.05], size=2)
        toy_rpy = np.array([0, 0, 0],)
        toy_rpy[-1] = np.random.uniform(-180, 180, 1)
        self.set_object_pose(physics, 'toy', toy_pos, toy_rpy)

        basket_pos = np.array([0.0, 0.0, 0.0])
        basket_pos[0] += np.random.uniform(-0.25, -0.15, 1)
        basket_rpy = np.array([0, 0, 0],)
        basket_rpy[-1] = np.random.uniform(-180, 180, 1)
        self.set_object_pose(physics, 'basket', basket_pos, basket_rpy)
        super().initialize_episode(physics) ## always last

    def get_reward(self, physics):
        ## [condition, counter]
        pos_toy, _ = self.get_object_pose(physics, 'toy')
        pos_basket, _ = self.get_object_pose(physics, 'basket')
        reward_condition_list = [
            [self.get_touch_condition(physics, 'basket', 'toy') and
              self.get_pos_condition(physics, pos_toy, pos_basket, 0.05), 10],
        ]
        return super().get_reward(physics, reward_condition_list) ### always first
    
    def get_instruction(self, reward):
        return 'Pick up the corocodile toy and put into the basket'
    
class StackPot(AlohaTask):
    def __init__(self, random=None):
        super().__init__(random=random, single_arm=True, single_arm_dir='right') ## always first
        self.add_object('pot1', 'Cole_Hardware_Flower_Pot_1025', pos=[0.0, 0.15, 0.00], rpy=[0, 0, 0], scale=[0.4, 0.4, 0.4])
        self.add_object('pot2', 'Cole_Hardware_Electric_Pot_Assortment_55', pos=[0.0, 0.00, 0.00], rpy=[0, 0, 0], scale=[0.6, 0.6, 0.6])
        self.add_object('pot3', 'Cole_Hardware_Electric_Pot_Cabana_55', pos=[0.0, -0.15, 0.00], rpy=[0, 0, 0], scale=[0.6, 0.6, 0.6])

    def initialize_episode(self, physics):
        pot1_pos = np.array([0.05, -0.15, 0.0])
        pot2_pos = np.array([0.05, 0.15, 0.0])
        pot3_pos = np.array([0.05, 0.00, 0.0])
        pot1_pos[:2] += np.random.uniform([-0.05, -0.05], [0.05, 0.05], size=2)
        pot2_pos[:2] += np.random.uniform([-0.05, -0.05], [0.05, 0.05], size=2)
        pot3_pos[:2] += np.random.uniform([-0.05, -0.05], [0.05, 0.05], size=2)
        
        toy_rpy = np.array([0, 0, 0],)
        toy_rpy[-1] = np.random.uniform(-180, 180, 1)

        self.set_object_pose(physics, 'pot1', pot1_pos, toy_rpy)
        self.set_object_pose(physics, 'pot2', pot2_pos, toy_rpy)
        self.set_object_pose(physics, 'pot3', pot3_pos, toy_rpy)
        super().initialize_episode(physics) ## always last

    def get_reward(self, physics):
        ## [condition, counter]
        reward_condition_list = [
            [self.get_touch_condition(physics, 'pot1', 'pot2') and self.get_touch_condition(physics, 'pot2', 'pot3')
             or self.get_touch_condition(physics, 'pot1', 'pot3') and self.get_touch_condition(physics, 'pot2', 'pot3'), 10],
        ]
        return super().get_reward(physics, reward_condition_list) ### always first
    
    def get_instruction(self, reward):
        return 'Stack the pots'

ALOHA_TASK_CONFIGS = {
    'aloha_dish_drainer': {
        'task_class': DishDrainer,
        'episode_len': 1200,
    },
    'aloha_upright_mug': {
        'task_class': UprightMug,
        'episode_len': 600,
    },
    'aloha_toy_basket': {
        'task_class': ToyBasket,
        'episode_len': 600,
    },
    'aloha_stack_pot': {
        'task_class': StackPot,
        'episode_len': 600,
    }
}
