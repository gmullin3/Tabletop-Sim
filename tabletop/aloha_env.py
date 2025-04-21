import numpy as np
import random
from tabletop.constants import *
from tabletop.utils import *
from tabletop.wrappers import GSOWrapper
from tabletop.aloha_env_base import AlohaTask

class DishDrainer(AlohaTask):
    def __init__(self, random=None):
        super().__init__(random=random, single_arm=False) ## always first
        self.add_object('drainer', 'Rubbermaid_Large_Drainer', pos=[-0.1, 0.1, 0.01], rpy=[0, 0, -60], scale=[0.8, 0.8, 0.8])
        self.add_object('plate', 'Threshold_Bistro_Ceramic_Dinner_Plate_Ruby_Ring', pos=[0.1, 0, 0.01], scale=[0.8, 0.8, 0.8], mass=0.2)

    def initialize_episode(self, physics):
        random_vector = np.random.randn(2)
        plate_pos = np.array([0.18, -0.15, 0.01])
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
    
class PutHat(AlohaTask):
    def __init__(self, random=None):
        super().__init__(random=random, single_arm=False) ## always first
        self.add_object('cabinet', 'Threshold_Basket_Natural_Finish_Fabric_Liner_Small', pos=[-0.1, 0.1, 0.01], scale=[1.5, 1.5, 1.2])
        self.add_object('dog', 'Dog', pos=[0.1, 0, 0.01], scale=[0.8, 0.8, 0.8], mass=0.2)
        self.add_object('hat', 'DPC_tropical_Trends_Hat', pos=[0.1, 0, 0.01], scale=[0.45, 0.45, 0.45], mass=0.1)

    def initialize_episode(self, physics):
        # Position for the cabinet
        cabinet_pos = np.array([0.0, 0.0, 0.01])
        cabinet_pos[0] = np.random.uniform(-0.1, 0.1)  # x-axis between -0.1 and 0.1
        
        # For y-axis, choose either [-0.2, -0.18] or [0.18, 0.2]
        if np.random.random() < 0.5:
            cabinet_pos[1] = np.random.uniform(-0.2, -0.18)
        else:
            cabinet_pos[1] = np.random.uniform(0.18, 0.2)
            
        self.set_object_pose(physics, 'cabinet', pos=cabinet_pos, rpy=np.array([0, 0, 0]))
        
        # Get cabinet position for reference
        cabinet_pos, _ = self.get_object_pose(physics, 'cabinet')
        
        # Place dog on top of the cabinet
        dog_pos = np.array([cabinet_pos[0], cabinet_pos[1], 0.15])  # Slightly above the cabinet
        dog_pos[:2] += np.random.uniform(-0.05, 0.05, size=2)  # Small deviation but still on cabinet
        dog_pos[2] += 0.1
        self.set_object_pose(physics, 'dog', pos=dog_pos, rpy=np.random.uniform(-np.pi, np.pi, 3))
        
        # Place hat outside the cabinet
        while True:
            hat_pos = np.array([0.0, 0.0, 0.05])
            hat_pos[:2] = np.random.uniform(-0.25, 0.25, size=2)
            hat_pos[2] = 0.2
            
            # Check if hat is far enough from cabinet center
            if np.linalg.norm(hat_pos[:2] - cabinet_pos[:2]) > 0.15:  # Assuming cabinet radius is about 0.15
                break
        
        self.set_object_pose(physics, 'hat', pos=hat_pos, rpy=np.random.uniform(-np.pi, np.pi, 3))
        
        super().initialize_episode(physics) ## always last

    def get_reward(self, physics):
        ## [condition, counter]
        dog_pos, _ = self.get_object_pose(physics, 'dog')
        hat_pos, _ = self.get_object_pose(physics, 'hat')
        reward_condition_list = [
            [self.get_pos_condition(physics, hat_pos, dog_pos, 0.05), 10],
        ]
        return super().get_reward(physics, reward_condition_list) ### always first
    
    def get_instruction(self, reward):
        return 'Put all honey dippers in the turtle-shaped holder'


#####Single Arm Tasks##############################
class UprightMug(AlohaTask):
    def __init__(self, random=None):
        super().__init__(random=random, single_arm=True, single_arm_dir='right') ## always first
        self.add_object('mug', 'Threshold_Porcelain_Coffee_Mug_All_Over_Bead_White', pos=[0.3, 0.1, 0.04], rpy=[0, 0, 20], scale=[0.8, 0.8, 0.8], mass=0.15)

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
        rpy = quat_to_rpy(*quat)
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
        self.add_object('toy', 'My_First_Wiggle_Crocodile', pos=[-0.2, -0.3, 0.00], rpy=[0, 0, 20], scale=[0.6, 0.6, 0.6], mass=0.15)

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
        self.add_object('pot1', 'Cole_Hardware_Flower_Pot_1025', pos=[0.0, 0.15, 0.00], rpy=[0, 0, 0], scale=[0.4, 0.4, 0.4], mass=0.2)
        self.add_object('pot2', 'Cole_Hardware_Electric_Pot_Assortment_55', pos=[0.0, 0.00, 0.00], rpy=[0, 0, 0], scale=[0.6, 0.6, 0.6], mass=0.2)
        self.add_object('pot3', 'Cole_Hardware_Electric_Pot_Cabana_55', pos=[0.0, -0.15, 0.00], rpy=[0, 0, 0], scale=[0.6, 0.6, 0.6], mass=0.2)

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
        'episode_len': 1200,
    },
    'aloha_stack_pot': {
        'task_class': StackPot,
        'episode_len': 600,
    },
    'aloha_put_hat': {
        'task_class': PutHat,
        'episode_len': 600,
    },
}
