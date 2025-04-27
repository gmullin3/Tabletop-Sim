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
        plate_pos = np.array([0.15, -0.17, 0.01])
        plate_pos[:2] += random_vector * 0.015
        plate_rpy = np.array([0, 0, 0],)
        self.set_object_pose(physics, 'plate', pos=plate_pos, rpy=plate_rpy)
        super().initialize_episode(physics) ## always last

    def get_reward(self, physics):
        ## [condition, counter]
        reward_condition_list = [
            [self.get_touch_condition(physics, 'drainer', 'plate'), 20],
        ]
        return super().get_reward(physics, reward_condition_list) ### always first
    
    def get_instruction(self, reward):
        return 'Pick up the dish and put on to the drainer'
    
class HandoverBox(AlohaTask):
    def __init__(self, random=None):
        super().__init__(random=random, single_arm=False) ## always first
        self.add_object('basket', 'Room_Essentials_Fabric_Cube_Lavender', pos=[0.2, 0.0, 0.01], rpy=[0, 0, -40], scale=[0.6, 0.6, 0.6])
        self.add_object('box', 'Fresca_Peach_Citrus_Sparkling_Flavored_Soda_12_PK', pos=[-0.2, -0.2, 0.01], rpy=[0, 0, 90], scale=[0.4, 0.25, 0.4], mass=0.3)

    def initialize_episode(self, physics):
        random_vector = np.random.randn(2)
        basket_pos = np.array([0.2, 0.1, 0.01])
        basket_pos[:2] += random_vector * 0.02
        basket_rpy = np.array([0.0, 0.0, 0.0],)

        random_vector = np.random.randn(2)
        box_pos = np.array([-0.2, -0.2, 0.01])
        box_pos[:2] += random_vector * 0.025
        box_rpy = np.array([0.0, 0.0, 0.0],)
        random_vector = np.random.randn(1) / (2 * np.pi)
        box_rpy[0] += random_vector
        self.set_object_pose(physics, 'basket', pos=basket_pos, rpy=basket_rpy)
        self.set_object_pose(physics, 'box', pos=box_pos, rpy=box_rpy)
        super().initialize_episode(physics) ## always last

    def get_reward(self, physics):
        ## [condition, counter]
        reward_condition_list = [
            [self.get_touch_condition(physics, 'box', 'basket'), 20],
        ]
        return super().get_reward(physics, reward_condition_list) ### always first
    
    def get_instruction(self, reward):
        return 'Handover the box and place into the pink basket'

class ShoesTable(AlohaTask):
    def __init__(self, random=None):
        super().__init__(random=random, single_arm=False) ## always first
        self.add_object('toy_table', '3D_Dollhouse_TablePurple', pos=[0.0, 0.1, 0.01], rpy=[0, 0, -80], scale=[3.0, 4.0, 2.0], mass=10.0)
        self.add_object('shoe_right', 'Womens_Canvas_Bahama_in_Black_vnJULsDVyq5', pos=[0.2, -0.2, 0.01], rpy=[0, 0, 0], scale=[0.65, 0.65, 0.65], mass=0.3)
        self.add_object('shoe_left', 'Womens_Canvas_Bahama_in_Black', pos=[-0.2, -0.2, 0.01], rpy=[0, 0, -90], scale=[0.65, 0.65, 0.65], mass=0.3)

    def initialize_episode(self, physics):
        random_vector = np.random.randn(2)
        tabletop_pos = np.array([0.0, 0.1, 0.01])
        tabletop_pos[:2] += random_vector * 0.01
        tabletop_rpy = np.array([-100/180.0*np.pi, 0, 0])

        shoe_right_pos = np.array([0.15, -0.15, 0.01])
        random_vector = np.random.randn(2)
        shoe_right_pos[:2] += random_vector * 0.05
        shoe_right_rpy = np.array([-np.pi, 0, 0])
        random_vector = np.random.randn(1)
        shoe_right_rpy[0] += random_vector * (1 / 9) * np.pi
        shoe_left_pos = np.array([-0.15, -0.15, 0.01])
        random_vector = np.random.randn(2)
        shoe_left_pos[:2] += random_vector * 0.05
        shoe_left_rpy = np.array([-90/180.0*np.pi, 0, 0])
        random_vector = np.random.randn(1)
        shoe_left_rpy[0] += random_vector * (1 / 9) * np.pi
        self.set_object_pose(physics, 'toy_table', tabletop_pos, tabletop_rpy)
        self.set_object_pose(physics, 'shoe_right', shoe_right_pos, shoe_right_rpy)
        self.set_object_pose(physics, 'shoe_left', shoe_left_pos, shoe_left_rpy)
        super().initialize_episode(physics) ## always last

    def get_reward(self, physics):
        ## [condition, counter]
        reward_condition_list = [
            [self.get_touch_condition(physics, 'shoe_right', 'toy_table'), 20] and [self.get_touch_condition(physics, 'shoe_left', 'toy_table'), 20],
        ]
        return super().get_reward(physics, reward_condition_list) ### always first
    
    def get_instruction(self, reward):
        return 'Pick up the black shoes and put them side by side on the purple table'

class LiftBoxBall(AlohaTask):
    def __init__(self, random=None):
        super().__init__(random=random, single_arm=False) ## always first
        self.add_object('box', 'Perricone_MD_Hypoallergenic_Firming_Eye_Cream_05_oz', pos=[0.0, 0.0, 0.1], rpy=[0, 0, 0], scale=[3, 3, 3], mass=0.5)
        self.add_object('ball', 'Toys_R_Us_Treat_Dispenser_Smart_Puzzle_Foobler', pos=[0.0, 0.0, 0.2], rpy=[0, 0, 0], scale=[0.4, 0.4, 0.4], mass=0.3)

    def initialize_episode(self, physics):
        # Generate random position for box within [-1.0, 1.0] range
        box_pos = np.array([
            np.random.uniform(-0.2, 0.2),  # x-coordinate
            np.random.uniform(-0.1, 0.27),  # y-coordinate
            0.02  # z-coordinate (kept at default)
        ])
        
        # Generate random position for ball within [-2.5, 2.5] range
        # Keep generating until it's sufficiently far from the box
        while True:
            ball_pos = np.array([
                np.random.uniform(-0.25, 0.25),  # x-coordinate
                np.random.uniform(-0.25, min(box_pos[1] + 0.049, 0.25)),  # y-coordinate must be less than box_pos[1] + 0.05
                0.01  # z-coordinate (kept at default)
            ])
            
            # Check if ball is far enough from box (using Euclidean distance)
            if np.linalg.norm(ball_pos[:2] - box_pos[:2]) > 0.1:  # Minimum distance threshold
                break
        
        # Set the object poses
        self.set_object_pose(physics, 'box', pos=box_pos, rpy=[0, 0, 0])
        self.set_object_pose(physics, 'ball', pos=ball_pos, rpy=[0, 0, 0])
        
        # Always call the parent's initialize_episode at the end
        super().initialize_episode(physics)  # always last

    def get_reward(self, physics):
        ## [condition, counter]
        reward_condition_list = [
            [self.get_touch_condition(physics, 'box', 'ball') and (self.get_object_pose(physics, 'box')[0][2] > 0.05), 10],
        ]
        return super().get_reward(physics, reward_condition_list)
    
    def get_instruction(self, reward):
        return 'Put ball top of the box and lift the box'    

ALOHA_TASK_CONFIGS = {
    'aloha_dish_drainer': {
        'task_class': DishDrainer,
        'episode_len': 10,
    },
    'aloha_handover_box': {
        'task_class':HandoverBox,
        'episode_len': 15,
    },
    'aloha_shoes_table': {
        'task_class': ShoesTable,
        'episode_len': 15,
    },
    'aloha_lift_box_ball': {
        'task_class': LiftBoxBall,
        'episode_len': 15,
    },
}