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
        plate_pos[:2] += random_vector * 0.03
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
            [self.get_touch_condition(physics, 'right_arm', 'box'), 10],
            [self.get_touch_condition(physics, 'left_arm', 'box'), 10],
            [self.get_touch_condition(physics, 'box', 'basket'), 10],
        ]
        return super().get_reward(physics, reward_condition_list) ### always first
    
    def get_instruction(self, reward):
        return 'Handover the obx and place into the pink basket'
    
class PrepareMeal(AlohaTask):
    def __init__(self, random=None):
        super().__init__(random=random, single_arm=False)
        self.add_object('pan', 'Chefmate_8_Frypan', pos=[0.16, 0.1, 0.02], scale=[1, 1, 1], rpy=[0, 0, 120], mass=0.1)
        self.add_object('spatula', 'OXO_Cookie_Spatula', pos=[-0.18, 0.05, 0.02], scale=[0.8, 0.8, 0.8], rpy=[0, 0, 220],mass=0.1)
        self.add_object('meal', 'SANDWICH_MEAL', pos=[0.12, 0.05, 0.04], scale=[1, 1, 1], mass=0.01)
        self.add_object('plate', 'Threshold_Salad_Plate_Square_Rim_Porcelain', pos=[0.0, -0.2, 0.01], scale=[0.8, 0.8, 0.8], mass=1)

    def initialize_episode(self, physics):
        plate_pos = np.array([0.0, -0.25, 0.01])
        plate_pos[:2] += np.random.randn(2) * 0.04
        self.set_object_pose(physics, 'plate', pos=plate_pos, rpy=[0.0, 0.0, 0.0])
        super().initialize_episode(physics)

    def get_reward(self, physics):
        reward_condition_list = [
            [self.get_touch_condition(physics, 'meal', 'plate'), 100]
        ]
        return super().get_reward(physics, reward_condition_list)

    def get_instruction(self, reward):
        return 'Move the sandwitch to the plate'

class LiftLargeBook(AlohaTask):
    def __init__(self, random=None):
        super().__init__(random=random, single_arm=False)
        self.add_object('book', 'Eat_to_Live_The_Amazing_NutrientRich_Program_for_Fast_and_Sustained_Weight_Loss_Revised_Edition_Book', pos=[0.0, 0.0, 0.02], scale=[1.2, 0.9, 0.15], mass=0.6)

    def initialize_episode(self, physics):
        random_vector = np.random.randn(2) * 0.03
        self.set_object_pose(physics, 'book', pos=[0.0 + random_vector[0], 0.0 + random_vector[1], 0.02], rpy=[0, 0, 0])
        super().initialize_episode(physics)

    def get_reward(self, physics):
        book_pos, _ = self.get_object_pose(physics, 'book')
        reward_condition_list = [
            [self.get_touch_condition(physics, 'right_arm', 'book'), 10],
            [self.get_touch_condition(physics, 'left_arm', 'book'), 10],
            [book_pos[2] > 0.3, 100],
        ]
        return super().get_reward(physics, reward_condition_list)

    def get_instruction(self, reward):
        return 'Use both arms to lift the large book.'

class MoveBookToShelf(AlohaTask):
    def __init__(self, random=None):
        super().__init__(random=random, single_arm=False)
        self.add_object('book', 'Eat_to_Live_The_Amazing_NutrientRich_Program_for_Fast_and_Sustained_Weight_Loss_Revised_Edition_Book', pos=[0.2, -0.2, 0.02], scale=[1.2, 0.9, 0.15], mass=0.6)
        self.add_object('shelf', 'Threshold_Tray_Rectangle_Porcelain', pos=[-0.2, 0.2, 0.1], scale=[1.5, 0.9, 0.15])

    def initialize_episode(self, physics):
        random_vector = np.random.randn(2) * 0.03
        self.set_object_pose(physics, 'book', pos=[0.2 + random_vector[0], -0.2 + random_vector[1], 0.02], rpy=[0, 0, 0])
        random_vector = np.random.randn(2) * 0.04
        self.set_object_pose(physics, 'shelf', pos=[-0.2 + random_vector[0], 0.2 + random_vector[1], 0.1], rpy=[0, 0, 0])
        super().initialize_episode(physics)

    def get_reward(self, physics):
        book_pos, _ = self.get_object_pose(physics, 'book')
        shelf_pos, _ = self.get_object_pose(physics, 'shelf')
        reward_condition_list = [
            [self.get_touch_condition(physics, 'right_arm', 'book'), 10],
            [self.get_touch_condition(physics, 'left_arm', 'book'), 10],
            [self.get_pos_condition(physics, book_pos[:2], shelf_pos[:2], delta=0.6) and book_pos[2] > shelf_pos[2] - 0.15 and book_pos[2] < shelf_pos[2] + 0.3, 100],
        ]
        return super().get_reward(physics, reward_condition_list)

    def get_instruction(self, reward):
        return 'Use both arms to pick up the book and place it on the porcelain tray shelf.'

class HoldLargeMug(AlohaTask):
    def __init__(self, random=None):
        super().__init__(random=random, single_arm=False)
        self.add_object('mug', 'BIA_Cordon_Bleu_White_Porcelain_Utensil_Holder_900028', pos=[0.0, 0.0, 0.02], scale=[0.8, 0.8, 1.0], mass=0.4)

    def initialize_episode(self, physics):
        random_vector = np.random.randn(2) * 0.03
        self.set_object_pose(physics, 'mug', pos=[0.0 + random_vector[0], 0.0 + random_vector[1], 0.02], rpy=[0, 0, 0])
        super().initialize_episode(physics)

    def get_reward(self, physics):
        mug_pos, _ = self.get_object_pose(physics, 'mug')
        reward_condition_list = [
            [self.get_touch_condition(physics, 'right_arm', 'mug'), 20],
            [self.get_touch_condition(physics, 'left_arm', 'mug'), 20],
            [mug_pos[2] > 0.3, 100],
        ]
        return super().get_reward(physics, reward_condition_list)

    def get_instruction(self, reward):
        return 'Use both arms to lift and hold the large utensil holder.'

class PlaceMugOnTable(AlohaTask):
    def __init__(self, random=None):
        super().__init__(random=random, single_arm=False)
        self.add_object('mug', 'ACE_Coffee_Mug_Kristen_16_oz_cup', pos=[0.3, 0.0, 0.02], scale=[1.0, 1.0, 1.0], mass=0.3)

    def initialize_episode(self, physics):
        random_vector = np.random.randn(2) * 0.03
        self.set_object_pose(physics, 'mug', pos=[0.3 + random_vector[0], 0.0 + random_vector[1], 0.02], rpy=[0, 0, 0])
        super().initialize_episode(physics)

    def get_reward(self, physics):
        mug_pos, _ = self.get_object_pose(physics, 'mug')
        reward_condition_list = [
            [self.get_touch_condition(physics, 'right_arm', 'mug'), 10],
            [self.get_touch_condition(physics, 'left_arm', 'mug'), 10],
            [mug_pos[2] < 0.1, 100],
        ]
        return super().get_reward(physics, reward_condition_list)

    def get_instruction(self, reward):
        return 'Use both arms to gently place the coffee mug on the table.'

class MoveBowlWithBothHands(AlohaTask):
    def __init__(self, random=None):
        super().__init__(random=random, single_arm=False)
        self.add_object('bowl', 'Bradshaw_International_11642_7_Qt_MP_Plastic_Bowl', pos=[-0.3, 0.0, 0.02], scale=[1.0, 1.0, 1.0], mass=0.25)

    def initialize_episode(self, physics):
        random_vector = np.random.randn(2) * 0.03
        self.set_object_pose(physics, 'bowl', pos=[-0.3 + random_vector[0], 0.0 + random_vector[1], 0.02], rpy=[0, 0, 0])
        super().initialize_episode(physics)

    def get_reward(self, physics):
        bowl_pos, _ = self.get_object_pose(physics, 'bowl')
        reward_condition_list = [
            [self.get_touch_condition(physics, 'right_arm', 'bowl'), 10],
            [self.get_touch_condition(physics, 'left_arm', 'bowl'), 10],
            [self.get_pos_condition(physics, bowl_pos[:2], [0.3, 0.0], delta=0.45), 100],
        ]
        return super().get_reward(physics, reward_condition_list)

    def get_instruction(self, reward):
        return 'Use both arms to move the plastic bowl to the right side of the table.'

class LiftAndHoldPlate(AlohaTask):
    def __init__(self, random=None):
        super().__init__(random=random, single_arm=False)
        self.add_object('plate', 'Threshold_Bistro_Ceramic_Dinner_Plate_Ruby_Ring', pos=[0.0, -0.2, 0.02], scale=[1.0, 1.0, 1.0], mass=0.35)

    def initialize_episode(self, physics):
        random_vector = np.random.randn(2) * 0.03
        self.set_object_pose(physics, 'plate', pos=[0.0 + random_vector[0], -0.2 + random_vector[1], 0.02], rpy=[0, 0, 0])
        super().initialize_episode(physics)

    def get_reward(self, physics):
        plate_pos, _ = self.get_object_pose(physics, 'plate')
        reward_condition_list = [
            [self.get_touch_condition(physics, 'right_arm', 'plate'), 20],
            [self.get_touch_condition(physics, 'left_arm', 'plate'), 20],
            [plate_pos[2] > 0.25, 100],
        ]
        return super().get_reward(physics, reward_condition_list)

    def get_instruction(self, reward):
        return 'Use both arms to lift the dinner plate and hold it steadily in the air.'

class PlacePlateOnAnotherPlate(AlohaTask):
    def __init__(self, random=None):
        super().__init__(random=random, single_arm=False)
        self.add_object('bottom_plate', 'Threshold_Bistro_Ceramic_Dinner_Plate_Ruby_Ring', pos=[0.0, 0.0, 0.02], scale=[1.0, 1.0, 1.0], mass=0.35)
        self.add_object('top_plate', 'Threshold_Bistro_Ceramic_Dinner_Plate_Ruby_Ring', pos=[0.3, 0.3, 0.02], scale=[1.0, 1.0, 1.0], mass=0.35)

    def initialize_episode(self, physics):
        random_vector = np.random.randn(2) * 0.03
        self.set_object_pose(physics, 'bottom_plate', pos=[0.0 + random_vector[0], 0.0 + random_vector[1], 0.02], rpy=[0, 0, 0])
        random_vector = np.random.randn(2) * 0.03
        self.set_object_pose(physics, 'top_plate', pos=[0.3 + random_vector[0], 0.3 + random_vector[1], 0.02], rpy=[0, 0, 0])
        super().initialize_episode(physics)

    def get_reward(self, physics):
        bottom_plate_pos, _ = self.get_object_pose(physics, 'bottom_plate')
        top_plate_pos, _ = self.get_object_pose(physics, 'top_plate')
        reward_condition_list = [
            [self.get_touch_condition(physics, 'right_arm', 'top_plate'), 10],
            [self.get_touch_condition(physics, 'left_arm', 'top_plate'), 10],
            [self.get_pos_condition(physics, top_plate_pos[:2], bottom_plate_pos[:2], delta=0.15) and top_plate_pos[2] > bottom_plate_pos[2] + 0.01, 100],
        ]
        return super().get_reward(physics, reward_condition_list)

    def get_instruction(self, reward):
        return 'Use both arms to carefully place the second dinner plate on top of the first one.'

class CarryLargeBottle(AlohaTask):
    def __init__(self, random=None):
        super().__init__(random=random, single_arm=False)
        self.add_object('bottle', 'Nestle_Pure_Life_Exotics_Sparkling_Water_Strawberry_Dragon_Fruit_8_count_12_fl_oz_can', pos=[0.0, 0.0, 0.02], scale=[0.6, 0.6, 1.5], mass=0.7)

    def initialize_episode(self, physics):
        random_vector = np.random.randn(2) * 0.03
        self.set_object_pose(physics, 'bottle', pos=[0.0 + random_vector[0], 0.0 + random_vector[1], 0.02], rpy=[0, 0, 0])
        super().initialize_episode(physics)

    def get_reward(self, physics):
        bottle_pos, _ = self.get_object_pose(physics, 'bottle')
        reward_condition_list = [
            [self.get_touch_condition(physics, 'right_arm', 'bottle'), 20],
            [self.get_touch_condition(physics, 'left_arm', 'bottle'), 20],
            [self.get_pos_condition(physics, bottle_pos[:2], [0.3, -0.3], delta=0.45) and bottle_pos[2] > 0.2, 100],
        ]
        return super().get_reward(physics, reward_condition_list)

    def get_instruction(self, reward):
        return 'Use both arms to lift and carry the large sparkling water bottle to the front right.'

class PlaceBottleInHolder(AlohaTask):
    def __init__(self, random=None):
        super().__init__(random=random, single_arm=False)
        self.add_object('bottle', 'Nestle_Pure_Life_Exotics_Sparkling_Water_Strawberry_Dragon_Fruit_8_count_12_fl_oz_can', pos=[0.3, -0.1, 0.02], scale=[0.6, 0.6, 1.5], mass=0.7)
        self.add_object('holder', 'BIA_Cordon_Bleu_White_Porcelain_Utensil_Holder_900028', pos=[-0.3, 0.1, 0.01], scale=[0.9, 0.9, 1.2])

    def initialize_episode(self, physics):
        random_vector = np.random.randn(2) * 0.03
        self.set_object_pose(physics, 'bottle', pos=[0.3 + random_vector[0], -0.1 + random_vector[1], 0.02], rpy=[0, 0, 0])
        random_vector = np.random.randn(2) * 0.04
        self.set_object_pose(physics, 'holder', pos=[-0.3 + random_vector[0], 0.1 + random_vector[1], 0.01], rpy=[0, 0, 0])
        super().initialize_episode(physics)

    def get_reward(self, physics):
        bottle_pos, _ = self.get_object_pose(physics, 'bottle')
        holder_pos, _ = self.get_object_pose(physics, 'holder')
        reward_condition_list = [
            [self.get_touch_condition(physics, 'right_arm', 'bottle'), 10],
            [self.get_touch_condition(physics, 'left_arm', 'bottle'), 10],
            [self.get_pos_condition(physics, bottle_pos[:2], holder_pos[:2], delta=0.4) and bottle_pos[2] < holder_pos[2] + 0.8 and bottle_pos[2] > holder_pos[2] - 0.2, 100],
        ]
        return super().get_reward(physics, reward_condition_list)

    def get_instruction(self, reward):
        return 'Use both arms to place the sparkling water bottle inside the white utensil holder.'

class MoveTwoBowls(AlohaTask):
    def __init__(self, random=None):
        super().__init__(random=random, single_arm=False)
        self.add_object('bowl1', 'Bradshaw_International_11642_7_Qt_MP_Plastic_Bowl', pos=[0.3, 0.1, 0.02], scale=[0.8, 0.8, 0.8], mass=0.2)
        self.add_object('bowl2', 'Bradshaw_International_11642_7_Qt_MP_Plastic_Bowl', pos=[-0.3, -0.1, 0.02], scale=[0.8, 0.8, 0.8], mass=0.2)

    def initialize_episode(self, physics):
        random_vector = np.random.randn(2) * 0.03
        self.set_object_pose(physics, 'bowl1', pos=[0.3 + random_vector[0], 0.1 + random_vector[1], 0.02], rpy=[0, 0, 0])
        random_vector = np.random.randn(2) * 0.03
        self.set_object_pose(physics, 'bowl2', pos=[-0.3 + random_vector[0], -0.1 + random_vector[1], 0.02], rpy=[0, 0, 0])
        super().initialize_episode(physics)

    def get_reward(self, physics):
        bowl1_pos, _ = self.get_object_pose(physics, 'bowl1')
        bowl2_pos, _ = self.get_object_pose(physics, 'bowl2')
        target_pos = np.array([0.0, 0.0, 0.1])
        reward_condition_list = [
            [self.get_touch_condition(physics, 'right_arm', 'bowl1'), 10],
            [self.get_touch_condition(physics, 'left_arm', 'bowl2'), 10],
            [self.get_pos_condition(physics, bowl1_pos, target_pos, delta=0.3) and self.get_pos_condition(physics, bowl2_pos, target_pos, delta=0.3), 100],
        ]
        return super().get_reward(physics, reward_condition_list)

    def get_instruction(self, reward):
        return 'Use both arms to move both plastic bowls towards the center of the table.'
    
ALOHA_TASK_CONFIGS = {
    'aloha_dish_drainer': {
        'task_class': DishDrainer,
        'episode_len': 10,
    },
    'aloha_handover_box': {
        'task_class':HandoverBox,
        'episode_len': 10,
    },
    'aloha_prepare_meal': {
        'task_class':PrepareMeal,
        'episode_len': 10,
    },
    'aloha_lift_large_book': {
        'task_class': LiftLargeBook,
        'episode_len': 600,
    },
    'aloha_move_book_to_shelf': {
        'task_class': MoveBookToShelf,
        'episode_len': 1200,
    },
    'aloha_hold_large_mug': {
        'task_class': HoldLargeMug,
        'episode_len': 600,
    },
    'aloha_place_mug_on_table': {
        'task_class': PlaceMugOnTable,
        'episode_len': 600,
    },
    'aloha_move_bowl_with_both_hands': {
        'task_class': MoveBowlWithBothHands,
        'episode_len': 1200,
    },
    'aloha_lift_and_hold_plate': {
        'task_class': LiftAndHoldPlate,
        'episode_len': 600,
    },
    'aloha_place_plate_on_another_plate': {
        'task_class': PlacePlateOnAnotherPlate,
        'episode_len': 1200,
    },
    'aloha_carry_large_bottle': {
        'task_class': CarryLargeBottle,
        'episode_len': 1200,
    },
    'aloha_place_bottle_in_holder': {
        'task_class': PlaceBottleInHolder,
        'episode_len': 1200,
    },
    'aloha_move_two_bowls': {
        'task_class': MoveTwoBowls,
        'episode_len': 1200,
    },
}