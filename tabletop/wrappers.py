import numpy as np
from tabletop.utils import *
import xml.etree.ElementTree as ET

class GSOWrapper:
    def __init__(self, name, pos, quat=[1, 0, 0, 0], scale=[1, 1, 1], mass=1.0, id=0, inertial=[0, 0, 0]):
        self.name = name
        self.pos = pos
        self.quat = quat
        self.scale = scale
        self.mass = mass
        self.gso_dir = f'mujoco_scanned_objects/models/{name}/'
        self.id = id
        self.inertial = inertial
        
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
        ET.SubElement(body, "inertial", pos=f"{self.inertial[0]} {self.inertial[1]} {self.inertial[2]}", mass=str(self.mass), diaginertia="0.002 0.002 0.002")
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
    


        


            