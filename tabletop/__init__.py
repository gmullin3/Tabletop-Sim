from tabletop.ee_sim_env import make_ee_sim_env
from tabletop.sim_env import make_sim_env
from tabletop.aloha_env import ALOHA_TASK_CONFIGS
from tabletop.constants import *
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
import numpy as np
from dm_control import mujoco
from dm_control.rl import control

# This function should be a global method to get
# all the task env with different action space
## joint_pos, joint_vel, ee_quat_pos, ee_rpy_pos, ee_quat_vel, ee_rpy_vel
def env(task_name, action_space):
    action_method = action_space.split('_')[0]
    task_aloha = task_name.split('_')[0] == 'aloha'
    if not task_aloha: ## Deprecated Envs
        if action_method == 'joint':
            env = make_sim_env(task_name, action_space)
        elif action_method == 'ee':
            env = make_ee_sim_env(task_name, action_space)
        else:
            raise NotImplementedError
    else: ## Aloha
        ## LOAD XML, ADD Control method, ADD objects
        tree = ET.parse(os.path.join(ALOHA_XML_DIR, 'scene.xml'))
        root = tree.getroot()
        worldbody = root.find('worldbody')
        # mujoco_root = root.find('mujoco')
        if action_method == 'joint':
            ET.SubElement(root, 'include', file="joint_position_actuators.xml")
        else:
            ET.SubElement(root, 'include', file="filtered_cartesian_actuators.xml")
        task = ALOHA_TASK_CONFIGS[task_name]['task_class']()
        for obj in task.obj_dict.values():
            asset, body = obj.generate_xml()
            root.append(asset)
            worldbody.append(body)

        rough_string = ET.tostring(root, encoding="unicode")
        parsed = minidom.parseString(rough_string)
        output_xml = parsed.toprettyxml(indent="  ")
        # tree.write(os.path.join(ALOHA_XML_DIR, 'aloha_temp.xml'), encoding="utf-8", xml_declaration=True)
        with open(os.path.join(ALOHA_XML_DIR, 'aloha_temp.xml'), "w", encoding="utf-8") as f:
            f.write(output_xml)
        # xml_string = ET.tostring(root, encoding="unicode", method="xml")
        physics = mujoco.Physics.from_xml_path(os.path.join(ALOHA_XML_DIR, 'aloha_temp.xml'))
        
        env = env = control.Environment(physics, task, time_limit=task.time_limit, control_timestep=DT, n_sub_steps=None, flat_observation=False)
    return env