from tabletop.aloha_env import ALOHA_TASK_CONFIGS
from tabletop.constants import *
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
from dm_control import mujoco
from dm_control.rl import control

def env(task_name, action_space, test=False):
    assert action_space in ['joint_pos', 'ee_quat_pos', 'ee_6d_pos'], f'Invalid action space {action_space}'
    
    ## LOAD XML, ADD Control method, ADD objects
    task = ALOHA_TASK_CONFIGS[task_name]['task_class']()
    if task.single_arm:
        tree = ET.parse(os.path.join(ALOHA_XML_DIR, f'scene_{task.single_arm_dir}.xml'))
    else:
        tree = ET.parse(os.path.join(ALOHA_XML_DIR, 'scene.xml'))
    root = tree.getroot()
    worldbody = root.find('worldbody')
    if task.single_arm:
        ET.SubElement(root, 'include', file=f"joint_position_actuators_{task.single_arm_dir}.xml")
    else:
        ET.SubElement(root, 'include', file="joint_position_actuators.xml")

    for obj in task.obj_dict.values():
        asset, body = obj.generate_xml()
        root.append(asset)
        worldbody.append(body)

    if test:
        ET.SubElement(root, 'include', file="grid.xml")

    rough_string = ET.tostring(root, encoding="unicode")
    parsed = minidom.parseString(rough_string)
    output_xml = parsed.toprettyxml(indent="  ")
    with open(os.path.join(ALOHA_XML_DIR, 'aloha_temp.xml'), "w", encoding="utf-8") as f:
        f.write(output_xml)
    physics = mujoco.Physics.from_xml_path(os.path.join(ALOHA_XML_DIR, 'aloha_temp.xml'))
    task.action_space = action_space
    env = env = control.Environment(physics, task, time_limit=task.time_limit, control_timestep=DT, n_sub_steps=None, flat_observation=False)
    return env