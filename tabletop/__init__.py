from tabletop.aloha_env import ALOHA_TASK_CONFIGS
from tabletop.constants import *
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
from dm_control import mujoco
from dm_control.rl import control
import numpy as np

def env(task_name, action_space, test=False):
    assert action_space in ['joint_pos', 'ee_quat_pos', 'ee_6d_pos'], f'Invalid action space {action_space}'
    
    ## LOAD XML, ADD Control method, ADD objects
    task = ALOHA_TASK_CONFIGS[task_name]['task_class']()
    if task.single_arm:
        tree = ET.parse(os.path.join(ALOHA_XML_DIR, f'scene_{task.single_arm_dir}.xml'))
    else:
        tree = ET.parse(os.path.join(ALOHA_XML_DIR, 'scene.xml'))

    root = tree.getroot()

    ## Set the table color if specified in the task config
    table_color =  ALOHA_TASK_CONFIGS[task_name].get('table_color', None)
    if table_color is not None:
        table_texture = f'table_{table_color}'
        for texture in root.findall(".//texture"):
            if texture.get("file") == 'small_meta_table_diffuse' + ".png":
                texture.set("file", table_texture + ".png")
        for material in root.findall(".//material"):
            if material.get("texture") == 'small_meta_table_diffuse':
                material.set("texture", table_texture)

    ## Add the control method
    worldbody = root.find('worldbody')
    if task.single_arm:
        ET.SubElement(root, 'include', file=f"joint_position_actuators_{task.single_arm_dir}.xml")
    else:
        ET.SubElement(root, 'include', file="joint_position_actuators.xml")

    ## Add the objects
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
    task.time_limit = ALOHA_TASK_CONFIGS[task_name]['episode_len']
    original_task = getattr(task, 'original_task', False)
    if original_task:
        if os.path.exists(os.path.join(BENCHMARK_INFO_DIR, f'{original_task}_benchmark_info.npy')):
            task.benchmark_info = np.load(os.path.join(BENCHMARK_INFO_DIR, f'{original_task}_benchmark_info.npy'), allow_pickle=True)
    else:
        if os.path.exists(os.path.join(BENCHMARK_INFO_DIR, f'{task_name}_benchmark_info.npy')):
            task.benchmark_info = np.load(os.path.join(BENCHMARK_INFO_DIR, f'{task_name}_benchmark_info.npy'), allow_pickle=True)
    env = control.Environment(physics, task, time_limit=task.time_limit, control_timestep=DT, n_sub_steps=None, flat_observation=False)
    return env