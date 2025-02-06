# from gello_ros import GelloEnv

import time
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import h5py

from tabletop.constants import *
from tabletop import make_ee_sim_env, make_sim_env
from tabletop.sim_env import make_sim_env, BOX_POSE
from tabletop.wrappers import quat_to_rpy, rpy_to_quat
from pyquaternion import Quaternion

from mujoco.glfw import glfw
import mujoco as mj

def main(args):
    task_name = args['task_name']
    dataset_dir = args['dataset_dir']
    num_episodes = args['num_episodes']
    render_cam_name = 'angle'
    real_idx = 0

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    camera_names = SIM_TASK_CONFIGS[task_name]['camera_names']

    # gello = GelloEnv()

    ## ENV INIT
    env = make_ee_sim_env(task_name)
    ts = env.reset()
    model = env.physics.model
    data = env.physics.data
    cam = mj.MjvCamera()
    opt = mj.MjvOption()

    ### GLFW SETUP ######################
    glfw.init()
    window = glfw.create_window(1200, 900, "Tabletop Simulation", None, None)
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    mj.mjv_defaultCamera(cam)
    mj.mjv_defaultOption(opt)
    scene = mj.MjvScene(model, maxgeom=10000)
    context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)
    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
    #####################################

    success = []
    episode_idx = 0

    while not glfw.window_should_close(window):
        while episode_idx < num_episodes:
            env = make_ee_sim_env(task_name)
            ts = env.reset()
            episode = [ts]

            # Waiting for start
            while not False:
                mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, self.scene)
                mj.mjr_render(viewport, self.scene, self.context)
                glfw.swap_buffers(self.window)
                glfw.poll_events()

    glfw.terminate()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--dataset_dir', action='store', type=str, help='dataset saving dir', required=True)
    parser.add_argument('--num_episodes', action='store', type=int, help='num_episodes', required=False)
    parser.add_argument('--onscreen_render', action='store_true')
    
    main(vars(parser.parse_args()))