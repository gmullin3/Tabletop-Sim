# import cv2
import sys
import os
import numpy as np
import argparse
import h5py
import tabletop
from tabletop.utils import save_images_to_video

def replay(task_name, action_type, episode_dir, num_episode, save_dir):
    env = tabletop.env(task_name, action_type)

    episode = h5py.File(f'{episode_dir}/{task_name}/episode_{num_episode}.hdf5', 'r')
    init_state = episode['/observations/states/env_state'][0]
    actions = episode[f'/actions/{action_type}']
    
    ts = env.reset()
    # init states
    np.copyto(env.physics.data.qpos[env.task.robot_offset:], init_state)

    replay_images = []
    for a in actions:
        ts = env.step(a)
        replay_images.append(ts.observation['images']['front'])
    save_images_to_video(replay_images, f'{save_dir}/episode_{num_episode}.mp4')
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task_name', action='store', type=str, default='aloha_dish_drainer', required=False)
    parser.add_argument('-a', '--action_type', action='store', type=str, default='ee_6d_pos', required=False)
    parser.add_argument('-d', '--episode_dir', action='store', type=str, default='datasets')
    parser.add_argument('-n', '--num_episode', action='store', type=int, default=0)
    parser.add_argument('-s', '--save_dir', action='store', type=str, default='.')
    
    args = parser.parse_args()

    replay(args.task_name, args.action_type, args.episode_dir, args.num_episode, args.save_dir)