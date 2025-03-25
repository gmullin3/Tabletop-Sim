import sys
import os
import numpy as np
import argparse
import h5py
from gello_ros import GelloEnv
from tabletop.constants import *
import tabletop
from tabletop.wrappers import quat_to_rpy, rpy_to_quat
from pyquaternion import Quaternion
import dm_env
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
import time
import rclpy

class RenderThread(QThread):
    image_signal = pyqtSignal(np.ndarray)
    reward_signal = pyqtSignal(np.ndarray)  # Signal to send reward values

    def __init__(self, env, physics, height, width, gello, num_episode, save_dir):
        super().__init__()
        self.env = env
        self.physics = physics
        self.height = height
        self.width = width
        self.gello = gello
        self.running = True
        self.reset_flag = True
        self.num_episode = num_episode
        self.save_dir = save_dir
        self.episode_count = 0
        self.terminate_signal = False
        self.episode = []
        self.episode_action = []

    def run(self):
        while self.running:
            reset_flag = True
            while self.gello.start and self.running:
                start_time = time.time()
                action = self.process_action(self.gello.action)
                self.episdoe_action.append(action)

                ts = self.env.step(action)
                self.reward_signal.emit(np.array([ts.reward, self.env.task.max_reward]))  # Send reward to UI

                # RENDER
                img = self.physics.render(self.height, self.width, camera_id=0)
                img = np.ascontiguousarray(img[:self.height, :self.width, :3].astype(np.uint8))
                self.image_signal.emit(img)

                self.episode.append(ts)
                ##

                ## TERMINATE
                if ts.step_type == dm_env.StepType.LAST:
                    self.gello.start = False
                    self.terminate_signal = True
                ##

                ## Deal with latency
                delta_time=  time.time() - start_time
                time.sleep(DT - delta_time if DT - delta_time > 0 else 0)

            while not self.gello.start and self.running:
                if reset_flag == False:
                    self.save_demo()
                    ts = self.env.reset()
                    self.episode = [ts]
                    self.episode_action = []
                    reset_flag = True
                    self.terminate_signal = False
                img = self.physics.render(self.height, self.width, camera_id=0)
                img = np.ascontiguousarray(img[:self.height, :self.width, :3].astype(np.uint8))
                self.image_signal.emit(img)
                time.sleep(DT)

    def process_action(self):
        left_grp = self.gello.action['left_gripper_command']
        right_grp = self.gello.action['right_gripper_command']

        if self.action_type.split('_')[0] == 'ee':
            ## Assuming quaternion control
            if self.single_arm:
                right_pos = self.gello.action['right_pose']
                right_xyz = right_pos[:3]
                right_quat = quat_to_rpy(*right_pos[3:6])
                action = np.concatenate([right_xyz, right_quat, 1 - right_grp])
            else:
                left_pos = self.gello.action['left_pose']
                left_xyz = left_pos[:3]
                left_quat = quat_to_rpy(*left_pos[3:6])
                right_pos = self.gello.action['right_pose']
                right_xyz = right_pos[:3]
                right_quat = quat_to_rpy(*right_pos[3:6])
                action = np.concatenate([left_xyz, left_quat, 1 - left_grp, right_xyz, right_quat, 1 - right_grp])
        else:
            if self.task.single_arm:
                right_joint = self.gello.action['right_qpos']
                remap_idx = [0, 1, 2, 4, 3, 5]
                right_joint = right_joint[remap_idx]
                right_joint[0] = right_joint[0] - np.pi / 2
                right_joint[1] = right_joint[1] - np.pi / 2
                right_joint[2] = right_joint[2] + np.pi / 2
                action = np.concatenate([right_joint, 1 - right_grp])
            else:
                # Joint control
                left_joint = self.gello.action['left_qpos']
                right_joint = self.gello.action['right_qpos']
                ## Joint Remapping - Unitree Z1 to Aloha
                remap_idx = [0, 1, 2, 4, 3, 5]
                left_joint = left_joint[remap_idx]
                right_joint = right_joint[remap_idx]
                left_joint[0] = left_joint[0] - np.pi / 2
                right_joint[0] = right_joint[0] - np.pi / 2
                left_joint[1] = left_joint[1] - np.pi / 2
                right_joint[1] = right_joint[1] - np.pi / 2
                left_joint[2] = left_joint[2] + np.pi / 2
                right_joint[2] = right_joint[2] + np.pi / 2
                action = np.concatenate([left_joint, 1 - left_grp, right_joint, 1 - right_grp])
        return action

    def save_demo(self):
        if self.terminate_signal:
            return
        num = self.episode_count
        data_dict = {
            '/observations/state/qpos': [],
            '/observations/state/qvel': [],
            '/observations/state/ee_pos': [],
            '/observations/state/ee_rpy_pos': [],
            '/observations/language_instruction': [],
            '/observations/state/env_state': [],
            '/observations/images/front': [],
            '/observations/images/back': [],
            '/action' : [],
        }
        if self.task.single_arm:
            data_dict[f'/observations/images/wrist'] = []
        else:
            data_dict[f'/observations/images/wrist_right'] = []
            data_dict[f'/observations/images/wrist_left'] = []
            
        max_timesteps = len(self.episode_action)
        for i in range(max_timesteps):
            ts = self.episode.pop(0)
            action = self.episode_action.pop(0)
            data_dict['/observations/state/qpos'].append(ts.observation['qpos'])
            data_dict['/observations/state/qvel'].append(ts.observation['qvel'])
            data_dict['/observations/state/env_state'].append(ts.observation['env_state'])
            data_dict['/observations/state/ee_pos'].append(ts.observation['ee_pos'])
            data_dict['/observations/state/ee_rpy_pos'].append(ts.observation['ee_rpy_pos'])
            data_dict['/observations/images/front'].append(ts.observation['images']['front'])
            data_dict['/observations/images/back'].append(ts.observation['images']['back'])
            if self.task.single_arm:
                data_dict['/observations/images/wrist'].append(ts.observation['images'][f'wrist_{self.task.single_arm_dir}'])
            else:
                data_dict['/observations/images/wrist_right'].append(ts.observation['images']['wrist_right'])
                data_dict['/observations/images/wrist_left'].append(ts.observation['images']['wrist_left'])
            data_dict['/observations/language_instruction'].append(ts.observation['language_instruction'])
            data_dict['/action'].append(action)
        
        dataset_path = os.path.join(self.save_dir, f'episode_{num}.hdf5')
  
        with h5py.File(dataset_path, 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            obs = root.create_group('observations')
            state = obs.create_group('state')
            qpos = state.create_dataset('qpos', (max_timesteps, 7 if self.task.single_arm else 14))
            qvel = state.create_dataset('qvel', (max_timesteps, 7 if self.task.single_arm else 14))
            ee_pos = state.create_dataset('ee_pos', (max_timesteps, 8 if self.task.single_arm else 16))
            ee_rpy_pos = state.create_dataset('ee_rpy_pos', (max_timesteps, 7 if self.task.single_arm else 14))
            env_state = state.create_dataset('env_state', (max_timesteps, data_dict['/observations/state/env_state'][0].shape[0]))
            instructions = state.create_dataset('language_instructions', (max_timesteps,), dtype=h5py.string_dtype(encoding='utf-8'))
            image = obs.create_group('images')
            image_front = image.create_dataset('front', (max_timesteps, 480, 640, 3), dtype='uint8', chunks=(1, 480, 640, 3), )
            image_back = image.create_dataset('back', (max_timesteps, 480, 640, 3), dtype='uint8', chunks=(1, 480, 640, 3), )
            if self.task.single_arm:
                image_wrist = image.create_dataset('wrist', (max_timesteps, 240, 320, 3), dtype='uint8', chunks=(1, 240, 320, 3), )
            else:
                image_wrist_right = image.create_dataset('wrist_right', (max_timesteps, 240, 320, 3), dtype='uint8', chunks=(1, 240, 320, 3), )
                image_wrist_left = image.create_dataset('wrist_left', (max_timesteps, 240, 320, 3), dtype='uint8', chunks=(1, 240, 320, 3), )
            action = root.create_dataset('action', (max_timesteps, data_dict['/action'][0].shape[0]))

            for name, array in data_dict.items():
                root[name][...] = array
        print(f'Saved {dataset_path}')
        num += 1

    def stop(self):
        self.running = False
        self.quit()
        self.wait()


class SimulationUI(QWidget):
    def __init__(self, task_name, action_type, num_episodes, save_dir, width=1600, height=900):
        super().__init__()
        self.task_name = task_name
        self.num_episodes = num_episodes
        self.action_type = action_type
        self.save_dir = f'{save_dir}/{task_name}/'
        self.width = width
        self.height = height

        self.gello = GelloEnv()
        self.env = tabletop.env(task_name, self.action_type)
        self.physics = self.env.physics
        self.physics.model.vis.global_.offwidth = self.width
        self.physics.model.vis.global_.offheight = self.height
        self.initUI()

        self.render_thread = RenderThread(
            self.env, 
            self.physics, 
            self.height, 
            self.width, 
            self.gello,
            self.num_episodes,
            self.save_dir
        )
        self.render_thread.image_signal.connect(self.display_image)
        self.render_thread.reward_signal.connect(self.set_current_reward)
        self.render_thread.start()

    def initUI(self):
        self.setWindowTitle("Tabletop Simulation")
        self.showMaximized()
        # Main layout
        self.layout = QVBoxLayout()  # layout to split image & buttons
        # Left Side - Image
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)  # Center align the image

        # Overlay Text Label (Big Text)
        self.text_label = QLabel("Reward", self)
        self.text_label.setAlignment(Qt.AlignCenter)
        self.text_label.setFont(QFont("Arial", 40, QFont.Bold))
        self.text_label.setStyleSheet("color: white; background-color: rgba(0, 0, 0, 150);")  # Semi-transparent background
        self.text_label.setGeometry(0, 0, self.width, 80)  # Position at the top

        self.layout.addWidget(self.text_label)
        self.layout.addWidget(self.image_label, stretch=3)  # Give it more space
        self.setLayout(self.layout)

    def display_image(self, img):
        h, w, ch = img.shape
        bytes_per_line = ch * w
        q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap)
        self.image_label.repaint()

    def closeEvent(self, event):
        self.render_thread.stop()
        event.accept()

    def set_current_reward(self, rewards):
        """ Updates the file name label. """
        self.text_label.setText(f"Reward : {rewards[0]} / {rewards[1]}")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task_name', action='store', type=str, default='aloha_dish_drainer', required=False)
    parser.add_argument('-a', '--action_type', action='store', type=str, default='ee_quat_pos', required=False)
    parser.add_argument('-n', '--num_episodes', action='store', type=int, default=1)
    parser.add_argument('-d', '--save_dir', action='store', type=str, default='datasets/')
    
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    ui = SimulationUI(args.task_name, args.action_type, args.num_episodes, args.save_dir)
    ui.show()
    sys.exit(app.exec_())
