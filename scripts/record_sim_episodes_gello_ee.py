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

    def __init__(self, env, physics, height, width, gello):
        super().__init__()
        self.env = env
        self.physics = physics
        self.height = height
        self.width = width
        self.gello = gello
        self.running = True
        self.reset_flag = True

    def run(self):
        while self.running:
            reset_flag = False
            while self.gello.start and self.running:
                start_time = time.time()
                # left_pos = self.gello.action['left_pose']
                # right_pos = self.gello.action['right_pose']
                left_grp = self.gello.action['left_gripper_command']
                right_grp = self.gello.action['right_gripper_command']
                left_joint = self.gello.action['left_qpos']
                right_joint = self.gello.action['right_qpos']

                ## Joint Remapping (haha)
                remap_idx = [0, 1, 2, 4, 3, 5]
                left_joint = left_joint[remap_idx]
                right_joint = right_joint[remap_idx]
                left_joint[0] = left_joint[0] - np.pi / 2
                right_joint[0] = right_joint[0] - np.pi / 2
                left_joint[1] = left_joint[1] - np.pi / 2
                right_joint[1] = right_joint[1] - np.pi / 2
                left_joint[2] = left_joint[2] + np.pi / 2
                right_joint[2] = right_joint[2] + np.pi / 2
                ##
                ##

                action = np.concatenate([left_joint, 1 - left_grp, right_joint, 1 - right_grp])
                ts = self.env.step(action)
                self.reward_signal.emit(np.array([ts.reward, self.env.task.max_reward]))  # Send reward to UI

                # RENDER
                img = self.physics.render(self.height, self.width, camera_id=0)
                img = np.ascontiguousarray(img[:self.height, :self.width, :3].astype(np.uint8))
                self.image_signal.emit(img)
                ##

                ## TERMINATE
                if ts.step_type == dm_env.StepType.LAST:
                    self.gello.start = False
                ##
                delta_time=  time.time() - start_time
                time.sleep(DT - delta_time if DT - delta_time > 0 else 0)
            while not self.gello.start and self.running:
                if reset_flag == False:
                    ts = self.env.reset()
                    reset_flag = True
                img = self.physics.render(self.height, self.width, camera_id=0)
                img = np.ascontiguousarray(img[:self.height, :self.width, :3].astype(np.uint8))
                self.image_signal.emit(img)
                time.sleep(DT)


    def stop(self):
        self.running = False
        self.quit()
        self.wait()


class SimulationUI(QWidget):
    def __init__(self, task_name, num_episodes, width=1600, height=900):
        super().__init__()
        self.task_name = task_name
        self.num_episodes = num_episodes
        self.width = width
        self.height = height

        self.gello = GelloEnv()
        self.env = tabletop.env(task_name, 'joint_pos')
        self.physics = self.env.physics
        self.physics.model.vis.global_.offwidth = self.width
        self.physics.model.vis.global_.offheight = self.height
        self.initUI()

        self.render_thread = RenderThread(self.env, self.physics, self.height, self.width, self.gello)
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
    parser.add_argument('--task_name', action='store', type=str, default='aloha_dish_drainer', required=False)
    parser.add_argument('--num_episodes', action='store', type=int, default=1)
    
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    ui = SimulationUI(args.task_name, args.num_episodes)
    ui.show()
    sys.exit(app.exec_())
