from absl import logging
import rclpy 
from std_msgs.msg import Float64MultiArray, Bool
import numpy as np
import threading
import time

class GelloEnv:
    def __init__(self):
        rclpy.init() # initialize ROS2 node
        self._node = rclpy.create_node('gello_env_node')
        self._subscriber_bringup()
        self.hz = 20 # control frequency
        self.timer_thread = threading.Thread(target=rclpy.spin, args=(self._node,), daemon=True)
        self.timer_thread.start()
        self.start = False
        logging.set_verbosity(logging.INFO)
        logging.info('Successfully initialized.')
        self.xyz_scale = np.array([1.0, 1.0, 1])
        self.xyz_offset = np.array([0.1, 0.0, -0.1])

    def ros_close(self):
        self.timer_thread.stop()
        self._node.destroy_node()
        rclpy.shutdown()

    def get_action(self, action_space='ee'):
        return self.action

    def _subscriber_bringup(self):
        '''
        Note: This function creates all the subscribers \
              for reading joint and gripper states.
        '''
        ###### Initial Setup ##### 
        self.action = {} 

        ###### ACTION ######
        self._node.create_subscription(Float64MultiArray, '/right/joint_command', self.right_joint_command_callback, 10) # 6dim joint(Unitree Z1) gripper
        self._node.create_subscription(Float64MultiArray, '/left/joint_command', self.left_joint_command_callback, 10)  # 6dim joint(Unitree Z1) gripper
        self._node.create_subscription(Float64MultiArray, '/right/rexel/pose_states', self.right_pose_command_callback, 10) # xyz quat gripper
        self._node.create_subscription(Float64MultiArray, '/left/rexel/pose_states', self.left_pose_command_callback, 10) # xyz quat gripper
        self.action['right_qpos'] = np.zeros(shape=(6,), dtype=np.float64)
        self.action['left_qpos'] = np.zeros(shape=(6,), dtype=np.float64)
        self.action['right_pose'] = np.zeros(shape=(7,), dtype=np.float64) # xyz quat
        self.action['left_pose'] = np.zeros(shape=(7,), dtype=np.float64) # xyz quat
        self.action['right_gripper_command'] = np.zeros(shape=(1,), dtype=np.float64)
        self.action['left_gripper_command'] = np.zeros(shape=(1,), dtype=np.float64)

        ##### TRIGGER ##### 
        self._node.create_subscription(Bool, '/done', self.done_callback, 10)

    # callback functions 
    #### ACTIONS ########
    def right_joint_command_callback(self, msg):
        self.action['right_qpos'] = np.array(msg.data[:6])
        if msg.data[6] >= 3.0:
            grp = 1.0
        else:
            grp = -1.0
        self.action['right_gripper_command'] = np.array([grp])

    def left_joint_command_callback(self, msg):
        self.action['left_qpos'] = np.array(msg.data[:6])
        if msg.data[6] >= 3.0:
            grp = 1.0
        else:
            grp = -1.0
        self.action['left_gripper_command'] = np.array([grp])

    def right_pose_command_callback(self, msg):
        action_pose = np.array(msg.data)
        action_pose[0:3] = action_pose[0:3] * self.xyz_scale + self.xyz_offset
        self.action['right_pose'] = action_pose
                       
    def left_pose_command_callback(self, msg):
        action_pose = np.array(msg.data)
        action_pose[0:3] = action_pose[0:3] * self.xyz_scale + self.xyz_offset
        self.action['left_pose'] = action_pose

    def done_callback(self, msg):
        if not self.start:
            self.start = True
        else:
            self.start = False

if __name__ == '__main__':
    env = GelloEnv()

    try:
        while True: 
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        env.ros_close()