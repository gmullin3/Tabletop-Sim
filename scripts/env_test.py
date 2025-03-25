from tabletop import env
import argparse
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser(description="Test environment rendering.")
parser.add_argument("-t", "--task_name", type=str, help="Name of the task.")
parser.add_argument("-a", "--action_space", type=str, default="ee_quat_pos", help="Action space type (default: ee_quat_pos).")
parser.add_argument("-r", "--reset", action="store_true", help="Whether to reset the environment.")
args = parser.parse_args()

task_name = args.task_name
action_space = args.action_space
reset = args.reset

test_env = env(task_name, action_space, True)
if reset:
    test_env.reset()
img = test_env.physics.render(480, 640, camera_id=0)
output_path = f"{task_name}_{action_space}_test.png"
image = Image.fromarray(np.array(img, dtype=np.uint8))  # Convert the grid to a PIL image
image.save(output_path)