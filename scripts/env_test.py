from tabletop import env
import argparse
from PIL import Image
import numpy as np
from tabletop.aloha_env import ALOHA_TASK_CONFIGS

def run_test(task_name, action_space, reset_env=False):
    """
    Run environment test for a specific task.
    """
    try:
        print(f"Testing environment creation for task: {task_name}")
        test_env = env(task_name, action_space, True)
        if reset_env:
            test_env.reset()
        img = test_env.physics.render(480, 640, camera_id=0)
        output_path = f"{task_name}_{action_space}_test.png"
        image = Image.fromarray(np.array(img, dtype=np.uint8))  # Convert the grid to a PIL image
        image.save(output_path)
        print(f"Saved test image to {output_path}")
    except Exception as e:
        print(f"Failed to run test for task {task_name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test environment rendering.")
    parser.add_argument("-t", "--task_name", type=str, help="Name of the task. If not provided, tests all tasks.")
    parser.add_argument("-a", "--action_space", type=str, default="ee_quat_pos", help="Action space type (default: ee_quat_pos).")
    parser.add_argument("-r", "--reset", action="store_true", help="Whether to reset the environment.")
    args = parser.parse_args()

    task_name = args.task_name
    action_space = args.action_space
    reset = args.reset

    if task_name is None:
        print("No task name provided. Testing all available tasks...")
        for k in ALOHA_TASK_CONFIGS.keys():
            run_test(k, action_space, reset)
    else:
        run_test(task_name, action_space, reset)