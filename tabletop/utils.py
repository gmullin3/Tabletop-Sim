import numpy as np
import torch
import os

import IPython
e = IPython.embed



### env utils

def sample_box_pose():
    x_range = [-0.2, 0.2]
    y_range = [0.3, 0.7]
    z_range = [0.1, 0.1]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

import cv2
import numpy as np
import os

def save_images_to_video(image_list, output_video_path, fps=20):
    """
    Saves a list of NumPy arrays (representing images) into a video file.

    Args:
        image_list (list of np.ndarray): A list where each element is a NumPy
                                         array representing an image frame
                                         (e.g., shape (height, width, 3) for color).
        output_video_path (str): Path to save the output video file (e.g., 'output.mp4').
        fps (int): Frames per second for the output video.
    """
    if not image_list:
        print("Error: Empty list of images provided.")
        return

    # Get the dimensions of the first frame
    first_frame = image_list[0]
    height, width, channels = first_frame.shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print(f"Saving video to: {output_video_path} at {fps} FPS...")

    for frame in image_list:
        if frame is None or not isinstance(frame, np.ndarray) or frame.shape != (height, width, channels):
            print("Warning: Skipping invalid frame in the image list.")
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.write(frame)

    # Release the VideoWriter object
    out.release()
    print("Video saved successfully!")
