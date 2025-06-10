import torch
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

def degree_to_radian(x, y, z):
    return np.array([x / 180 * np.pi, y / 180 * np.pi, z / 180 * np.pi])

def quat_to_rpy(w, x, y, z, mode='abs'):
    try:
        r = Rotation.from_quat([w, x, y, z], scalar_first=True)
    except:
        return np.array([0.0, 0.0, 0.0])
    return np.array(r.as_euler('zyx', degrees=False))

def rpy_to_quat(roll, pitch, yaw, degrees=False):
    try:
        r = Rotation.from_euler('zyx', [roll, pitch, yaw], degrees=degrees)
    except:
        return np.array([0.0, 0.0, 0.0, 0.0])
    return np.array(r.as_quat(scalar_first=True))

def mat_to_rpy(mat):
    return Rotation.from_matrix(mat).as_euler('zyx', degrees=False)

def mat_to_quat(mat):
    return Rotation.from_matrix(mat).as_quat()

def quat_to_6d(w, x, y, z):
    """
    Converts a quaternion to a 6D representation.

    Args:
        w, x, y, z (float): Components of the quaternion.

    Returns:
        np.ndarray: A 6D representation of the quaternion.
    """
    r = Rotation.from_quat([w, x, y, z], scalar_first=True)
    mat = r.as_matrix()
    col1, col2 = mat[:, 0], mat[:, 1]
    return np.concatenate((col1, col2), axis=0)

def sixd_to_quat(sixd):
    """
    Converts a 6D representation back to a quaternion.

    Args:
        sixd (np.ndarray): A 6D representation of a rotation.

    Returns:
        np.ndarray: A quaternion [w, x, y, z].
    """
    col1, col2 = sixd[:3], sixd[3:]
    mat = np.column_stack((col1, col2, np.cross(col1, col2)))
    r = Rotation.from_matrix(mat)
    return r.as_quat(scalar_first=True)

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
