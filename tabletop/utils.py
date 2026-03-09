import numpy as np
import cv2
from scipy.spatial.transform import Rotation

def quat_to_rpy(w, x, y, z, mode='abs'):
    """Convert quaternion [w, x, y, z] to Euler angles [Roll, Pitch, Yaw] in radians.
    Roll = rotation about X-axis, Pitch = rotation about Y-axis, Yaw = rotation about Z-axis.
    """
    try:
        r = Rotation.from_quat([w, x, y, z], scalar_first=True)
    except:
        return np.array([0.0, 0.0, 0.0])
    return np.array(r.as_euler('XYZ', degrees=False))

def rpy_to_quat(roll, pitch, yaw, degrees=False):
    """Convert Euler angles [Roll, Pitch, Yaw] to quaternion [w, x, y, z].
    Roll = rotation about X-axis, Pitch = rotation about Y-axis, Yaw = rotation about Z-axis.
    """
    try:
        r = Rotation.from_euler('XYZ', [roll, pitch, yaw], degrees=degrees)
    except:
        return np.array([0.0, 0.0, 0.0, 0.0])
    return np.array(r.as_quat(scalar_first=True))

def quat_to_6d(w, x, y, z):
    """
    Converts a quaternion to a 6D representation.

    Args:
        w, x, y, z (float): Components of the quaternion (scalar first).

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
        np.ndarray: A quaternion [w, x, y, z] (scalar first).
    """
    col1, col2 = sixd[:3], sixd[3:]
    mat = np.column_stack((col1, col2, np.cross(col1, col2)))
    r = Rotation.from_matrix(mat)
    return r.as_quat(scalar_first=True)

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
