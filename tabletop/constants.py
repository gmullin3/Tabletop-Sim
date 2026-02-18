
import pathlib

### Task parameters
# DATA_DIR removed as it was unused and empty

### Simulation envs fixed constants
DT = 0.04

# Absolute paths to assets and benchmark info
ALOHA_XML_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/assets/aloha' 
BENCHMARK_INFO_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/benchmark_info'

# Gripper normalization constants
ALOHA_GRIPPER_CLOSE = 0.002
PROPRIO_GRIPPER_CLOSE = 0.009
ALOHA_GRIPPER_OPEN = 0.037

############################ Helper functions ############################

def ALOHA_GRIPPER_UNNORMALIZE_FN(x):
    """
    Assume gripper is normalized to [-1, 1] range, unnorm to [0, 1] range.
    Actually based on physical limits ALOHA_GRIPPER_CLOSE/OPEN.
    """
    return (x + 1) / 2 * (ALOHA_GRIPPER_OPEN - ALOHA_GRIPPER_CLOSE) + ALOHA_GRIPPER_CLOSE

def ALOHA_GRIPPER_NORMALIZE_FN(x):
    """
    Assume gripper input is in physical range, normalize to [-1, 1] range.
    """
    return 2 * ((x - PROPRIO_GRIPPER_CLOSE) / (ALOHA_GRIPPER_OPEN - PROPRIO_GRIPPER_CLOSE)) - 1

def ALOHA_GRIPPER_VELOCITY_NORMALIZE_FN(x):
    """
    Normalize gripper velocity.
    """
    return x / (ALOHA_GRIPPER_OPEN - PROPRIO_GRIPPER_CLOSE)