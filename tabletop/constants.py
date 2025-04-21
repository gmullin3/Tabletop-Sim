import pathlib

### Task parameters
DATA_DIR = ''

### Simulation envs fixed constants
DT = 0.04

ALOHA_XML_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/assets/aloha' # note: absolute path

ALOHA_GRIPPER_CLOSE = 0.002
ALOHA_GRIPPER_OPEN = 0.037

############################ Helper functions ############################
ALOHA_GRIPPER_UNNORMALIZE_FN = lambda x: x * (ALOHA_GRIPPER_OPEN - ALOHA_GRIPPER_CLOSE) + ALOHA_GRIPPER_CLOSE
ALOHA_GRIPPER_NORMALIZE_FN = lambda x: (x - ALOHA_GRIPPER_CLOSE) / (ALOHA_GRIPPER_OPEN - ALOHA_GRIPPER_CLOSE)
ALOHA_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (ALOHA_GRIPPER_OPEN - ALOHA_GRIPPER_CLOSE)