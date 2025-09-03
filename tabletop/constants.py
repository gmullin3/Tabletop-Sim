import pathlib

### Task parameters
DATA_DIR = ''

### Simulation envs fixed constants
DT = 0.04

ALOHA_XML_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/assets/aloha' # note: absolute path
BENCHMARK_INFO_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/benchmark_info' # note: absolute path

ALOHA_GRIPPER_CLOSE = 0.009
ALOHA_GRIPPER_OPEN = 0.037

############################ Helper functions ############################
## Assume gripper is normalized to [-1, 1] range, unnorm to [0, 1] range
ALOHA_GRIPPER_UNNORMALIZE_FN = lambda x: (x + 1) / 2 * (ALOHA_GRIPPER_OPEN - ALOHA_GRIPPER_CLOSE) + ALOHA_GRIPPER_CLOSE

## Assume gripper input is normalized to [-1, 1] range
ALOHA_GRIPPER_NORMALIZE_FN = lambda x: 2 * ((x - ALOHA_GRIPPER_CLOSE) / (ALOHA_GRIPPER_OPEN - ALOHA_GRIPPER_CLOSE)) - 1 # - 0.115) / (1.185 - 0.115) - 1
ALOHA_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (ALOHA_GRIPPER_OPEN - ALOHA_GRIPPER_CLOSE)