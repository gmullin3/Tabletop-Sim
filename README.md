# Tabletop Simulation
This repo is Aloha based mujoco simulation framework to evaluate bimanual policy in simulation.

## Installation
    pip install -r requirements.txt
    pip install -e .

## Download datasets

TBD

## Task list
### Biamanual Tasks
1. ``aloha_dish_drainer``
2. ``aloha_handover_box``
3. ``aloha_shoes_table``
4. ``aloha_lift_box``

### Single Arm Tasks

TBD

## Example Usage

TBD

## Make your own Tasks

### Task definition

You can define your own aloha task in ``tabletop/aloha_env.py``.

```
class TaskName(AlohaTask):
    def __init__(self, random=None):
        super().__init__(random=random, single_arm=False) ## always first
        self.add_object('drainer', 'Rubbermaid_Large_Drainer', pos=[-0.1, 0.1, 0.01], rpy=[0, 0, -60], scale=[0.8, 0.8, 0.8])
        self.add_object('plate', 'Threshold_Bistro_Ceramic_Dinner_Plate_Ruby_Ring', pos=[0.1, 0, 0.01], scale=[0.8, 0.8, 0.8], mass=0.2)

    def initialize_episode(self, physics):
        random_vector = np.random.randn(2)
        plate_pos = np.array([0.15, -0.17, 0.01])
        plate_pos[:2] += random_vector * 0.015
        plate_rpy = np.array([0, 0, 0],)
        self.set_object_pose(physics, 'plate', pos=plate_pos, rpy=plate_rpy)
        super().initialize_episode(physics) ## always last

    def get_reward(self, physics):
        ## [condition, counter]
        reward_condition_list = [
            [self.get_touch_condition(physics, 'drainer', 'plate'), 20],
        ]
        return super().get_reward(physics, reward_condition_list) ### always first
    
    def get_instruction(self, reward):
        return 'Pick up the dish and put on to the drainer'
```

1. When ``__init__()``, you can choose whether to use single arm or bimanual. Set ``single_arm=False`` to use bimanual, or ``single_arm='left'``, ``single_arm='right'`` to use single arm.
2. After ``super().__init__(random=random, single_arm=False)``, you can initialize the object you want to include. Example code as below.
    ```
    self.add_object(NickName, ObjectName, pos=[x, y, z], rpy=[r, p, y], scale=[x, y, z], mass=kg)
    ```
    ``NickName`` can be arbitrary, but for ``ObjectName`` should be same as GSO dataset, since we are using it. You can see the catalog of the possible object list from here. [Google Scanned Objects](https://app.gazebosim.org/GoogleResearch/fuel/collections/Scanned%20Objects%20by%20Google%20Research)
3. ``initialize_episode()`` defines the initalization of the task. You can randomize the initial position of the objects here. Use ``self.set_object_pose``. Please use numpy.random since we are fixing the numpy random seed for reproducibility.
4. ``get_reward()`` defines the reward signal. You should fill the ``reward_condition_list``, this will applied sequentially. Total length of the ``reward_condition_list`` would be the ``max_reward`` of this task. we implemented some helper function for reward (``self.get_touch_condition(physics, NickName1, NickName2)`` which checks whether two objects are collided or not. You can also use reserved keyword (``table, right_arm, left_arm``) as a NickName and ``self.get_object_pose()``.. etc).
5. ``get_instruction(self, reward)`` defines the instruction. Since you input reward, you can change the language instruciton by the reward, enabling multi-stage long-horizon task.
6. Register your task. At the bottom of the ``tabletop/aloha_env.py``, you can see the dictionary ``ALOHA_TASK_CONFIGS``. You can register your task class with the task name, episode_len. Note that this episode_len unit is second, since we are using 20Hz, the total number of timesteps would be ``episode_len * 20``.

### Debugging your task
You can run ``scripts/env_test.py`` to visualize your task. The visualized image will be saved. There is the grids spaced at 0.05m. Before run this code, make sure you registered at ``ALOHA_TASK_CONFIGS`` in ``tabletop/aloha_env.py``.

```
python scripts/env_test.py -t aloha_dish_drainer
```

If you add --reset flag, it will visualize the scene after ``initialize_episode()``

## Demo collection

### Collecting demos using GELLO

We used GELLO for collecting demos inside simulation. You can refer ``scripts/gello_ros.py`` and ``scripts/record_sim_episodes_gello.py``. Change ``scripts/gello_ros.py`` to match your gello ROS setup. Note that we used ROS2 for that.

```
python scripts/record_sim_episodes_gello.py -t aloha_dish_drainer
```

By doing this, you can collect demos and save into hdf5 file format.

### Replay collected demos

You can replay the collected hdf5 file using ``scripts/replay_episodes.py``

```
python scripts/replay_episdoes.py -t aloha_dish_drainer -n 30
```

## Citation

If you feel this useful, please cite this work with:
```bibtex
@misc{im2025tabletop,
    author = {Hokyun, Im},
    title = {Tabletop Simulation}
    howpublished = "\url{https://github.com/jellyho/Tabletop-Sim}",
    year = {2025}
}
```