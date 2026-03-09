#!/bin/bash

# TASK_NAME="aloha_dish_drainer"
# python convert_hdf5_to_lerobot.py \
#     --input-dir /data5/twinvla/tabletop-sim/hdf5/$TASK_NAME \
#     --lerobot-dir  /data5/twinvla/tabletop-sim/lerobot/$TASK_NAME \
#     --hub-dir jellyho/$TASK_NAME \
#     --fps 20

# TASK_NAME="aloha_handover_box"
# python convert_hdf5_to_lerobot.py \
#     --input-dir /data5/twinvla/tabletop-sim/hdf5/$TASK_NAME \
#     --lerobot-dir  /data5/twinvla/tabletop-sim/lerobot/$TASK_NAME \
#     --hub-dir jellyho/$TASK_NAME \
#     --fps 20

# TASK_NAME="aloha_lift_box"
# python convert_hdf5_to_lerobot.py \
#     --input-dir /data5/twinvla/tabletop-sim/hdf5/$TASK_NAME \
#     --lerobot-dir  /data5/twinvla/tabletop-sim/lerobot/$TASK_NAME \
#     --hub-dir jellyho/$TASK_NAME \
#     --fps 20

# TASK_NAME="aloha_shoes_table"
# python convert_hdf5_to_lerobot.py \
#     --input-dir /data5/twinvla/tabletop-sim/hdf5/$TASK_NAME \
#     --lerobot-dir  /data5/twinvla/tabletop-sim/lerobot/$TASK_NAME \
#     --hub-dir jellyho/$TASK_NAME \
#     --fps 20

# TASK_NAME="aloha_box_into_pot_easy"
# python convert_hdf5_to_lerobot.py \
#     --input-dir /data5/twinvla/tabletop-sim/hdf5/$TASK_NAME \
#     --lerobot-dir  /data5/twinvla/tabletop-sim/lerobot/$TASK_NAME \
#     --hub-dir jellyho/$TASK_NAME \
#     --fps 20

TASK_NAME="anubis_brush_to_pan"
python convert_anubis_hdf5_to_lerobot.py \
    --input-dir /data5/twinvla/anubis/hdf5_reform_v2/$TASK_NAME \
    --lerobot-dir  /data5/twinvla/anubis/lerobot/$TASK_NAME \
    --hub-dir jellyho/$TASK_NAME \
    --fps 20

TASK_NAME="anubis_carrot_to_bag"
python convert_anubis_hdf5_to_lerobot.py \
    --input-dir /data5/twinvla/anubis/hdf5_reform_v2/$TASK_NAME \
    --lerobot-dir  /data5/twinvla/anubis/lerobot/$TASK_NAME \
    --hub-dir jellyho/$TASK_NAME \
    --fps 20

TASK_NAME="anubis_fold_towel"
python convert_anubis_hdf5_to_lerobot.py \
    --input-dir /data5/twinvla/anubis/hdf5_reform_v2/$TASK_NAME \
    --lerobot-dir  /data5/twinvla/anubis/lerobot/$TASK_NAME \
    --hub-dir jellyho/$TASK_NAME \
    --fps 20

TASK_NAME="anubis_pullout_wrench"
python convert_anubis_hdf5_to_lerobot.py \
    --input-dir /data5/twinvla/anubis/hdf5_reform_v2/$TASK_NAME \
    --lerobot-dir  /data5/twinvla/anubis/lerobot/$TASK_NAME \
    --hub-dir jellyho/$TASK_NAME \
    --fps 20

TASK_NAME="anubis_put_into_pot"
python convert_anubis_hdf5_to_lerobot.py \
    --input-dir /data5/twinvla/anubis/hdf5_reform_v2/$TASK_NAME \
    --lerobot-dir  /data5/twinvla/anubis/lerobot/$TASK_NAME \
    --hub-dir jellyho/$TASK_NAME \
    --fps 20

TASK_NAME="anubis_towel_kirby"
python convert_anubis_hdf5_to_lerobot.py \
    --input-dir /data5/twinvla/anubis/hdf5_reform_v2/$TASK_NAME \
    --lerobot-dir  /data5/twinvla/anubis/lerobot/$TASK_NAME \
    --hub-dir jellyho/$TASK_NAME \
    --fps 20