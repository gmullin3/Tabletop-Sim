import argparse
from pathlib import Path
import traceback

import h5py
import numpy as np
import os 

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES
from features import features
import shutil # for deleting existing lerobot dir 

def _open_dataset(lerobot_dir: Path, fps: int,
                  image_writer_processes: int = 8, image_writer_threads: int = 16,
                  repo_id: str = None, features=DEFAULT_FEATURES) -> LeRobotDataset:
    
    assert repo_id is not None, "repo_id must be specified"
    lerobot_dir = Path(lerobot_dir)

    if lerobot_dir.exists(): 
        should_delete = input(f"{lerobot_dir} already exists. \nDelete and recreate? (y/n): ")
        if should_delete.lower() == 'y':
            shutil.rmtree(lerobot_dir)
        else: 
            raise FileExistsError(f"{lerobot_dir} already exists. Aborting.")
             
    print(f"[create] {lerobot_dir}")
    return LeRobotDataset.create(
        repo_id=repo_id,
        root=str(lerobot_dir),
        features=features,
        fps=fps,
        use_videos=True,
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
    )


def convert_one(hdf5_path: Path, ds: LeRobotDataset) -> None:
    hdf5_path = Path(hdf5_path)
    print(f"Converting: {hdf5_path}")

    # Read once (keeps the dataset writer lifetime clean and short)
    with h5py.File(hdf5_path, "r") as f:
        action_joint_pos = f["actions"]["joint_pos"][()]
        action_eef_6d_pos = f["actions"]["ee_6d_pos"][()]
        action_eef_quat_pos = f["actions"]["ee_quat_pos"][()]

        state_joint_pos = f["observations"]["states"]["joint_pos"][()]
        state_ee_6d_pos = f["observations"]["states"]["ee_6d_pos"][()]
        state_ee_quat_pos = f["observations"]["states"]["ee_quat_pos"][()]

        agentview   = f["observations"]["images"]["back"][()]
        wrist_left  = f["observations"]["images"]["wrist_left"][()]
        wrist_right = f["observations"]["images"]["wrist_right"][()]

        task = f["observations"]["states"]["language_instruction"][()]
    
    try:
        n = len(action_joint_pos)

        for i in range(n):
            frame = {
                "observation.state.joint_pos": state_joint_pos[i],
                "observation.state.ee_6d_pos": state_ee_6d_pos[i],
                "observation.state.ee_quat_pos": state_ee_quat_pos[i],
                "observation.images.agentview": agentview[i],
                "observation.images.wrist_left": wrist_left[i],
                "observation.images.wrist_right": wrist_right[i],
                "action.joint_pos": action_joint_pos[i],
                "action.ee_6d_pos": action_eef_6d_pos[i],
                "action.ee_quat_pos": action_eef_quat_pos[i],
                "task": task[i],

                # For value learning
                "done": np.array([False], dtype=bool) if i < n - 1 else np.array([True], dtype=bool),
                "reward": np.array([0], dtype=float) if i < n - 1 else np.array([1], dtype=float)
            }

            ds.add_frame(frame)

        ds.save_episode()
        print(f"✅ Saved episode from {hdf5_path.name}")
    finally:
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory containing .hdf5 files")
    parser.add_argument("--lerobot-dir", required=True, type=Path, help="Output LeRobot dataset directory")
    parser.add_argument("--hub-dir", required=True, type=str, help="Repo ID for LeRobot dataset")
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    hdf5_files = sorted([p for p in Path(args.input_dir).glob("*.hdf5") if p.is_file()],
                         key=lambda x: int(x.name.split("_")[-1].split(".")[0]))
    if not hdf5_files:
        print(f"No .hdf5 files found in {args.input_dir}")
        return

    print(f"Found {len(hdf5_files)} file(s), fps={args.fps}")

    features.update(DEFAULT_FEATURES)
    ft = features
    ds = _open_dataset(args.lerobot_dir, fps=args.fps, repo_id=args.hub_dir, features=ft)

    for _, p in enumerate(hdf5_files):
        try:
            convert_one(p, ds)
        except Exception as e:
            print(f"❌ Failed: {p} -> {e}")
            traceback.print_exc()
    ds.finalize()
    ds.push_to_hub()



if __name__ == "__main__":
    main()
