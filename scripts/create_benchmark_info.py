import h5py
import glob
import os
import argparse
import numpy as np
from tqdm import tqdm

def list_hdf5_files(folder_path):
    hdf5_files = glob.glob(os.path.join(folder_path, '*.hdf5'))
    hdf5_files = sorted(hdf5_files)  # 알파벳 순 정렬 (옵션)
    return hdf5_files

def create_benchmark_info(task_name, folder_path):
    hdf5_files = list_hdf5_files(os.path.join(folder_path, task_name))
    benchmark_info = []

    for hdf5_file in tqdm(hdf5_files):
        with h5py.File(hdf5_file, 'r') as f:
            # Assuming the structure of the HDF5 file is known
            init_state = f['/observations/states/env_state'][0]
            benchmark_info.append(init_state)
    return np.stack(benchmark_info)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create benchmark info from HDF5 files.")
    parser.add_argument('-d', '--dataset_dir', type=str, help="Path to the folder containing HDF5 files.")
    parser.add_argument('-t', '--task_name', type=str)
    args = parser.parse_args()
    benchmark_info = create_benchmark_info(args.task_name, args.dataset_dir)
    np.save(f'tabletop/benchmark_info/{args.task_name}_benchmark_info.npy', benchmark_info)
    print(f"Benchmark info saved to tabletop/benchmark_info/{args.task_name}_benchmark_info.npy")

# This script creates a benchmark info file that lists the number of episodes for each task in the specified folder.