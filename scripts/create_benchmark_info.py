import h5py
import glob
import os
import argparse
import numpy as np
from tqdm import tqdm

def list_hdf5_files(folder_path):
    """List all HDF5 files in a directory."""
    hdf5_files = glob.glob(os.path.join(folder_path, '*.hdf5'))
    hdf5_files = sorted(hdf5_files)  # Sort alphabetically
    return hdf5_files

def create_benchmark_info(task_name, folder_path):
    """
    Extract initial states from HDF5 episode files to create benchmark info.
    
    Args:
        task_name (str): Name of the task folder.
        folder_path (str): Parent directory containing task folders.
        
    Returns:
        np.array: Array of initial states.
    """
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
    parser.add_argument('-t', '--task_name', type=str, help="Name of the task.")
    args = parser.parse_args()
    
    # Ensure the output directory exists
    output_dir = 'tabletop/benchmark_info'
    os.makedirs(output_dir, exist_ok=True)
    
    benchmark_info = create_benchmark_info(args.task_name, args.dataset_dir)
    output_path = os.path.join(output_dir, f'{args.task_name}_benchmark_info.npy')
    np.save(output_path, benchmark_info)
    print(f"Benchmark info saved to {output_path}")

# This script creates a benchmark info file that contains the initial states for each episode in the specified folder.