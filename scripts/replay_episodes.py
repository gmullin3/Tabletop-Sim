import os
import glob
import argparse
import h5py
import tabletop
from tabletop.utils import save_images_to_video 
from tqdm import tqdm

def replay(task_name, action_type, mode, episode_dir, save_dir, num_episode=None):
    search_path = os.path.join(episode_dir, task_name, '*.hdf5')
    episode_files = glob.glob(search_path)
    episode_files.sort()

    total_files = len(episode_files)
    print(f"Found {total_files} episodes in {search_path}")

    if total_files == 0:
        print("No episode files found.")
        return
    target_files = []
    
    if num_episode is not None:
        if num_episode < 0 or num_episode >= total_files:
            raise ValueError(f"Index {num_episode} out of range. Only {total_files} files available.")
        
        print(f"Replaying specific episode index: {num_episode}")
        target_files = [episode_files[num_episode]]
    else:
        print(f"Replaying ALL {total_files} episodes...")
        target_files = episode_files

    env = tabletop.env(task_name, action_type)

    succ_list = []

    for file_path in tqdm(target_files):
        try:
            file_name = os.path.basename(file_path)
            base_name = os.path.splitext(file_name)[0]
            
            print(f"Processing: {file_name}")
            
            succ = False
            with h5py.File(file_path, 'r') as episode:
                init_state = episode['/observations/states/env_state'][0]
                if mode == 'action':
                    actions = episode[f'/actions/{action_type}']
                elif mode == 'state':
                    actions = episode[f'/observations/states/{action_type}']
                
                env.reset()
                ts = env.task.state_init(env.physics, init_state)

                replay_images = []

                for a in actions:
                    ts = env.step(a)
                    replay_images.append(ts.observation['images']['front'])

                if ts.reward == env.task.max_reward:
                    succ = True
                else:
                    succ = False
                succ_list.append(1 if succ else 0)
                    
                save_path = os.path.join(save_dir, f'{task_name}_{action_type}')
                os.makedirs(save_path, exist_ok=True)
                
                output_filename = os.path.join(save_path, f'{base_name}_{succ}.mp4')
                save_images_to_video(replay_images, output_filename)
                print(f"Saved to {output_filename}")
            print(f"Episode Success: {succ}, Total Success Count: {sum(succ_list)}")

        except Exception as e:
            print(f"Failed to replay {file_path}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task_name', action='store', type=str, default='aloha_dish_drainer', required=False)
    parser.add_argument('-a', '--action_type', action='store', type=str, default='ee_6d_pos', required=False)
    parser.add_argument('-d', '--episode_dir', action='store', type=str, default='datasets')
    
    parser.add_argument('-n', '--num_episode', action='store', type=int, default=None)
    parser.add_argument('-s', '--save_dir', action='store', type=str, default='.')
    parser.add_argument('-m', '--mode', action='store', type=str, default='action')
    
    args = parser.parse_args()

    replay(args.task_name, args.action_type, args.mode, args.episode_dir, args.save_dir, args.num_episode)