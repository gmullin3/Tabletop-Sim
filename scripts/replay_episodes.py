import sys
import os
import glob  # glob 추가
import numpy as np
import argparse
import h5py
import tabletop
# from tabletop.utils import save_images_to_video # (가정)

# save_images_to_video 함수가 없어서 임시로 주석 처리하거나
# 사용자의 환경에 맞게 import 경로를 유지하세요.
from tabletop.utils import save_images_to_video 

def replay(task_name, action_type, mode, episode_dir, save_dir, num_episode=None):
    # 1. glob을 사용하여 모든 hdf5 파일 경로를 리스트로 가져옴
    search_path = os.path.join(episode_dir, task_name, '*.hdf5')
    episode_files = glob.glob(search_path)
    
    # 2. 파일명 순서대로 정렬 (중요: glob은 순서를 보장하지 않음)
    # 윈도우/리눅스 파일 시스템에 따라 1, 10, 2 순서가 될 수 있으므로 
    # 필요하다면 숫자를 파싱해서 정렬하는 natsort 등을 써도 되지만, 여기선 기본 sort 사용
    episode_files.sort()

    total_files = len(episode_files)
    print(f"Found {total_files} episodes in {search_path}")

    if total_files == 0:
        print("No episode files found.")
        return

    # 3. 처리할 대상 파일 리스트 선정
    target_files = []
    
    if num_episode is not None:
        # 특정 인덱스(i번째) 파일 하나만 실행하고 싶은 경우
        if num_episode < 0 or num_episode >= total_files:
            raise ValueError(f"Index {num_episode} out of range. Only {total_files} files available.")
        
        print(f"Replaying specific episode index: {num_episode}")
        target_files = [episode_files[num_episode]]
    else:
        # 전체 실행
        print(f"Replaying ALL {total_files} episodes...")
        target_files = episode_files

    # 환경 설정 (한 번만 초기화하거나, 에피소드마다 리셋이 필요하면 루프 안으로 넣을 수도 있음)
    # 보통 task_name이 같다면 reset()으로 충분하므로 루프 밖에서 초기화
    env = tabletop.env(task_name, action_type)

    from tqdm import tqdm
    succ_list = []
    # 4. 파일 리스트 순회
    for file_path in tqdm(target_files):
        try:
            # 원본 파일명 추출 (예: episode_105.hdf5 -> episode_105)
            # 저장할 때 0, 1, 2 순서가 아니라 원본 파일 번호를 따라가기 위함
            file_name = os.path.basename(file_path)
            base_name = os.path.splitext(file_name)[0]
            
            print(f"Processing: {file_name}")
            
            succ = False
            with h5py.File(file_path, 'r') as episode:
                # 데이터 읽기
                init_state = episode['/observations/states/env_state'][0]
                if mode == 'action':
                    actions = episode[f'/actions/{action_type}']
                elif mode == 'state':
                    actions = episode[f'/observations/states/{action_type}']
                
                # 환경 리셋 및 초기화
                ts = env.reset()
                ts = env.task.state_init(env.physics, init_state)

                replay_images = []
                # 액션 수행
                for a in actions:
                    ts = env.step(a)
                    replay_images.append(ts.observation['images']['front'])

                if ts.reward == env.task.max_reward:
                    succ = True
                else:
                    succ = False
                succ_list.append(1 if succ else 0)
                    
                # 영상 저장
                save_path = os.path.join(save_dir, f'{task_name}_{action_type}')
                os.makedirs(save_path, exist_ok=True) # 저장 폴더가 없으면 생성
                
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
    
    # num_episode: 지정하면 해당 '순번(index)'의 파일만 실행, 안 쓰면 전체 실행
    parser.add_argument('-n', '--num_episode', action='store', type=int, default=None)
    parser.add_argument('-s', '--save_dir', action='store', type=str, default='.')
    parser.add_argument('-m', '--mode', action='store', type=str, default='action')
    
    args = parser.parse_args()

    replay(args.task_name, args.action_type, args.mode, args.episode_dir, args.save_dir, args.num_episode)