import h5py
import numpy as np
import os
import argparse
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def main():
    parser = argparse.ArgumentParser(description="Convert LeRobotDataset to ALOHA/ACT HDF5 format")
    parser.add_argument("--lerobot_dir", type=str, required=True, help="Path to your LeRobot dataset directory")
    parser.add_argument("--out_dir", type=str, required=True, help="Where to save the .hdf5 files")
    parser.add_argument("--camera_name", type=str, default="webcam", help="Name of the camera feature")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"Loading LeRobotDataset from {args.lerobot_dir} (this may take a moment to decode video)...")
    dataset = LeRobotDataset("local_teleop/imitation_dataset", root=args.lerobot_dir, video_backend="pyav")
    
    num_episodes = dataset.num_episodes
    print(f"Found {num_episodes} episodes to convert.")

    # We need to extract episode start/end indices.
    # In LeRobotDataset, `dataset.hf_dataset` holds the row items.
    # Actually, we can just group by the `episode_index` returned in each item.
    
    current_ep = -1
    qpos_list = []
    action_list = []
    img_list = []
    
    saved_episodes = 0
    
    # Iterate frame by frame
    for i in tqdm(range(len(dataset))):
        item = dataset[i]
        ep_idx = int(item['episode_index'].item())
        
        if ep_idx != current_ep:
            # We finished reading an episode and are starting a new one.
            if current_ep != -1:
                # Save the complete old episode
                save_episode(args.out_dir, saved_episodes, qpos_list, action_list, img_list, args.camera_name)
                saved_episodes += 1
                
            # Reset buffers for the new episode
            current_ep = ep_idx
            qpos_list = []
            action_list = []
            img_list = []

        # Extract features
        # state: shape [6]
        qpos_list.append(item['observation.state'].numpy())
        # action: shape [6]
        action_list.append(item['action'].numpy())
        
        # image: [3, H, W] float [0,1] -> Need [H, W, 3] uint8
        img = item[f'observation.images.{args.camera_name}'].numpy()
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
        img = np.transpose(img, (1, 2, 0)) # [C, H, W] -> [H, W, C]
        img_list.append(img)
        
    # Save the last episode
    if len(qpos_list) > 0:
        save_episode(args.out_dir, saved_episodes, qpos_list, action_list, img_list, args.camera_name)
        saved_episodes += 1

    print(f"Successfully converted {saved_episodes} episodes to {args.out_dir}")

def save_episode(out_dir, ep_idx, qpos_list, action_list, img_list, cam_name):
    out_path = os.path.join(out_dir, f"episode_{ep_idx}.hdf5")
    
    qpos_arr = np.array(qpos_list)
    action_arr = np.array(action_list)
    img_arr = np.array(img_list)
    # create dummy qvel as zeros of same shape as qpos
    qvel_arr = np.zeros_like(qpos_arr)
    
    with h5py.File(out_path, 'w') as root:
        root.attrs['sim'] = False # ALOHA code uses this
        root.attrs['compress'] = False # Not currently using compression script
        
        obs = root.create_group('observations')
        obs.create_dataset('qpos', data=qpos_arr)
        obs.create_dataset('qvel', data=qvel_arr)
        
        images = obs.create_group('images')
        images.create_dataset(cam_name, data=img_arr) # e.g. 'webcam'
        
        root.create_dataset('action', data=action_arr)

if __name__ == "__main__":
    main()
