import json
import torch
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from poses.preprocessing import normalize_keypoints, threshold_confidence
from utils import parse_vtt
from config import *


class DVCDataset(Dataset):
    def __init__(self, split, stride_ratio=0.5, max_tries=10, min_sentences=1, seed=42):
        '''
        PyTorch Dataset for DVC with on-the-fly sliding window sampling.
        Args:
            split: 'train', 'val', or 'test'
            stride_ratio: For val/test sequential sampling (e.g., 0.5 for 50% overlap)
            max_tries: Max resamples for train windows with < min_sentences
            min_sentences: Min full sentences per train window
            seed: For reproducibility in random sampling
        '''
        assert split in ['train', 'val', 'test'], f"Split must be 'train', 'val', or 'test', but got {split}"
        self.split = split
        self.window_size_frames = int(WINDOW_DURATION_SECONDS * FPS)
        self.stride = int(self.window_size_frames * stride_ratio)
        self.max_tries = max_tries
        self.min_sentences = min_sentences
        np.random.seed(seed)
        
        self.video_ids = self.load_subset(split)
        self.video_metadata = {} # Precomputed metadata per video for efficiency
        self.eval_windows = [] # Store windows for val/test splits
        self._build_video_metadata()
        print(f'Dataset initialized for {split}: {len(self.video_ids)} videos')
        print(f'Window size: {WINDOW_DURATION_SECONDS}s ({self.window_size_frames} frames @ {FPS} fps)')


    @staticmethod
    def load_subset(split): # Load the subset2episode.json to get train/val/test lists of video IDs
        try:
            with open(SUBSET_JSON, 'r') as f:
                splits = json.load(f)
        except FileNotFoundError:
            print('Error: Metadata file not found at', SUBSET_JSON)
            print('Please ensure the SUBSET_JSON path in config.py is correct.')

        video_ids = splits.get(split, [])
        if not video_ids: 
            print(f'No videos found in {split} split.')
            return []
        print(f'Found {len(video_ids)} videos in the {split} split.')
        return video_ids


    def _build_video_metadata(self): # Precompute for sampling efficiency
        for video_id in tqdm(self.video_ids, desc=f"Building video metadata for {self.split} split"):
            pose_dir = POSE_ROOT / video_id
            if not pose_dir.exists():
                raise FileNotFoundError(f'Pose directory not found: {pose_dir}')

            segment_paths = sorted(list(pose_dir.glob("*.npy")), key=lambda p: p.stem)
            if not segment_paths: 
                raise ValueError(f'No .npy files in {pose_dir}')

            frame_counts = [np.load(f, mmap_mode='r').shape[0] for f in segment_paths]
            total_frames = sum(frame_counts)
            self.video_metadata[video_id] = {
                'segment_paths': segment_paths,
                'frame_counts': np.array(frame_counts),
                'total_frames': total_frames,
                'cumulative_frames': np.cumsum([0] + frame_counts),
                'subtitles': parse_vtt(VTT_DIR / f'{video_id}.vtt')
            }
            
            if self.split != 'train': # For val/test: count fixed, non-overlapping windows
                for window_start_frame in range(0, total_frames, self.stride):
                    window_end_frame = window_start_frame + self.window_size_frames
                    if window_end_frame <= total_frames: # Ignore the last, smaller window if it's too short
                        self.eval_windows.append({
                            'video_id': video_id,
                            'window_start_frame': window_start_frame,
                            'window_end_frame': window_end_frame
                        })

    def __len__(self):
        if self.split == 'train': return len(self.video_metadata) # For train: One per video (sampling random per getitem call)
        return len(self.eval_windows) # For val/test: Number of sequential windows across all videos
        # eval_window_count = 0
        # for video_id in self.video_ids:
        #     total_frames = self.video_metadata[video_id]['total_frames']
        #     eval_window_count += max(1, (total_frames - self.window_size_frames) // self.stride + 1)
        # return eval_window_count
            
    
    def __getitem__(self, idx):
        # --- Random Sampling for Training ---
        if self.split == 'train': # idx is video index; sample random window
            video_id = self.video_ids[idx]
            max_start_frame = self.video_metadata[video_id]['total_frames'] - self.window_size_frames

            for try_num in range(self.max_tries):
                if max_start_frame <= 0: # Video is shorter than window, so we take the whole thing and will pad later
                    window_start_frame = 0
                    window_end_frame = self.video_metadata[video_id]['total_frames']
                else: # Randomly select a start frame for the window
                    window_start_frame = np.random.randint(0, max_start_frame)
                    window_end_frame = window_start_frame + self.window_size_frames
                
                window = self._get_window_data(video_id, window_start_frame, window_end_frame)
                if len(window[-1]) >= self.min_sentences: # Check events
                    print(f'Sampled valid window for {video_id} (try {try_num+1})')
                    return window
                
            print(f'Warning: Could not find window with >= {self.min_sentences} sentences for {video_id} after {self.max_tries} tries')
            return self._get_window_data(video_id, window_start_frame, window_end_frame)  # Return last anyway

        # --- Fixed Window for Evaluation ---
        else: # idx is global window index; find corresponding video_id and local window
            # cum_windows = 0
            # for video_id in self.video_ids:
            #     total_frames = self.video_metadata[video_id]['total_frames']
            #     num_windows = max(1, (total_frames - self.window_size_frames) // self.stride + 1)

            #     if idx < cum_windows + num_windows:
            #         local_idx = idx - cum_windows
            #         window_start_frame = local_idx * self.stride
            #         window_end_frame = window_start_frame + self.window_size_frames
            #         window = self._get_window_data(video_id, window_start_frame, window_end_frame)
            #         # print(f'Fixed window {local_idx}/{num_windows} for {video_id}')
            #         return window
            #     cum_windows += num_windows
            # raise IndexError('Invalid idx for val/test')
            eval_window = self.eval_windows[idx]
            return self._get_window_data(
                eval_window['video_id'],
                eval_window['window_start_frame'],
                eval_window['window_end_frame']
            )
            

    def _get_window_data(self, video_id, window_start_frame, window_end_frame):
        if window_start_frame >= window_end_frame:
            raise ValueError('Invalid window boundaries')

        # full_poses = self.load_poses_for_video(video_id)
        # window_poses = full_poses[window_start_frame:window_end_frame, :, :]
        window_poses = self.load_poses_for_window(video_id, window_start_frame, window_end_frame)
        window_poses = normalize_keypoints(window_poses)
        window_poses = threshold_confidence(window_poses)
        poses_tensor = torch.from_numpy(window_poses).float()  # (T, K, 3)
        
        # Filter subtitles in window
        events = []
        for sub in self.video_metadata[video_id]['subtitles']:
            sub_start_frame = int(sub['start'] * FPS)
            sub_end_frame = int(sub['end'] * FPS)

            # Subtitle must be FULLY contained within the window and have valid duration
            if sub_start_frame >= window_start_frame and sub_end_frame <= window_end_frame and \
                MIN_SUB_DURATION <= sub['duration'] <= MAX_SUB_DURATION: 
                # Normalize to [0, 1] relative to window
                rel_start = (sub_start_frame - window_start_frame) / self.window_size_frames
                rel_end = (sub_end_frame - window_start_frame) / self.window_size_frames
                events.append({'rel_start': rel_start, 'rel_end': rel_end, 'text': sub['text']})

        return video_id, window_start_frame, window_end_frame, poses_tensor, events

        
    def load_poses_for_video(self, video_id: str) -> np.ndarray:
        '''
        Load all .npy segments for a video, concatenate into 1 array.
        Uses memmap for efficiency on large videos.
        Returns np.array (total_frames, 133, 3)
        '''
        # segment_shapes = []
        pose_segments = []
        
        for seg_path in self.video_metadata[video_id]['segment_paths']:
            seg = np.load(seg_path, mmap_mode='r')
            # segment_shapes.append(seg.shape[0])
            # seg = np.load(seg_path)
            pose_segments.append(seg)
        full_poses = np.concatenate(pose_segments, axis=0)
        
        # Concatenate using memmap views
        # offset = 0
        # full_poses = np.empty((self.video_metadata[video_id][total_frames], 133, 3), dtype=np.float32)
        # for i, seg_path in enumerate(self.video_metadata[video_id]['segment_paths']):
        #     seg = np.load(seg_path, mmap_mode='r')
        #     full_poses[offset:offset + segment_shapes[i]] = seg
        #     offset += segment_shapes[i]
        
        print(f'Loaded poses for {video_id}: {full_poses.shape} from {len(pose_segments)} segments')
        return full_poses


    def load_poses_for_window(self, video_id: str, window_start_frame: int, window_end_frame: int) -> np.ndarray:
        '''
        Load all .npy segments for a given window, concatenate into 1 array.
        Returns np.array (total_frames, 133, 3)
        '''
        pose_segments = []
        cumulative_frames = self.video_metadata[video_id]['cumulative_frames']
        
        # Find which npy files this window intersects with
        start_file_idx = np.searchsorted(cumulative_frames, window_start_frame, side='right') - 1
        end_file_idx = np.searchsorted(cumulative_frames, window_end_frame - 1, side='right') - 1

        for i in range(start_file_idx, end_file_idx + 1):
            local_start = max(0, window_start_frame - cumulative_frames[i])
            local_end = min(self.video_metadata[video_id]['frame_counts'][i], window_end_frame - cumulative_frames[i])
            seg = np.load(self.video_metadata[video_id]['segment_paths'][i], mmap_mode='r')
            pose_segments.append(seg[local_start:local_end])
        return np.concatenate(pose_segments, axis=0)
        

def collate_fn(batch):
    '''
    Collate for variable lengths: Stack poses, list others.
    Fixed window_size, so no padding needed for poses.
    '''
    video_ids, window_start_frames, window_end_frames, poses, events = zip(*batch)
    return list(video_ids), list(window_start_frames), list(window_end_frames), torch.stack(poses), list(events)


def get_loader(split='train', stride_ratio=0.5, max_tries=10, min_sentences=1, seed=42, batch_size=32):
    # Create a data loader for a specific split
    dataset = DVCDataset(split=split, stride_ratio=stride_ratio, max_tries=max_tries, min_sentences=min_sentences, seed=seed)
    return DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True if split == 'train' else False, num_workers=2,
        pin_memory=True, collate_fn=collate_fn
    )
    

if __name__ == '__main__':
    train_loader = get_loader('train', batch_size=32)
    for video_ids, start_frames, end_frames, poses, batch_events in train_loader:
        print('Batch poses shape: ', poses.shape)
        for video_id, start_frame, end_frame, events in zip(video_ids, start_frames, end_frames, batch_events):
            print(f'\nVideo ID: {video_id}, Start Frame: {start_frame}, End Frame: {end_frame}')
            for i, event in enumerate(events): print(f'- Event {i}: {event}')
        break