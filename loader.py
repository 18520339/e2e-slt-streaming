import json
import torch
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from poses.preprocessing import normalize_keypoints, threshold_confidence
from utils import parse_vtt
from config import *


class DVCDataset(Dataset):
    def __init__(self, split, stride_ratio=0.5, max_caption_len=20, max_tries=10, 
                 min_events=1, tokenizer=None, load_by='window', seed=42):
        '''
        PyTorch Dataset for DVC with on-the-fly sliding window sampling.
        Args:
            split: 'train', 'val', or 'test'
            stride_ratio: For val/test sequential sampling (e.g., 0.5 for 50% overlap)
            max_caption_len: Max caption token length for padding/truncation
            max_tries: Max resamples for train windows with < min_events
            min_events: Min full events (subtitles) in a window
            tokenizer: HuggingFace tokenizer for text processing
            load_by: 'window' (default) or 'video' - whether to
                     load poses per window and concatenate or 
                     load full video poses at once and slice
            seed: For reproducibility in random sampling
        '''
        assert split in ['train', 'val', 'test'], f"Split must be 'train', 'val', or 'test', but got {split}"
        self.split = split
        self.window_size_frames = int(WINDOW_DURATION_SECONDS * FPS)
        self.stride = int(self.window_size_frames * stride_ratio)
        self.max_caption_len = max_caption_len
        self.max_tries = max_tries
        self.min_events = min_events
        self.load_by = load_by
        assert self.load_by in ['window', 'video'], "load_by must be 'window' or 'video'"
        np.random.seed(seed)

        self.tokenizer = tokenizer
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
        for video_id in tqdm(self.video_ids, desc=f'Building video metadata for {self.split} split'):
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
            if self.split != 'train': # For val/test: count fixed, overlapping windows
                for window_start_frame in range(0, total_frames, self.stride):
                    window_end_frame = window_start_frame + self.window_size_frames
                    
                    if window_end_frame <= total_frames: # Ignore the last, smaller window if it's too short
                        valid_events_count = 0 # Count valid events fully contained in a window (used for val filtering)
                        for sub in self.video_metadata[video_id]['subtitles']:
                            sub_start_frame = int(sub['start'] * FPS)
                            sub_end_frame = int(sub['end'] * FPS)
                            if sub_start_frame >= window_start_frame and sub_end_frame <= window_end_frame and \
                                MIN_SUB_DURATION <= sub['duration'] <= MAX_SUB_DURATION:
                                valid_events_count += 1
                                
                        if valid_events_count < self.min_events: continue
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
                if window[-1]['class_labels'].shape[0] >= self.min_events: # Check events
                    # print(f'Sampled valid window for {video_id} (try {try_num+1})')
                    return window
                
            print(f'Warning: Could not find window with >= {self.min_events} events for {video_id} after {self.max_tries} tries')
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
        if window_start_frame >= window_end_frame: raise ValueError('Invalid window boundaries')
        if self.load_by == 'video': # Load full video poses at once and slice
            full_poses = self.load_poses_for_video(video_id)
            window_poses = full_poses[window_start_frame:window_end_frame, :, :]
        elif self.load_by == 'window': # Load only the necessary segments for this window and concatenate
            window_poses = self.load_poses_for_window(video_id, window_start_frame, window_end_frame)
        
        # Preprocess poses: Normalize and threshold
        window_poses = normalize_keypoints(window_poses)
        window_poses = threshold_confidence(window_poses)
        poses_tensor = torch.from_numpy(window_poses).float()  # (T, K, 3)
        
        # Filter subtitles in window and build model-ready labels
        labels = {'class_labels': [], 'boxes': [], 'seq_tokens': []}
        for sub in self.video_metadata[video_id]['subtitles']:
            sub_start_frame = int(sub['start'] * FPS)
            sub_end_frame = int(sub['end'] * FPS)

            # Subtitle must be FULLY contained within the window and have valid duration
            if sub_start_frame >= window_start_frame and sub_end_frame <= window_end_frame and \
                MIN_SUB_DURATION <= sub['duration'] <= MAX_SUB_DURATION: 
                # Normalize to [0, 1] relative to window
                rel_start = (sub_start_frame - window_start_frame) / self.window_size_frames
                rel_end = (sub_end_frame - window_start_frame) / self.window_size_frames
                center = min(max(0.5 * (rel_start + rel_end), 0.0), 1.0)
                width = min(max(rel_end - rel_start, 0.0), 1.0)
                labels['class_labels'].append(0) # Default single class 0
                labels['boxes'].append([center, width])
                labels['seq_tokens'].append(sub['text'])
    
        # Convert to tensors
        if labels['class_labels']:
            labels['class_labels'] = torch.tensor(labels['class_labels'], dtype=torch.long)
            labels['boxes'] = torch.tensor(labels['boxes'], dtype=torch.float)
            labels['seq_tokens'] = self.tokenizer(
                labels['seq_tokens'], add_special_tokens=True, truncation=True, 
                padding='max_length', max_length=self.max_caption_len, return_tensors='pt'
            )['input_ids']
        else: # No valid subtitles in window
            labels['class_labels'] = torch.empty(0, dtype=torch.long)
            labels['boxes'] = torch.empty(0, 2, dtype=torch.float)
            labels['seq_tokens'] = torch.empty(0, self.max_caption_len, dtype=torch.long)
        return video_id, window_start_frame, window_end_frame, poses_tensor, labels

        
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
    video_ids, window_start_frames, window_end_frames, poses_tensor, labels = zip(*batch)
    T = poses_tensor[0].shape[0]
    assert all(p.shape[0] == T for p in poses_tensor), 'Variable T in batch; use batch_size=1 or add padding.'
    return {
        'video_ids': video_ids,
        'window_start_frames': window_start_frames,
        'window_end_frames': window_end_frames,
        'pixel_values': torch.stack(poses_tensor), # [B(N), T, 77(K), 3(C)] Channel-last for CoSign backbone
        'pixel_mask': torch.ones(len(poses_tensor), T, dtype=torch.bool), # All True since we have fixed-size windows
        'labels': labels # List of {'class_labels': (N_i, ), 'boxes': (N_i, 2), 'seq_tokens': (N_i, L)}
    }
    

def get_loader(split='train', batch_size=32, stride_ratio=0.5, max_caption_len=20, 
               max_tries=10, min_events=1, tokenizer=None, load_by='window', seed=42):
    dataset = DVCDataset( # Create a data loader for a specific split
        split=split, stride_ratio=stride_ratio, max_caption_len=max_caption_len, max_tries=max_tries,
        min_events=min_events, tokenizer=tokenizer, load_by=load_by, seed=seed
    )
    return DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True if split == 'train' else False, num_workers=2,
        pin_memory=True, collate_fn=collate_fn
    )
    
    
if __name__ == '__main__':
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base', use_fast=True)
    train_loader = get_loader('train', batch_size=4, tokenizer=tokenizer)
    
    for batch in train_loader:
        video_ids, start_frames, end_frames = batch['video_ids'], batch['window_start_frames'], batch['window_end_frames']
        poses, pixel_mask, labels = batch['pixel_values'], batch['pixel_mask'], batch['labels']
        print('Batch poses shape: ', poses.shape)
        
        for video_id, start_frame, end_frame, events in zip(video_ids, start_frames, end_frames, labels):
            print(f'\nVIDEO ID: {video_id}, Start Frame: {start_frame}, End Frame: {end_frame}')
            for i, (box, event_tokens) in enumerate(zip(events['boxes'], events['seq_tokens'])):
                print(f'[Event {i + 1}] center={box[0]:.3f}, width={box[1]:.3f}, caption length={event_tokens.shape}:\n'
                      f'- Tokens: {event_tokens.tolist()}\n'
                      f"- Text: {tokenizer.decode(event_tokens)}")
        break