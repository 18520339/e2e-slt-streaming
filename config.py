# import torch
from pathlib import Path

try:
    import webvtt
    USE_WEBVTT = True
except ImportError:
    USE_WEBVTT = False
    print('webvtt-py not installed; using custom VTT parser')

# -- Project Paths ----------------------------------------------------------
# NOTE: Please update these paths to match your local machine's setup.
PROJECT_ROOT = Path(__file__).parent
DATA_ROOT = Path('dataset/BOBSL') # TO BE FILLED BY USER
POSE_ROOT = DATA_ROOT / 'bobsl_dwpose'  # Directory with pose .npy files
CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints'
OUTPUT_DIR = PROJECT_ROOT / 'outputs'

# -- Dataset and Dataloader Configuration ------------------------------------
SUBSET_JSON = DATA_ROOT / 'original_data/metadata/test.json' # Train/val/test splits
VTT_DIR = DATA_ROOT / 'original_data/subtitles/audio-aligned'  # Directory with .vtt files

FPS = 12.5 # Downsampled FPS from original 25fps
WINDOW_DURATION_SECONDS = 18 # This will be determined by the analysis in `stats.ipynb`
MIN_SUB_DURATION = 1.0 # From LiTFiC, seconds
MAX_SUB_DURATION = 20.0 # From LiTFiC, seconds
POSE_DIMENSIONS = 2 # 2D coordinates (x, y)
POSE_SHAPE = (133, 3) # Keypoints x (x, y, confidence)

# Dataloader settings
BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
NUM_WORKERS = 4 # Number of CPU cores for data loading

# -- Pose Preprocessing (CoSign Inspired) -----------------------------------
# Define keypoint groups based on COCO-WholeBody (133 points: body 0-22, face 23-90, left hand 91-111, right hand 112-132)
# Upper body (9): nose(0), left eye(1), right eye(2), left shoulder(5), right shoulder(6), left elbow(7), right elbow(8), left wrist(9), right wrist(10)
# Mouth (8): inner lips approx 60-67 in face (face starts at 23, so 23+60-67 = 83-90, but adjust to 8)
# Face lower/cheek (18): contour approx 23+0 to 23+16 (17 points), plus nose
BODY_IDS = [0, 1, 2, 5, 6, 7, 8, 9, 10]  # 9 points
LEFT_HAND_IDS = list(range(91, 112))  # 21 points
RIGHT_HAND_IDS = list(range(112, 133))  # 21 points
MOUTH_IDS = list(range(83, 91))  # Inner mouth 8 points
FACE_IDS = list(range(23, 40)) + [53]  # First 18 as cheek/lower approx
ALL_SELECTED_IDS = BODY_IDS + LEFT_HAND_IDS + RIGHT_HAND_IDS + MOUTH_IDS + FACE_IDS  # Total 9+21+21+8+18=77
CONF_THRESHOLD = 0.5  # From supp: Keypoints with conf > 0.5 considered valid
NUM_KEYPOINTS = len(ALL_SELECTED_IDS)

# -- PDVC Baseline Model Architecture ----------------------------------------
# Pose Encoder (ST-GCN)
STGCN_CHANNELS = [3, 64, 64, 128, 128, 256, 256] # Input channels, then block channels

# Transformer
D_MODEL = 256 # Must match the last channel dim of the pose encoder
NHEAD = 8
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DIM_FEEDFORWARD = 1024
DROPOUT = 0.1

# Vocabulary settings (placeholders for now)
VOCAB_SIZE = 10000
PAD_IDX = 0

# -- Training Configuration --------------------------------------------------
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
CLIP_GRAD_NORM = 1.0 # Gradient clipping to prevent exploding gradients