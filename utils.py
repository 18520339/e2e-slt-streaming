import os
import cv2
import torch
from typing import List, Dict, Union
from config import *


def get_video_info(video_path: str):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f'Video file not found: {video_path}')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise ValueError(f'Cannot open video: {video_path}')

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = frame_count / fps if fps > 0 else 0
    cap.release()
    return fps, video_duration, frame_count


def parse_vtt(vtt_path) -> List[Dict[str, Union[float, str]]]:
    '''
    Parse VTT file to extract subtitles with start/end times (in seconds) and text.
    Filters out non-subtitle entries (e.g., '[Music]').
    Returns list of dicts: {'start': float, 'end': float, 'text': str, 'duration': float}
    '''
    if not os.path.exists(vtt_path):
        raise FileNotFoundError(f'VTT file not found: {vtt_path}')
    
    subtitles = []
    try:
        if USE_WEBVTT:
            for subtitle in webvtt.read(vtt_path):
                text = subtitle.text.strip()
                if not text or text.startswith('[') and text.endswith(']'): continue # Skip non-verbal like [Music]
                start_sec, end_sec = subtitle.start_in_seconds, subtitle.end_in_seconds
                if start_sec >= end_sec: continue # Invalid timing
                if text: subtitles.append({'start': start_sec, 'end': end_sec, 'text': text, 'duration': end_sec - start_sec})
        else:
            i = 0
            lines = open(vtt_path, 'r').readlines()
            while i < len(lines):
                line = lines[i].strip()
                if '-->' not in line: i += 1
                else: # Timestamp line, e.g., 00:00:01.000 --> 00:00:05.000
                    # i += 1
                    # text = ''
                    # while i < len(lines) and not '-->' in lines[i] and lines[i].strip() != '':
                    #     text += lines[i].strip() + ' '
                    #     i += 1
                    
                    text = lines[i+1].strip()
                    i += 2
                    
                    if not text or text.startswith('[') and text.endswith(']'): continue # Skip non-verbal like [Music]
                    times = line.split(' --> ')
                    start_str = times[0].replace(',', '.')
                    end_str = times[1].replace(',', '.')

                    # HH:MM:SS.ms to sec
                    start_sec = sum(float(x) * 60 ** j for j, x in enumerate(reversed(start_str.split(':')))) 
                    end_sec = sum(float(x) * 60 ** j for j, x in enumerate(reversed(end_str.split(':'))))
                    if start_sec >= end_sec: continue # Invalid timing
                    if text: subtitles.append({'start': start_sec, 'end': end_sec, 'text': text, 'duration': end_sec - start_sec})
    except Exception as e:
        raise ValueError(f'Error parsing VTT file {vtt_path}: {e}')
    return sorted(subtitles, key=lambda x: x['start'])  # Ensure chronological order


def time_to_seconds(time_str): # Convert HH:MM:SS.mmm to seconds float
    if ':' not in time_str: return float(time_str)
    parts = time_str.split(':')
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    elif len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    return float(time_str)


def cw_to_se(cw: torch.Tensor) -> torch.Tensor:
    '''
    Convert (center, length) to (start, end). Inputs are normalized to [0, 1].
    cw: (N, 2) or (..., 2)
    Returns: (..., 2) as (start, end)
    '''
    c = cw[..., 0]
    w = torch.clamp(cw[..., 1], min=1e-6)
    box = [c - 0.5 * w, c + 0.5 * w]
    return torch.stack(box, dim=-1)


def se_to_cw(se: torch.Tensor) -> torch.Tensor:
    '''
    Convert (start, end) in [0,1] to (center, width).
    se: (N, 2) or (..., 2)
    Returns: (..., 2) as (center, length)
    '''
    s = se[..., 0]
    e = se[..., 1]
    box = [0.5 * (s + e), torch.clamp(e - s, min=1e-6)]
    return torch.stack(box, dim=-1)


def ensure_cw_format(boxes: torch.Tensor) -> torch.Tensor:
    '''
    Ensure boxes are in (center, width) format. If input is (start, end), convert it.
    Returns: (N, 2) as (center, width)
    '''
    if boxes.shape[-1] != 2: raise ValueError('Boxes must have shape (N, 2)')
    s, e = boxes[:, 0], boxes[:, 1]
    if (s <= e).all() and boxes.min() >= 0.0 and boxes.max() <= 1.0:
        return se_to_cw(boxes)
    return boxes


def box_iou(pred_se: torch.Tensor, target_se: torch.Tensor) -> torch.Tensor:
    area1 = pred_se[:, 1] - pred_se[:, 0]
    area2 = target_se[:, 1] - target_se[:, 0]
    left_top = torch.max(pred_se[:, None, 0], target_se[:, 0])
    right_bottom = torch.min(pred_se[:, None, 1], target_se[:, 1]) 
    inter = (right_bottom - left_top).clamp(min=0) 
    union = area1[:, None] + area2 - inter
    iou = torch.where(union > 0, inter / union, torch.zeros_like(union))
    return iou, union


def generalized_box_iou(pred_se, target_se):
    # Degenerate boxes gives inf / nan results, so do an early check
    assert (pred_se[:, 1:] >= pred_se[:, :1]).all()
    assert (target_se[:, 1:] >= target_se[:, :1]).all()
    iou, union = box_iou(pred_se, target_se)
    top_left = torch.min(pred_se[:, None, 0], target_se[:, 0])
    right_bottom = torch.max(pred_se[:, None, 1], target_se[:, 1])
    area = (right_bottom - top_left).clamp(min=0)
    return torch.where(area > 0, iou - (area - union) / area, torch.zeros_like(area))