import numpy as np
import os
import torch
import pickle

from collections import defaultdict

from loguru import logger
import torch.nn as nn

from itertools import combinations

role_map = {
    "player": 0,
    "goalkeeper": 1,
    "referee": 2,
    "other": 3,
    "ball": 4,
    # default if something else
}

def compute_iou(array1, array2):
    """
    Computes IoU between bounding boxes in array1 and array2.

    array1: (1, 10, 4) - bounding boxes for a single track
    array2: (12, 10, 4) - bounding boxes for multiple tracks
    threshold: IoU threshold for suppression

    Returns:
        iou_mask: (12, 10, 1) - Boolean mask indicating where IoU >= threshold
    """
    # Extract coordinates for broadcasting
    x1_1, y1_1, w1, h1 = array1
    x1_2, y1_2, w2, h2 = array2
    x2_1, y2_1 = x1_1 + w1, y1_1 + h1  # Convert width & height to bottom-right coordinates
    x2_2, y2_2 = x1_2 + w2, y1_2 + h2
    # Compute intersection coordinates
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    # Compute intersection area
    intersection_width = max(x_right - x_left, 0)
    intersection_height = max(y_bottom - y_top, 0)
    intersection_area = intersection_width * intersection_height  # (12, 10, 1)

    # Compute areas of bounding boxes
    area1 = w1 * h1  # (1, 10, 1)
    area2 = w2 * h2  # (12, 10, 1)

    # Compute IoU
    union_area = area1 + area2 - intersection_area

    return intersection_area / union_area if union_area > 0 else 0

def calc_same_team(emb1, emb2, thres=0.5):
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    cos_sim = cos(torch.tensor(emb1), torch.tensor(emb2))
    if cos_sim > thres:
        return True
    else:
        return False

def check_double_bboxes(data, entire_data, iou_thres=None, ver_dist_thres=None):
    frame_id = list(set(data['FrameID']))[0]
    track_ids = data['TrackID']
    bounding_boxes = np.column_stack((data['X'], data['Y'], data['W'], data['H']))
    y_centers = data['Y'] + (data['H'] / 2)
    jersey_colors = data['JerseyColor']
    duplicate_tracks = []
    remove_line_list = []

    for i, j in combinations(range(len(track_ids)), 2):
        iou = compute_iou(bounding_boxes[i], bounding_boxes[j])
        y_diff = abs(y_centers[i] - y_centers[j])

        if iou > iou_thres and y_diff < ver_dist_thres and jersey_colors[i] == jersey_colors[j]:
            duplicate_tracks.append((frame_id, track_ids[i], track_ids[j]))
            remove_line = find_index(frame_id, track_ids[j], entire_data)    # j is created later than i. Hence j is subject to removal
            remove_line_list.append(remove_line)
    return duplicate_tracks, remove_line_list


def find_index(frame_id, track_id, data):
    index = np.where((data['FrameID'] == frame_id) & (data['TrackID'] == track_id))[0]
    return index[0] if index.size > 0 else -1  # Return -1 if not found

def rewrite_txt_file(file_path, rmv_lines, write_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    with open(write_path, "w") as file:
        for i, line in enumerate(lines):
            if i not in rmv_lines:
                file.write(line)

def remove_double_bbox(cfg, datasets):
    for split in datasets:
        seq_tracks_dir = os.path.join(cfg['DATA_DIR'], split)
        seqs_tracks = os.listdir(seq_tracks_dir)
        seqs_tracks.sort()
        for seq_idx, seq in enumerate(seqs_tracks):
            logger.info(f"Processing seq {seq_idx + 1} / {len(seqs_tracks)}")
            logger.info(f"Processing {seq}")
            file_path = os.path.join(seq_tracks_dir, seq, f'interpolate_{seq}.txt')
            dtype = [
                ("FrameID", int),
                ("TrackID", int),
                ("X", float),
                ("Y", float),
                ("W", float),
                ("H", float),
                ("Score", float),
                ("Role", "U10"),  # Unicode string, max 10 characters
                ("JerseyNumber", int),
                ("JerseyColor", "U10"),
                ("Team", "U10")
            ]

            # Load data into structured NumPy array
            data = np.genfromtxt(file_path, delimiter=",", dtype=dtype, encoding="utf-8")
            initial_trklets_num = len(set(data['TrackID']))
            print(
                f"----------------Number of tracklets before removing double bboxes: {initial_trklets_num}----------------")
            max_num_frames = 750
            rmv_line_entire = []
            for i in range(1, max_num_frames + 1):  # for every frame
                frame_i_info = data[data['FrameID'] == i]
                try:
                    duplicate_tracks, remove_line_list = check_double_bboxes(frame_i_info, data, iou_thres=0.15, ver_dist_thres=3)
                except:
                    print(f"Failed case in sample: {seq} at frame: {i}")
                    continue
                rmv_line_entire.extend(remove_line_list)
            write_to = os.path.join(seq_tracks_dir, seq, f'rmved_{seq}.txt')
            rewrite_txt_file(file_path=file_path, rmv_lines=rmv_line_entire, write_path=write_to)
            new_data = np.genfromtxt(write_to, delimiter=",", dtype=dtype, encoding="utf-8")
            new_trklets_num = len(set(new_data['TrackID']))
            print(
                f"----------------Number of tracklets after removing double bboxes: {new_trklets_num}----------------")



if __name__ == "__main__":
    import yaml
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    data_sets = cfg['DATA_SETS']
    remove_double_bbox(cfg, data_sets)

