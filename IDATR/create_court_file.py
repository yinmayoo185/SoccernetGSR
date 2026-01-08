import glob
import os
import time
import pandas as pd
from pathlib import Path
import cv2
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from utils.sys_utils import read_yaml_file


def pixel_to_meter_all(x_bottom_left, y_bottom_left, x_bottom_right, y_bottom_right, x_bottom_middle, y_bottom_middle,
                       center_x_px, center_y_px):
    """
    Convert pixel coordinates back to meters.
    Args:
        x_px (float): The x-coordinate in pixels.
        y_px (float): The y-coordinate in pixels.
        center_x_px (float): The x-coordinate of the center of the image in pixels.
        center_y_px (float): The y-coordinate of the center of the image in pixels.

    Returns:
        tuple: The corresponding (x, y) coordinates in meters.
    """
    # Calculate the meter coordinates based on the center
    x_left = x_bottom_left - center_x_px
    y_left = y_bottom_left - center_y_px  # Inverting the y coordinate back
    x_right = x_bottom_right - center_x_px
    y_right = y_bottom_right - center_y_px  # Inverting the y coordinate back
    x_middle = x_bottom_middle - center_x_px
    y_middle = y_bottom_middle - center_y_px  # Inverting the y coordinate back

    return x_left, y_left, x_right, y_right, x_middle, y_middle


def post_process(cfg, split):
    """
    reads final track from GTA & calculates new homography using frame num, matrix
    then to calculate pixel and meter values
    """
    sfr_cfg = cfg['SFR']
    idatr_cfg = cfg['IDATR']

    # The columns for your refined track
    key_names = ["frame", "track_id", "x", "y", "w", "h", "score", "role", "jersey", "color", "team"]

    # Output column names for meter/pixel files
    court_names_meter = [
        "frame", "track_id",
        "x_left", "y_left",
        "x_right", "y_right",
        "x_middle", "y_middle",
        "role", "jersey", "color", "team"
    ]
    court_names_pixel = [
        "frame", "track_id",
        "x", "y",
        "role", "jersey", "color", "team"
    ]

    # Load the pitch template for coordinate transformations
    template = cv2.imread(sfr_cfg['TEMPLATE_PATH'])
    center_x_px = template.shape[1] / 2
    center_y_px = template.shape[0] / 2
    data_path = os.path.join(cfg['DATA_DIR'], split)
    refined_track_paths = sorted(glob.glob(os.path.join(data_path, '**/*refined_SNGS*.txt'), recursive=True))
    for refined_track_path in refined_track_paths:
        # Prepare data structures to store final results
        player_position_results_meter = {key: [] for key in court_names_meter}
        data = refined_track_path.split('/')[-2]

        homo_file_path = sorted(
            glob.glob(os.path.join(data_path, data, "img1/*.npy"))
        )

        with open(refined_track_path, "r") as refined_track_file:
            for line in refined_track_file:
                # Each line should have 11 columns:
                # frame, track_id, l, t, w, h, conf, role, jersey, color, team
                vals = line.strip().split(",")
                if len(vals) < 11:
                    continue  # skip malformed lines if any

                # Parse numeric fields
                frame_num = int(float(vals[0]))
                tid       = int(vals[1])
                l         = float(vals[2])
                t         = float(vals[3])
                w         = float(vals[4])
                h         = float(vals[5])
                conf      = float(vals[6])
                role   = vals[7]
                jersey = vals[8]
                color  = vals[9]
                team = vals[10]

                # Load the corresponding homography
                if (frame_num - 1) < 0 or (frame_num - 1) >= len(homo_file_path):
                    # If homography file is missing or out of range, skip or handle
                    continue
                homography = np.load(homo_file_path[frame_num - 1])

                # Convert to bounding corners
                x_e = l + w
                y_e = t + h
                x_center = l + (w // 2)  # or (w / 2)

                # Build pitch_position array
                x_bottom_left  = l
                y_bottom_left  = y_e
                x_bottom_right = x_e
                y_bottom_right = y_e
                x_bottom_middle = x_center
                y_bottom_middle = y_e

                pitch_position = np.array([
                    [x_bottom_left,  y_bottom_left,  1.],
                    [x_bottom_right, y_bottom_right, 1.],
                    [x_bottom_middle,y_bottom_middle,1.]
                ])
                pitch_position_img = pitch_position.T
                transformed_pitch = np.dot(homography, pitch_position_img)
                player_position_pitch = (transformed_pitch[:2] / transformed_pitch[2]).T

                # Convert to np array
                player_position_pitch = np.array(player_position_pitch)
                # Convert pixel coords to "meter" coords
                x_left, y_left, x_right, y_right, x_middle, y_middle = pixel_to_meter_all(
                    player_position_pitch[0][0], player_position_pitch[0][1],
                    player_position_pitch[1][0], player_position_pitch[1][1],
                    player_position_pitch[2][0], player_position_pitch[2][1],
                    center_x_px, center_y_px
                )

                # 1) Store in the main "results" dict
                track_result = [
                    frame_num, tid, l, t, w, h, conf, role, jersey, color, team
                ]

                # 2) Build pixel-level result
                court_result_pixel = [
                    frame_num, tid,
                    int(player_position_pitch[2][0]),
                    int(player_position_pitch[2][1]),
                    role, jersey, color, team
                ]

                # 3) Build meter-level result
                court_result_meter = [
                    frame_num, tid,
                    x_left, y_left,
                    x_right, y_right,
                    x_middle, y_middle,
                    role, jersey, color, team
                ]
                for k, d in zip(court_names_meter, court_result_meter):
                    player_position_results_meter[k].append(d)

        # Convert to DataFrames
        meter_df = pd.DataFrame(player_position_results_meter)

        # Build filenames
        meter_filename = os.path.join(data_path, data, f"court_meter_{data}.txt")

        print(f"Saving refined track to {meter_filename}")
        meter_df['frame'] = meter_df['frame'].astype(int).astype(str).str.zfill(6)
        meter_df.to_csv(meter_filename, index=False, header=False)



if __name__ == "__main__":
    import yaml
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    data_sets = cfg['DATA_SETS']

    for split in data_sets:
        post_process(cfg, split)

