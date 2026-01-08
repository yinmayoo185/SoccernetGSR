import os
import pandas as pd
import numpy as np
import json
from collections import Counter, defaultdict

from urllib.request import urlopen

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from PIL import Image


# Correct column names for the court meter file.
# (Assuming the file contains 12 comma‐separated values per row.)
court_names_meter = [
    "frame", "track_id", "x_left", "y_left", "x_right", "y_right",
    "x_middle", "y_middle", "role", "jersey", "color", "team"
]

def load_court_meter_data(file_path):
    # Read the file without a header so that our provided names are used.
    df = pd.read_csv(file_path, header=None, names=court_names_meter)
    # Convert numeric columns to numbers
    numeric_cols = ["frame", "track_id", "x_left", "y_left", "x_right", "y_right",
                    "x_middle", "y_middle", "jersey", "team"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df


def majority_jersey(series, threshold=0.97):
    """
    Determine the majority jersey number for a tracklet, with special handling for jersey 100.

    Args:
        series (pd.Series): Series of jersey numbers.
        threshold (float): Proportion threshold to accept 100 as the majority.

    Returns:
        The chosen jersey number (as numeric) or None if no valid votes.
    """
    counts = Counter(series)
    total = sum(counts.values())
    if total == 0:
        return None
    count_100 = counts.get(100, 0)
    prop_100 = count_100 / total
    if prop_100 >= threshold:
        return 100
    else:
        # Exclude jersey 100 from consideration.
        if 100 in counts:
            del counts[100]
        if len(counts) == 0:
            return 100
        return counts.most_common(1)[0][0]

def majority_voting(df):
    # Only consider Player and Goalkeeper for jersey voting
    player_df = df[df["role"].isin(["Player", "Goalkeeper"])]
    # Apply our custom majority_jersey function on each track_id group.
    tracklet_jersey = player_df.groupby("track_id")["jersey"].apply(majority_jersey)
    # For roles, use the most common role overall per track_id.
    tracklet_roles = df.groupby("track_id")["role"].apply(
        lambda x: Counter(x).most_common(1)[0][0]
    )
    tracklet_color = df.groupby("track_id")["color"].apply(
        lambda x: Counter(x).most_common(1)[0][0]
    )
    
    return tracklet_jersey, tracklet_roles, tracklet_color

def determine_team_sides(df):
    from collections import Counter
    # -----------------------
    # STEP 1: Assign players using your original approach
    # -----------------------
    valid_df = df[df.team.isin([0, 1])]
    print(valid_df.groupby(['role', 'color']).size().reset_index(name='count').sort_values(by='count', ascending=False))
    if not valid_df.empty:
        team_a = valid_df[valid_df.team == 0]
        team_b = valid_df[valid_df.team == 1]
        avg_a = np.nanmean(team_a["x_middle"])
        avg_b = np.nanmean(team_b["x_middle"])
        # Map team based on which group has the lower average x_middle
        if avg_a < avg_b:
            mapping = {0: "left", 1: "right"}
        else:
            mapping = {0: "right", 1: "left"}

        threshold = (avg_a + avg_b) / 2
    else:
        threshold = np.nanmedian(df["x_middle"])
        mapping = {}

    def assign_team_side(row):
        # If this detection has a valid team value, use the mapping
        if row["team"] in mapping:
            return mapping[row["team"]]
        else:
            # Otherwise compare x_middle to threshold
            return "left" if row["x_middle"] < threshold else "right"

    # Apply to all rows initially
    df["team_side"] = df.apply(assign_team_side, axis=1)

    # -----------------------
    # STEP 2: Override logic for goalkeepers
    # -----------------------
    # If a goalkeeper's x_middle is > 0, force it to "right", else "left"
    # (assuming your pitch center is x=0).
    # This is a simple approach for keepers who stand near their respective goals.
    is_gk = (df["role"] == "Goalkeeper") & (~df["x_middle"].isna())
    df.loc[is_gk & (df["x_middle"] > 0), "team_side"] = "right"
    df.loc[is_gk & (df["x_middle"] <= 0), "team_side"] = "left"

    # -----------------------
    # STEP 3: Majority team side per tracklet
    # -----------------------
    tracklet_team_side = (
        df[df["role"].isin(["Player", "Goalkeeper"])]
        .groupby("track_id")["team_side"]
        .apply(lambda x: Counter(x.dropna()).most_common(1)[0][0] if len(x.dropna()) > 0 else None)
    )

    return tracklet_team_side


def generate_tracklet_json(tracklet_jersey, tracklet_roles, tracklet_team_side, tracklet_color):
    # Allowed roles in final JSON
    allowed_roles = {"other", "player", "goalkeeper", "referee", "ball"}
    tracklet_results = {}
    for track_id in tracklet_roles.index:
        # Convert role to lowercase.
        role = str(tracklet_roles[track_id]).lower()
        if role not in allowed_roles:
            role = "other"

        # For ball and referee, jersey and team are set to None.
        if role in ["ball", "referee"]:
            tracklet_results[str(track_id)] = {"role": role, "jersey": None, "team": None, "color": None}
        else:
            jersey_val = tracklet_jersey[track_id] if track_id in tracklet_jersey.index else None
            jersey_str = str(jersey_val) if jersey_val is not None else None
            if jersey_str == "100":
                jersey_str = None
            team = tracklet_team_side.get(track_id, None)
            color = tracklet_color[track_id] if track_id in tracklet_color.index else None
            tracklet_results[str(track_id)] = {
                "role": role,
                "jersey": jersey_str,
                "team": team,
                "color": color
            }
    return tracklet_results


def postprocess_ball_only(predictions, max_ball_area=800):
    """
    For each track_id, compute the average bounding-box area across all frames.
    If the average area is below 'max_ball_area', override the entire track's role to 'ball'.
    Otherwise, keep whatever role(s) the pipeline assigned.

    Args:
        predictions (list of dict): The final detection list from write_predictions_json.
        max_ball_area (float): Threshold below which we force the entire track to 'ball'.

    Returns:
        list of dict: Updated predictions with track-level overrides for ball only.
    """
    from collections import defaultdict

    # 1) Gather bounding-box areas per track
    track_areas = defaultdict(list)
    for det in predictions:
        track_id = det["track_id"]
        w = det["bbox_image"]["w"]
        h = det["bbox_image"]["h"]
        area = w * h
        track_areas[track_id].append(area)

    # 2) Decide if the track is 'ball' or not
    #    If average area < max_ball_area => force 'ball'
    #    Else => do not override
    track_is_ball = {}
    for track_id, areas in track_areas.items():
        avg_area = sum(areas) / len(areas)
        if avg_area < max_ball_area:
            track_is_ball[track_id] = True
        else:
            track_is_ball[track_id] = False

    # 3) Override role to 'ball' only if track_is_ball[track_id] is True
    for det in predictions:
        track_id = det["track_id"]
        if track_is_ball[track_id]:
            det["attributes"]["role"] = "ball"

    return predictions

# Define the structure for the prediction
def get_most_frequent_color_by_team(predictions):
    # Dictionary to store the color counts for each team
    color_count = defaultdict(int)
    color_count_lr = {
        "left": defaultdict(int),
        "right": defaultdict(int)
    }
    team_count = defaultdict(int)

    # Iterate through each prediction and count the colors based on the team
    for prediction in predictions:
        team = prediction['attributes']['team']
        color = prediction['attributes']['color']
        color_count[color] += 1
        if team == "left" or team == "right":
            color_count_lr[team][color] += 1
        team_count[team] += 1

    def get_most_frequent_color(team):
        most_frequent_color = ""
        max_count = 0
        for color, count in color_count_lr[team].items():
            if count > max_count:
                most_frequent_color = color
                max_count = count
        return most_frequent_color
    
    most_frequent_left_color = get_most_frequent_color("left")
    most_frequent_right_color = get_most_frequent_color("right")

    # Sort the color counts and team counts to determine the most frequent
    sorted_colors = sorted(
        ((color, count) for color, count in color_count.items() if color is not None),
        key=lambda x: x[1],
        reverse=True
    )

    sorted_teams = sorted(
        ((team, count) for team, count in team_count.items() if team is not None),
        key=lambda x: x[1],
        reverse=True
    )
    print(sorted_colors, sorted_teams)
    # Identify the most frequent color and the second and third most frequent
    most_frequent_color = sorted_colors[0][0] if sorted_colors else None

    second_most_frequent_color = sorted_colors[1][0] if len(sorted_colors) > 1 else None
    third_most_frequent_color = sorted_colors[2][0] if len(sorted_colors) > 2 else None

    most_frequent_team = sorted_teams[0][0] if sorted_teams else None
    second_most_frequent_team = sorted_teams[1][0] if len(sorted_teams) > 1 else None

    # Iterate through the predictions and update the team based on the conditions
    updated_predictions = []
    for prediction in predictions:
        team = prediction['attributes']['team']
        color = prediction['attributes']['color']
        role = prediction['attributes']['role']

        # If role is "player", we need to check conditions to change the team
        if role == "player":
            if team == "left" and color != most_frequent_left_color:
                # Change left to right and vice versa
                prediction['attributes']['team'] = "right"
            elif team == "right" and color != most_frequent_right_color:
                prediction['attributes']['team'] = "left"

            # If color is the third most frequent, change the team to second most frequent team
            if color == third_most_frequent_color:
                if second_most_frequent_team:
                    print(color, second_most_frequent_team)
                    prediction['attributes']['team'] = second_most_frequent_team

        # Add the updated prediction to the list
        updated_predictions.append(prediction)
    
    return updated_predictions

# 4) Filter out referees who are far outside the pitch
def is_inside_pitch(bbox_pitch, half_width=52.5, half_height=34, margin_x=10, margin_y=5):
    min_x = -half_width - margin_x
    max_x = half_width + margin_x
    min_y = -half_height - margin_y
    max_y = half_height + margin_y
    x_mid = bbox_pitch["x_bottom_middle"]
    y_mid = bbox_pitch["y_bottom_middle"]
    return (min_x <= x_mid <= max_x) and (min_y <= y_mid <= max_y)

def compute_and_fix_team_labels(data):
    """
    Computes the average x coordinate for each team in a list of objects and,
    if the left team's average is not lower than the right team's, swaps the team labels.

    Each object should have:
      - 'attributes': a dict with keys 'team' (either 'left' or 'right') and 'role' (e.g., 'player')
      - 'bbox_pitch': a dict with keys 'x_bottom_middle' and 'y_bottom_middle'
    """
    # Calculate pitch dimensions.
    pitchWidth = 105 + 2 * 10     # 125

    def normalize_x(x: float) -> float:
        # Provided normalization formula.
        return ((x + pitchWidth / 2) / pitchWidth) * pitchWidth

    # Filter out only players for each team.
    left_players = [
        obj for obj in data
        if obj.get('attributes', {}).get('team') == 'left' and obj.get('attributes', {}).get('role') == 'player'
    ]
    right_players = [
        obj for obj in data
        if obj.get('attributes', {}).get('team') == 'right' and obj.get('attributes', {}).get('role') == 'player'
    ]

    # Check for division by zero in case one team has no players.
    if not left_players or not right_players:
        print("One of the teams has no players. Defaulting to original assignment.")
        return data

    # Compute average normalized x coordinate for each team.
    mean_x_left = sum(normalize_x(obj['bbox_pitch']['x_bottom_middle']) for obj in left_players) / len(left_players)
    mean_x_right = sum(normalize_x(obj['bbox_pitch']['x_bottom_middle']) for obj in right_players) / len(right_players)

    print(f"Computed mean x - Left team: {mean_x_left:.2f}, Right team: {mean_x_right:.2f}")

    # If the left team's mean is NOT lower than the right team's, swap the team labels.
    if mean_x_left >= mean_x_right:
        print("Left team's average x is not lower than the right team's. Swapping team labels...")
        for obj in data:
            if obj.get('attributes', {}).get('role') == 'player':
                team = obj.get('attributes', {}).get('team')
                if team == 'left':
                    obj['attributes']['team'] = 'right'
                elif team == 'right':
                    obj['attributes']['team'] = 'left'
    else:
        print("Team labels are correct. No swap needed.")

    return data


def check_duplicate_jerseys_across_track_ids(track_data_list, df):
    # Create a dictionary to store the jersey numbers per team and track_id
    team_jersey_track = {}

    for data in track_data_list:
        # Only process entries where the role is "player" and jersey is not None
        if data["attributes"]["role"] == "player" and data["attributes"]["jersey"] is not None:
            # Extract relevant information
            team = data["attributes"]["team"]
            role = data["attributes"]["role"]
            jersey = data["attributes"]["jersey"]
            track_id = data["track_id"]

            # If the team doesn't exist in the dictionary, create it
            if team not in team_jersey_track:
                team_jersey_track[team] = {}

            # If the jersey number doesn't exist for the current team, create a set for tracking track_ids
            if jersey not in team_jersey_track[team]:
                team_jersey_track[team][jersey] = set()

            # Add the track_id to the set for this jersey (set ensures uniqueness)
            team_jersey_track[team][jersey].add(track_id)

    # Now check and print only if there are more than one unique track_id for the same jersey
    for team in team_jersey_track:
        for jersey, track_ids in team_jersey_track[team].items():
            if len(track_ids) > 1:
                # print(f"Duplicate jersey found: Team: {team}, Jersey: {jersey}, Track IDs: {sorted(track_ids)}")

                # Perform majority voting for each track_id's jersey based on df
                track_id_jerseys = df[df['track_id'].isin(track_ids)]['jersey']
                majority_jersey_value = majority_jersey(track_id_jerseys)

                # print(f"Majority voted jersey for track ids {sorted(track_ids)}: {majority_jersey_value}")

                # Step 3: Identify which track_id has the most frequent majority value
                majority_counts = {}
                for track_id in track_ids:
                    track_id_jersey = df[df['track_id'] == track_id]['jersey'].iloc[0]
                    if track_id_jersey == majority_jersey_value:
                        majority_counts[track_id] = track_id_jersey

                if len(majority_counts) > 0:
                    # Track IDs with the most frequent majority jersey
                    max_majority_track_id = max(majority_counts, key=majority_counts.get)
                    # print(f"Track ID {max_majority_track_id} has the most majority jersey: {majority_jersey_value}")
                    # For the other track_ids, we will perform second majority voting
                    remaining_track_ids = track_ids - set([max_majority_track_id])

                    if remaining_track_ids:
                        for track_id in remaining_track_ids:
                            # Perform majority voting again for the second most common jersey
                            track_id_jerseys_remaining = df[df['track_id'] == track_id]['jersey']
                            track_id_jerseys_remaining = track_id_jerseys_remaining[
                                track_id_jerseys_remaining != majority_jersey_value]
                            second_majority_jersey_value = majority_jersey(track_id_jerseys_remaining)
                            second_majority_jersey_value = second_majority_jersey_value if second_majority_jersey_value != 100 else None
                            # print(f"Track ID {track_id} has second majority jersey: {second_majority_jersey_value}")

                            # Update the jersey in df with second majority value
                            for data in track_data_list:
                                if data["track_id"] == track_id:
                                    data["attributes"]["jersey"] = second_majority_jersey_value
    return track_data_list

def write_predictions_json(tracklet_results, court_meter_df, sngs_df, output_json_path, video_id):
    predictions = []
    # Use enumerate to get an incremental id for each detection (line) in the tracking file.
    for i, (_, image_data) in enumerate(sngs_df.iterrows()):
        track_id = int(image_data["track_id"])
        frame_num = int(image_data["frame"])

        # Try to match by both track_id and frame
        matching_rows = court_meter_df[
            (court_meter_df["track_id"] == track_id) &
            (court_meter_df["frame"] == frame_num)
            ]
        if matching_rows.empty:
            # Fall back to the first occurrence for that track_id
            matching_rows = court_meter_df[court_meter_df["track_id"] == track_id]
            if matching_rows.empty:
                continue
        track_data = matching_rows.iloc[0]

        # Build the image_id, for example "2" + video_id + zero-padded frame number
        image_id = f"2{video_id}{str(frame_num).zfill(7)}"

        # Get the track-level attributes if available, otherwise use defaults.
        if str(track_id) in tracklet_results:
            track_info = tracklet_results[str(track_id)]
        else:
            track_info = {"role": "other", "jersey": None, "team": None, "color": None}

        prediction = {
            "bbox_pitch": {
                "x_bottom_left": track_data['x_left'],
                "y_bottom_left": track_data['y_left'],
                "x_bottom_right": track_data['x_right'],
                "y_bottom_right": track_data['y_right'],
                "x_bottom_middle": track_data['x_middle'],
                "y_bottom_middle": track_data['y_middle']
            },
            "bbox_image": {
                "x": float(image_data["x"]),
                "y": float(image_data["y"]),
                "w": float(image_data["w"]),
                "h": float(image_data["h"])
            },
            "category_id": 1.0,
            "image_id": image_id,
            "video_id": video_id,
            "track_id": track_id,
            "supercategory": "object",
            "attributes": {
                "role": track_info["role"],
                "jersey": track_info["jersey"],
                "team": track_info["team"],
                "color": track_info["color"]
            },
            "id": str(i)  # Incremental id starting from 0 (line number in tracking result file)
        }
        predictions.append(prediction)

    # Now apply the postprocessing step:
    predictions = postprocess_ball_only(predictions, max_ball_area=800)

    predictions = get_most_frequent_color_by_team(predictions)
    # predictions = compute_and_fix_team_labels(predictions)
    # predictions = check_duplicate_jerseys_across_track_ids(predictions, sngs_df)

    # --- Filtering Step ---
    # Remove predictions with role "other"
    filtered_predictions = [pred for pred in predictions if pred["attributes"]["role"] != "other"]

    # --- Filtering Step 2: Remove short tracklets for players and referee only.
    # Compute the number of detections per track_id.
    min_frame_threshold = 20
    track_frame_counts = defaultdict(int)
    for pred in filtered_predictions:
        track_frame_counts[pred["track_id"]] += 1

    # For tracklets whose role is "player" or "referee", only keep them if they have at least min_frame_threshold detections.
    filtered_predictions = [
        pred for pred in filtered_predictions
        if (pred["attributes"]["role"].lower() not in ["player", "referee"])
           or (track_frame_counts[pred["track_id"]] >= min_frame_threshold)
    ]

    # Optionally, remove predictions with role "ball" if remove_ball is True
    # if remove_ball:
    #     filtered_predictions = [pred for pred in filtered_predictions if pred["attributes"]["role"] != "ball"]

    final_filtered_predictions = [
        pred for pred in filtered_predictions if is_inside_pitch(pred["bbox_pitch"])
    ]

    # --- Remove the "color" attribute before writing JSON ---
    for pred in final_filtered_predictions:
        if "color" in pred["attributes"]:
            del pred["attributes"]["color"]

    # 5) Write JSON
    final_output = {"predictions": final_filtered_predictions}
    with open(output_json_path, 'w') as f:
        json.dump(final_output, f, indent=4)
    print(f"Predictions have been written to {output_json_path}")

def main(vdo_clip_path, vdo_path, vdo_name):
    # Assume court-meter file is named "court_meter_<vdo_name>.txt"
    court_meter_path = os.path.join(vdo_path, f'court_meter_{vdo_name}.txt')
    output_json_path = os.path.join(vdo_path, f"{vdo_name}.json")

    # The image-space tracking file (sngs file)
    sngs_path = os.path.join(vdo_path, f'refined_{vdo_name}.txt')

    if not os.path.exists(court_meter_path):
        print(f"Missing files for {vdo_name}, skipping...")
        return

    print(f"Processing tracking results for {vdo_name} in {vdo_path}...")
    # Load the court-meter data
    court_meter_df = load_court_meter_data(court_meter_path)

    # Read the image-space (sngs) file with explicit column names.
    sngs_df = pd.read_csv(sngs_path, header=None,
                          names=['frame', 'track_id', 'x', 'y', 'w', 'h', 'score', 'role', 'jersey', 'color', 'team'])
    # Ensure track_id is numeric so that matching works correctly.
    sngs_df["track_id"] = pd.to_numeric(sngs_df["track_id"], errors="coerce")

    # Compute majority voting for jersey numbers and roles.
    tracklet_jersey, tracklet_roles, tracklet_color = majority_voting(court_meter_df)
    # Determine team side (left/right) per tracklet.
    tracklet_team_side = determine_team_sides(court_meter_df)
    # Build tracklet-level JSON structure.
    tracklet_results = generate_tracklet_json(tracklet_jersey, tracklet_roles, tracklet_team_side, tracklet_color)

    # Extract video_id from vdo_name (e.g., if vdo_name is "SNGS-116", then video_id is "116")
    video_id = vdo_name.split('-')[-1]
    write_predictions_json(tracklet_results, court_meter_df, sngs_df, output_json_path, video_id)


if __name__ == "__main__":
    import yaml
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    data_sets = ['test']

    for split in data_sets:
        folder_path = os.path.join(cfg['DATA_DIR'], split)
        for vdo_name in sorted(os.listdir(folder_path)):
            vdo_path = os.path.join(folder_path, vdo_name)
            if not os.path.isdir(vdo_path):
                continue
            for folder_name in sorted(os.listdir(vdo_path)):
                vdo_clip_path = os.path.join(vdo_path, folder_name)
                if not os.path.isdir(vdo_clip_path):
                    continue
                print(f"Processing {vdo_clip_path}")
                main(vdo_clip_path, vdo_path, vdo_name)
