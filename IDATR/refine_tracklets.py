import numpy as np
import os
import glob
import torch
import pickle
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time

from collections import defaultdict, Counter

import matplotlib.pyplot as plt
import seaborn as sns

from loguru import logger
from tqdm import tqdm

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from Tracklet import Tracklet


def find_consecutive_segments(track_times):
    """
    Identifies and returns the start and end indices of consecutive segments in a list of times.

    Args:
        track_times (list): A list of frame times (integers) representing when a tracklet was detected.

    Returns:
        list of tuples: Each tuple contains two integers (start_index, end_index) representing the start and end of a consecutive segment.
    """
    segments = []
    start_index = 0
    end_index = 0
    for i in range(1, len(track_times)):
        if track_times[i] == track_times[end_index] + 1:
            end_index = i
        else:
            segments.append((start_index, end_index))
            start_index = i
            end_index = i
    segments.append((start_index, end_index))
    return segments


def query_subtracks(seg1, seg2, track1, track2):
    """
    Processes and pairs up segments from two different tracks to form valid subtracks based on their temporal alignment.

    Args:
        seg1 (list of tuples): List of segments from the first track where each segment is a tuple of start and end indices.
        seg2 (list of tuples): List of segments from the second track similar to seg1.
        track1 (Tracklet): First track object containing times and bounding boxes.
        track2 (Tracklet): Second track object similar to track1.

    Returns:
        list: Returns a list of subtracks which are either segments of track1 or track2 sorted by time.
    """
    subtracks = []  # List to store valid subtracks
    while seg1 and seg2:  # Continue until seg1 or seg1 is empty
        s1_start, s1_end = seg1[0]  # Get the start and end indices of the first segment in seg1
        s2_start, s2_end = seg2[0]  # Get the start and end indices of the first segment in seg2
        '''Optionally eliminate false positive subtracks
        if (s1_end - s1_start + 1) < 30:
            seg1.pop(0)  # Remove the first element from seg1
            continue
        if (s2_end - s2_start + 1) < 30:
            seg2.pop(0)  # Remove the first element from seg2
            continue
        '''

        subtrack_1 = track1.extract(s1_start, s1_end)
        subtrack_2 = track2.extract(s2_start, s2_end)

        s1_startFrame = track1.times[s1_start]  # Get the starting frame of subtrack 1
        s2_startFrame = track2.times[s2_start]  # Get the starting frame of subtrack 2

        # print("track 1 and 2 start frame:", s1_startFrame, s2_startFrame)
        # print("track 1 and 2 end frame:", track1.times[s1_end], track2.times[s2_end])

        if s1_startFrame < s2_startFrame:  # Compare the starting frames of the two subtracks
            assert track1.times[s1_end] <= s2_startFrame
            subtracks.append(subtrack_1)
            subtracks.append(subtrack_2)
        else:
            assert s1_startFrame >= track2.times[s2_end]
            subtracks.append(subtrack_2)
            subtracks.append(subtrack_1)
        seg1.pop(0)
        seg2.pop(0)

    seg_remain = seg1 if seg1 else seg2
    track_remain = track1 if seg1 else track2
    while seg_remain:
        s_start, s_end = seg_remain[0]
        if (s_end - s_start) < 30:
            seg_remain.pop(0)
            continue
        subtracks.append(track_remain.extract(s_start, s_end))
        seg_remain.pop(0)

    return subtracks  # Return the list of valid subtracks sorted ascending temporally


def get_subtrack(track, s_start, s_end):
    """
    Extracts a subtrack from a given track.

    Args:
    track (STrack): The original track object from which the subtrack is to be extracted.
    s_start (int): The starting index of the subtrack.
    s_end (int): The ending index of the subtrack.

    Returns:
    STrack: A subtrack object extracted from the original track object, containing the specified time intervals
            and bounding boxes. The parent track ID is also assigned to the subtrack.
    """
    subtrack = Tracklet()
    subtrack.times = track.times[s_start: s_end + 1]
    subtrack.bboxes = track.bboxes[s_start: s_end + 1]
    subtrack.parent_id = track.track_id

    return subtrack


def get_spatial_constraints(tid2track, factor):
    """
    Calculates and returns the maximal spatial constraints for bounding boxes across all tracks.

    Args:
        tid2track (dict): Dictionary mapping track IDs to their respective track objects.
        factor (float): Factor by which to scale the calculated x and y ranges.

    Returns:
        tuple: Maximal x and y range scaled by the given factor.
    """

    min_x = float('inf')
    max_x = -float('inf')
    min_y = float('inf')
    max_y = -float('inf')

    for track in tid2track.values():
        for bbox in track.bboxes:
            assert len(bbox) == 4
            x, y, w, h = bbox[0:4]  # x, y is coordinate of top-left point of bounding box
            x += w / 2  # get center point
            y += h / 2  # get center point
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)

    x_range = abs(max_x - min_x) * factor
    y_range = abs(max_y - min_y) * factor

    return x_range, y_range


def display_Dist(Dist, seq_name=None, isMerged=False, isSplit=False):
    """
    Displays a heatmap for the distances between tracklets for one or more sequences.

    Args:
        seq2Dist (dict): A dictionary mapping sequence names to their corresponding distance matrices.
        seq_name (str, optional): Specific sequence name to display the heatmap for. If None, displays for all sequences.
        isMerged (bool): Flag indicating whether the distances are post-merge.
        isSplit (bool): Flag indicating whether the distances are post-split.
    """
    split_info = " After Split" if isSplit else " Before Split"
    merge_info = " After Merge" if isMerged else " Before Merge"
    info = split_info + merge_info

    plt.figure(figsize=(10, 8))  # Optional: adjust the size of the heatmap

    # Plot the heatmap
    sns.heatmap(Dist, cmap='Blues')

    plt.title(f"{seq_name}{info}")
    plt.show()


def get_distance_matrix(tid2track, use_jn=False):
    """
    Constructs and returns a distance matrix between all tracklets based on overlapping times and feature similarities.

    Args:
        tid2track (dict): Dictionary mapping track IDs to their respective track objects.

    Returns:
        ndarray: A square matrix where each element (i, j) represents the calculated distance between track i and track j.
    """
    # print("number of tracks:", len(tid2track))
    Dist = np.zeros((len(tid2track), len(tid2track)))

    for i, (track1_id, track1) in enumerate(tid2track.items()):
        assert len(track1.times) == len(track1.bboxes)
        for j, (track2_id, track2) in enumerate(tid2track.items()):
            if j < i:
                Dist[i][j] = Dist[j][i]
            else:
                Dist[i][j] = get_distance(track1_id, track2_id, track1, track2, use_jn=use_jn)
    return Dist

def majority_vote(arr, jersey=False):
    if not jersey:
        return Counter(arr).most_common(1)[0][0]
    else:
        jn = Counter(arr).most_common(1)[0][0]
        if jn == '100':
            if len(set(arr)) == 1:  # if predicted jn is only 100
                return '100'
            else:
                filtered_jn_lst = [x for x in arr if x != '100']    # what's the next most appeared jn after removing 100 from the list
                jn = Counter(filtered_jn_lst).most_common(1)[0][0]
        return jn

def get_distance(track1_id, track2_id, track1, track2, use_jn=False):
    """
    Calculates the cosine distance between two tracks using PyTorch for efficient computation.
    *** calculates the cosine distance based on ReID features (appearance features) and temporal constraints.
    -> if there is a temporal overlap between two tracklets, then cos dist is max because same track id cannot exist at the same time
    *** add additional constraints: role, team, jersey number (if True)
    Args:
        track1_id (int): ID of the first track.
        track2_id (int): ID of the second track.
        track1 (Tracklet): First track object.
        track2 (Tracklet): Second track object.

    Returns:
        float: Cosine distance between the two tracks. -> closer to 0: similar / closer to 1: dissimilar
    """
    assert track1_id == track1.track_id and track2_id == track2.track_id  # debug line
    doesOverlap = False
    diff_role = False
    diff_team = False
    diff_jn = False
    if (track1_id != track2_id):
        """ 
        temporal constraint: if there is a temporal overlap, then these two track id cannot be merged
        because a player can only exist once per frame.
        -> set the cosine dist matrix to max value
        """
        doesOverlap = set(track1.times) & set(track2.times)
        """
        put constraint based on role and team (jersey number is not reliable enough)
        -> set the cosine dist matrix to max if two tracks have different role or team
        """
        track1_role = Counter(track1.role).most_common(1)[0][0]
        track2_role = Counter(track2.role).most_common(1)[0][0]
        track1_team = Counter(track1.team).most_common(1)[0][0]
        track2_team = Counter(track2.team).most_common(1)[0][0]
        diff_role = track1_role != track2_role  # True if different role
        diff_team = track1_team != track2_team  # True if different team
        if use_jn:
            track1_jn = majority_vote(track1.jn, jersey=True)
            track2_jn = majority_vote(track2.jn, jersey=True)
            diff_jn = track1_jn != track2_jn  # True if different team
    if doesOverlap or diff_role or diff_team or diff_jn:
        """if any of those constraint is met, then assign cos dist to max"""
        return 1  # make the cosine distance between two tracks maximum, max = 1 *** or should it be 2?
    else:
        # calculate cosine distance between two tracks based on features
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        track1_features_tensor = torch.tensor(np.stack(track1.features), dtype=torch.float32).to(device)
        track2_features_tensor = torch.tensor(np.stack(track2.features), dtype=torch.float32).to(device)
        count1 = len(track1_features_tensor)
        count2 = len(track2_features_tensor)

        cos_sim_Numerator = torch.matmul(track1_features_tensor, track2_features_tensor.T)
        track1_features_dist = torch.norm(track1_features_tensor, p=2, dim=1, keepdim=True)
        track2_features_dist = torch.norm(track2_features_tensor, p=2, dim=1, keepdim=True)
        cos_sim_Denominator = torch.matmul(track1_features_dist, track2_features_dist.T)
        cos_Dist = 1 - cos_sim_Numerator / cos_sim_Denominator

        total_cos_Dist = cos_Dist.sum()
        result = total_cos_Dist / (count1 * count2)
        return result


def check_spatial_constraints(trk_1, trk_2, max_x_range, max_y_range):
    """
    Checks if two tracklets meet spatial constraints for potential merging.

    Args:
        trk_1 (Tracklet): The first tracklet object containing times and bounding boxes.
        trk_2 (Tracklet): The second tracklet object containing times and bounding boxes, to be evaluated
                        against trk_1 for merging possibility.
        max_x_range (float): The maximum allowed distance in the x-coordinate between the end of trk_1 and
                             the start of trk_2 for them to be considered for merging.
        max_y_range (float): The maximum allowed distance in the y-coordinate under the same conditions as
                             the x-coordinate.

    Returns:
        bool: True if the spatial constraints are met (the tracklets are close enough to consider merging),
              False otherwise.
    """
    inSpatialRange = True
    seg_1 = find_consecutive_segments(trk_1.times)
    seg_2 = find_consecutive_segments(trk_2.times)
    '''Debug
    assert((len(seg_1) + len(seg_2)) > 1)         # debug line, delete later
    print(seg_1)                                  # debug line, delete later
    print(seg_2)                                  # debug line, delete later
    '''

    subtracks = query_subtracks(seg_1, seg_2, trk_1, trk_2)
    # assert(len(subtracks) > 1)                    # debug line, delete later
    subtrack_1st = subtracks.pop(0)
    # print("Entering while loop")
    while subtracks:
        # print("Subtracks remaining: ", len(subtracks))
        subtrack_2nd = subtracks.pop(0)
        if subtrack_1st.parent_id == subtrack_2nd.parent_id:
            subtrack_1st = subtrack_2nd
            continue
        x_1, y_1, w_1, h_1 = subtrack_1st.bboxes[-1][0: 4]
        x_2, y_2, w_2, h_2 = subtrack_2nd.bboxes[0][0: 4]
        x_1 += w_1 / 2
        y_1 += h_1 / 2
        x_2 += w_2 / 2
        y_2 += h_2 / 2
        dx = abs(x_1 - x_2)
        dy = abs(y_1 - y_2)

        # check the distance between exit location of track_1 and enter location of track_2
        if dx > max_x_range or dy > max_y_range:
            inSpatialRange = False
            # print(f"dx={dx}, dy={dy} out of range max_x_range = {max_x_range}, max_y_range  = {max_y_range}")    # debug line, delete later
            break
        else:
            subtrack_1st = subtrack_2nd
    # print("Exit while loop")
    return inSpatialRange


def merge_tracklets(tracklets, seq2Dist, Dist, seq_name=None, max_x_range=None, max_y_range=None,
                    merge_dist_thres=None, use_jn=False):
    seq2Dist[seq_name] = Dist
    idx2tid = {idx: tid for idx, tid in enumerate(tracklets.keys())}

    # Hierarchical Clustering
    # While there are still values (exclude diagonal) in distance matrix lower than merging distance threshold
    #   Step 1: find minimal distance for tracklet pair
    #   Step 2: merge tracklet pair
    #   Step 3: update distance matrix
    diagonal_mask = np.eye(Dist.shape[0], dtype=bool)
    non_diagonal_mask = ~diagonal_mask

    while (np.any(Dist[non_diagonal_mask] < merge_dist_thres)):
        # print(np.sum(np.any(Dist[non_diagonal_mask] < merge_dist_thres)))
        # Get the indices of the minimum value considering the mask
        min_index = np.argmin(Dist[non_diagonal_mask])
        min_value = np.min(Dist[non_diagonal_mask])
        # Translate this index to the original array's indices
        masked_indices = np.where(non_diagonal_mask)
        track1_idx, track2_idx = masked_indices[0][min_index], masked_indices[1][min_index]
        # print("Tracks idx to merge:", track1_idx, track2_idx)
        # print(f"Minimum value in masked Dist: {min_value}")
        # print(f"Corresponding value in Dist using recalculated indices: {Dist[track1_idx, track2_idx]}")

        assert min_value == Dist[track1_idx, track2_idx] == Dist[track2_idx, track1_idx], "Values should match!"

        track1 = tracklets[idx2tid[track1_idx]]
        track2 = tracklets[idx2tid[track2_idx]]
        print(f"merge pair candidates before spatial constraints: T_id: {idx2tid[track1_idx]} and T_id: {idx2tid[track2_idx]}")

        inSpatialRange = check_spatial_constraints(track1, track2, max_x_range, max_y_range)
        # print("In spatial range:", inSpatialRange)
        if inSpatialRange:
            track1.features += track2.features  # Note: currently we merge track 2 to track 1 without creating a new track
            track1.times += track2.times
            track1.bboxes += track2.bboxes
            track1.scores += track2.scores
            track1.role += track2.role
            track1.jn += track2.jn
            track1.jc += track2.jc
            track1.team += track2.team

            # update tracklets dictionary
            tracklets[idx2tid[track1_idx]] = track1
            tracklets.pop(idx2tid[track2_idx])

            # Remove the merged tracklet (track2) from the distance matrix
            Dist = np.delete(Dist, track2_idx, axis=0)  # Remove row for track2
            Dist = np.delete(Dist, track2_idx, axis=1)  # Remove column for track2
            # update idx2tid
            idx2tid = {idx: tid for idx, tid in enumerate(tracklets.keys())}

            # Update distance matrix only for the merged tracklet's row and column
            for idx in range(Dist.shape[0]):
                Dist[track1_idx, idx] = get_distance(idx2tid[track1_idx], idx2tid[idx], tracklets[idx2tid[track1_idx]],
                                                     tracklets[idx2tid[idx]], use_jn=use_jn)
                Dist[idx, track1_idx] = Dist[track1_idx, idx]  # Ensure symmetry

            seq2Dist[seq_name] = Dist  # used to display Dist

            # update mask
            diagonal_mask = np.eye(Dist.shape[0], dtype=bool)
            non_diagonal_mask = ~diagonal_mask
        else:
            # change distance between track pair to threshold
            Dist[track1_idx, track2_idx], Dist[track2_idx, track1_idx] = merge_dist_thres, merge_dist_thres
    # print("Finish merge while loop")
    return tracklets


def detect_id_switch(frames, roles, jns, teams, gap_len):
    """
    Detects identity switches within a tracklet using role, JN, JC
    Returns:
        bool: True if an identity switch is detected, otherwise False.
    """
    gaps = np.where(np.diff(frames) > gap_len)[0]
    if len(gaps) == 0:
        return [(0, len(frames))]
    split_indices = [0] + (gaps + 1).tolist() + [len(frames)]

    # Step 2: Check identity consistency across chunks
    final_splits = [0]

    for i in range(1, len(split_indices) - 1):
        start_prev = final_splits[-1]
        end_prev = split_indices[i]
        start_curr = split_indices[i]
        end_curr = split_indices[i + 1]

        maj_role_prev = majority_vote(roles[start_prev:end_prev])
        maj_jns_prev = majority_vote(jns[start_prev:end_prev], jersey=True)
        maj_team_prev = majority_vote(teams[start_prev:end_prev])

        maj_role_curr = majority_vote(roles[start_curr:end_curr])
        maj_jns_curr = majority_vote(jns[start_curr:end_curr], jersey=True)
        maj_team_curr = majority_vote(teams[start_curr:end_curr])

        # Check for inconsistency
        if (maj_role_prev != maj_role_curr or
                maj_jns_prev != maj_jns_curr or
                maj_team_prev != maj_team_curr):
            final_splits.append(start_curr)

    final_splits.append(len(frames))  # End of the last chunk

    # Return list of split segments as index ranges
    chunks = [(final_splits[i], final_splits[i + 1]) for i in range(len(final_splits) - 1)]
    return chunks


def split_tracklets(tmp_trklets, gap_len=None):
    new_id = max(tmp_trklets.keys()) + 1
    tracklets = defaultdict()
    # Splitting algorithm to process every tracklet in a sequence
    for tid in tqdm(sorted(list(tmp_trklets.keys())), total=len(tmp_trklets), desc="Splitting tracklets"):
        trklet = tmp_trklets[tid]
        logger.info(f"Processing track id: {trklet.track_id}")
        embs = np.stack(trklet.features)
        frames = np.array(trklet.times)
        bboxes = np.stack(trklet.bboxes)
        scores = np.array(trklet.scores)
        roles = np.array(trklet.role)
        jns = np.array(trklet.jn)
        jcs = np.array(trklet.jc)
        teams = np.array(trklet.team)
        # Perform DBSCAN clustering
        split_ranges = detect_id_switch(frames, roles, jns, teams, gap_len)
        if len(split_ranges) == 1:
            tracklets[tid] = trklet
        else:
            for rng in split_ranges:
                tmp_embs = embs[rng[0]: rng[1]]
                tmp_frames = frames[rng[0]: rng[1]]
                tmp_bboxes = bboxes[rng[0]: rng[1]]
                tmp_scores = scores[rng[0]: rng[1]]
                tmp_roles = roles[rng[0]: rng[1]]
                tmp_jns = jns[rng[0]: rng[1]]
                tmp_jcs = jcs[rng[0]: rng[1]]
                tmp_teams = teams[rng[0]: rng[1]]
                assert new_id not in tmp_trklets
                tracklets[new_id] = Tracklet(new_id, tmp_frames.tolist(), tmp_scores.tolist(), tmp_bboxes.tolist(),
                                             tmp_roles.tolist(), tmp_jns.tolist(), tmp_jcs.tolist(),
                                             tmp_teams.tolist(), tmp_embs.tolist())
                new_id += 1

    assert len(tracklets) >= len(tmp_trklets)
    return tracklets


def save_results(sct_output_path_txt, tracklets):
    """
    Saves the final tracklet results into a specified path.

    Args:
        sct_output_path (str): Path where the results will be saved.
        tracklets (dict): Dictionary of tracklets containing their final states.

    """
    # Create a new dictionary with reassigned keys
    new_tracklets = {}
    results = []

    # First pass: create results and new tracklets dictionary
    for i, old_tid in enumerate(sorted(tracklets.keys())):
        new_tid = i + 1  # This will be the new track ID
        track = tracklets[old_tid]

        # Store track with new ID
        new_tracklets[new_tid] = track

        # Create text file results with new ID
        for instance_idx, frame_id in enumerate(track.times):
            bbox = track.bboxes[instance_idx]
            role = track.role[instance_idx]
            jn = track.jn[instance_idx]
            jc = track.jc[instance_idx]
            team = track.team[instance_idx]
            results.append(
                [frame_id, new_tid, bbox[0], bbox[1], bbox[2], bbox[3], 1, role, jn, jc, team]
            )

    # Sort and save text results
    results = sorted(results, key=lambda x: x[0])
    txt_results = []
    for line in results:
        txt_results.append(
            f"{line[0]},{line[1]},{line[2]:.2f},{line[3]:.2f},{line[4]:.2f},{line[5]:.2f},{line[6]},{line[7]},{line[8]},{line[9]},{line[10]}\n"
        )

    with open(sct_output_path_txt, 'w') as f:
        f.writelines(txt_results)
    logger.info(f"save SCT results to {sct_output_path_txt}")


def refine_tracklets(cfg, datasets):
    idatr_cfg = cfg['IDATR']
    for split in datasets:
        data_path = os.path.join(cfg['DATA_DIR'], split)
        seqs_tracks = sorted(glob.glob(os.path.join(data_path, '**/*.pkl'), recursive=True))
        seq2Dist = dict()
        process_limit = 10000  # debug line, delete later
        for seq_idx, seq in enumerate(seqs_tracks):
            start = time.time()
            if seq_idx >= process_limit:  # debug line, delete later
                break  # debug line, delete later
            seq_name = seq.split('/')[-2]
            logger.info(f"Processing seq {seq_idx + 1} / {len(seqs_tracks)}")
            logger.info(f"Processing {seq_name}")
            with open(seq, 'rb') as pkl_f:
                tmp_trklets = pickle.load(pkl_f)  # dict(key:track id, value:tracklet)

            max_x_range, max_y_range = get_spatial_constraints(tmp_trklets, idatr_cfg['SPATIAL_FACTOR'])

            print(f"----------------Number of tracklets before splitting: {len(tmp_trklets)}----------------")
            splitTracklets = split_tracklets(tmp_trklets, gap_len=idatr_cfg['GAP_LEN'])

            Dist = get_distance_matrix(splitTracklets, use_jn=idatr_cfg['USE_JN'])
            print(f"----------------Number of tracklets before merging: {len(splitTracklets)}----------------")

            mergedTracklets = merge_tracklets(splitTracklets, seq2Dist, Dist, seq_name=seq_name, max_x_range=max_x_range,
                                            max_y_range=max_y_range, merge_dist_thres=idatr_cfg['MERGE_DIST_THRES'],
                                            use_jn=idatr_cfg['USE_JN'])

            end = time.time()
            print(f"Elapsed time: {end - start:.2f} seconds")
            print(f"----------------Number of tracklets after merging: {len(mergedTracklets)}----------------")
            new_sct_output_path_txt = os.path.join(os.path.dirname(seq), f'refined_{seq_name}.txt')
            save_results(new_sct_output_path_txt, mergedTracklets)



if __name__ == "__main__":
    import yaml
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    data_sets = cfg['DATA_SETS']
    refine_tracklets(cfg, data_sets)

