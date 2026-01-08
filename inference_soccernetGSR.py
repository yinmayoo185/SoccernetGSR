# from unsloth import FastVisionModel (Moved to local import)
import glob
import os
import os.path as osp
import time
from collections import Counter
from pathlib import Path
import json
import warnings

from reid.torchreid.utils import FeatureExtractor
from utils.sys_utils import to_torch, normalize
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from natsort import natsorted
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import AgglomerativeClustering
# Tracking and preprocessing imports
from yolox.EIoU_tracker.Deep_EIoU import Deep_EIoU
from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.tracking_utils.timer import Timer
from yolox.utils import postprocess, get_model_info
from yolox.utils.transforms import get_transforms

# LLaMA-Vision (jersey/role) related imports
from transformers import TextStreamer
from jersey_model.CLIPFinetune import CLIPFinetune
import clip

warnings.filterwarnings("ignore")


##########################################
#   PREDICTOR CLASS (with batched inference)
##########################################

class Predictor(object):
    def __init__(
            self,
            model,
            exp,
            trt_file=None,
            decoder=None,
            device=torch.device("cpu"),
            fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        # Original single-image inference (kept for compatibility)
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img_proc, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img_tensor = torch.from_numpy(img_proc).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img_tensor = img_tensor.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img_tensor)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
        return outputs, img_info

    def inference_batch(self, imgs):
        """
        Performs batched inference for a list of images.
        Args:
            imgs (list of np.array): Each image in BGR (as read by cv2).
        Returns:
            outputs: List (batch) of outputs (as returned by postprocess).
            img_infos: List of dictionaries containing original image info.
        """
        batch = []
        img_infos = []
        for img in imgs:
            info = {}
            h, w = img.shape[:2]
            info["height"] = h
            info["width"] = w
            info["raw_img"] = img
            proc_img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
            info["ratio"] = ratio
            batch.append(proc_img)
            img_infos.append(info)
        batch = np.stack(batch, axis=0)
        batch = torch.from_numpy(batch).float().to(self.device)
        if self.fp16:
            batch = batch.half()
        with torch.no_grad():
            outputs = self.model(batch)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
        return outputs, img_infos


##########################################
#   UTILITY FUNCTIONS
##########################################

def get_number(digit1, digit2):
    """
    10: null or invisible
    """
    if digit1 == 10:
        if digit2 == 10:
            number = 100
        else:
            number = digit2
    else:
        number = digit1 * 10 + digit2
    return number

def predict_role_and_jersey_batch_clip(images, jersey_model, device):
    """
    Process each image one by one to avoid batch size issues with TextStreamer.
    Returns a list of tuples: (role, jersey, jersey_color) for each image.
    """
    results = []
    role_mapping = {0: "Player", 1: "Goalkeeper", 2: "Referee", 3: "Ball", 4: "Other"}
    common_colors = ["Red", "Blue", "Green", "Yellow", "White", "Black", "Orange", "Purple", "Pink",
                     "Brown", "Grey", "Charcoal", "Neon yellow"]
    color_tokens = clip.tokenize(common_colors).to(device)
    color_features = jersey_model.clip_model.encode_text(color_tokens)
    color_features = F.normalize(color_features, p=2, dim=1)
    for image in images:
        # Preprocess image for CLIP
        image = jersey_model.preprocess(image).unsqueeze(0).to(device)
        outputs = jersey_model(image)
        role_logit = outputs['role_logits']
        digit1_logit = outputs['digit1_logits']
        digit2_logit = outputs['digit2_logits']
        color_embedding = outputs['color_embedding']
        # Normalize for cosine similarity
        color_embedding = F.normalize(color_embedding, p=2, dim=1)
        # Predictions
        _, role_pred = torch.max(role_logit, 1)
        role = role_mapping[role_pred.item()]
        _, digit1_pred = torch.max(digit1_logit, 1)
        _, digit2_pred = torch.max(digit2_logit, 1)
        number = get_number(digit1_pred.item(), digit2_pred.item())
        # Cosine similarity for each sample in batch
        similarities = torch.matmul(color_embedding.float(), color_features.T.float())
        color_idx = torch.argmax(similarities, dim=1)  # shape: [B]
        color = [common_colors[i] for i in color_idx.tolist()]
        results.append((role, number, color[0]))
    return results


def predict_role_and_jersey_batch(images, jersey_model, jersey_tokenizer,
                                  instruction="Classify Role, Jersey Number and Jersey Color"):
    """
    Process each image one by one to avoid batch size issues with TextStreamer.
    Returns a list of tuples: (role, jersey, jersey_color) for each image.
    """
    results = []
    for image in images:
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": image}
            ]}
        ]
        input_text = jersey_tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = jersey_tokenizer(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to("cuda")

        # Create a new TextStreamer for each image
        text_streamer = TextStreamer(jersey_tokenizer, skip_prompt=True)

        output = jersey_model.generate(
            **inputs,
            streamer=text_streamer,
            max_new_tokens=128,
            use_cache=True,
            temperature=1.0,
            min_p=0.1
        )
        prediction = jersey_tokenizer.decode(output[0], skip_special_tokens=True)
        extracted_text = prediction.split("\n")[-1].strip()
        parts = extracted_text.split(',')
        if len(parts) >= 3:
            role = parts[0].strip()
            try:
                jersey_number = int(parts[1].strip())
            except ValueError:
                jersey_number = 100
            jersey_color = parts[2].strip()
            results.append((role, jersey_number, jersey_color))
        else:
            results.append(("Player", 100, "Unknown"))
    return results

def print_model_info(model, name="Model"):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{name}: Total parameters: {total_params:,}, Trainable: {trainable_params:,}")

def print_gpu_memory_usage():
    if torch.cuda.is_available():
        mem_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
        mem_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f"GPU Memory Allocated: {mem_alloc:.2f} MB")
        print(f"GPU Memory Reserved: {mem_reserved:.2f} MB")

def bbox_distance(bbox1, bbox2):
    """Calculates the Euclidean distance between the centers of two bounding boxes."""
    center_x1, center_y1 = bbox1[:2] + bbox1[2:] / 2
    center_x2, center_y2 = bbox2[:2] + bbox2[2:] / 2
    return np.sqrt((center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2)


def pixel_to_meter_all(x_bottom_left, y_bottom_left, x_bottom_right, y_bottom_right, x_bottom_middle, y_bottom_middle,
                       center_x_px, center_y_px):
    """
    Convert pixel coordinates back to meters.
    """
    x_left = x_bottom_left - center_x_px
    y_left = y_bottom_left - center_y_px
    x_right = x_bottom_right - center_x_px
    y_right = y_bottom_right - center_y_px
    x_middle = x_bottom_middle - center_x_px
    y_middle = y_bottom_middle - center_y_px

    return x_left, y_left, x_right, y_right, x_middle, y_middle


##########################################
#   FUNCTIONS FOR ROLE, JERSEY NUMBER & JERSEY COLOR CLASSIFICATION
##########################################

def unify_team_assignments(track_df):
    """
    Enforce a single (consistent) team assignment across all frames for each track_id,
    but only for rows where role == 'player'.

    - If a track_id has at least one frame labeled 0 or 1 (among 'player' rows), pick the majority.
    - If all 'player' rows are -1 (or there are no 'player' rows), remain -1.
    - Non-player rows for the same track_id are not modified.
    """
    # Group by track_id, but we'll look only at the subset of rows where role == 'player'
    grouped = track_df.groupby("track_id")
    final_team_map = {}

    for tid, group in grouped:
        # Filter to only rows with role='player'
        player_rows = group[group["role"].str.lower() == "player"]
        if player_rows.empty:
            # No player rows for this track_id => do nothing
            continue

        # Among those 'player' rows, look only at team in {0,1}
        assigned = player_rows[player_rows["team"].isin([0, 1])]["team"]
        if len(assigned) == 0:
            # No frames were assigned => remain -1 for all player rows
            final_team_map[tid] = -1
        else:
            # Majority vote among {0,1}
            counts = assigned.value_counts()
            final_team = counts.idxmax()  # the team with the highest count
            final_team_map[tid] = final_team

    # Now overwrite 'team' only for role='player' rows
    for tid, team_val in final_team_map.items():
        is_player_mask = (
            (track_df["track_id"] == tid)
            & (track_df["role"].str.lower() == "player")
        )
        track_df.loc[is_player_mask, "team"] = team_val

    return track_df

def predict_role_and_jersey(image, jersey_model, jersey_tokenizer, instruction="Classify Role, Jersey Number and Jersey Color"):
    """
    Predict the role, jersey number and jersey color for a single image using the LLaMA vision model.
    The image is expected as a PIL image.
    """
    try:
        # Prepare the input conversation (as required by the model)
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": image}  # Pass the PIL image directly
            ]}
        ]
        input_text = jersey_tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = jersey_tokenizer(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to("cuda")
        text_streamer = TextStreamer(jersey_tokenizer, skip_prompt=True)
        output = jersey_model.generate(
            **inputs,
            streamer=text_streamer,
            max_new_tokens=128,
            use_cache=True,
            temperature=1.0,
            min_p=0.1
        )
        prediction = jersey_tokenizer.decode(output[0], skip_special_tokens=True)
        extracted_text = prediction.split("\n")[-1].strip()
        parts = extracted_text.split(',')
        if len(parts) >= 3:
            role = parts[0].strip()
            try:
                jersey_number = int(parts[1].strip())
            except ValueError:
                jersey_number = 100
            jersey_color = parts[2].strip()
            return role, jersey_number, jersey_color
        else:
            return "Player", 100, "Unknown"
    except Exception as e:
        # On error, return default predictions
        return "Player", 100, "Unknown"

##########################################
#   VIDEO DATASET FOR BATCH LOADING
##########################################

class VideoDataset(Dataset):
    def __init__(self, video_path):
        self.img_paths = natsorted(glob.glob(os.path.join(video_path, '*.jpg')))
        # Exclude files ending with '_result.jpg'
        self.img_paths = [p for p in self.img_paths]
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        path = self.img_paths[idx]
        frame = cv2.imread(path)
        frame_id = int(Path(path).stem)
        return frame, frame_id, path

def global_color_team_assignment(track_df):
    """
    1) Gather all color predictions among rows where role=='player'.
    2) Find the two most common colors in the entire video.
    3) Assign those colors to team=0 and team=1.
       All leftover colors => team=-1.
    4) Overwrite the 'team' column accordingly.
    """
    # Ensure jersey colors are normalized
    track_df["color"] = track_df["color"].str.strip().str.lower()

    # Filter to only 'player' rows
    player_mask = track_df["role"].str.lower() == "player"
    players = track_df[player_mask].copy()

    if players.empty:
        # No player rows => do nothing
        return track_df

    # Count color frequencies among players
    color_counts = players["color"].value_counts()
    if len(color_counts) == 1:
        # Only one color in the entire video => everything is team=0
        main_color = color_counts.index[0]
        track_df.loc[player_mask & (track_df["color"] == main_color), "team"] = 0
    elif len(color_counts) >= 2:
        # Take the top 2 most common colors
        color_a, color_b = color_counts.index[0], color_counts.index[1]
        # color_a => 0, color_b => 1
        track_df.loc[player_mask & (track_df["color"] == color_a), "team"] = 0
        track_df.loc[player_mask & (track_df["color"] == color_b), "team"] = 1

        # For leftover colors, set team = -1
        leftover_mask = player_mask & ~track_df["color"].isin([color_a, color_b])
        track_df.loc[leftover_mask, "team"] = -1

    return track_df


##########################################
#   GSRPipeline CLASS
##########################################

class GSRPipeline:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.track_cfg = cfg['TRACKER']
        self.reid_cfg = cfg['REID']
        self.sfr_cfg = cfg['SFR']
        self.llama_cfg = cfg['LLAMA']
        self.jersey_mode = cfg.get('JERSEY_MODE', 'LLAMA')

        self._init_models()

    def _init_models(self):
        # 1) Detection Model (YOLOX)
        name = "yolo-x"
        exp = get_exp(self.track_cfg['EXP_FILE'], name)
        if self.track_cfg['CONF'] is not None:
            exp.test_conf = self.track_cfg['CONF']
        if self.track_cfg['NMS_THRESH'] is not None:
            exp.nmsthre = self.track_cfg['NMS_THRESH']
        if self.track_cfg['TSIZE'] is not None:
            exp.test_size = (800, 1440)

        detection_model = exp.get_model().to(self.device)
        detection_model.eval()

        ckpt_file = self.track_cfg['MODEL_PATH']
        ckpt = torch.load(ckpt_file, weights_only=False)
        detection_model.load_state_dict(ckpt["model"])

        self.predictor = Predictor(
            model=detection_model,
            exp=exp,
            trt_file=None,
            decoder=None,
            device=self.device,
            fp16=self.track_cfg['FP16']
        )

        # 2) Re‐ID Model (OSNet)
        self.reid_model = FeatureExtractor(
            model_name='osnet_x1_0',
            model_path=self.reid_cfg['MODEL_PATH'],
            device=str(self.device)
        )
        
        # 3) SFR Keypoint Model (Removed as we use pre-computed .npy files)
        # self.kpts_model = ... 

        # 4) Jersey/Role Model (LLaMA or CLIP)
        if self.jersey_mode == 'CLIP':
            clip_cfg = self.cfg['CLIP']
            model_dir = clip_cfg['MODEL_PATH']
            self.jersey_model = CLIPFinetune()
            self.jersey_model.load_state_dict(torch.load(model_dir, weights_only=False))
            self.jersey_model = self.jersey_model.to(self.device)
            self.jersey_model.eval()
            self.jersey_tokenizer = None
        else:
            from unsloth import FastVisionModel
            # Default to LLAMA
            model_dir = self.llama_cfg['MODEL_PATH']
            self.jersey_model, self.jersey_tokenizer = FastVisionModel.from_pretrained(
                model_name=model_dir,
                load_in_4bit=True
            )
            FastVisionModel.for_inference(self.jersey_model)

    def process_video(self, video_path, vis_folder, batch_size=4):
        # Updated key names to include "color"
        key_names = ["frame", "track_id", "x", "y", "w", "h", "score", "role", "jersey", "color", "team"]
        # Updated court names with the desired order.
        court_names_meter = ["frame", "track_id", "x_left", "y_left", "x_right", "y_right", "x_middle", "y_middle", "role", "jersey", "color", "team"]
        court_names_pixel = ["frame", "track_id", "x", "y", "role", "jersey", "color", "team"]
        instruction = "Classify Role, Jersey Number and Jersey Color"

        results = {key: [] for key in key_names}
        player_position_results_meter = {key: [] for key in court_names_meter}
        player_position_results_pixel = {key: [] for key in court_names_pixel}

        # Prepare SFR template info
        template_kpts = np.load(self.sfr_cfg['KPTS_PATH'])
        template = cv2.imread(self.sfr_cfg['TEMPLATE_PATH'])
        # Use a transform that resizes images to a fixed size (e.g., 256x128) for re‑ID features
        im_transforms = get_transforms(image_size=(256, 128)) 
        center_x_px = template.shape[1] / 2
        center_y_px = template.shape[0] / 2

        # Create DataLoader for batched image reading
        dataset = VideoDataset(video_path)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, collate_fn=lambda x: x)

        tracker = Deep_EIoU(self.track_cfg, frame_rate=25)
        timer = Timer()

        for batch in tqdm(dataloader, desc=f"Processing {os.path.basename(video_path)}"):
            # Each batch is a list of tuples: (frame, frame_id, path)
            frames, frame_ids, paths = zip(*batch)
            # Run batched inference for detection
            outputs_batch, img_infos = self.predictor.inference_batch(list(frames))
            for i in range(len(frames)):
                # Initialize per-frame variables before processing detections for the frame.
                batch_images = []
                batch_indices = []
                detection_predictions = {}

                frame = frames[i]
                frame_id = frame_ids[i]
                img_info = img_infos[i]

                img_path = paths[i]
                homo_path = os.path.splitext(img_path)[0] + ".npy"
                if os.path.exists(homo_path):
                    homography = np.load(homo_path)
                else:
                    # print(f"Homography file {homo_path} not found for frame {frame_id}. Using identity.")
                    homography = np.eye(3, dtype=np.float32)

                height, width, _ = frame.shape
                output = outputs_batch[i]
                if output is None or len(output) == 0:
                    continue

                # Convert detection tensor to numpy array and adjust scale
                det = output.cpu().detach().numpy()
                scale = min(1440 / width, 800 / height)
                det /= scale
                rows_to_remove = np.any(det[:, 0:4] < 1, axis=1)
                det = det[~rows_to_remove]

                # Crop detection boxes for re‑ID (batched per frame)
                cropped_imgs = [frame[max(0, int(y1)):min(height, int(y2)),
                                        max(0, int(x1)):min(width, int(x2))]
                                for x1, y1, x2, y2, _, _, _ in det]
                if not cropped_imgs:
                    continue

                # Obtain re‑ID features (even though we will not use them for team clustering)
                # Convert crops from BGR to RGB for OSNet/Transforms
                images_tensor = []
                for crop in cropped_imgs:
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    images_tensor.append(im_transforms(image=crop_rgb)['image'])
                
                images_tensor = torch.stack(images_tensor, dim=0).to(self.device, non_blocking=True)
                embs = self.reid_model(images_tensor)
                embs = F.normalize(embs, p=2, dim=1)
                embs = embs.cpu().detach().numpy()

                # Update tracker to obtain online targets
                online_targets = tracker.update(det, embs)
                detection_info = [(j, t, t.last_tlwh, t.track_id) for j, t in enumerate(online_targets)]

                for j, t in enumerate(online_targets):
                    tlwh = t.last_tlwh
                    tid = t.track_id
                    # Filter out small boxes
                    if tlwh[2] * tlwh[3] <= self.track_cfg['MIN_BOX_AREA']:
                        continue

                    x, y, w, h = tlwh
                    x_e = x + w
                    y_e = y + h

                    # Crop the detection region for classification
                    x1_crop = int(max(0, x))
                    y1_crop = int(max(0, y))
                    x2_crop = int(min(width, x_e))
                    y2_crop = int(min(height, y_e))

                    if x2_crop > x1_crop and y2_crop > y1_crop:
                        crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]
                        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        batch_images.append(crop_pil)
                        batch_indices.append(j)
                    else:
                        detection_predictions[j] = ("Player", 100, "Unknown")

                # Run batch inference for jersey classification if any crops are available.
                if batch_images:
                    if self.jersey_mode == 'CLIP':
                        batch_results = predict_role_and_jersey_batch_clip(batch_images, self.jersey_model, self.device)
                    else:
                        batch_results = predict_role_and_jersey_batch(batch_images, self.jersey_model, self.jersey_tokenizer,
                                                                      instruction)
                    for idx, result in zip(batch_indices, batch_results):
                        detection_predictions[idx] = result

                # --- Build temporary results using the classification predictions ---
                temp_results = []
                player_candidates = []  # list of (target_index, predicted_color)
                for (j, t, tlwh, tid) in detection_info:
                    # If no prediction exists, assign defaults.
                    role, jersey, jersey_color = detection_predictions.get(j, ("Player", 100, "Unknown"))
                    normalized_color = jersey_color.strip().lower()
                    temp_result = [frame_id, tid, tlwh[0], tlwh[1], tlwh[2], tlwh[3], t.score, role, jersey,
                                   normalized_color, -1]
                    temp_results.append((j, temp_result))
                    if role.lower() == "player":
                        player_candidates.append((j, normalized_color))

                # Dynamic team assignment based solely on predicted jersey color text
                if player_candidates:
                    colors = [color for (_, color) in player_candidates]
                    color_counts = Counter(colors)
                    if len(color_counts) == 1:
                        # Only one unique color detected: assign team 0 for all.
                        for (j, color) in player_candidates:
                            for (k, res) in temp_results:
                                if k == j:
                                    res[-1] = 0
                    else:
                        # Two or more unique colors; select the two most common.
                        most_common = color_counts.most_common(2)
                        color_a, _ = most_common[0]
                        color_b, _ = most_common[1]
                        for (j, color) in player_candidates:
                            # Assign team based on which common color the prediction matches.
                            team_id = 0 if color == color_a else (1 if color == color_b else -1)
                            for (k, res) in temp_results:
                                if k == j:
                                    res[-1] = team_id

                embs_for_frame = {i: embs[i] for i in range(len(online_targets))}

                # b) Gather embeddings for each assigned team (0 and 1), and leftover
                team0_embs = []
                team1_embs = []
                leftover_players = []  # list of (detection_index, temp_result)

                for (detection_idx, res_list) in temp_results:
                    assigned_team = res_list[-1]  # the 'team' field is last in your temp_result
                    role_str = str(res_list[7]).lower()  # res_list[7] = role
                    if role_str == "player":
                        if assigned_team == 0:
                            team0_embs.append(embs_for_frame[detection_idx])
                        elif assigned_team == 1:
                            team1_embs.append(embs_for_frame[detection_idx])
                        elif assigned_team == -1:
                            leftover_players.append((detection_idx, res_list))

                # c) If we have at least one detection in team=0 and at least one in team=1, compute centroids
                if len(team0_embs) > 0 and len(team1_embs) > 0:
                    centroid0 = np.mean(team0_embs, axis=0)
                    centroid1 = np.mean(team1_embs, axis=0)

                    for (detection_idx, res_list) in leftover_players:
                        emb = embs_for_frame[detection_idx]

                        # Example: use Euclidean distance
                        dist0 = np.linalg.norm(emb - centroid0)
                        dist1 = np.linalg.norm(emb - centroid1)

                        # Assign whichever is closer
                        if dist0 < dist1:
                            res_list[-1] = 0
                        else:
                            res_list[-1] = 1

                # Append temporary results into the final results dictionary.
                for (_, res) in temp_results:
                    for key, data in zip(key_names, res):
                        results[key].append(data)

                # Compute court coordinates for each online target and add them to the output.
                for (j, res) in temp_results:
                    t = online_targets[j]
                    tlwh = t.last_tlwh
                    x, y, w, h = tlwh
                    x_e = x + w
                    y_e = y + h
                    x_center = x + ((x_e - x) // 2)
                    x_bottom_left = x
                    y_bottom_left = y_e
                    x_bottom_right = x_e
                    y_bottom_right = y_e
                    x_bottom_middle = x_center
                    y_bottom_middle = y_e
                    pitch_position = np.array([[x_bottom_left, y_bottom_left, 1.],
                                               [x_bottom_right, y_bottom_right, 1.],
                                               [x_bottom_middle, y_bottom_middle, 1.]])
                    pitch_position_img = pitch_position.T
                    transformed_pitch = np.dot(homography, pitch_position_img)
                    player_position_pitch = (transformed_pitch[:2] / transformed_pitch[2]).T
                    player_position_pitch = np.array(player_position_pitch)
                    x_left, y_left, x_right, y_right, x_middle, y_middle = pixel_to_meter_all(
                        player_position_pitch[0][0],
                        player_position_pitch[0][1],
                        player_position_pitch[1][0],
                        player_position_pitch[1][1],
                        player_position_pitch[2][0],
                        player_position_pitch[2][1],
                        center_x_px, center_y_px)
                    # Build court outputs.
                    # For pixel: [frame, track_id, x, y, role, jersey, color, team]
                    court_result_pixel = [frame_id, t.track_id,
                                          int(player_position_pitch[2][0]),
                                          int(player_position_pitch[2][1]),
                                          res[7], res[8], res[9], res[10]]
                    # For meter: [frame, track_id, x_left, y_left, x_right, y_right, x_middle, y_middle, role, jersey, color, team]
                    court_result_meter = [frame_id, t.track_id,
                                          x_left, y_left, x_right, y_right, x_middle, y_middle,
                                          res[7], res[8], res[9], res[10]]
                    for key, data in zip(court_names_meter, court_result_meter):
                        player_position_results_meter[key].append(data)
                    for key, data in zip(court_names_pixel, court_result_pixel):
                        player_position_results_pixel[key].append(data)
                timer.toc()

        # ---------------------------
        # Convert dict -> DataFrame
        track_df = pd.DataFrame(results)
        pixel_df = pd.DataFrame(player_position_results_pixel)
        meter_df = pd.DataFrame(player_position_results_meter)

        # ---------------------------
        # Apply global color -> team assignment
        # (Optional, if you want consistent color->team in all 3 dataframes)
        track_df = global_color_team_assignment(track_df)
        pixel_df = global_color_team_assignment(pixel_df)
        meter_df = global_color_team_assignment(meter_df)

        # Return them
        return track_df, pixel_df, meter_df, list(dataset.img_paths)

    def run(self, data_sets):
        data_dir = self.cfg['DATA_DIR']
        output_dir = self.cfg['IMG_SAVE_DIR']
        
        for dataset_name in data_sets:
            dataset_path = os.path.join(data_dir, dataset_name)
            # Find all video folders (e.g., SNGS-xxx)
            # Assuming structure: data_dir/dataset_name/SNGS-xxx/img1/*.jpg
            # We look for folders containing 'img1'
            video_folders = glob.glob(os.path.join(dataset_path, "*"))
            video_folders = [f for f in video_folders if os.path.isdir(f)]
            
            for video_folder in video_folders:
                video_name = os.path.basename(video_folder)
                img1_path = os.path.join(video_folder, "img1")
                if not os.path.exists(img1_path):
                    continue
                
                print(f"Processing video: {video_name}")
                
                # Define output directory for this video
                video_output_dir = os.path.join(self.cfg['DATA_DIR'], dataset_name, video_name)
                os.makedirs(video_output_dir, exist_ok=True)
                
                # Run processing
                track_df, initial_positions_pixel_df, initial_positions_meter_df, frame_list = self.process_video(
                    img1_path, 
                    video_output_dir, 
                    batch_size=4
                )
                
                # Unify team assignments
                track_df = unify_team_assignments(track_df)
                initial_positions_pixel_df = unify_team_assignments(initial_positions_pixel_df)
                initial_positions_meter_df = unify_team_assignments(initial_positions_meter_df)

                # Define filenames
                # Using video_output_dir as the output_dir context
                interpolate_track_filename = osp.join(video_output_dir, f'interpolate_{Path(video_output_dir).stem}.txt')
                interpolate_court_pixel_filename = osp.join(video_output_dir, f'court_pixel_{Path(video_output_dir).stem}.txt')
                interpolate_court_meter_filename = osp.join(video_output_dir, f'court_meter_{Path(video_output_dir).stem}.txt')

                # Format frame column
                track_df['frame'] = track_df['frame'].astype(str).str.zfill(6)
                initial_positions_pixel_df['frame'] = initial_positions_pixel_df['frame'].astype(str).str.zfill(6)
                initial_positions_meter_df['frame'] = initial_positions_meter_df['frame'].astype(str).str.zfill(6)

                # Save results
                initial_positions_pixel_df.to_csv(interpolate_court_pixel_filename, index=False, header=False)
                initial_positions_meter_df.to_csv(interpolate_court_meter_filename, index=False, header=False)
                track_df.to_csv(interpolate_track_filename, index=False, header=False)
                
                print(f"Saved tracking results to {interpolate_track_filename}")


if __name__ == "__main__":
    import yaml
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print_gpu_memory_usage()

    pipeline = GSRPipeline(cfg, device)
    
    # Define datasets to process
    data_sets = ['test'] # or ['challenge'], ['validation']
    
    pipeline.run(data_sets)
