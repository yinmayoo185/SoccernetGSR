# This script produces tracklets given tracking results and original sequence frame as RGB images.
import argparse
from torchreid.utils import FeatureExtractor

import os
from tqdm import tqdm
from loguru import logger
from PIL import Image

import pickle
import numpy as np
import glob

import torch
import torchvision.transforms as T

from Tracklet import Tracklet


def generate_tracklets(cfg, datasets):
    idatr_cfg = cfg['IDATR']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path=idatr_cfg['MODEL_PATH'],
        device=device
    )
    val_transforms = T.Compose([
    T.Resize([256, 128]),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    for split in datasets:
        data_path = os.path.join(cfg['DATA_DIR'], split)
        seqs = sorted(glob.glob(os.path.join(data_path, '**/*rmved_SNGS*.txt'), recursive=True))
        for s_id, seq in tqdm(enumerate(seqs, 1), total=len(seqs), desc='Processing Seqs'):
            seq_name = seq.split("_")[-1].replace('.txt', '')
            seq_dir = os.path.dirname(seq)
            imgs = sorted(glob.glob(os.path.join(seq_dir, 'img1', '*.jpg')))
            track_res = np.genfromtxt(seq, dtype=str, delimiter=',', encoding="utf-8")
            last_frame = int(track_res[-1][0])
            seq_tracks = {}

            for frame_id in range(1, last_frame + 1):
                if frame_id % 100 == 0:
                    logger.info(f'Processing frame {frame_id}/{last_frame}')
                inds = track_res[:, 0].astype(int) == frame_id
                frame_res = track_res[inds]
                img = Image.open(imgs[int(frame_id) - 1])

                input_batch = None  # input batch to speed up processing
                tid2idx = {}

                for idx, (frame, track_id, l, t, w, h, score, role, jn, jc, team) in enumerate(frame_res):  # jn: jersey number, jc: jersey color
                    # Update tracklet with detection
                    frame, track_id, l, t, w, h, score = \
                        int(frame), int(track_id), float(l), float(t), float(w), float(h), float(score)
                    bbox = [l, t, w, h]
                    if track_id not in seq_tracks:
                        seq_tracks[track_id] = Tracklet(track_id, frame, score, bbox, role, jn, jc, team)
                    else:
                        seq_tracks[track_id].append_det(frame, score, bbox, role, jn, jc, team)
                    tid2idx[track_id] = idx

                    im = img.crop((l, t, l + w, t + h)).convert('RGB')
                    im = val_transforms(im).unsqueeze(0)
                    if input_batch is None:
                        input_batch = im
                    else:
                        input_batch = torch.cat([input_batch, im], dim=0)

                if input_batch is not None:
                    features = extractor(input_batch)  # len(features) == len(frame_res)
                    feats = features.cpu().detach().numpy()

                    # update tracklets with feature
                    for tid, idx in tid2idx.items():
                        feat = feats[tid2idx[tid]]
                        feat /= np.linalg.norm(feat)
                        seq_tracks[tid].append_feat(feat)
                else:
                    print(f"No detection at frame: {frame_id}")

            # save seq_tracks into pickle file
            track_output_path = os.path.join(seq_dir, f'{seq_name}.pkl')
            with open(track_output_path, 'wb') as f:
                pickle.dump(seq_tracks, f)
            logger.info(f"save tracklets info to {track_output_path}")


if __name__ == "__main__":
    import yaml
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    data_sets = cfg['DATA_SETS']
    generate_tracklets(cfg, data_sets)