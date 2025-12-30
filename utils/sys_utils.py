import os
from torchvision import transforms
import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm


def read_yaml_file(yaml_path):
    class DictAsMember(dict):
        def __getattr__(self, name):
            value = self[name]
            if isinstance(value, dict):
                value = DictAsMember(value)
            return value

    with open(yaml_path, "r") as yaml_file:
        cfg = DictAsMember(yaml.load(yaml_file.read(), Loader=yaml.FullLoader))

    return cfg


def to_torch(np_array):
    # configs = parse_configs()
    tensor = torch.from_numpy(np_array).float()
    return torch.autograd.Variable(tensor, requires_grad=False)


def normalize(video_tensor):
    """
    Undoes mean/standard deviation normalization, zero to one scaling,
    and channel rearrangement for a batch of images.
    args:
        video_tensor: a (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    return normalize(video_tensor) / 255.0

def clip_video_opencv(input_video, start_frame, end_frame, output_path):
    # Open the input video
    cap = cv2.VideoCapture(input_video)

    # Get the original video's width, height, and FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 file

    # Open a VideoWriter to save the clipped video
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Set the start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Read and save frames from start_frame to end_frame
    for frame_idx in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    # Release resources
    cap.release()
    out.release()


def saving_rally_data(input_video: str, temp_folder: str, kpt_preds, kpt_templ):
    segment = 0
    img_num = 0
    cons = False
    start_frame = None

    # Open the input video with OpenCV to get the FPS
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Process keypoints and create segments
    for idx in tqdm(range(len(kpt_preds)), desc="Saving rallies"):
        num_keypoints = len(kpt_preds[idx])

        # If keypoints condition is met, accumulate frames for the segment
        if num_keypoints >= 10:
            cons = False

            # Start a new segment if it's the first frame
            if start_frame is None:
                start_frame = idx  # Set the start frame for the segment
                img_num = 0  # Reset the image number for the segment

            # Calculate and save the homography matrix
            pts_sel = np.array(kpt_preds[idx])
            template_sel = np.array(kpt_templ[idx])
            homography, _ = cv2.findHomography(
                pts_sel, template_sel, cv2.RANSAC, ransacReprojThreshold=10, maxIters=5000
            )

            save_dir = os.path.join(temp_folder, f"rally_{str(segment).zfill(6)}")
            os.makedirs(save_dir, exist_ok=True)
            homo_path = os.path.join(save_dir, f"{str(img_num).zfill(6)}.npy")

            # Save the homography matrix if valid
            if homography is not None and np.linalg.det(homography):
                np.save(homo_path, homography)
            else:
                print(f"homo is iden")
                np.save(homo_path, np.eye(3))

            img_num += 1

        else:
            # If a segment was in progress, finalize it
            if not cons and start_frame is not None:
                end_frame = idx - 1  # Set the end frame for the segment

                output_clip_path = os.path.join(temp_folder, f"rally_{str(segment).zfill(6)}.mp4")
                clip_video_opencv(input_video, start_frame, end_frame, output_clip_path)

                # Reset start_frame and increment the segment count
                start_frame = None
                segment += 1
                cons = True

    # Finalize the last segment if it was in progress
    if start_frame is not None:
        end_frame = len(kpt_preds) - 1  # Use the last frame of the input video

        output_clip_path = os.path.join(temp_folder, f"rally_{str(segment).zfill(6)}.mp4")
        # Save the final video segment using OpenCV
        clip_video_opencv(input_video, start_frame, end_frame, output_clip_path)


def read_video(video_path):
    frame_list = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Unable to open video file: {video_path}")

    while True:
        ret, img = cap.read()
        if not ret:
            break

        # save frame
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame_list.append(img)

    cap.release()
    return frame_list
