#!/usr/bin/env python
"""
Combined Tennis Court Detection Script
All required components combined into a single file for easy deployment.
"""

import glob
import os
import time
import math
from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchvision.models import EfficientNet_V2_S_Weights
from scipy.optimize import least_squares
from skimage.morphology import skeletonize

# ========================= UTILITY FUNCTIONS =========================


def print_line(name, length=80):
    """Print a formatted line with the given name"""
    length_name = len(name) + 2
    rest = length - length_name
    left = rest // 2
    right = rest - left
    print("\n{}[{}]{}".format(left * "-", name, right * "-"))


# ========================= MODEL COMPONENTS =========================


def weights_init(m):
    """Initialize filters with Gaussian random weights"""
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


class _NonLocalBlockND(nn.Module):
    """Non-local block for capturing long-range dependencies"""

    def __init__(
        self,
        in_channels,
        inter_channels=None,
        dimension=3,
        sub_sample=True,
        bn_layer=True,
    ):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(
                    in_channels=self.inter_channels,
                    out_channels=self.in_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
                bn(self.in_channels),
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(
                in_channels=self.inter_channels,
                out_channels=self.in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.phi = conv_nd(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, return_nl_map=False):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z


class NONLocalBlock2D(_NonLocalBlockND):
    """2D Non-local block"""

    def __init__(
        self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True
    ):
        super(NONLocalBlock2D, self).__init__(
            in_channels,
            inter_channels=inter_channels,
            dimension=2,
            sub_sample=sub_sample,
            bn_layer=bn_layer,
        )


class DoubleConv(nn.Module):
    """Double convolution block"""

    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.channel_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        x = self.conv(input)
        return x + self.channel_conv(input)


class Attention_block(nn.Module):
    """Attention gate block"""

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class EfficientNet(nn.Module):
    """EfficientNet backbone"""

    def __init__(self, n_channels):
        super(EfficientNet, self).__init__()

        self.efficient_model = torchvision.models.efficientnet_v2_s(
            weights=EfficientNet_V2_S_Weights.DEFAULT
        )

        self.layer1 = self.efficient_model.features[0:2]
        self.layer2 = self.efficient_model.features[2:3]
        self.layer3 = self.efficient_model.features[3:4]
        self.layer4 = self.efficient_model.features[4:6]
        self.layer5 = self.efficient_model.features[6:9]

        self.nl_1 = NONLocalBlock2D(in_channels=64)
        self.nl_2 = NONLocalBlock2D(in_channels=160)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x3 = self.nl_1(x3)
        x4 = self.layer4(x3)
        x4 = self.nl_2(x4)
        x5 = self.layer5(x4)

        return x1, x2, x3, x4, x5


class UpConvBlock(nn.Module):
    """Upsampling convolution block"""

    def __init__(self, ch_in, ch_out, size):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(size=size, mode="bilinear", align_corners=True),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)


class Unet(nn.Module):
    """U-Net model with EfficientNet backbone and attention gates"""

    def __init__(self, out_ch, num_lines=21):
        super(Unet, self).__init__()
        self.pretrained_net = EfficientNet(3)
        self.up6 = UpConvBlock(ch_in=1280, ch_out=512, size=(24, 24))
        self.att1 = Attention_block(512, 160, 160)
        self.conv6 = DoubleConv(160 + 512, 512)
        self.up7 = UpConvBlock(ch_in=512, ch_out=256, size=(48, 48))
        self.att2 = Attention_block(256, 64, 64)
        self.conv7 = DoubleConv(256 + 64, 256)
        self.up8 = UpConvBlock(ch_in=256, ch_out=128, size=(96, 96))
        self.att3 = Attention_block(128, 48, 48)
        self.conv8 = DoubleConv(128 + 48, 128)
        self.up9 = UpConvBlock(ch_in=128, ch_out=64, size=(192, 192))
        self.att4 = Attention_block(64, 24, 24)
        self.conv9 = DoubleConv(64 + 24, 64)
        self.up10 = UpConvBlock(ch_in=64, ch_out=64, size=(384, 384))
        self.att5 = Attention_block(64, 3, 1)
        self.conv10 = DoubleConv(67, 64)
        self.conv11 = nn.Conv2d(64, out_ch, kernel_size=1)
        self.line_head = nn.Conv2d(64, num_lines, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # Weights initialize
        self.conv6.apply(weights_init)
        self.conv7.apply(weights_init)
        self.conv8.apply(weights_init)
        self.conv9.apply(weights_init)
        self.conv10.apply(weights_init)
        self.conv11.apply(weights_init)
        self.line_head.apply(weights_init)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.pretrained_net(x)

        up_6 = self.up6(x5)
        x4 = self.att1(up_6, x4)
        merge6 = torch.cat([up_6, x4], dim=1)
        c6 = self.conv6(merge6)

        up_7 = self.up7(c6)
        x3 = self.att2(up_7, x3)
        merge7 = torch.cat([up_7, x3], dim=1)
        c7 = self.conv7(merge7)

        up_8 = self.up8(c7)
        x2 = self.att3(up_8, x2)
        merge8 = torch.cat([up_8, x2], dim=1)
        c8 = self.conv8(merge8)

        up_9 = self.up9(c8)
        x1 = self.att4(up_9, x1)
        merge9 = torch.cat([up_9, x1], dim=1)
        c9 = self.conv9(merge9)

        up_10 = self.up10(c9)
        x = self.att5(up_10, x)
        merge10 = torch.cat([up_10, x], dim=1)
        c10 = self.conv10(merge10)
        c11 = self.conv11(c10)

        out = self.sigmoid(c11)
        line_output = self.line_head(c10)
        line_out = self.sigmoid(line_output)

        return out, line_out


# ========================= INFERENCE FUNCTIONS =========================


def get_max_preds_torch(batch_heatmaps):
    """Get predictions from score maps using PyTorch"""
    assert isinstance(
        batch_heatmaps, torch.Tensor
    ), "batch_heatmaps should be a torch.Tensor"
    assert (
        batch_heatmaps.dim() == 4
    ), f"batch_heatmaps should be 4-dimensional, found {batch_heatmaps.dim()} dimensions"

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    width = batch_heatmaps.size(3)

    heatmaps_reshaped = batch_heatmaps.view(batch_size, num_joints, -1)
    idx = torch.argmax(heatmaps_reshaped, dim=2)
    maxvals = torch.max(heatmaps_reshaped, dim=2)[0]

    maxvals = maxvals.view(batch_size, num_joints, 1)
    idx = idx.view(batch_size, num_joints, 1)

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = preds[:, :, 0] % width
    preds[:, :, 1] = torch.floor(preds[:, :, 1] / width)

    pred_mask = (maxvals > 0.0).repeat(1, 1, 2).float()
    preds *= pred_mask

    return preds, maxvals


def taylor(hm, coord):
    """Taylor expansion for sub-pixel accuracy"""
    heatmap_height = hm.shape[0]
    heatmap_width = hm.shape[1]
    px = int(coord[0])
    py = int(coord[1])
    if 1 < px < heatmap_width - 2 and 1 < py < heatmap_height - 2:
        dx = 0.5 * (hm[py][px + 1] - hm[py][px - 1])
        dy = 0.5 * (hm[py + 1][px] - hm[py - 1][px])
        dxx = 0.25 * (hm[py][px + 2] - 2 * hm[py][px] + hm[py][px - 2])
        dxy = 0.25 * (
            hm[py + 1][px + 1]
            - hm[py - 1][px + 1]
            - hm[py + 1][px - 1]
            + hm[py - 1][px - 1]
        )
        dyy = 0.25 * (hm[py + 2 * 1][px] - 2 * hm[py][px] + hm[py - 2 * 1][px])
        derivative = np.matrix([[dx], [dy]])
        hessian = np.matrix([[dxx, dxy], [dxy, dyy]])
        if dxx * dyy - dxy**2 != 0:
            hessianinv = hessian.I
            offset = -hessianinv * derivative
            offset = np.squeeze(np.array(offset.T), axis=0)
            coord += offset
    return coord


def gaussian_blur_torch(hm, kernel_size):
    """Apply Gaussian blur using PyTorch"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(hm, np.ndarray):
        hm = torch.from_numpy(hm).float()
    hm = hm.to(device)

    batch_size, num_joints, height, width = hm.shape
    padding = (kernel_size - 1) // 2

    # Create Gaussian kernel
    sigma = kernel_size / 6.0
    x = torch.arange(-padding, padding + 1, device=device).float()
    y = x.view(-1, 1)
    x_grid, y_grid = x.repeat(kernel_size, 1), y.repeat(1, kernel_size)
    kernel = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()

    # Reshape kernel to [out_channels, in_channels/groups, kH, kW]
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(num_joints, 1, 1, 1)

    # Apply padding
    hm_padded = F.pad(
        hm, (padding, padding, padding, padding), mode="constant", value=0
    )

    # Store original maximum values
    origin_max = hm.amax(dim=(2, 3), keepdim=True)

    # Apply Gaussian filter using group convolution
    hm_blurred = F.conv2d(hm_padded, kernel, groups=num_joints)

    # Normalize to maintain the original maximum value
    max_blurred = (
        hm_blurred.amax(dim=(2, 3), keepdim=True) + 1e-6
    )  # Avoid division by zero
    hm_blurred = hm_blurred * (origin_max / max_blurred)

    return hm_blurred.cpu().numpy()


def get_final_preds_torch(hm):
    """Get final predictions with post-processing"""
    coords, maxvals = get_max_preds_torch(hm)
    coords = coords.cpu().numpy()
    maxvals = maxvals.cpu().numpy()

    heatmap_height = hm.shape[2]
    heatmap_width = hm.shape[3]
    BLUR_KERNEL = 11

    # post-processing
    hm = gaussian_blur_torch(hm, BLUR_KERNEL)
    hm = np.maximum(hm, 1e-10)
    hm = np.log(hm)
    for n in range(coords.shape[0]):
        for p in range(coords.shape[1]):
            coords[n, p] = taylor(hm[n][p], coords[n][p])

    preds = coords.copy()

    return preds, maxvals


# ========================= TRANSFORMS =========================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _get_img_transforms(crop_dim=384, is_eval=False):
    """Get image transforms for training or evaluation"""
    p = 0.0 if is_eval else 0.3  # Probability for color jittering only during training

    # Geometric transforms that will be applied to both images and masks
    geometric_transforms = []
    if not is_eval:
        print("Using train transforms")
        geometric_transforms.append(
            transforms.RandomChoice(
                [
                    transforms.RandomPerspective(
                        distortion_scale=0.3, p=0.3
                    ),  # Reduced distortion
                    transforms.RandomRotation(
                        degrees=(0, 10)
                    ),  # Reduced rotation angle
                    transforms.RandomAffine(
                        degrees=(0, 10), translate=(0.05, 0.1), scale=(0.8, 1.0)
                    ),
                    transforms.Resize((crop_dim, crop_dim), antialias=True),  # No-op
                ]
            )
        )

    # Add Resize to make sure the output has the correct shape (both for images and masks)
    geometric_transforms.append(transforms.Resize((crop_dim, crop_dim), antialias=True))

    img_transforms = []

    if not is_eval:
        img_transforms += [
            transforms.RandomApply(
                nn.ModuleList([transforms.ColorJitter(hue=0.2)]), p=p
            ),
            transforms.RandomApply(
                nn.ModuleList([transforms.ColorJitter(saturation=(0.3, 0.5))]), p=p
            ),
        ]

    # Convert to tensor and normalize
    img_transforms += [
        transforms.ToImage(),  # Converts image to tensor and scales to [0, 1]
        transforms.Normalize(
            mean=IMAGENET_MEAN, std=IMAGENET_STD
        ),  # Normalize for pre-trained models
    ]

    # Compose final transformations
    return transforms.Compose(img_transforms + geometric_transforms)


# ========================= DATASET =========================


class SoccernetDataset(Dataset):
    """Soccernet dataset for keypoint detection"""

    def __init__(
        self,
        dataset_dir,
        frame_sequence=1,
        frame_stride=1,
        aug_transforms=None,
        hm_size=(384, 384),
    ):
        self.dataset_dir = dataset_dir
        self.frame_sequence = frame_sequence
        self.frame_stride = frame_stride
        self.aug_transforms = aug_transforms
        self.hm_size = hm_size
        # Get all image paths first
        self.all_image_paths = sorted(
            glob.glob(os.path.join(self.dataset_dir, "*.jpg"))
        )
        # Process only key frames (every frame_stride frames)
        self.key_frames = [
            self.all_image_paths[i]
            for i in range(0, len(self.all_image_paths), self.frame_stride)
        ]
        self.video_name = os.path.basename(os.path.dirname(self.dataset_dir))

    def __len__(self):
        return len(self.key_frames)

    def __getitem__(self, idx):
        # Get the key frame path
        key_frame_path = self.key_frames[idx]

        # Load and process the key frame
        image = torchvision.io.read_image(key_frame_path).float() / 255.0
        image = TF.resize(image, self.hm_size, antialias=True)

        # Apply transforms if needed
        if self.aug_transforms:
            image_tensor = image.unsqueeze(0)
            image_tensor, _ = self.aug_transforms(image_tensor, None)
            image = image_tensor.squeeze(0)

        # Find all frames in this batch
        start_idx = idx * self.frame_stride
        end_idx = min(start_idx + self.frame_stride, len(self.all_image_paths))
        batch_frames = self.all_image_paths[start_idx:end_idx]

        return image.unsqueeze(0), batch_frames, self.video_name


# ========================= LINE HOMOGRAPHY HELPERS =========================

def extract_line_points_from_heatmap(
    heatmap: np.ndarray,
    all_lines_heatmap: np.ndarray = None,
    threshold: float = 0.8,
    high_intensity_threshold: float = 0.96,
    num_samples: int = 100,
    num_random_points: int = 100,
    min_length: float = 100,
    all_lines_weight: float = 0.5
) -> np.ndarray:
    """
    Robustly extract high‐intensity skeleton points from a line heatmap,
    then throw away any that lie more than `max_outlier_dist` px from
    the fitted infinite line.

    Returns:
        np.ndarray of shape (M,3) containing [x, y, intensity].
    """
    # outlier pruning params
    max_outlier_dist = 3.0   # px away from the fitted line
    min_inliers      = 20    # must keep at least this many

    # 1) Optionally fuse with the “all lines” channel
    hm = heatmap.astype(np.float32)
    if all_lines_heatmap is not None:
        all_hm = all_lines_heatmap.astype(np.float32)
        hm = (1 - all_lines_weight) * hm + all_lines_weight * all_hm

    # 2) normalize & threshold
    hm = (hm - hm.min()) / (np.ptp(hm) + 1e-8)
    bw = (hm > threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 3) skeletonize the largest connected blob
    num_cc, labels = cv2.connectedComponents(bw, connectivity=8)
    if num_cc <= 1:
        return np.zeros((0,3), dtype=np.float32)
    sizes = [(labels==c).sum() for c in range(1, num_cc)]
    best = np.argmax(sizes) + 1
    if sizes[best-1] < min_length:
        return np.zeros((0,3), dtype=np.float32)
    mask = (labels == best).astype(np.uint8)*255
    # skel = cv2.ximgproc.thinning(mask)
    # Use skimage skeletonize instead
    skel = skeletonize(mask > 0)
    skel = skel.astype(np.uint8) * 255

    # 4) pick only high‐intensity skeleton points
    ys, xs = np.nonzero((skel>0) & (hm>=high_intensity_threshold))
    if len(xs) < min_inliers:
        return np.zeros((0,3), dtype=np.float32)

    pts    = np.stack([xs, ys], axis=1).astype(np.float32)  # (N,2)
    intens = hm[ys, xs].astype(np.float32)                  # (N,)

    # 5) fit infinite line
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    p0 = np.array([x0[0], y0[0]], dtype=np.float32)
    v  = np.array([vx[0], vy[0]], dtype=np.float32)
    v /= np.linalg.norm(v)

    # 6) discard outliers > max_outlier_dist
    diffs = pts - p0[None,:]
    # cross‐product magnitude → distance to line
    dists = np.abs(diffs[:,0]*v[1] - diffs[:,1]*v[0])
    inliers = dists <= max_outlier_dist
    if inliers.sum() < min_inliers:
        return np.zeros((0,3), dtype=np.float32)

    # 7) randomly subsample inliers
    idxs = np.nonzero(inliers)[0]
    if idxs.size > num_random_points:
        idxs = np.random.choice(idxs, size=num_random_points, replace=False)

    xs_f = pts[idxs, 0]
    ys_f = pts[idxs, 1]
    i_f  = intens[idxs]

    return np.stack([xs_f, ys_f, i_f], axis=1)  # (M,3)


def extract_circle_points_from_heatmap(heatmap: np.ndarray, threshold: float = 0.8, high_intensity_threshold: float = 0.96,
                                       num_samples: int = 50, num_random_points: int = 50) -> np.ndarray:
    """
    Extracts robust high-intensity circle points with geometric outlier removal.
    """
    max_outlier_dist: float = 3.0
    min_inliers: int = 20

    # 1) Normalize heatmap
    hm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # 2) Threshold and clean
    bw = (hm > threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 3) High-intensity mask
    high_mask = (hm >= high_intensity_threshold) & (bw > 0)
    ys, xs = np.nonzero(high_mask)

    if len(xs) < min_inliers:
        return np.zeros((0, 3), dtype=np.float32)

    # 4) Randomly subsample
    idx = np.random.choice(len(xs), size=min(len(xs), num_random_points), replace=False)
    pts = np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)

    # 5) Fit ellipse
    if len(pts) < 5:
        return np.zeros((0, 3), dtype=np.float32)
    ellipse = cv2.fitEllipse(pts)
    center = np.array(ellipse[0])
    axes = np.array(ellipse[1]) / 2.0
    angle = np.deg2rad(ellipse[2])

    # 6) Compute distances from ellipse perimeter
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    delta = pts - center
    rotated = np.stack([
        cos_a * delta[:, 0] + sin_a * delta[:, 1],
       -sin_a * delta[:, 0] + cos_a * delta[:, 1]
    ], axis=1)
    normed = rotated / axes
    dists = np.abs(np.linalg.norm(normed, axis=1) - 1.0)

    # 7) Filter by distance to ellipse
    inlier_mask = dists < (max_outlier_dist / np.mean(axes))
    if inlier_mask.sum() < min_inliers:
        return np.zeros((0, 3), dtype=np.float32)

    inlier_pts = pts[inlier_mask]
    inlier_x = inlier_pts[:, 0]
    inlier_y = inlier_pts[:, 1]
    inlier_i = heatmap[inlier_y.astype(int), inlier_x.astype(int)]

    # 8) Return in homogeneous form
    return np.stack([inlier_x, inlier_y, inlier_i], axis=1)


def get_line(p0, p1):
    """
    Calculate the line passing through two points in homogeneous coordinates.
    Returns normalized line coefficients [a, b, c] where ax + by + c = 0
    """
    l = np.cross(p0, p1)
    return l / np.linalg.norm(l[:2])


def normalization_transform_pts(pts: np.ndarray) -> np.ndarray:
    """
    Normalize an (N,2) point set so that the centroid is at the origin
    and the average distance to the origin is sqrt(2).
    Returns a 3×3 homography matrix.
    """
    # centroid
    m = pts.mean(axis=0)
    # average distance to centroid
    d = np.linalg.norm(pts - m, axis=1).mean()
    # scale factor so that mean distance becomes sqrt(2)
    s = np.sqrt(2) / (d if d>0 else 1.0)
    T = np.array([
        [ s,  0,  -s*m[0]],
        [ 0,  s,  -s*m[1]],
        [ 0,  0,     1   ],
    ], dtype=float)
    return T


def get_line_from_points(points_h):
    points_h = np.array(points_h)
    points = points_h[:, :2] / points_h[:, 2][:, np.newaxis]

    A = np.c_[points, np.ones(points.shape[0])]
    _, _, Vt = np.linalg.svd(A)
    line = Vt[-1]
    line /= np.linalg.norm(line[:2])
    return line


def fit_line_to_points(points):
    """Fit a line to a set of points using least squares."""
    if len(points) < 2:
        return None, None
    points = np.array(points)[:, :2]
    vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    p0 = np.array([x0[0], y0[0]])
    v = np.array([vx[0], vy[0]])
    return p0, v


def circle_conic_matrix(center, rad):
    C = np.array([[1.0, 0.0, -center[0]],
                  [0.0, 1.0, -center[1]],
                  [-center[0], -center[1], center[0] ** 2 + center[1] ** 2 - rad ** 2]])
    return C


def convert_and_normalize_keypoints(
    refined_pts: np.ndarray,
    template_kpts: np.ndarray,
    T_i: np.ndarray,
    T_p: np.ndarray
):
    """
    Take *all* non-zero refined_pts, match to template_kpts,
    normalize each, and bundle into (p_i, p_t, conf).
    """
    pairs = []
    for idx, (x, y, conf) in enumerate(refined_pts):
        # skip missing detections
        if x == 0 and y == 0:
            continue

        # ————— normalize image pt —————
        p_i = np.array([x, y, 1.0], dtype=float)
        p_i = T_i @ p_i
        p_i /= p_i[2]

        # ————— normalize template pt —————
        # if your template_kpts[idx] is length-3, just take the first two
        xt, yt = template_kpts[idx][:2]
        p_t = np.array([xt, yt, 1.0], dtype=float)
        p_t = T_p @ p_t
        p_t /= p_t[2]

        pairs.append((p_i, p_t, float(conf)))

    return pairs


def get_inv_residual_line_circle_points_middle_pair(
    h_inv,
    line_points_pair,
    line_confidences,
    C,
    circle_points,
    img_h,      # ← (M,3) array of [x_img, y_img, 1]
    tmp_h,      # ← (M,3) array of [x_tmp, y_tmp, 1]
    confs       # ← (M,) array of confidences
):
    # rebuild inverse homography
    H_inv = np.append(h_inv, 1.0).reshape(3, 3)
    res = []

    if len(line_points_pair) >= 5:
        for (line, points), w in zip(line_points_pair, line_confidences):
            l2 = H_inv.T @ line
            for x, y, intensity in points:
                raw = float(np.dot([x, y, 1.0], l2)) 
                res.append(intensity * raw)


    if len(circle_points):
        C_tr = H_inv.T @ C @ H_inv
        norm = np.linalg.norm(C_tr)
        for x, y, intensity in circle_points:
            p = np.array([x, y, 1.0])
            raw = float(p @ C_tr @ p)
            res.append((raw / norm))

    if len(line_points_pair) < 5 and img_h.shape[0] >= 4:
        # project all image points → template at once
        proj = (H_inv @ img_h.T).T  # (M,3)
        xy = proj[:, :2] / proj[:, 2:3]  # (M,2)
        true = tmp_h[:, :2] / tmp_h[:, 2:3]  # (M,2)
        dists = np.linalg.norm(xy - true, axis=1)

        # weigh each reprojection distance by its confidence
        for dist, conf in zip(dists, confs):
            res.append(conf * dist)

    return np.array(res)


def estimate_inv_homography_line_circle_points_middle_pair(
    line_points_pair,
    line_confidences,
    C,
    circle_points,
    middle_point_pairs
):
    # 1) Precompute the homogeneous‐point arrays ONCE
    img_h  = np.stack([p for (p,_,_) in middle_point_pairs], axis=0)  # (M,3)
    tmp_h  = np.stack([t for (_,t,_) in middle_point_pairs], axis=0)  # (M,3)
    confs  = np.array([c for (_,_,c) in middle_point_pairs], dtype=float)  # (M,)

    # # choose initial guess:
    if img_h.shape[0] >= 4:
        # DLT from your 4+ pts
        src = img_h[:, :2].astype(float)
        dst = tmp_h[:, :2].astype(float)
        H0, _ = cv2.findHomography(src, dst, cv2.RANSAC, ransacReprojThreshold=10)
        h0 = H0.flatten()[:8]
    else:
        # no middle pts → just identity
        h0 = np.array([1, 0, 0, 0, 1, 0, 0, 0], dtype=float)

    # 3) Call least_squares passing in our prebuilt arrays
    result = least_squares(
        fun=get_inv_residual_line_circle_points_middle_pair,
        x0=h0,
        args=(line_points_pair,
              line_confidences,
              C,
              circle_points,
              img_h,
              tmp_h,
              confs),
        method='lm',
    )

    # 4) Reassemble full 3×3 inverse homography
    return np.append(result.x, 1.0).reshape(3, 3)


# ========================= INFERENCE FUNCTION =========================



def predict_soccernet_inference(
    model,
    dataloader,
    device,
    result_dir,
    filename,
    template_image,
    template_numpy,
    num_keypoints,
    verbose=True,
    save_viz=True,  # New argument to control saving result.jpg
    npy_subfolder="npy_files",  # Subfolder for .npy
    viz_subfolder="viz",  # Subfolder for result.jpg
):
    """Main inference function for tennis court detection"""
    model.eval()

    if verbose:
        bar = tqdm(
            dataloader,
            total=len(dataloader),
            ascii=True,
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            desc="Test ",
        )
    else:
        bar = dataloader

    start_time = time.time()

    # Load template and template keypoints
    template = cv2.imread(template_image)
    if template is None:
        raise ValueError("Could not load template image from " + template_image)
    template_kpts = np.load(template_numpy)
    hm_size = (384, 384)
    distance_threshold = 30  # Adjust as needed

    # Define line definitions
    pitch_line_endpoints = {
        'Big rect. left top': [[10.0, 18.84, 1], [26.50, 18.84, 1]],
        'Big rect. left bottom': [[10.0, 59.16, 1], [26.50, 59.16, 1]],
        'Big rect. left main': [[26.50, 59.16, 1], [26.50, 18.84, 1]],
        'Big rect. right top': [[98.50, 18.84, 1], [115.0, 18.84, 1]],
        'Big rect. right bottom': [[98.50, 59.16, 1], [115.0, 59.16, 1]],
        'Big rect. right main': [[98.50, 59.16, 1], [98.50, 18.84, 1]],
        'Small rect. left top': [[10.0, 29.84, 1], [15.5, 29.84, 1]],
        'Small rect. left bottom': [[10.0, 48.16, 1], [15.5, 48.16, 1]],
        'Small rect. left main': [[15.5, 48.16, 1], [15.5, 29.84, 1]],
        'Small rect. right top': [[109.50, 29.84, 1], [115.0, 29.84, 1]],
        'Small rect. right bottom': [[109.50, 48.16, 1], [115.0, 48.16, 1]],
        'Small rect. right main': [[109.50, 48.16, 1], [109.50, 29.84, 1]],
        'Side line bottom': [[10.0, 73.0, 1], [115.0, 73.0, 1]],
        'Side line top': [[10.0, 5.0, 1], [115.0, 5.0, 1]],
        'Side line left': [[10.0, 73.0, 1], [10.0, 5.0, 1]],
        'Side line right': [[115.0, 73.0, 1], [115.0, 5.0, 1]],
        'Middle line': [[62.50, 73.0, 1], [62.50, 5.0, 1]],
        'Circle left': [[26.50, 46.31, 1], [26.50, 31.69, 1]],
        'Circle right': [[98.50, 46.31, 1], [98.50, 31.69, 1]],
    }

    line_names = [
        "Big rect. left bottom", "Big rect. left main", "Big rect. left top",
        "Big rect. right bottom", "Big rect. right main", "Big rect. right top",
        "Circle central", "Middle line", "Side line bottom", "Side line left",
        "Side line right", "Side line top", "Small rect. left bottom",
        "Small rect. left main", "Small rect. left top", "Small rect. right bottom",
        "Small rect. right main", "Small rect. right top", "Circle left", "Circle right",
        "All lines"
    ]

    # Process frames in batches of 30, only computing homography for first frame in each batch
    frame_skip = 1  # Process one frame, then skip 29 frames

    # First, collect all image paths from the dataloader
    all_images = []
    for _, (_, img_paths, _) in enumerate(bar):
        # Handle both list and single path cases
        if isinstance(img_paths, (list, tuple)):
            all_images.extend(img_paths)
        else:
            all_images.append(img_paths)

    # Flatten nested lists if any
    all_images = [
        path[0] if isinstance(path, (list, tuple)) else path for path in all_images
    ]

    # Create progress bar for frame processing
    if verbose:
        process_bar = tqdm(
            range(0, len(all_images), frame_skip), desc="Processing frames", ascii=True
        )
    else:
        process_bar = range(0, len(all_images), frame_skip)

    # Process frames with stride 30
    for frame_idx in process_bar:
        # Get the key frame path
        key_frame_path = all_images[frame_idx]

        # Get all frames in this batch
        batch_end = min(frame_idx + frame_skip, len(all_images))
        batch_frames = all_images[frame_idx:batch_end]

        # Load the key frame
        image = cv2.imread(key_frame_path)
        if image is None:
            print(f"Could not read image: {key_frame_path}")
            continue

        img_height, img_width, _ = image.shape

        # Create a tensor from the key frame for model input
        with torch.no_grad():
            # Prepare the image for the model (single frame)
            img_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            img_tensor = torch.nn.functional.interpolate(
                img_tensor.unsqueeze(0),
                size=hm_size,
                mode="bilinear",
                align_corners=False,
            )

            # Apply normalization if needed
            if (
                hasattr(dataloader.dataset, "aug_transforms")
                and dataloader.dataset.aug_transforms is not None
            ):
                img_tensor, _ = dataloader.dataset.aug_transforms(img_tensor, None)

            img_tensor = img_tensor.to(device)

            # Get predictions for single frame
            pred_kpts, pred_lines = model(img_tensor)

            # Get final keypoint coordinates
            preds_batch, maxvals_batch = get_final_preds_torch(pred_kpts)

            # Extract the directory path and create results directory
            f_fn = Path(key_frame_path).stem
            # directory_path = os.path.basename(os.path.dirname(key_frame_path))
            
            # Save .npy files in the same directory as the images
            npy_directory = os.path.dirname(key_frame_path)
            
            viz_directory = os.path.join(result_dir, viz_subfolder)
            # os.makedirs(npy_directory, exist_ok=True) # Directory should already exist
            if save_viz:
                os.makedirs(viz_directory, exist_ok=True)

            # Scale the keypoints to match the original image dimensions
            scale_x = img_width / hm_size[0]
            scale_y = img_height / hm_size[1]

            # Filter predicted keypoints
            filtered_keypoints = []
            for kp_idx in range(num_keypoints):
                conf = maxvals_batch[0, kp_idx]  # Batch size is 1
                if conf >= 0.7:
                    pred_kp = preds_batch[0, kp_idx]
                    x = int(np.rint(pred_kp[0] * scale_x))
                    y = int(np.rint(pred_kp[1] * scale_y))
                    keypoint = (x, y, float(conf.item()))
                    if not filtered_keypoints:
                        filtered_keypoints.append(keypoint)
                    else:
                        distances = [
                            np.linalg.norm(np.array(keypoint[:2]) - np.array(kp[:2]))
                            for kp in filtered_keypoints
                        ]
                        if min(distances) > distance_threshold:
                            filtered_keypoints.append(keypoint)
                        else:
                            filtered_keypoints.append((0, 0, 0.0))
                else:
                    filtered_keypoints.append((0, 0, 0.0))
            pts = np.array(filtered_keypoints)

            # --- LINE-BASED HOMOGRAPHY LOGIC ---
            
            # Calculate confidence scores for each line
            confidence_scores = torch.max(pred_lines.view(pred_lines.size(0),
                                                        pred_lines.size(1), -1),
                                          dim=2)[0]
            
            line_detected = {}
            for line_name in line_names:
                if line_name == "All lines":
                    continue
                line_idx = line_names.index(line_name)
                conf = confidence_scores[0, line_idx].item()
                line_detected[line_name] = (conf >= 0.8)

            # Extract "All lines" heatmap
            all_lines_idx = line_names.index("All lines")
            all_lines_heatmap = pred_lines[0, all_lines_idx].cpu().numpy()

            detected_line_points = {}
            raw_circle_pts = []

            for line_name in line_names:
                if line_name == "All lines":
                    continue
                if not line_detected[line_name]:
                    continue
                
                line_idx = line_names.index(line_name)
                heatmap = pred_lines[0, line_idx].cpu().numpy()

                if line_name == "Circle central":
                    points = extract_circle_points_from_heatmap(heatmap, threshold=0.9,
                                                                high_intensity_threshold=0.96,
                                                                num_samples=0, num_random_points=100)
                    if len(points) > 0:
                        # Scale points
                        scaled_points = [[pt[0] * scale_x, pt[1] * scale_y, pt[2]] for pt in points]
                        raw_circle_pts = scaled_points
                elif line_name != "All lines":
                    points = extract_line_points_from_heatmap(heatmap, all_lines_heatmap=all_lines_heatmap,
                                                                threshold=0.93, high_intensity_threshold=0.9,
                                                                num_samples=100, num_random_points=100,
                                                                min_length=15,
                                                                all_lines_weight=0.3)
                    if len(points) > 0:
                        scaled_points = [[pt[0] * scale_x, pt[1] * scale_y, pt[2]] for pt in points]
                        detected_line_points[line_name] = np.array(scaled_points)

            # Collect all points for normalization
            valid_idx = [i for i, (x, y, _) in enumerate(pts) if x != 0 or y != 0]
            refined_pts = pts # Using pts as refined_pts for now
            
            all_img_pts = []
            if len(valid_idx) > 0:
                all_img_pts += refined_pts[valid_idx, :2].tolist()
            for pts_arr in detected_line_points.values():
                all_img_pts += pts_arr[:, :2].tolist()
            all_img_pts += [p[:2] for p in raw_circle_pts]
            all_img_pts = np.array(all_img_pts, float)

            # Collect all template points
            all_tmp_pts = []
            for ep in pitch_line_endpoints.values():
                for x, y, _ in ep:
                    all_tmp_pts.append([x, y])
            if len(valid_idx) > 0:
                all_tmp_pts += template_kpts[valid_idx, :2].tolist()
            all_tmp_pts = np.array(all_tmp_pts, float)

            homography_matrix = None
            keypoints_data = None

            if len(all_img_pts) > 0:
                # Build normalizers
                T_i = normalization_transform_pts(all_img_pts)
                T_p = normalization_transform_pts(all_tmp_pts)

                # Normalize circle samples
                circle_points = []
                for x, y, i in raw_circle_pts:
                    p = np.array([x, y, 1.0])
                    p_n = T_i @ p
                    p_n /= p_n[2]
                    circle_points.append([p_n[0], p_n[1], i])

                # Normalize template line endpoints
                normalized_pitch_line_endpoints = {}
                for key, value in pitch_line_endpoints.items():
                    pts_list = []
                    for pt in value:
                        pts_list.append(T_p @ pt)
                    normalized_pitch_line_endpoints[key] = pts_list

                pitch_line = {}
                for key, value in normalized_pitch_line_endpoints.items():
                    pitch_line[key] = get_line(value[0], value[1])

                # Normalize detected line points
                normalized_detected_line_points = {}
                for key, pts_arr in detected_line_points.items():
                    normed = []
                    for x, y, intensity in pts_arr:
                        p_h = np.array([x, y, 1.0], dtype=float)
                        p_n = T_i @ p_h
                        p_n /= p_n[2]
                        normed.append([p_n[0], p_n[1], intensity])
                    normalized_detected_line_points[key] = np.array(normed, dtype=float)

                line_points_pair = []
                line_confidences = []

                for key in normalized_detected_line_points:
                    if key in pitch_line and line_detected.get(key, False):
                        line_idx = line_names.index(key)
                        line_confidence = float(confidence_scores[0, line_idx].cpu())
                        
                        line = pitch_line[key]
                        line /= np.linalg.norm(line[:2])
                        points = normalized_detected_line_points[key]
                        line_points_pair.append((line, [np.array(p, dtype=float) for p in points]))
                        line_confidences.append(line_confidence)

                circle_center = T_p @ np.array([62.50, 39.00, 1.0])
                circle_rad = T_p[0, 0] * 9.14
                C = circle_conic_matrix(circle_center, circle_rad)

                point_pairs = convert_and_normalize_keypoints(refined_pts, template_kpts, T_i, T_p)

                try:
                    H_inv_normalized = estimate_inv_homography_line_circle_points_middle_pair(
                        line_points_pair, line_confidences, C, circle_points, point_pairs
                    )
                    H_normalized = np.linalg.inv(H_inv_normalized)
                    orig_homography = np.linalg.inv(T_i) @ H_normalized @ T_p
                    orig_homography = np.linalg.inv(orig_homography)
                    
                    if np.linalg.det(orig_homography) != 0:
                        homography_matrix = orig_homography.astype(np.float32)
                    else:
                        print(f"Singular matrix for frame {f_fn}")
                        homography_matrix = np.eye(3, dtype=np.float32)
                except Exception as e:
                    print(f"Homography estimation failed for {f_fn}: {e}")
                    homography_matrix = np.eye(3, dtype=np.float32)
            else:
                # Fallback to point-based if no lines/points found (or just use identity)
                homography_matrix = np.eye(3, dtype=np.float32)

            # Save homography
            np.save(os.path.join(npy_directory, f"{f_fn}.npy"), homography_matrix)

            # Visualization
            if verbose and save_viz:
                try:
                    init_H_inv = np.linalg.inv(homography_matrix)
                    im_out = cv2.warpPerspective(
                        template, init_H_inv, (image.shape[1], image.shape[0])
                    )

                    vis_image = image.copy()
                    valid_index = im_out[:, :, 0] > 0
                    overlay = (
                        (
                            image[valid_index].astype("float32")
                            + im_out[valid_index].astype("float32")
                        )
                        / 2
                    ).astype(np.uint8)
                    vis_image[valid_index] = overlay
                    


                    cv2.imwrite(
                        os.path.join(viz_directory, f"{f_fn}_result.jpg"),
                        vis_image,
                    )
                except Exception as e:
                    print(f"Visualization failed for {f_fn}: {e}")

            # Now copy the homography to all subsequent frames in the batch
            if homography_matrix is not None:
                for i in range(1, len(batch_frames)):
                    frame_path = batch_frames[i]
                    frame_name = Path(frame_path).stem

                    # Copy homography
                    np.save(
                        os.path.join(npy_directory, f"{frame_name}.npy"),
                        homography_matrix,
                    )

                    # Visualization for all frames
                    if verbose and save_viz:
                        frame_image = cv2.imread(frame_path)
                        if frame_image is None:
                            continue

                        try:
                            init_H_inv = np.linalg.inv(homography_matrix)
                            im_out = cv2.warpPerspective(
                                template,
                                init_H_inv,
                                (frame_image.shape[1], frame_image.shape[0]),
                            )

                            vis_image = frame_image.copy()
                            valid_index = im_out[:, :, 0] > 0
                            overlay = (
                                (
                                    frame_image[valid_index].astype("float32")
                                    + im_out[valid_index].astype("float32")
                                )
                                / 2
                            ).astype(np.uint8)
                            vis_image[valid_index] = overlay

                            cv2.imwrite(
                                os.path.join(viz_directory, f"{frame_name}_result.jpg"),
                                vis_image,
                            )
                        except Exception as e:
                            pass

    total_duration = time.time() - start_time
    print(f"Total inference time: {total_duration:.2f} seconds")


# ========================= CONFIGURATION =========================


@dataclass
class Configuration:
    model: str = ("efficientNet", "imagenet")
    img_size: tuple = (384, 384)
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)
    num_keypoints: int = 97
    num_lines: int = 21
    num_frames: int = 1
    frame_stride: int = 1
    sports: str = "Soccer"
    batch_size: int = 1
    checkpoints: str = "checkpoints/SoccernetGSR_EfficientNet_Best.pth"
    data_dir: str = "Data/"
    result_dir: str = "Results/"
    verbose: bool = True
    num_workers: int = 0 if os.name == "nt" else 8
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"


# ========================= MAIN PREDICTION FUNCTION =========================


def predict(
    img_path,
    result_dir,
    checkpoints="checkpoints/SoccernetGSR_EfficientNet_Best.pth",
    template_image="template/Pitch_Radar.png",
    template_npy="template/soccernet_template_97.npy",
    save_viz=False,
    verbose=True,
    npy_subfolder="npy_files",
):
    """
    Main prediction function for tennis court detection

    Args:
        img_path (str): Path to the directory containing image data
        result_dir (str): Path to save results
        checkpoints (str): Path to model weights
        template_image (str): Path to template image
        template_npy (str): Path to template keypoints
        save_viz (bool): Whether to save visualization images
        verbose (bool): Whether to show progress and debug info
    """
    # Create configuration with custom paths
    config = Configuration()
    config.data_dir = img_path
    config.result_dir = result_dir
    config.checkpoints = checkpoints
    config.verbose = verbose

    print("\nModel: {}".format(config.model))
    model = Unet(out_ch=config.num_keypoints + 1, num_lines=config.num_lines)
    print("\nEfficientNet Single Model: {}".format(config.model))
    print_line(name=config.checkpoints, length=80)
    model_dict = torch.load(config.checkpoints, map_location=config.device)
    model.load_state_dict(model_dict, strict=True)
    model = model.to(config.device)

    print("\nImage Size:", config.img_size)
    print("Mean:", config.mean)
    print("Std:", config.std)

    # Set up transforms for inference.
    val_transforms = _get_img_transforms(crop_dim=384, is_eval=True)

    # Process the image folder directly
    clip_path = config.data_dir
    clip_folder = os.path.basename(clip_path)
    print(f"Processing clip_path: {clip_path}")

    # Create an inference dataset using the image folder
    inference_dataset = SoccernetDataset(
        clip_path,
        frame_sequence=config.num_frames,
        frame_stride=config.frame_stride,
        aug_transforms=val_transforms,
        hm_size=config.img_size,
    )
    inference_loader = DataLoader(
        inference_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=True,
    )
    print("len test_loader: {}".format(len(inference_loader)))

    # Run prediction with specified parameters
    predict_soccernet_inference(
        model,
        inference_loader,
        config.device,
        config.result_dir,
        clip_folder,
        template_image,
        template_npy,
        num_keypoints=config.num_keypoints,
        verbose=config.verbose,
        save_viz=save_viz,
        npy_subfolder=npy_subfolder,
        viz_subfolder="viz",
    )

    print("Inference complete.")


# ========================= MAIN EXECUTION =========================

if __name__ == "__main__":
    import glob
    
    test_root = "data/SoccerNetGS/test"
    # Get all subdirectories
    video_dirs = sorted(glob.glob(os.path.join(test_root, "*")))
    
    for video_dir in video_dirs:
        if not os.path.isdir(video_dir):
            continue
            
        video_name = os.path.basename(video_dir)
        img_path = os.path.join(video_dir, "img1")
        
        if not os.path.exists(img_path):
            print(f"Skipping {video_name}: img1 folder not found")
            continue
            
        print(f"Processing {video_name}...")
        
        # Create a specific result directory for this video
        result_dir = os.path.join("Results", video_name)
        
        predict(
            img_path=img_path,
            result_dir=result_dir,
            checkpoints="checkpoints/SoccernetGSR_EfficientNet_Best.pth",
            template_image="template/Pitch_Radar.png",
            template_npy="template/soccernet_template_97.npy",
            save_viz=False,
            verbose=True,
        )
