import torch.nn as nn
import torchvision.transforms.v2 as transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def _get_img_transforms(crop_dim=224, is_eval=False):
    p = 0.0 if is_eval else 0.3  # Reduced probability for color jittering

    geometric_transforms = []
    if not is_eval:
        print("Using train transforms")
        geometric_transforms.append(
            transforms.RandomChoice(
                [
                    transforms.RandomCrop((crop_dim, crop_dim)),
                    transforms.RandomResizedCrop((crop_dim, crop_dim), antialias=True),
                    transforms.Resize((crop_dim, crop_dim), antialias=True),  # No-op
                ]
            )
        )
        geometric_transforms.append(
            transforms.RandomChoice(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomZoomOut(p=0.3),  # Lowered probability
                    transforms.RandomPerspective(distortion_scale=0.3, p=0.3),  # Reduced distortion
                    transforms.RandomRotation(degrees=(0, 15)),  # Reduced rotation angle
                    transforms.RandomAffine(
                        degrees=(0, 10), translate=(0.05, 0.1), scale=(0.8, 1.0)  # Less translation and scale distortion
                    ),
                    transforms.Resize((crop_dim, crop_dim), antialias=True),  # No-op
                ]
            ),
        )

    geometric_transforms.append(transforms.Resize((crop_dim, crop_dim), antialias=True))

    # Include color transforms only during training
    img_transforms = []
    if not is_eval:
        img_transforms += [
            transforms.RandomApply(nn.ModuleList([transforms.ColorJitter(hue=0.2)]), p=p),
            transforms.RandomApply(nn.ModuleList([transforms.ColorJitter(saturation=(0.3, 0.5))]), p=p),
            transforms.RandomApply(nn.ModuleList([transforms.ColorJitter(brightness=(0.3, 0.5))]), p=p),
            transforms.RandomApply(nn.ModuleList([transforms.ColorJitter(contrast=(0.3, 0.5))]), p=p),
            transforms.RandomApply(nn.ModuleList([transforms.RandomAdjustSharpness(2)]), p=p),
            transforms.RandomApply(nn.ModuleList([transforms.GaussianBlur(3)]), p=0.2),
        ]

    # Normalization applies for both train and eval
    img_transforms.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))

    return transforms.Compose(geometric_transforms + img_transforms)