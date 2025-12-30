import clip
import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPFinetune1(nn.Module):
    def __init__(
        self,
        clip_model_name: str = "ViT-L/14",
        num_classes_role: int = 5,
        num_classes_number: int = 11,  # 0-10
        color_embedding_dim: int = 768,  # 512 for vit-B/32,
        freeze_clip: bool = True
    ):
        super().__init__()
        # Load CLIP model
        self.clip_model, self.preprocess = clip.load(clip_model_name)
        self.clip_feature_dim = self.clip_model.visual.output_dim

        # Freeze CLIP parameters if specified
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        # Task-specific heads
        # 1. Role classifier (player, goalkeeper, referee, ball, other)
        self.role_classifier = nn.Sequential(
            nn.Linear(self.clip_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes_role)
        )

        # 2. Jersey number classifiers: one for unit (ones) and the other for tens digits (each: 0-10)
        self.digit1_classifier = nn.Sequential(
            nn.Linear(self.clip_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes_number)
        )
        self.digit2_classifier = nn.Sequential(
            nn.Linear(self.clip_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes_number)
        )

        # 3. Jersey color encoder
        # We'll use CLIP's text encoder to create color embeddings
        self.color_projection = nn.Linear(self.clip_feature_dim, color_embedding_dim)

        # Feature extractor layers to get embeddings for contrastive learning
        # self.role_feature = nn.Linear(self.clip_feature_dim, 128)
        # self.number_feature = nn.Linear(self.clip_feature_dim, 128)

    def encode_image(self, image):
        with torch.no_grad() if self.clip_model.visual.parameters().__next__().requires_grad == False else torch.enable_grad():
            image_features = self.clip_model.encode_image(image)
        return image_features

    def encode_color_text(self, color_text):
        with torch.no_grad():
            color_tokens = clip.tokenize(color_text).to(next(self.parameters()).device)
            color_features = self.clip_model.encode_text(color_tokens)
        return color_features

    def forward(self, images):
        # Get image features from CLIP
        image_features = self.encode_image(images)
        image_features = image_features.float()

        # Get predictions for each task
        role_logits = self.role_classifier(image_features)
        digit1_logits = self.digit1_classifier(image_features)
        digit2_logits = self.digit2_classifier(image_features)
        color_embedding = self.color_projection(image_features)

        # Get feature embeddings for contrastive learning
        # role_embedding = self.role_feature(image_features)
        # number_embedding = self.number_feature(image_features)

        return {
            'role_logits': role_logits,
            'digit1_logits': digit1_logits,
            'digit2_logits': digit2_logits,
            'color_embedding': color_embedding,
            # 'role_embedding': F.normalize(role_embedding, p=2, dim=1),
            # 'number_embedding': F.normalize(number_embedding, p=2, dim=1)
        }


class CLIPFinetune(nn.Module):
    def __init__(
        self,
        clip_model_name: str = "ViT-L/14",
        num_classes_role: int = 5,
        num_classes_number: int = 11,  # 0-10
        color_embedding_dim: int = 768,  # 512 for vit-B/32,
        jn_len: int = 3,
        num_classes_jn: int = 100,
        freeze_clip: bool = True
    ):
        super().__init__()
        # Load CLIP model
        self.clip_model, self.preprocess = clip.load(clip_model_name)
        self.clip_feature_dim = self.clip_model.visual.output_dim

        # Freeze CLIP parameters if specified
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        # Task-specific heads
        # 1. Role classifier (player, goalkeeper, referee, ball, other)
        self.role_classifier = nn.Sequential(
            nn.Linear(self.clip_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes_role)
        )

        # 2. Jersey number classifiers: one for unit (ones) and the other for tens digits (each: 0-10)
        self.digit1_classifier = nn.Sequential(
            nn.Linear(self.clip_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes_number)
        )
        self.digit2_classifier = nn.Sequential(
            nn.Linear(self.clip_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes_number)
        )
        self.jn_classifier = nn.Sequential(
            nn.Linear(self.clip_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes_jn)
        )

        self.jn_len= nn.Sequential(
            nn.Linear(self.clip_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, jn_len)
        )

        # 3. Jersey color encoder
        # We'll use CLIP's text encoder to create color embeddings
        self.color_projection = nn.Linear(self.clip_feature_dim, color_embedding_dim)

        # Feature extractor layers to get embeddings for contrastive learning
        # self.role_feature = nn.Linear(self.clip_feature_dim, 128)
        # self.number_feature = nn.Linear(self.clip_feature_dim, 128)

    def encode_image(self, image):
        with torch.no_grad() if self.clip_model.visual.parameters().__next__().requires_grad == False else torch.enable_grad():
            image_features = self.clip_model.encode_image(image)
        return image_features

    def encode_color_text(self, color_text):
        with torch.no_grad():
            color_tokens = clip.tokenize(color_text).to(next(self.parameters()).device)
            color_features = self.clip_model.encode_text(color_tokens)
        return color_features

    def forward(self, images):
        # Get image features from CLIP
        image_features = self.encode_image(images)
        image_features = image_features.float()

        # Get predictions for each task
        role_logits = self.role_classifier(image_features)
        digit1_logits = self.digit1_classifier(image_features)
        digit2_logits = self.digit2_classifier(image_features)
        color_embedding = self.color_projection(image_features)

        number_logits = self.jn_classifier(image_features)
        length_logits = self.jn_len(image_features)

        # Get feature embeddings for contrastive learning
        # role_embedding = self.role_feature(image_features)
        # number_embedding = self.number_feature(image_features)

        return {
            'role_logits': role_logits,
            'digit1_logits': digit1_logits,
            'digit2_logits': digit2_logits,
            'color_embedding': color_embedding,
            'length_logits': length_logits,
            'number_logits': number_logits,
            # 'role_embedding': F.normalize(role_embedding, p=2, dim=1),
            # 'number_embedding': F.normalize(number_embedding, p=2, dim=1)
        }
