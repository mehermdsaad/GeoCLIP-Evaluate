import warnings

import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoTokenizer, CLIPModel

# warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub.*")


class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.CLIP = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.image_processor = AutoProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )

        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        self.mlp = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 512),
        )

        # Freeze CLIP
        for param in self.CLIP.parameters():
            param.requires_grad = False

    def preprocess_image(self, image):
        x = self.image_processor(images=image, return_tensors="pt", padding=True)[
            "pixel_values"
        ]

        return x

    def encode_text(self, texts):
        with torch.no_grad():
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.CLIP.config.text_config.max_position_embeddings,
            )
            text_features = self.CLIP.get_text_features(**inputs)

        return text_features

    def forward(self, image_tensor, texts):
        image_features = self.CLIP.get_image_features(pixel_values=image_tensor)
        text_features = self.encode_text(texts)

        combined_features = 1 * image_features + 0 * text_features

        encoded_output = self.mlp(combined_features)
        return encoded_output
