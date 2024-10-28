import torch
import torch.nn as nn
from torchvision import models
from transformers import ViTModel, ViTConfig

class MultiModelClassifier(nn.Module):
    def __init__(self, model_type='efficientnet_b1', num_images=13, num_classes=3):
        super(MultiModelClassifier, self).__init__()
        
        # CNN Backbone 설정
        self.model_type = model_type
        self.num_images = num_images
        cnn_output_dim = 0

        if model_type == 'efficientnet_b1':
            self.backbone = models.efficientnet_b1(pretrained=True)
            self.backbone.classifier = nn.Identity()
            cnn_output_dim = 1280
        elif model_type == 'convnext_large':
            self.backbone = models.convnext_tiny(pretrained=True)
            self.backbone.classifier = nn.Identity()
            cnn_output_dim = 768
        elif model_type == 'vit':
            config = ViTConfig.from_pretrained('google/vit-base-patch16-224-in21k')
            self.backbone = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k', config=config)
            cnn_output_dim = config.hidden_size
        else:
            raise ValueError(f"Unsupported model type: {model_type}")


        # Classification Layer
        self.fc = nn.Sequential(
            nn.Linear(cnn_output_dim * num_images, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, img_inputs):
        batch_size = img_inputs.size(0)
        img_features = []

        # Backbone Feature Extraction
        for i in range(self.num_images):
            x = img_inputs[:, i, :, :, :]  # 각 이미지 추출
            if self.model_type == 'vit':
                vit_output = self.backbone(x).pooler_output
                img_features.append(vit_output)
            else:
                cnn_output = self.backbone.features(x) if hasattr(self.backbone, 'features') else self.backbone(x)
                pooled_output = torch.flatten(nn.AdaptiveAvgPool2d((1, 1))(cnn_output), 1)
                img_features.append(pooled_output)

        # Concatenate All Image Features
        img_features = torch.cat(img_features, dim=1)

        # Fully Connected Layer for Classification
        output = self.fc(img_features)

        return output

# Model Instantiation
def get_model(model_type='efficientnet_b0'):
    model = MultiModelClassifier(model_type=model_type)
    return model
