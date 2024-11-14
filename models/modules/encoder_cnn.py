import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.inception import Inception3

class EncoderCNN(nn.Module):
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super(EncoderCNN, self).__init__()
        # Initialize the InceptionV3 model with pretrained weights
        self.inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT).to(device)
        
        # Dictionary to store activation from the desired layer
        self.activation = {}

        # Register forward hook on the Mixed_7c layer to capture features
        self.inception.Mixed_7c.register_forward_hook(self.get_activation("Mixed_7c"))

        # Pooling layer to convert the feature map to 1x1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def get_activation(self, name):
        # Helper function to store activation from the hook
        def hook(model, input, output):
            self.activation[name] = output
        return hook

    def forward(self, images):
        # Forward pass through InceptionV3
        _ = self.inception(images)  # Just to trigger the forward hook on Mixed_7c
        
        # Get features from the activation dictionary
        features = self.activation["Mixed_7c"]
        
        # Apply global average pooling
        features = self.pool(features)
        
        # Flatten and pass through the fully connected layer
        features = features.view(features.size(0), -1)  # Flatten to [batch_size, 2048]        

        return features