import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from models.modules.encoder_cnn import EncoderCNN
from models.modules.transformer import DecoderTransformer

class ViT(nn.Module):
    def __init__(self):
        super(ViT, self).__init__()
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        self.vit.head = nn.Identity() # Remove the classification head
        self.vit_transform = transforms.Compose([transforms.Resize((224, 224))])
        
    def forward(self, images):
        vit_features = self.vit(self.vit_transform(images))
        return vit_features


class VITAttentionModel(DecoderTransformer):
    def __init__(self, embed_size, vocab_size, num_heads, num_layers, dropout=0.1, max_seq_length=50):
        super(VITAttentionModel, self).__init__(embed_size, vocab_size, num_heads, num_layers, dropout, max_seq_length)
        self.vit = ViT()
        self.vit_out_size = 1000
        self.fc_vit = torch.nn.Linear(1000, embed_size)

        self.dropout = nn.Dropout(dropout)
        self.batchnorm = nn.BatchNorm1d(embed_size)
    
    def precompute_image(self, images):
        with torch.no_grad():
            enc_output = self.vit.forward(images)
        return enc_output
    
    def forward(self, images, captions, mode):
        with torch.no_grad():
            if mode == "precomputed":
                enc_output = images
            else:
                enc_output = self.vit.forward(images) # shape: (batch_size, 8*64*64 + 2048)

        enc_output = self.fc_vit(enc_output)
        enc_output = self.dropout(enc_output)
        enc_output = self.batchnorm(enc_output)
        enc_output = enc_output.unsqueeze(1)
        enc_output = enc_output.reshape(enc_output.size(0), 1, -1)
        
        return self.decoder_forward(enc_output, captions)
        
    def caption_images(self, images, vocabulary, mode="precomputed", max_length=40):
        self.eval()
        with torch.no_grad():
            # Encode the image
            if mode == "precomputed":
                enc_output = images
            else:
                enc_output = self.vit.forward(images)
                
            enc_output = self.fc_vit(enc_output)
            enc_output = self.dropout(enc_output)
            enc_output = self.batchnorm(enc_output)
            enc_output = enc_output.unsqueeze(1)
            enc_output = enc_output.reshape(enc_output.size(0), 1, -1)

            return self.greedy_inference(enc_output, vocabulary, max_length)

    def caption_images_beam_search(self, images, vocabulary, beam_width=3, mode="precomputed", max_length=50):
        self.eval()
        batch_size = images.size(0)  # Get the batch size from images
        # print("Batch size: ", batch_size)
        
        with torch.no_grad():
            # Encode all images in the batch
            if mode == "precomputed":
                enc_output = images
            else:
                enc_output = self.vit.forward(images)

            enc_output = self.fc_vit(enc_output)
            enc_output = self.dropout(enc_output)
            enc_output = self.batchnorm(enc_output)
            enc_output = enc_output.unsqueeze(1)
            enc_output = enc_output.reshape(enc_output.size(0), 1, -1)
            enc_output = enc_output.unsqueeze(1).expand(-1, beam_width, -1, -1)
            enc_output = enc_output.reshape(batch_size * beam_width, 1, -1)
            
            return self.beam_search_inference(enc_output, batch_size, vocabulary, beam_width, max_length)