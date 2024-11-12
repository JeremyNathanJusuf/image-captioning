import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.encoder_cnn import EncoderCNN 
from models.modules.transformer import DecoderTransformer

class CNNAttentionModel(DecoderTransformer):
    def __init__(self, embed_size, vocab_size, num_heads, num_layers, dropout=0.1, max_seq_length=50):
        super(CNNAttentionModel, self).__init__(embed_size, vocab_size, num_heads, num_layers, dropout, max_seq_length)
        self.encoderCNN = EncoderCNN()
        self.fc1 = nn.Linear(2048, embed_size, bias = False)
        self.dropout = nn.Dropout(dropout)
        self.batchnorm = nn.BatchNorm1d(embed_size)
    
    def precompute_image(self, images):
        with torch.no_grad():
            features = self.encoderCNN(images)
        return features
    
    def forward(self, images, captions, mode='precomputed'):
        if mode == 'precomputed':
            enc_output = images
        else:
            with torch.no_grad():
                enc_output = self.encoderCNN(images)
            
        enc_output = self.fc1(enc_output)
        enc_output = self.dropout(enc_output)
        enc_output = self.batchnorm(enc_output)
        enc_output = enc_output.unsqueeze(1)

        return self.decoder_forward(enc_output, captions)
        
    def caption_images(self, images, vocabulary, mode="precomputed", max_length=50):
        self.eval()
        with torch.no_grad():
            # Encode the image
            if mode == "precomputed":
                enc_output = images
            else:
                enc_output = self.encoderCNN(images)
            enc_output = self.fc1(enc_output)
            enc_output = self.dropout(enc_output)
            enc_output = self.batchnorm(enc_output)
            enc_output = enc_output.unsqueeze(1)  # Expand dimensions for transformer input

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
                enc_output = self.encoderCNN(images)
                
            enc_output = self.fc1(enc_output)
            enc_output = self.dropout(enc_output)
            enc_output = self.batchnorm(enc_output) 
            enc_output = enc_output.unsqueeze(1)
            enc_output = enc_output.expand(-1, beam_width, -1)
            enc_output = enc_output.reshape(batch_size * beam_width, 1, -1)

        # enc output shape: (batch_size * beam_width, 1, embed_size)
        return self.beam_search_inference(enc_output, batch_size, vocabulary, beam_width, max_length)