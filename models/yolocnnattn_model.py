import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ultralytics import YOLO
from torchvision import transforms
from models.modules.encoder_cnn import EncoderCNN
from models.modules.transformer import DecoderTransformer

class YOLOCNNModel():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo_model = YOLO("yolov8n.pt").to(self.device)
        self.yolo_layers = torch.nn.modules.container.Sequential = self.yolo_model.model.__dict__["_modules"]["model"]
        self.cnn = EncoderCNN().to(self.device)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, images):
        images = images.to(self.device)
        
        detect_output = {}
        def hook_fn(module, input, output):
            detect_output["preds"] = output
        
        detect_layer = self.yolo_layers[-1]  # The last layer is the Detect layer
        hook = detect_layer.register_forward_hook(hook_fn)
        
        with torch.no_grad():
            yolo_transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
            yolo_images = yolo_transform(images).to(self.device)
            # clip to [0, 1]
            yolo_images = torch.clamp(yolo_images, 0, 1)
            _ = self.yolo_model(yolo_images, verbose=False)

        hook.remove()

        raw_predictions = detect_output["preds"][0]
        argmax_predictions = self.softmax(raw_predictions[:, 5:]).argmax(dim=1)
        predictions = torch.cat((raw_predictions[:, :5], argmax_predictions.unsqueeze(1).float()), dim=1)
        
        predictions = raw_predictions.view(predictions.size(0), -1)
        
        cnn_output = self.cnn(images).to(self.device)
        return torch.cat((predictions, cnn_output), dim=1)

class YOLOCNNAttentionModel(DecoderTransformer):
    def __init__(self, embed_size, vocab_size, num_heads, num_layers, dropout=0.1, max_seq_length=50):
        super(YOLOCNNAttentionModel, self).__init__(embed_size, vocab_size, num_heads, num_layers, dropout, max_seq_length)

        self.yolo_output_size = 84 * 1029
        self.yolocnn = YOLOCNNModel()
        self.fc_yolo = nn.Linear(self.yolo_output_size, embed_size)
        self.fc_cnn = nn.Linear(2048, embed_size)
        
        self.dropout = nn.Dropout(dropout)
        self.batchnorm = nn.BatchNorm1d(2*embed_size)
    
    def precompute_image(self, images):
        with torch.no_grad():
            enc_output = self.yolocnn.forward(images)
        return enc_output
    
    def forward(self, images, captions, mode):
        with torch.no_grad():
            if mode == "precomputed":
                enc_output = images
            else:
                enc_output = self.yolocnn.forward(images) # shape: (batch_size, 8*64*64 + 2048)

        yolo_output = self.fc_yolo(enc_output[:, :self.yolo_output_size])
        cnn_output = self.fc_cnn(enc_output[:, self.yolo_output_size:])
        enc_output = torch.cat((yolo_output, cnn_output), dim=1)
        enc_output = self.dropout(enc_output)
        enc_output = self.batchnorm(enc_output)
        enc_output = enc_output.unsqueeze(1)
        enc_output = enc_output.reshape(enc_output.size(0), 2, -1)
        
        return self.decoder_forward(enc_output, captions)
        
    def caption_images(self, images, vocabulary, mode="precomputed", max_length=40):
        self.eval()
        with torch.no_grad():
            # Encode the image
            if mode == "precomputed":
                enc_output = images
            else:
                enc_output = self.yolocnn.forward(images)
                
            yolo_output = self.fc_yolo(enc_output[:, :self.yolo_output_size])
            cnn_output = self.fc_cnn(enc_output[:, self.yolo_output_size:])
            enc_output = torch.cat((yolo_output, cnn_output), dim=1)
            enc_output = self.dropout(enc_output)
            enc_output = self.batchnorm(enc_output)
            enc_output = enc_output.unsqueeze(1)
            enc_output = enc_output.reshape(enc_output.size(0), 2, -1)
        
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
                enc_output = self.yolocnn.forward(images)

            yolo_output = self.fc_yolo(enc_output[:, :self.yolo_output_size])
            cnn_output = self.fc_cnn(enc_output[:, self.yolo_output_size:])
            enc_output = torch.cat((yolo_output, cnn_output), dim=1)
            enc_output = self.dropout(enc_output)
            enc_output = self.batchnorm(enc_output)
            enc_output = enc_output.unsqueeze(1)
            enc_output = enc_output.reshape(enc_output.size(0), 2, -1)
            enc_output = enc_output.unsqueeze(1).expand(-1, beam_width, -1, -1)
            enc_output = enc_output.reshape(batch_size * beam_width, 2, -1)
            
            return self.beam_search_inference(enc_output, batch_size, vocabulary, beam_width, max_length)