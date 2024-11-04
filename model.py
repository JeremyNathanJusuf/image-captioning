import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

import torch
import torch.nn as nn
import torchvision.models as models

import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        # Initialize the InceptionV3 model with pretrained weights
        self.inception = models.inception_v3(pretrained=True)
        
        # Dictionary to store activation from the desired layer
        self.activation = {}

        # Register forward hook on the Mixed_7c layer to capture features
        self.inception.Mixed_7c.register_forward_hook(self.get_activation("Mixed_7c"))

        # Pooling layer to convert the feature map to 1x1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer to map to the desired embedding size
        self.fc = nn.Linear(2048, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.batchnorm = nn.BatchNorm1d(embed_size)

        for param in self.inception.parameters():
            param.requires_grad = False

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
        features = self.dropout(features)
        features = self.fc(features)
        features = self.batchnorm(features)
        
        return features



class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        # features: [batch_size, embed_size]
        # captions: [batch_size, seq_len-1] (S0, ... S(n-1))

        embeddings = self.embed(captions)  # [batch_size, seq_len-1, embed_size]
        # put to dropout
        embeddings = self.dropout(embeddings)
        
        _, states = self.lstm(features.unsqueeze(1))  # [batch_size, seq_len-1, hidden_size]

        # Run the LSTM with the image features as the initial hidden state
        outputs, _ = self.lstm(embeddings, states)

        # Pass through the dropout and fully connected layers
        outputs = self.dropout(outputs)  # Shape: (batch_size, sequence_length, hidden_size)
        outputs = self.fc(outputs)       # Shape: (batch_size, sequence_length, vocab_size)
        return outputs

class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)
        return outputs

    def caption_images(self, images, vocabulary, max_length=20):
        self.eval()
        batch_size = images.size(0)  # Get the batch size from images
        result_captions = [[] for _ in range(batch_size)]  # Initialize captions list for each image in the batch
        done = [False] * batch_size  # Tracks completion for each item in the batch

        with torch.no_grad():
            states = None

            # Encode all images in the batch
            img_features = self.encoderCNN(images)  # img_features shape: (batch_size, feature_dim)
            hiddens, states = self.decoderRNN.lstm(img_features, states)

            # Initialize the first input with <SOS> token for each batch item
            first_inputs = torch.tensor([vocabulary.stoi["<SOS>"]] * batch_size).to(images.device)  # Shape: (batch_size)
            emb = self.decoderRNN.embed(first_inputs)  # Embedding shape: (batch_size, embed_dim)

            for _ in range(max_length):
                output, states = self.decoderRNN.lstm(emb, states)  # LSTM step for the whole batch
                output = self.decoderRNN.fc(self.decoderRNN.dropout(output))  # Shape: (batch_size, vocab_size)

                # Sample the token with log softmax probabilities
                # Apply log softmax to get log probabilities
                log_probs = F.log_softmax(output, dim=-1)  # Shape: (batch_size, vocab_size)

                # Sample from the distribution based on probabilities
                predicted = torch.multinomial(log_probs.exp(), 1).squeeze(1)  # Shape: (batch_size)


                # Append the predicted token to each corresponding caption in the batch
                for i in range(batch_size):
                    if not done[i]:  # Only proceed if the caption is not yet marked as done
                        token = vocabulary.itos[predicted[i].item()]
                        if token == "<EOS>":
                            done[i] = True  # Mark this sequence as complete
                        else:
                            result_captions[i].append(predicted[i].item())

                # If all items are done, break early
                if all(done):
                    break

                # Update the embeddings with the latest predictions for the next step
                emb = self.decoderRNN.embed(predicted)  # Shape: (batch_size, embed_dim)

        # Convert token indices to words for each caption in the batch
        captions_text = [' '.join([vocabulary.itos[idx] for idx in caption]) for caption in result_captions]
        return captions_text
