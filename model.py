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
        self.inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        
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



class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.fc1 = nn.Linear(2048, embed_size, bias = False)
        self.dropout = nn.Dropout(0.5)
        self.batchnorm = nn.BatchNorm1d(embed_size)

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc2 = nn.Linear(hidden_size, vocab_size)
        

    def forward(self, features, captions):
        # features: [batch_size, embed_size]
        # captions: [batch_size, seq_len-1] (S0, ... S(n-1))
        
        # Project the features back to the embedding size
        features = self.dropout(features)
        features = self.fc1(features)  # [batch_size, embed_size]
        features = self.batchnorm(features)
        features = features.unsqueeze(1)

        embeddings = self.embed(captions)
        embeddings = self.dropout(embeddings)
        _, states = self.lstm(features)
        outputs, _ = self.lstm(embeddings, states)

        # Pass through the dropout and fully connected layers
        outputs = self.fc2(outputs)
        return outputs

class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size)

    def forward(self, images, captions):
        with torch.no_grad():
            features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)
        return outputs
    
    def caption_images(self, images, vocabulary, max_length=40):
        self.eval()
        batch_size = images.size(0)  # Get the batch size from images
        result_captions = [[] for _ in range(batch_size)]  # Initialize captions list for each image in the batch
        done = [False] * batch_size  # Tracks completion for each item in the batch

        with torch.no_grad():
            # Encode all images in the batch
            img_features = self.encoderCNN(images)  # img_features shape: (batch_size, feature_dim)
            img_features = self.decoderRNN.dropout(img_features)
            img_features = self.decoderRNN.fc1(img_features)  # [batch_size, embed_size]
            img_features = self.decoderRNN.batchnorm(img_features)
            img_features = img_features.unsqueeze(1)

            # Initialize LSTM states with the encoded image features
            _, states = self.decoderRNN.lstm(img_features)

            # Initialize the first input with <SOS> token for each batch item
            first_inputs = torch.full((batch_size,), vocabulary.stoi["<SOS>"], device=images.device)
            emb = self.decoderRNN.embed(first_inputs)  # Embedding shape: (batch_size, embed_size)

            
            for i in range(max_length):
                output, states = self.decoderRNN.lstm(emb.unsqueeze(1), states)  # LSTM step for the whole batch
                output = self.decoderRNN.fc2(self.decoderRNN.dropout(output.squeeze(1)))  # Shape: (batch_size, vocab_size)

                # Apply log softmax to get log probabilities
                log_probs = F.log_softmax(output, dim=-1)  # Shape: (batch_size, vocab_size)

                # Select the highest probability token for each item in the batch
                predicted = log_probs.argmax(dim=-1)  # Shape: (batch_size)

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
                emb = self.decoderRNN.embed(predicted)  # Shape: (batch_size, embed_size)

        # Convert token indices to words for each caption in the batch
        captions_text = [' '.join([vocabulary.itos[idx] for idx in caption]) for caption in result_captions]
        return captions_text
    
    def caption_images_beam_search(self, images, vocabulary, beam_width=20, max_length=40):
        self.eval()
        batch_size = images.size(0)  # Get the batch size from images
        
        with torch.no_grad():
            # Encode all images in the batch
            img_features = self.encoderCNN(images)  # img_features shape: (batch_size, feature_dim)
            img_features = self.decoderRNN.dropout(img_features)
            img_features = self.decoderRNN.fc1(img_features)  # [batch_size, embed_size]
            img_features = self.decoderRNN.batchnorm(img_features)
            img_features = img_features.unsqueeze(1)

            # Initialize LSTM states with the encoded image features
            _, states = self.decoderRNN.lstm(img_features)
            
            done = [[False]*beam_width for _ in range(batch_size)]
            sequences = [[([vocabulary.stoi["<SOS>"]], 0.0, 
                           (states[0][:, batch_idx, :], states[1][:, batch_idx, :])
                        )] for batch_idx in range(batch_size)]  # (sequence, score, states)

            for length in range(max_length):
                all_candidates = [[] for _ in range(batch_size)]
                
                for batch_idx in range(batch_size):
                    candidates = sequences[batch_idx]
                    
                    if all(done[batch_idx]):
                        all_candidates[batch_idx] = candidates
                        continue
                    
                    for seq, score, states in candidates:
                        if seq[-1] == vocabulary.stoi["<EOS>"]:
                            all_candidates[batch_idx].append((seq, score, states))
                            continue
                        
                        embedding = self.decoderRNN.embed(torch.tensor([seq[-1]], device=images.device))
                        output, states = self.decoderRNN.lstm(embedding, states)
                        
                        output = self.decoderRNN.fc2(self.decoderRNN.dropout(output.squeeze(1)))
                        log_probs = F.log_softmax(output, dim=-1)
                        top_log_probs, top_tokens = log_probs.topk(beam_width, dim=-1)

                        for i in range(beam_width):
                            pred_token = top_tokens[0][i].item()
                            if pred_token == vocabulary.stoi["<EOS>"]:
                                done[batch_idx][i] = True
                                
                            candidate = (
                                seq + [pred_token], 
                                # (score*length + top_log_probs[0][i].item())/(length+1),
                                score + top_log_probs[0][i].item(),
                                states
                            )
                            all_candidates[batch_idx].append(candidate)
                
                terminate = True
                for batch_idx in range(batch_size):
                    ordered = sorted(all_candidates[batch_idx], key=lambda x: x[1], reverse=True)
                    sequences[batch_idx] = ordered[:beam_width]
                    terminate = terminate and all(done[batch_idx])
                    
                if terminate:
                    break

        result_captions = [seq[0][0] for seq in sequences]
        captions_text = [' '.join([vocabulary.itos[idx] for idx in caption[1:-1]]) for caption in result_captions]
        return captions_text
