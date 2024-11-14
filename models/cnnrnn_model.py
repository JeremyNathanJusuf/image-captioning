import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.encoder_cnn import EncoderCNN
   
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, dropout=0.2):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.fc1 = nn.Linear(2048, embed_size, bias = False)
        self.dropout = nn.Dropout(dropout)
        self.batchnorm = nn.BatchNorm1d(embed_size)

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc2 = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions):
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
        self.encoderCNN = EncoderCNN()
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size)

    def forward(self, images, captions, mode='precomputed'):
        if mode == 'precomputed':
            features = images
        else:
            with torch.no_grad():
                features = self.encoderCNN(images)
                
        outputs = self.decoderRNN(features, captions)
        return outputs
    
    def precompute_image(self, images):
        with torch.no_grad():
            features = self.encoderCNN(images)
        return features
    
    def caption_images(self, images, vocabulary, mode="precomputed", max_length=40):
        self.eval()
        batch_size = images.size(0)  # Get the batch size from images
        result_captions = [[] for _ in range(batch_size)]  # Initialize captions list for each image in the batch
        done = [False] * batch_size  # Tracks completion for each item in the batch

        with torch.no_grad():
            # Encode all images in the batch
            if mode == 'precomputed':
                img_features = images
            else:
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
    
    def caption_images_beam_search(self, images, vocabulary, beam_width=3, mode="precomputed", max_length=50):
        self.eval()
        batch_size = images.size(0)  # Get the batch size from images
        # print("Batch size: ", batch_size)
        
        with torch.no_grad():
            # Encode all images in the batch
            if mode == 'precomputed':
                img_features = images
            else:
                img_features = self.encoderCNN(images)
            img_features = self.decoderRNN.dropout(img_features)
            img_features = self.decoderRNN.fc1(img_features)
            img_features = self.decoderRNN.batchnorm(img_features) 
            img_features = img_features.unsqueeze(1)

            # Initialize LSTM states with the encoded image features
            _, (states_h, states_c) = self.decoderRNN.lstm(img_features) # Shape: (1, batch_size, hidden_size)

            sequences = torch.Tensor([[vocabulary.stoi["<SOS>"]]]).repeat(batch_size, beam_width, 1, 1).long().to(images.device)
            scores = torch.zeros(batch_size, beam_width, dtype=torch.float, device=images.device)
            states_h = states_h.unsqueeze(2).repeat(1, 1, beam_width, 1)
            states_c = states_c.unsqueeze(2).repeat(1, 1, beam_width, 1)
            done = torch.zeros(batch_size, beam_width, dtype=torch.bool, device=images.device) # Shape: (batch_size, beam_width)
            lengths = torch.zeros(batch_size, beam_width, dtype=torch.long, device=images.device) # Shape: (batch_size, beam_width)

            for i in range(max_length):

                seq_inp = sequences.reshape(batch_size * beam_width, -1, 1)  # Shape: (batch_size * beam_width, seq_len, 1)
                states_c = states_c.reshape(1, batch_size * beam_width, -1)  # Shape: (1, batch_size * beam_width, hidden_size)
                states_h = states_h.reshape(1, batch_size * beam_width, -1) # Shape: (1, batch_size * beam_width, hidden_size)

                embedding = self.decoderRNN.embed(seq_inp[:, -1, :]) # Shape: (batch_size * beam_width, 1, embed_size)
                output, (states_h, states_c) = self.decoderRNN.lstm(embedding, (states_h, states_c))
                output = self.decoderRNN.fc2(self.decoderRNN.dropout(output.squeeze(1)))  # Shape: (batch_size * beam_width, vocab_size)
                output = output.reshape(batch_size, beam_width, -1)  # Shape: (batch_size, beam_width, vocab_size)
                states_h = states_h.reshape(1, batch_size, beam_width, -1) # Shape: (1, batch_size, beam_width, hidden_size)
                states_c = states_c.reshape(1, batch_size, beam_width, -1)  # Shape: (1, batch_size, beam_width, hidden_size)

                log_probs = F.log_softmax(output, dim=2) # Shape: (batch_size, beam_width, vocab_size)

                # take top beam_width sequences for each batch
                top_log_probs, top_indices = log_probs.topk(beam_width, dim=2)  # Shapes: (batch_size, beam_width, beam_width)

                new_scores = (
                    scores.unsqueeze(-1) + (1 - done.unsqueeze(-1).float()) * top_log_probs
                )

                # if done for 1 batch and 1 beam, only keep 1 best score and set others to -inf
                mask = done.unsqueeze(-1)
                mask = mask.expand(-1, -1, beam_width)
                mask[:, :, 0] = False # keep the best score
                if i == 0:
                    bool_mask = torch.zeros_like(mask, dtype=torch.bool)
                    bool_mask[:, 1:, :] = True
                    mask = mask | bool_mask
                new_scores = new_scores.masked_fill(mask, float("-inf"))
                new_scores = new_scores.reshape(batch_size, -1)  # Shape: (batch_size, beam_width*beam_width)

                # Get the top beam_width sequences (take sequences, scores and states)
                top_scores, all_top_indices = new_scores.topk(beam_width, dim=-1)  # Shapes: (batch_size, beam_width)
                scores = top_scores # Shape: (batch_size, beam_width)

                # all top indices from [0, beam_width*beam_width)
                beam_indices = all_top_indices // beam_width # previous beam index
                token_indices = all_top_indices % beam_width # current token index
                batch_indices = torch.arange(batch_size).unsqueeze(-1).expand(-1, beam_width).to(beam_indices.device)
                new_tokens = top_indices[batch_indices, beam_indices, token_indices]
                new_tokens = new_tokens.unsqueeze(-1).unsqueeze(-1)  # Shape: (batch_size, beam_width, 1, 1)

                prv_seq_indices = beam_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, sequences.size(2), -1)  # Shape: (batch_size, beam_width, seq_len, 1)
                prv_seq_tokens = sequences.gather(dim=1, index=prv_seq_indices)
                sequences = torch.cat((prv_seq_tokens, new_tokens), dim=2)

                prv_state_indices = beam_indices.unsqueeze(-1).unsqueeze(0).expand(-1, -1, -1, states_h.size(3))
                states_h = states_h.gather(dim=2, index=prv_state_indices)
                states_c = states_c.gather(dim=2, index=prv_state_indices)
                
                # update done based on last token if it is <EOS>
                done = done.gather(dim=1, index=beam_indices) 
                done = done | (new_tokens.reshape(done.shape) == vocabulary.stoi["<EOS>"])
                lengths = lengths.gather(dim=1, index=beam_indices)
                lengths += done.logical_not().long()

                if done.all():
                    break

        result_captions = []
        for i in range(batch_size):
            # stop at the first <EOS> token
            caption = sequences[i][0].squeeze(1).tolist()
            if vocabulary.stoi["<EOS>"] in caption:
                caption = caption[1:caption.index(vocabulary.stoi["<EOS>"])]
            else:
                caption = caption[1:]
            result_captions.append(caption)
        captions_text = [' '.join([vocabulary.itos[idx] for idx in caption]) for caption in result_captions]
        return captions_text
