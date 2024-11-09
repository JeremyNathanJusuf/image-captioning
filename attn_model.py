import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math

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

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output
    
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, None)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class CNNAttentionModel(nn.Module):
    def __init__(self, embed_size, vocab_size, num_heads, num_layers, dropout=0.1, max_seq_length=50):
        super(CNNAttentionModel, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size)

        self.fc1 = nn.Linear(2048, embed_size, bias = False)
        self.dropout = nn.Dropout(dropout)
        self.batchnorm = nn.BatchNorm1d(embed_size)

        self.decoder_layers = nn.ModuleList([DecoderLayer(embed_size, num_heads, embed_size, dropout) for _ in range(num_layers)])
        self.positional_encoding = PositionalEncoding(embed_size, max_seq_length)
        self.fc2 = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.decoder_embedding = nn.Embedding(vocab_size, embed_size)
        # self.softmax = nn.Softmax(dim=2)
        
    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool().to(tgt.device)
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask
    
    def forward(self, images, captions):
        with torch.no_grad():
            enc_output = self.encoderCNN(images)
        enc_output = self.fc1(enc_output)
        enc_output = self.dropout(enc_output)
        enc_output = self.batchnorm(enc_output)

        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(captions)))
        enc_output = enc_output.unsqueeze(1)
        enc_output = enc_output.expand(-1, tgt_embedded.size(1), -1) 
        src_mask, tgt_mask = self.generate_mask(enc_output, captions)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc2(dec_output)
        # output = self.softmax(output)
        # print("dec output:", output.size())
        return output
        
    def caption_images(self, images, vocabulary, max_length=50):
        self.eval()
        with torch.no_grad():
            # Encode the image
            enc_output = self.encoderCNN(images)
            enc_output = self.fc1(enc_output)
            enc_output = self.dropout(enc_output)
            enc_output = self.batchnorm(enc_output)
            enc_output = enc_output.unsqueeze(1)  # Expand dimensions for transformer input

            # Initialize the caption with the <SOS> token
            batch_size = images.size(0)
            caption = torch.full((batch_size, 1), vocabulary.stoi["<SOS>"], device=images.device)

            # Prepare list to hold generated captions for each image in batch
            result_captions = [[] for _ in range(batch_size)]
            done = [False] * batch_size

            # Iterate to generate each word
            for _ in range(max_length):
                # Embed the current sequence and apply positional encoding
                tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(caption)))

                # Generate masks
                src_mask, tgt_mask = self.generate_mask(enc_output, caption)

                # Pass through decoder layers
                dec_output = tgt_embedded
                for layer in self.decoder_layers:
                    dec_output = layer(dec_output, enc_output, src_mask, tgt_mask)

                # Get the last token output for prediction
                output = self.fc2(dec_output[:, -1, :])  # Shape: (batch_size, vocab_size)

                # Select the token with the highest probability
                predicted = output.argmax(dim=-1)

                # Append the predicted token to each caption
                for i in range(batch_size):
                    if not done[i]:  # Only proceed if caption generation is not complete
                        token = vocabulary.itos[predicted[i].item()]
                        if token == "<EOS>":
                            done[i] = True
                        else:
                            result_captions[i].append(predicted[i].item())

                # If all captions are complete, exit early
                if all(done):
                    break

                # Update the input sequence with the predicted tokens for the next iteration
                caption = torch.cat([caption, predicted.unsqueeze(1)], dim=1)

        # Convert the list of token indices to words
        captions_text = [' '.join([vocabulary.itos[idx] for idx in caption]) for caption in result_captions]
        return captions_text

    def caption_images_beam_search(self, images, vocabulary, beam_width=3, max_length=50):
        self.eval()
        with torch.no_grad():
            # Encode the image
            enc_output = self.encoderCNN(images)
            enc_output = self.fc1(enc_output)
            enc_output = self.dropout(enc_output)
            enc_output = self.batchnorm(enc_output)
            enc_output = enc_output.unsqueeze(1)  # Expand dimensions for transformer input

            # Initialize the caption with the <SOS> token
            batch_size = images.size(0)
            
            done = [[False]*beam_width for _ in range(batch_size)]
            sequences = [[([vocabulary.stoi["<SOS>"]], 0.0, 
                        )] for batch_idx in range(batch_size)]  # (sequence, score)

            for length in range(max_length):
                all_candidates = [[] for _ in range(batch_size)]
                
                for batch_idx in range(batch_size):
                    candidates = sequences[batch_idx]
                    
                    if all(done[batch_idx]):
                        all_candidates[batch_idx] = candidates
                        continue
                    
                    for seq, score in candidates:
                        if seq[-1] == vocabulary.stoi["<EOS>"]:
                            all_candidates[batch_idx].append((seq, score))
                            continue
                        
                        tensor_seq = torch.tensor([seq], device=images.device)
                        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tensor_seq)))
                        src_mask, tgt_mask = self.generate_mask(enc_output, tensor_seq)
                        
                        dec_output = tgt_embedded
                        for layer in self.decoder_layers:
                            dec_output = layer(dec_output, enc_output, src_mask, tgt_mask)
                        
                        output = self.fc2(dec_output[:, -1, :]) # Shape: (batch_size, vocab_size)
                        log_probs = F.log_softmax(output, dim=-1)
                        top_log_probs, top_tokens = log_probs.topk(beam_width, dim=-1)

                        for i in range(beam_width):
                            pred_token = top_tokens[0][i].item()
                            if pred_token == vocabulary.stoi["<EOS>"]:
                                done[batch_idx][i] = True
                                
                            candidate = (
                                seq + [pred_token], 
                                # (score*length + top_log_probs[0][i].item())/(length+1),
                                score + top_log_probs[0][i].item()
                            )
                            all_candidates[batch_idx].append(candidate)
                
                terminate = False
                for batch_idx in range(batch_size):
                    ordered = sorted(all_candidates[batch_idx], key=lambda x: x[1], reverse=True)
                    sequences[batch_idx] = ordered[:beam_width]
                    terminate = terminate or all(done[batch_idx])
                    
                if terminate:
                    break

        result_captions = [seq[0][0] for seq in sequences]
        captions_text = [' '.join([vocabulary.itos[idx] for idx in caption]) for caption in result_captions]
        return captions_text