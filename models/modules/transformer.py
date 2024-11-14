import torch
import torch.nn as nn
import math
from abc import ABC, abstractmethod
from torch.nn import functional as F

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
    
class DecoderTransformer(ABC, nn.Module):
    def __init__(self, embed_size, vocab_size, num_heads, num_layers, dropout, max_seq_length):
        super(DecoderTransformer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.decoder_layers = nn.ModuleList([DecoderLayer(embed_size, num_heads, embed_size, dropout) for _ in range(num_layers)])
        self.positional_encoding = PositionalEncoding(embed_size, max_seq_length)
        self.fc2 = nn.Linear(embed_size, vocab_size)
        self.decoder_embedding = nn.Embedding(vocab_size, embed_size)
    
    @abstractmethod
    def precompute_image(self, images):
        pass
        
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def caption_images(self, *args, **kwargs):
        pass

    @abstractmethod
    def caption_images_beam_search(self, *args, **kwargs):
        pass
    
    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool().to(tgt.device)
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask
    
    def decoder_forward(self, enc_output, captions, take_last=False):
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(captions)))
        
        src_mask, tgt_mask = self.generate_mask(enc_output, captions)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        if take_last:
            dec_output = dec_output[:, -1, :]

        output = self.fc2(dec_output)
        return output


    def greedy_inference(self, enc_output, vocabulary, max_length):
        # Initialize the caption with the <SOS> token
        batch_size = enc_output.size(0)
        caption = torch.full((batch_size, 1), vocabulary.stoi["<SOS>"], device=enc_output.device)

        # Prepare list to hold generated captions for each image in batch
        result_captions = [[] for _ in range(batch_size)]
        done = [False] * batch_size

        # Iterate to generate each word
        for _ in range(max_length):
            output = self.decoder_forward(enc_output, caption, take_last=True)

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

    def beam_search_inference(self, enc_output, batch_size, vocabulary, beam_width, max_length):
        
        sequences = torch.Tensor([[vocabulary.stoi["<SOS>"]]]).repeat(batch_size, beam_width, 1, 1).long().to(enc_output.device)
        scores = torch.zeros(batch_size, beam_width, dtype=torch.float, device=enc_output.device)
        done = torch.zeros(batch_size, beam_width, dtype=torch.bool, device=enc_output.device) # Shape: (batch_size, beam_width)
        lengths = torch.zeros(batch_size, beam_width, dtype=torch.long, device=enc_output.device) # Shape: (batch_size, beam_width)

        for i in range(max_length):
            seq_inp = sequences.reshape(batch_size * beam_width, -1)  # Shape: (batch_size * beam_width, seq_len, 1)
            output = self.decoder_forward(enc_output, seq_inp, take_last=True)  # Shape: (batch_size * beam_width, seq_len, embed_size)
            output = output.reshape(batch_size, beam_width, -1)  # Shape: (batch_size, beam_width, vocab_size)
        
            log_probs = F.log_softmax(output, dim=-1) # Shape: (batch_size, beam_width, vocab_size)

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

