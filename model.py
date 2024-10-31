import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        self.inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, aux_logits=True)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)

        for param in self.inception.parameters():
            param.requires_grad = False
        
        for param in self.inception.fc.parameters():
            param.requires_grad = True

    def forward(self, images, is_training=True):
        if is_training:
            features = self.inception(images).logits

        else:
            self.inception.eval()  
            with torch.no_grad():
                features = self.inception(images)
 
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features, captions):
        batch_size = features.size(0)
        # features: [batch_size, embed_size]
        # captions: [batch_size, seq_len-1] (S0, ... S(n-1))
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.lstm(embeddings)

        # [batch_size, seq_len, hidden_size]
        hiddens = hiddens.reshape(-1, self.hidden_size)
        # [batch_size * seq_len, hidden_size]
        outputs = self.linear(hiddens)
        # [batch_size * seq_len, vocab_size]
        outputs = self.softmax(outputs)
        # [batch_size * seq_len, vocab_size]

        outputs = outputs.reshape(batch_size, -1, self.vocab_size)
        # [batch_size, seq_len, vocab_size]
        return outputs


class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions, is_training=True):
        features = self.encoderCNN(images, is_training)
        outputs = self.decoderRNN(features, captions)
        return outputs

    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []

        with torch.no_grad():
            x = self.encoderCNN(image, False).unsqueeze(0)
            states = None

            for _ in range(max_length):
                hiddens, states = self.decoderRNN.lstm(x, states)
                output = self.decoderRNN.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)
                result_caption.append(predicted.item())
                x = self.decoderRNN.embed(predicted).unsqueeze(0)

                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break

        return [vocabulary.itos[idx] for idx in result_caption]