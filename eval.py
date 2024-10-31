import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from utils import load_checkpoint, transform
from get_loader import get_loader
from model import CNNtoRNN
from nlgmetricverse import NLGMetricverse, load_metric
import yaml

# Configurations
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

embed_size = int(config['model']['embed_size'])
hidden_size = int(config['model']['hidden_size'])
num_layers = int(config['model']['num_layers'])

learning_rate = float(config['training']['learning_rate'])
num_epochs = int(config['training']['num_epochs'])
num_workers = int(config['training']['num_workers'])
batch_size = int(config['training']['batch_size'])

load_model = bool(config['checkpoint']['load_model'])
save_model = bool(config['checkpoint']['save_model'])

def eval():

    _, _, test_loader, train_dataset, _, _ = get_loader(
        root_folder="./flickr30k/images",
        annotation_file="./flickr30k/captions.txt",
        transform=transform,
        num_workers=num_workers,
        batch_size=batch_size
    )
    vocab_size = len(train_dataset.vocab)

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initialize model, loss etc
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    load_checkpoint(torch.load("./checkpoints/checkpoint_epoch_50.pth.tar"), model, optimizer)

    bleu = NLGMetricverse(metrics=load_metric("bleu"))
    meteor = NLGMetricverse(metrics=load_metric("meteor"))
    cider = NLGMetricverse(metrics=load_metric("cider"))
        
    test_loss = 0
    test_bleu, test_meteor, test_cider = 0, 0, 0
        
    model.eval()
    with torch.no_grad():
        for idx, (imgs, captions, caption_tokens) in tqdm(
            enumerate(test_loader), total=len(test_loader), leave=False
        ):
            print(f"Batch {idx}")
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs, captions[:-1], is_training=False)
            loss = criterion(
                outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1)
            )
            
            test_loss += loss.item()
            predicted_indices = outputs.argmax(dim=2)
            
            pred_tokens = []
            for i, batch in enumerate(predicted_indices.transpose(0, 1)):  # Iterate over each example in batch
                sentence = [train_dataset.vocab.itos[idx.item()] for idx in batch]
                pred_tokens.append(' '.join(sentence))
                caption_tokens[i] = ' '.join(caption_tokens[i][:-1])

                print(f"\nCORRECT: {caption_tokens[i]}")
                print(f"OUTPUT: {pred_tokens[i]}\n")
            
            # Filter out empty predictions and references
            filtered_pred_tokens = []
            filtered_caption_tokens = []

            for pred, ref in zip(pred_tokens, caption_tokens):
                if len(pred) > 0 and len(ref) > 0:  # Only include non-empty sequences
                    filtered_pred_tokens.append(pred)
                    filtered_caption_tokens.append(ref)
        
            # Calculate metrics with filtered lists
            if len(filtered_pred_tokens) > 0 and len(filtered_caption_tokens) > 0:
                test_bleu += bleu(predictions=filtered_pred_tokens, 
                                references=filtered_caption_tokens, 
                                reduce_fn='mean')['bleu']['score']

                test_meteor += meteor(predictions=filtered_pred_tokens, 
                                    references=filtered_caption_tokens, 
                                    reduce_fn='mean')['meteor']['score']    

                test_cider += cider(predictions=filtered_pred_tokens, 
                                    references=filtered_caption_tokens, 
                                    reduce_fn='mean')['cider']['score']
                
        test_loss /= len(test_loader)
        test_bleu /= len(test_loader)
        test_meteor /= len(test_loader)
        test_cider /= len(test_loader)
        
        print(f"[test] loss: {test_loss:.4f} | bleu: {test_bleu:.4f} | meteor: {test_meteor:.4f} | cider: {test_cider:.4f}")


if __name__ == "__main__":
    eval()