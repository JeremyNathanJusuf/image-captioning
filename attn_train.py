import os
import json
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint, load_checkpoint, transform
from get_loader import get_loader
from attn_model import JoshNameThisModel
from nlgmetricverse import NLGMetricverse, load_metric
import yaml

# TODO merge with the train.py file

# Configurations
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

embed_size = int(config['model']['embed_size'])
hidden_size = int(config['model']['hidden_size'])
num_layers = int(config['model']['num_layers'])

learning_rate = float(config['training']['learning_rate'])
num_epochs = int(config['training']['num_epochs'])
num_workers = int(config['training']['num_workers'])
batch_size = 64

load_model = bool(config['checkpoint']['load_model'])
save_model = bool(config['checkpoint']['save_model'])



def train():
    
    train_loader, val_loader, _, train_dataset, _, _ = get_loader(
        root_folder="./flickr30k/images",
        annotation_file="./flickr30k/captions.txt",
        transform=transform,
        num_workers=num_workers,
        batch_size=batch_size
    )
    vocab_size = len(train_dataset.vocab)

    # torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # for tensorboard
    writer = SummaryWriter("runs/flickr")
    step = 0

    # initialize model, loss etc
    dropout = 0.2
    num_heads = 16
    max_seq_length = 50
    model = JoshNameThisModel(embed_size, vocab_size, num_heads, num_layers, dropout, max_seq_length).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # for param in model.encoderCNN.parameters():
    #     param.requires_grad = False
    # for param in model.encoderCNN.inception.fc.parameters():
    #     param.requires_grad = True
    # for param in model.encoderCNN.fc_last.parameters():
    #     param.requires_grad = True


    if load_model:
        step = load_checkpoint(torch.load("./checkpoints/checkpoint_epoch_60.pth.tar"), model, optimizer)

    train_losses, val_losses = [], []
    train_bleus, val_bleus = [], []
    train_meteors, val_meteors = [], []
    train_ciders, val_ciders = [], []
    

    bleu = NLGMetricverse(metrics=load_metric("bleu"))
    meteor = NLGMetricverse(metrics=load_metric("meteor"))
    cider = NLGMetricverse(metrics=load_metric("cider"))

    eval_every = 5 

    for epoch in range(num_epochs):
        # Uncomment the line below to see a couple of test cases
        # print_examples(model, device, dataset)
        
        train_loss, val_loss = 0, 0
        train_bleu, train_meteor, train_cider = 0, 0, 0
        val_bleu, val_meteor, val_cider = 0, 0, 0
        
        print(f"[Epoch {epoch} / {num_epochs}]")
        
        model.train()

        for idx, (imgs, captions, caption_tokens) in tqdm(
            enumerate(train_loader), total=len(train_loader), leave=False
        ):
            imgs = imgs.to(device)
            captions = captions.to(device)
            
            outputs = model(imgs, captions[:, :-1], is_training=True)
            # print("outputs:", outputs.size())
            # print("captions:", captions.size())
            loss = criterion(
                outputs.reshape(-1, outputs.shape[2]), captions[:, 1:].reshape(-1)
            )
            
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()

            if (epoch+1) % eval_every == 0:
                # Get predicted indices
                predicted_indices = outputs.argmax(dim=2)  # Shape: (Batch_size, Timestep)

                # Convert indices to words for each sequence in the batch
                pred_tokens = []
                for i, batch in enumerate(predicted_indices):  # Iterate over each example in batch
                    sentence = [train_dataset.vocab.itos[idx.item()] for idx in batch]
                    pred_tokens.append(' '.join(sentence))
                    caption_tokens[i] = ' '.join(caption_tokens[i])
                
                # Filter out empty predictions and references
                filtered_pred_tokens = []
                filtered_caption_tokens = []

                for pred, ref in zip(pred_tokens, caption_tokens):
                        if len(pred) > 0 and len(ref) > 0:  # Only include non-empty sequences
                            filtered_pred_tokens.append(pred)
                            filtered_caption_tokens.append(ref)
            
                # Calculate metrics with filtered lists
                if len(filtered_pred_tokens) > 0 and len(filtered_caption_tokens) > 0:
                    train_bleu += bleu(predictions=filtered_pred_tokens, 
                                    references=filtered_caption_tokens, 
                                    reduce_fn='mean')['bleu']['score']

                    train_meteor += meteor(predictions=filtered_pred_tokens, 
                                        references=filtered_caption_tokens, 
                                        reduce_fn='mean')['meteor']['score']    

                    train_cider += cider(predictions=filtered_pred_tokens, 
                                        references=filtered_caption_tokens, 
                                        reduce_fn='mean')['cider']['score']
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        writer.add_scalar("Training loss", train_loss, global_step=epoch)

        if epoch % eval_every == 0:
            train_bleu /= len(train_loader)
            train_meteor /= len(train_loader)
            train_cider /= len(train_loader)
            train_bleus.append(train_bleu)
            train_meteors.append(train_meteor)
            train_ciders.append(train_cider)
            writer.add_scalar("Training bleu", train_bleu, global_step=epoch)
            writer.add_scalar("Training meteor", train_meteor, global_step=epoch)
            writer.add_scalar("Training cider", train_cider, global_step=epoch)

        
        model.eval()
        with torch.no_grad():
            for idx, (imgs, captions, caption_tokens) in tqdm(
                enumerate(val_loader), total=len(val_loader), leave=False
            ):
                imgs = imgs.to(device)
                captions = captions.to(device)

                outputs = model(imgs, captions[:, :-1], is_training=False)
                loss = criterion(
                    outputs.reshape(-1, outputs.shape[2]), captions[:, 1:].reshape(-1)
                )
                
                # Get predicted indices
                predicted_indices = outputs.argmax(dim=2)  # Shape: (Timestep, Batch_size)
                val_loss += loss.item()
                
                pred_tokens = []
                for i, batch in enumerate(predicted_indices):  # Iterate over each example in batch
                    sentence = [train_dataset.vocab.itos[idx.item()] for idx in batch]
                    pred_tokens.append(' '.join(sentence))
                    caption_tokens[i] = ' '.join(caption_tokens[i])
                
                # Filter out empty predictions and references
                filtered_pred_tokens = []
                filtered_caption_tokens = []

                for pred, ref in zip(pred_tokens, caption_tokens):
                    if len(pred) > 0 and len(ref) > 0:  # Only include non-empty sequences
                        filtered_pred_tokens.append(pred)
                        filtered_caption_tokens.append(ref)
            
                # Calculate metrics with filtered lists
                if len(filtered_pred_tokens) > 0 and len(filtered_caption_tokens) > 0:
                    val_bleu += bleu(predictions=filtered_pred_tokens, 
                                    references=filtered_caption_tokens, 
                                    reduce_fn='mean')['bleu']['score']

                    val_meteor += meteor(predictions=filtered_pred_tokens, 
                                        references=filtered_caption_tokens, 
                                        reduce_fn='mean')['meteor']['score']    

                    val_cider += cider(predictions=filtered_pred_tokens, 
                                        references=filtered_caption_tokens, 
                                        reduce_fn='mean')['cider']['score']
                
        val_loss /= len(val_loader)
        val_bleu /= len(val_loader)
        val_meteor /= len(val_loader)
        val_cider /= len(val_loader)
        val_losses.append(val_loss)
        val_bleus.append(val_bleu)
        val_meteors.append(val_meteor)
        val_ciders.append(val_cider)

        writer.add_scalar("Validation loss", val_loss, global_step=epoch)
        writer.add_scalar("Validation bleu", val_bleu, global_step=epoch)
        writer.add_scalar("Validation meteor", val_meteor, global_step=epoch)
        writer.add_scalar("Validation cider", val_cider, global_step=epoch)
        
        print(f"[Train] loss: {train_loss:.4f} | bleu: {train_bleu:.4f} | meteor: {train_meteor:.4f} | cider: {train_cider:.4f}")
        print(f"[Val]   loss: {val_loss:.4f} | bleu: {val_bleu:.4f} | meteor: {val_meteor:.4f} | cider: {val_cider:.4f}")
        
        if (epoch+1) % 10 == 0 and save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
            filename = f"./checkpoints/checkpoint_epoch_{epoch + 1}.pth.tar"
            save_checkpoint(checkpoint, filename)
            
            metrics = {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_bleus': train_bleus,
                'val_bleus': val_bleus,
                'train_meteors': train_meteors,
                'val_meteors': val_meteors,
                'train_ciders': train_ciders,
                'val_ciders': val_ciders
            }
            
            # Specify the file path
            metrics_file_path = f'./metric_logs/train_val_to_epoch_{epoch+1}.json'

            # Extract the directory from the file path
            metrics_dir = os.path.dirname(metrics_file_path)

            # Create the directory if it doesn't exist
            os.makedirs(metrics_dir, exist_ok=True)

            # Save the metrics dictionary to a JSON file
            with open(metrics_file_path, 'w') as json_file:
                json.dump(metrics, json_file, indent=4)

            print(f"Metrics successfully saved to {metrics_file_path}")

    
    print("Training complete!")
    

    

if __name__ == "__main__":
    train()