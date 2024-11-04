import os
import json
import torch
import yaml
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator  # Added import
from utils import save_checkpoint, load_checkpoint, transform
from get_loader import get_loader
from nlgmetricverse import NLGMetricverse, load_metric

from model import CNNtoRNN
from attn_model import CNNAttentionModel

# Configurations
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

embed_size = int(config['model']['embed_size'])
hidden_size = int(config['model']['hidden_size'])
num_layers = int(config['model']['num_layers'])

num_heads = int(config['attn_model']['num_heads'])
dropout = float(config['attn_model']['dropout'])
max_length = int(config['attn_model']['max_length'])

learning_rate = float(config['training']['learning_rate'])
num_epochs = int(config['training']['num_epochs'])
num_workers = int(config['training']['num_workers'])
batch_size = int(config['training']['batch_size'])

load_model = bool(config['checkpoint']['load_model'])
save_model = bool(config['checkpoint']['save_model'])


def train():
    train_loader, val_loader, _, train_dataset, _, _ = get_loader(
        transform=transform,
        num_workers=num_workers,
        batch_size=batch_size,
        dataset="mscoco"
    )
    vocab_size = len(train_dataset.vocab)
    print("Vocabulary size:", vocab_size)

    accelerator = Accelerator()  # Initialize Accelerator
    device = accelerator.device  # Use accelerator's device
    print(f"Using device: {device}")

    # Initialize SummaryWriter only on the main process
    if accelerator.is_main_process:
        writer = SummaryWriter("runs/flickr")
    else:
        writer = None
    step = 0

    model = CNNtoRNN(embed_size, hidden_size, vocab_size).to(device)
    # model = CNNAttentionModel(embed_size, vocab_size, num_heads, num_layers, dropout, max_length).to(device)

    pad_idx = train_dataset.vocab.stoi['<PAD>']
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(lr=learning_rate, params=model.parameters())
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

    if load_model and accelerator.is_main_process:
        step = load_checkpoint(torch.load("./checkpoints/checkpoint_epoch_60.pth.tar"), model, optimizer)

    # Prepare everything with accelerator
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    train_losses, val_losses = [], []
    train_bleus, val_bleus = [], []
    train_meteors, val_meteors = [], []
    train_ciders, val_ciders = [], []

    bleu = NLGMetricverse(metrics=load_metric("bleu"))
    meteor = NLGMetricverse(metrics=load_metric("meteor"))
    cider = NLGMetricverse(metrics=load_metric("cider"))

    eval_every = 1

    for epoch in range(num_epochs):
        print(f"[Epoch {epoch+1} / {num_epochs}]")
        
        model.train()
        train_loss = 0
        for idx, (imgs, captions, _) in tqdm(
            enumerate(train_loader), total=len(train_loader), leave=False):
            
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs, captions[:, :-1])
            captions = captions[:, 1:]
            loss = criterion(
                outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1)
            )

            train_loss += loss.item()

            optimizer.zero_grad()
            accelerator.backward(loss)  # Use accelerator's backward
            optimizer.step()

        train_loss /= len(train_loader)
        if accelerator.is_main_process:
            train_losses.append(train_loss)
            writer.add_scalar("Training loss", train_loss, global_step=epoch)
            
            print(f"[Training] loss: {train_loss:.4f}")

        # Evaluation
        if (epoch + 1) % eval_every == 0:
            model.eval()
            val_loss = 0

            # Accumulate predictions and references
            all_pred_tokens = []
            all_caption_tokens = []

            with torch.no_grad():
                for idx, (imgs, captions, ref_captions) in tqdm(
                    enumerate(val_loader), total=len(val_loader), leave=False
                ):
                    imgs = imgs.to(device)
                    captions = captions.to(device)

                    outputs = model(imgs, captions[:, :-1])
                    captions = captions[:, 1:]
                    loss = criterion(
                        outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1)
                    )
                    val_loss += loss.item()

                    generated_captions = model.caption_images(imgs, train_dataset.vocab)
                    caption_tokens = ref_captions

                    # print("Images: ", imgs)
                    print(f"Predicted: {generated_captions[0]}")
                    print(f"Target: {caption_tokens[0]}")
                    
                    all_pred_tokens.extend(generated_captions)
                    all_caption_tokens.extend(caption_tokens)
                    

            val_loss /= len(val_loader)
            if accelerator.is_main_process:
                # Compute metrics on the main process
                val_bleu_score = bleu(
                    predictions=all_pred_tokens,
                    references=all_caption_tokens,
                    reduce_fn='mean')['bleu']['precisions'][0]

                val_meteor_score = meteor(
                    predictions=all_pred_tokens,
                    references=all_caption_tokens,
                    reduce_fn='mean')['meteor']['score']

                val_cider_score = cider(
                    predictions=all_pred_tokens,
                    references=all_caption_tokens,
                    reduce_fn='mean')['cider']['score']

                val_losses.append(val_loss)
                val_bleus.append(val_bleu_score)
                val_meteors.append(val_meteor_score)
                val_ciders.append(val_cider_score)

                writer.add_scalar("Validation loss", val_loss, global_step=epoch)
                writer.add_scalar("Validation BLEU", val_bleu_score, global_step=epoch)
                writer.add_scalar("Validation METEOR", val_meteor_score, global_step=epoch)
                writer.add_scalar("Validation CIDEr", val_cider_score, global_step=epoch)

                print(f"[Validation] loss: {val_loss:.4f} | BLEU: {val_bleu_score:.4f} | "
                      f"METEOR: {val_meteor_score:.4f} | CIDEr: {val_cider_score:.4f}")

        scheduler.step()

        # Checkpoint saving
        if (epoch + 1) % 10 == 0 and save_model and accelerator.is_main_process:
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
                'val_bleus': val_bleus,
                'val_meteors': val_meteors,
                'val_ciders': val_ciders
            }

            # Save metrics to a JSON file
            metrics_file_path = f'./metric_logs/train_val_to_epoch_{epoch+1}.json'
            os.makedirs(os.path.dirname(metrics_file_path), exist_ok=True)
            with open(metrics_file_path, 'w') as json_file:
                json.dump(metrics, json_file, indent=4)

            print(f"Metrics successfully saved to {metrics_file_path}")

    if accelerator.is_main_process:
        print("Training complete!")


if __name__ == "__main__":
    train()
