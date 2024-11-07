import os
import json
import torch
import yaml
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
from utils import save_checkpoint, load_checkpoint, transform
from get_loader import get_loader
from nlgmetricverse import NLGMetricverse, load_metric

from model import CNNtoRNN
from attn_model import CNNAttentionModel

# Configurations
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

learning_rate = float(config['training']['learning_rate'])
num_epochs = int(config['training']['num_epochs'])
num_workers = int(config['training']['num_workers'])
batch_size = int(config['training']['batch_size'])
step_size = int(config['training']['step_size'])
gamma = float(config['training']['gamma'])
model_arch = config['training']['model_arch']
dataset = config['training']['dataset']
inference_type = config['training']['inference_type']
beam_width = int(config['training']['beam_width'])
save_model = bool(config['training']['save_model'])
load_model = bool(config['training']['load_model'])
checkpoint_dir = config['training']['checkpoint_dir']

if 'rnn_model' in config:
    rnn_embed_size = int(config['rnn_model']['embed_size'])
    rnn_hidden_size = int(config['rnn_model']['hidden_size'])

if 'attn_model' in config:
    attn_embed_size = int(config['attn_model']['embed_size'])
    attn_num_layers = int(config['attn_model']['num_layers'])
    attn_num_heads = int(config['attn_model']['num_heads'])


def train(model_arch=model_arch, dataset=dataset):
    train_loader, val_loader, _, train_dataset, _, _ = get_loader(
        transform=transform,
        num_workers=num_workers,
        batch_size=batch_size,
        dataset=dataset
    )
    vocab_size = len(train_dataset.vocab)
    print("Vocabulary size:", vocab_size)

    accelerator = Accelerator()  # Initialize Accelerator
    device = accelerator.device  # Use accelerator's device
    print(f"Using device: {device}")

    # Initialize SummaryWriter only on the main process
    if accelerator.is_main_process:
        if not os.path.exists(f"runs/{dataset}/{model_arch}"):
            os.makedirs(f"runs/{dataset}/{model_arch}")
        writer = SummaryWriter(f"runs/{dataset}/{model_arch}")
    else:
        writer = None
    step = 0

    if model_arch == "cnn-rnn":
        model = CNNtoRNN(rnn_embed_size, rnn_hidden_size, vocab_size).to(device)
    elif model_arch == "cnn-attn":
        model = CNNAttentionModel(attn_embed_size, vocab_size, attn_num_heads, attn_num_layers).to(device)
    else:
        raise ValueError("Model not recognized")
    
    
    pad_idx = train_dataset.vocab.stoi['<PAD>']
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(lr=learning_rate, params=model.parameters())
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
    
    if load_model:
        step = load_checkpoint(torch.load(checkpoint_dir), model, optimizer)
        
    # Prepare everything with accelerator
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    train_losses = []
    val_bleus = []
    val_meteors = []
    val_ciders = []

    bleu = NLGMetricverse(metrics=load_metric("bleu"))
    meteor = NLGMetricverse(metrics=load_metric("meteor"))
    cider = NLGMetricverse(metrics=load_metric("cider"))

    eval_every = 1

    for epoch in range(num_epochs):
        print(f"[Epoch {epoch+1} / {num_epochs}]")
        
        # model.train()
        # train_loss = 0
        # for idx, (imgs, captions, _) in tqdm(
        #     enumerate(train_loader), total=len(train_loader), leave=False):
            
        #     imgs = imgs.to(device)
        #     captions = captions.to(device)

        #     outputs = model(imgs, captions[:, :-1])
        #     captions = captions[:, 1:]
        #     loss = criterion(
        #         outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1)
        #     )

        #     train_loss += loss.item()

        #     optimizer.zero_grad()
        #     accelerator.backward(loss)  # Use accelerator's backward
        #     optimizer.step()

        # train_loss /= len(train_loader)
        # if accelerator.is_main_process:
        #     train_losses.append(train_loss)
        #     writer.add_scalar("Training loss", train_loss, global_step=epoch)
            
        #     print(f"[Training] loss: {train_loss:.4f}")

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
                    if inference_type == 'greedy':
                        generated_captions = model.caption_images(imgs, train_dataset.vocab)
                    elif inference_type == 'beam':
                        generated_captions = model.caption_images_beam_search(imgs, train_dataset.vocab, beam_width)
                    else:
                        raise ValueError("Inference type not recognized")

                    # print("Images: ", imgs)
                    print(f"Predicted: {generated_captions[0]}")
                    print(f"Target: {ref_captions[0]}")
                    
                    all_pred_tokens.extend(generated_captions)
                    all_caption_tokens.extend(ref_captions)
                    

            # val_loss /= len(val_loader)
            if accelerator.is_main_process:
                # Compute metrics on the main process
                val_bleu_score = bleu(
                    predictions=all_pred_tokens,
                    references=all_caption_tokens,
                    reduce_fn='mean')['bleu']['score']

                val_meteor_score = meteor(
                    predictions=all_pred_tokens,
                    references=all_caption_tokens,
                    reduce_fn='mean')['meteor']['score']

                val_cider_score = cider(
                    predictions=all_pred_tokens,
                    references=all_caption_tokens,
                    reduce_fn='mean')['cider']['score']

                # val_losses.append(val_loss)
                val_bleus.append(val_bleu_score)
                val_meteors.append(val_meteor_score)
                val_ciders.append(val_cider_score)

                # writer.add_scalar("Validation loss", val_loss, global_step=epoch)
                writer.add_scalar("Validation BLEU", val_bleu_score, global_step=epoch)
                writer.add_scalar("Validation METEOR", val_meteor_score, global_step=epoch)
                writer.add_scalar("Validation CIDEr", val_cider_score, global_step=epoch)

                print(f"BLEU: {val_bleu_score:.4f} | METEOR: {val_meteor_score:.4f} | CIDEr: {val_cider_score:.4f}")

        scheduler.step()

        # Checkpoint saving
        if save_model:
            if (epoch + 1) % 10 == 0 and accelerator.is_main_process:
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                }
                if not os.path.exists(f"./checkpoints/{model_arch}/{dataset}"):
                    os.makedirs(f"./checkpoints/{model_arch}/{dataset}")
                filename = f"./checkpoints/{model_arch}/{dataset}/checkpoint_epoch_{epoch + 1}.pth.tar"
                save_checkpoint(checkpoint, filename)

                metrics = {
                    'train_losses': train_losses,
                    'val_bleus': val_bleus,
                    'val_meteors': val_meteors,
                    'val_ciders': val_ciders
                }

                # Save metrics to a JSON file
                if not os.path.exists(f'./metric_logs/{model_arch}/{dataset}'):
                    os.makedirs(f'./metric_logs/{model_arch}/{dataset}')
                metrics_file_path = f'./metric_logs/{model_arch}/{dataset}/train_val_to_epoch_{epoch+1}.json'
                os.makedirs(os.path.dirname(metrics_file_path), exist_ok=True)
                with open(metrics_file_path, 'w') as json_file:
                    json.dump(metrics, json_file, indent=4)

                print(f"Metrics successfully saved to {metrics_file_path}")

    if accelerator.is_main_process:
        print("Training complete!")


if __name__ == "__main__":
    train()
