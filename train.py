import os
import json
import torch
import yaml
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import pickle
import sys
import argparse

from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
from utils import save_checkpoint, load_checkpoint, transform
from get_loader import get_loader
from nlgmetricverse import NLGMetricverse, load_metric

from models.cnnrnn_model import CNNtoRNN
from models.cnnattn_model import CNNAttentionModel
from models.vitcnnattn_model import VITCNNAttentionModel
from models.vitattn_model import VITAttentionModel
from models.yoloattn_model import YOLOAttentionModel
from models.yolocnnattn_model import YOLOCNNAttentionModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--embed_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=-1)
    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument("--dataset", type=str, default="flickr")
    parser.add_argument("--model_arch", type=str, default="cnn-rnn")
    return parser.parse_args()

def precompute_images(
    model,
    model_arch,
    dataset,
    train_loader,
    val_loader,
    test_loader
):
    print("Precomputing images...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    with torch.no_grad():
        for idx, (img_ids, imgs, captions, _) in tqdm(
            enumerate(train_loader), total=len(train_loader), leave=False):
            
            imgs = imgs.to(device)
            outputs = model.precompute_image(imgs)
            # save computed encoded outputs to precomputed folder
            if not os.path.exists(f'precomputed/{model_arch}/{dataset}'):
                os.makedirs(f'precomputed/{model_arch}/{dataset}')
                
            for i in range(len(img_ids)):
                filepath = f'precomputed/{model_arch}/{dataset}/{img_ids[i].split(".")[0]}.pkl'
                print(filepath, os.path.exists(filepath))
                with open(filepath, 'wb') as f:
                    pickle.dump(outputs[i].cpu(), f)

        for idx, (img_ids, imgs, captions, ref_captions) in tqdm(
            enumerate(val_loader), total=len(val_loader), leave=False
        ):
            imgs = imgs.to(device)
            outputs = model.precompute_image(imgs)
            
            if not os.path.exists(f'precomputed/{model_arch}/{dataset}'):
                os.makedirs(f'precomputed/{model_arch}/{dataset}')
                
            for i in range(len(img_ids)):
                filepath = f'precomputed/{model_arch}/{dataset}/{img_ids[i].split(".")[0]}.pkl'
                print(filepath, os.path.exists(filepath))
                with open(filepath, 'wb') as f:
                    pickle.dump(outputs[i].cpu(), f)
                    
        for idx, (img_ids, imgs, captions, ref_captions) in tqdm(
            enumerate(test_loader), total=len(test_loader), leave=False
        ):
            imgs = imgs.to(device)
            outputs = model.precompute_image(imgs)
            
            if not os.path.exists(f'precomputed/{model_arch}/{dataset}'):
                os.makedirs(f'precomputed/{model_arch}/{dataset}')
                
            for i in range(len(img_ids)):
                filepath = f'precomputed/{model_arch}/{dataset}/{img_ids[i].split(".")[0]}.pkl'
                print(filepath, os.path.exists(filepath))
                with open(filepath, 'wb') as f:
                    pickle.dump(outputs[i].cpu(), f)
                    

def get_model(model_config, vocab_size, device):
    model_arch = model_config['model_arch']
    
    if model_arch == "cnn-rnn":
        rnn_embed_size = model_config['rnn_embed_size']
        rnn_hidden_size = model_config['rnn_hidden_size']
        return CNNtoRNN(rnn_embed_size, rnn_hidden_size, vocab_size).to(device)
    
    elif model_arch == "cnn-attn":
        attn_embed_size = model_config['attn_embed_size']
        attn_num_layers = model_config['attn_num_layers']
        attn_num_heads = model_config['attn_num_heads']
        return CNNAttentionModel(attn_embed_size, vocab_size, attn_num_heads, attn_num_layers).to(device)
    
    elif model_arch == "vitcnn-attn":
        vitcnn_embed_size = model_config['vitcnn_embed_size']
        vitcnn_num_layers = model_config['vitcnn_num_layers']
        vitcnn_num_heads = model_config['vitcnn_num_heads']
        return VITCNNAttentionModel(vitcnn_embed_size, vocab_size, vitcnn_num_heads, vitcnn_num_layers).to(device)
    
    elif model_arch == "vit-attn":
        vit_embed_size = model_config['vit_embed_size']
        vit_num_layers = model_config['vit_num_layers']
        vit_num_heads = model_config['vit_num_heads']
        return VITAttentionModel(vit_embed_size, vocab_size, vit_num_heads, vit_num_layers).to(device)
    
    elif model_arch == "yolo-attn":
        yolo_embed_size = model_config['yolo_embed_size']
        yolo_num_layers = model_config['yolo_num_layers']
        yolo_num_heads = model_config['yolo_num_heads']
        return YOLOAttentionModel(yolo_embed_size, vocab_size, yolo_num_heads, yolo_num_layers).to(device)
    
    elif model_arch == "yolocnn-attn":
        yolocnn_embed_size = model_config['yolocnn_embed_size']
        yolocnn_num_layers = model_config['yolocnn_num_layers']
        yolocnn_num_heads = model_config['yolocnn_num_heads']
        return YOLOCNNAttentionModel(yolocnn_embed_size, vocab_size, yolocnn_num_heads, yolocnn_num_layers).to(device)
    
    else:
        raise ValueError("Model not recognized")

def train(
    learning_rate,
    num_epochs,
    num_workers,
    batch_size,
    val_ratio,
    test_ratio,
    step_size,
    gamma,
    model_arch,
    mode,
    dataset,
    beam_width,
    save_model,
    load_model,
    checkpoint_dir,
    model_config,
    saved_name,
    save_every,
    eval_every
):
    
    if os.path.exists(f'./checkpoints/{model_arch}/{dataset}/{saved_name}'):
        exit(f"Model {model_arch}, {saved_name}, dataset {dataset} already trained")
        
    train_loader, val_loader, _, train_dataset, _, _ = get_loader(
        transform=transform,
        num_workers=num_workers,
        batch_size=batch_size,
        mode=mode,
        model_arch=model_arch,
        dataset=dataset,
        val_ratio=val_ratio,
        test_ratio=test_ratio
    )

    vocab_size = len(train_dataset.vocab)
    print("Vocabulary size:", vocab_size)

    accelerator = Accelerator()  # Initialize Accelerator
    device = accelerator.device  # Use accelerator's device
    print(f"Using device: {device}")

    model = get_model(model_config, vocab_size, device)
    print("Model initialized")
    
    if mode == 'precomputed' and not os.path.exists(f'precomputed/{model_arch}/{dataset}'):
        image_train_loader, image_val_loader, image_test_loader, _, _, _ = get_loader(
            transform=transform,
            num_workers=num_workers,
            batch_size=batch_size,
            mode='image',
            model_arch=model_arch,
            dataset=dataset,
            val_ratio=val_ratio,
            test_ratio=test_ratio
        )
        
        precompute_images(
            model,
            model_arch,
            dataset,
            image_train_loader,
            image_val_loader,
            image_test_loader
        )
        # remove datasets
        del image_train_loader, image_val_loader, image_test_loader

    # Initialize SummaryWriter only on the main process
    if accelerator.is_main_process:
        if not os.path.exists(f"runs/{model_arch}/{dataset}/{saved_name}"):
            os.makedirs(f"runs/{model_arch}/{dataset}/{saved_name}")
        writer = SummaryWriter(f"runs/{model_arch}/{dataset}/{saved_name}")
    else:
        writer = None
    step = 0
    
    pad_idx = train_dataset.vocab.stoi['<PAD>']
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(lr=learning_rate, params=model.parameters())
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    if load_model:
        step = load_checkpoint(torch.load(checkpoint_dir), model, optimizer)
        
    # Prepare everything with accelerator
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    train_losses = []
    val_bleus_greedy = []
    val_meteors_greedy = []
    val_ciders_greedy = []
    val_bleus_beam = []
    val_meteors_beam = []
    val_ciders_beam = []

    bleu = NLGMetricverse(metrics=load_metric("bleu"))
    meteor = NLGMetricverse(metrics=load_metric("meteor"))
    cider = NLGMetricverse(metrics=load_metric("cider"))

    print("Starting training...")
    for epoch in range(num_epochs):
        print(f"[Epoch {epoch+1} / {num_epochs}]")
        
        model.train()
        train_loss = 0
        for idx, (img_ids, imgs, captions, _) in tqdm(
            enumerate(train_loader), total=len(train_loader), leave=False):
            
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs, captions[:, :-1], mode=mode)
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

            # Accumulate predictions and references
            all_pred_tokens_greedy = []
            all_pred_tokens_beam = []
            all_caption_tokens = []

            with torch.no_grad():
                for idx, (img_ids, imgs, captions, ref_captions) in tqdm(
                    enumerate(val_loader), total=len(val_loader), leave=False
                ):
                    generated_captions_greedy = model.caption_images(imgs, train_dataset.vocab, mode=mode)
                    # print("Images: ", imgs)
                    print(f"Predicted (greedy): {generated_captions_greedy[0]}")
                    print(f"Target: {ref_captions[0]}")
                    
                    all_pred_tokens_greedy.extend(generated_captions_greedy)
                    all_caption_tokens.extend(ref_captions)

            if accelerator.is_main_process:
                # Compute metrics on the main process
                val_bleu_score_greedy = bleu(
                    predictions=all_pred_tokens_greedy,
                    references=all_caption_tokens,
                    reduce_fn='mean')['bleu']['score']

                val_meteor_score_greedy = meteor(
                    predictions=all_pred_tokens_greedy,
                    references=all_caption_tokens,
                    reduce_fn='mean')['meteor']['score']

                val_cider_score_greedy = cider(
                    predictions=all_pred_tokens_greedy,
                    references=all_caption_tokens,
                    reduce_fn='mean')['cider']['score']

                # val_losses.append(val_loss)
                val_bleus_greedy.append(val_bleu_score_greedy)
                val_meteors_greedy.append(val_meteor_score_greedy)
                val_ciders_greedy.append(val_cider_score_greedy)

                # writer.add_scalar("Validation loss", val_loss, global_step=epoch)
                writer.add_scalar("Validation Greedy BLEU", val_bleu_score_greedy, global_step=epoch)
                writer.add_scalar("Validation Greedy METEOR", val_meteor_score_greedy, global_step=epoch)
                writer.add_scalar("Validation Greedy CIDEr", val_cider_score_greedy, global_step=epoch)
                
                print("Greedy:")
                print(f"BLEU: {val_bleu_score_greedy:.4f} | METEOR: {val_meteor_score_greedy:.4f} | CIDEr: {val_cider_score_greedy:.4f}")
            
            all_caption_tokens = []
            with torch.no_grad():
                for idx, (img_ids, imgs, captions, ref_captions) in tqdm(
                    enumerate(val_loader), total=len(val_loader), leave=False
                ):
                    generated_captions_beam = model.caption_images_beam_search(imgs, train_dataset.vocab, beam_width, mode=mode)

                    # print("Images: ", imgs)
                    print(f"Predicted (beam): {generated_captions_beam[0]}")
                    print(f"Target: {ref_captions[0]}")
                    
                    all_pred_tokens_beam.extend(generated_captions_beam)
                    all_caption_tokens.extend(ref_captions)

            if accelerator.is_main_process:
                val_bleu_score_beam = bleu(
                    predictions=all_pred_tokens_beam,
                    references=all_caption_tokens,
                    reduce_fn='mean')['bleu']['score']

                val_meteor_score_beam = meteor(
                    predictions=all_pred_tokens_beam,
                    references=all_caption_tokens,
                    reduce_fn='mean')['meteor']['score']

                val_cider_score_beam = cider(
                    predictions=all_pred_tokens_beam,
                    references=all_caption_tokens,
                    reduce_fn='mean')['cider']['score']

                # val_losses.append(val_loss)
                val_bleus_beam.append(val_bleu_score_beam)
                val_meteors_beam.append(val_meteor_score_beam)
                val_ciders_beam.append(val_cider_score_beam)

                # writer.add_scalar("Validation loss", val_loss, global_step=epoch)
                writer.add_scalar("Validation beam BLEU", val_bleu_score_beam, global_step=epoch)
                writer.add_scalar("Validation beam METEOR", val_meteor_score_beam, global_step=epoch)
                writer.add_scalar("Validation beam CIDEr", val_cider_score_beam, global_step=epoch)

                print("beam:")
                print(f"BLEU: {val_bleu_score_beam:.4f} | METEOR: {val_meteor_score_beam:.4f} | CIDEr: {val_cider_score_beam:.4f}")
                
                
        scheduler.step()

        # Checkpoint saving
        if save_model:
            if (epoch + 1) % save_every == 0 and accelerator.is_main_process:
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                }
                if not os.path.exists(f"./checkpoints/{model_arch}/{dataset}/{saved_name}"):
                    os.makedirs(f"./checkpoints/{model_arch}/{dataset}/{saved_name}")
                filename = f"./checkpoints/{model_arch}/{dataset}/{saved_name}/checkpoint_epoch_{epoch + 1}.pth.tar"
                save_checkpoint(checkpoint, filename)

                metrics = {
                    'train_losses': train_losses,
                    'val_greedy_bleus': val_bleus_greedy,
                    'val_greedy_meteors': val_meteors_greedy,
                    'val_greedy_ciders': val_ciders_greedy,
                    'val_beam_bleus': val_bleus_beam,
                    'val_beam_meteors': val_meteors_beam,
                    'val_beam_ciders': val_ciders_beam
                }

                # Save metrics to a JSON file
                if not os.path.exists(f'./metric_logs/{model_arch}/{dataset}/{saved_name}'):
                    os.makedirs(f'./metric_logs/{model_arch}/{dataset}/{saved_name}')
                metrics_file_path = f'./metric_logs/{model_arch}/{dataset}/{saved_name}//train_val_to_epoch_{epoch+1}.json'
                os.makedirs(os.path.dirname(metrics_file_path), exist_ok=True)
                with open(metrics_file_path, 'w') as json_file:
                    json.dump(metrics, json_file, indent=4)

                print(f"Metrics successfully saved to {metrics_file_path}")

    if accelerator.is_main_process:
        print("Training complete!")

if __name__ == "__main__":
    args = parse_args()
    config_path = args.config_file
    batch_size = args.batch_size
    embed_size = args.embed_size
    num_layers = args.num_layers
    learning_rate = args.learning_rate

    with open(f'./configs/{config_path}', 'r') as file:
        config = yaml.safe_load(file)

    # learning_rate = float(config['training']['learning_rate'])
    # batch_size = int(config['training']['batch_size'])
    num_epochs = int(config['training']['num_epochs'])
    num_workers = int(config['training']['num_workers'])
    val_ratio = float(config['training']['val_ratio'])
    test_ratio = float(config['training']['test_ratio'])
    step_size = int(config['training']['step_size'])
    gamma = float(config['training']['gamma'])
    model_arch = config['training']['model_arch']
    mode = config['training']['mode']
    dataset = config['training']['dataset']
    beam_width = int(config['training']['beam_width'])
    eval_every = int(config['training']['eval_every'])
    save_every = int(config['training']['save_every'])
    save_model = bool(config['training']['save_model'])
    load_model = bool(config['training']['load_model'])

    if "checkpoint_dir" in config['training']:
        checkpoint_dir = config['training']['checkpoint_dir']
    else:
        checkpoint_dir = "./<insert_your_checkpoint>.pth.tar"

    model_config = {}
    model_config['model_arch'] = model_arch
    
    if 'rnn_model' in config:
        model_config['rnn_embed_size'] = embed_size
        model_config['rnn_hidden_size'] = int(config['rnn_model']['hidden_size'])

    if 'attn_model' in config:
        model_config['attn_embed_size'] = embed_size
        model_config['attn_num_layers'] = num_layers
        model_config['attn_num_heads'] = int(config['attn_model']['num_heads'])

    if 'vitcnn_attn_model' in config:
        model_config['vitcnn_embed_size'] = embed_size
        model_config['vitcnn_num_layers'] = num_layers
        model_config['vitcnn_num_heads'] = int(config['vitcnn_attn_model']['num_heads'])
    
    if 'vit_attn_model' in config:
        model_config['vit_embed_size'] = embed_size
        model_config['vit_num_layers'] = num_layers
        model_config['vit_num_heads'] = int(config['vit_attn_model']['num_heads'])
    
    if 'yolo_attn_model' in config:
        model_config['yolo_embed_size'] = embed_size
        model_config['yolo_num_layers'] = num_layers
        model_config['yolo_num_heads'] = int(config['yolo_attn_model']['num_heads'])
    
    if 'yolocnn_attn_model' in config:
        model_config['yolocnn_embed_size'] = embed_size
        model_config['yolocnn_num_layers'] = num_layers
        model_config['yolocnn_num_heads'] = int(config['yolocnn_attn_model']['num_heads'])
    
    if num_layers == -1:
        saved_name = f"bs{batch_size}_lr{learning_rate}_es{embed_size}"
    else:
        saved_name = f"bs{batch_size}_lr{learning_rate}_es{embed_size}_nl{num_layers}"

    print(f"Training model {model_arch}, {saved_name}, dataset {dataset}")

    train(
        learning_rate,
        num_epochs,
        num_workers,
        batch_size,
        val_ratio,
        test_ratio,
        step_size,
        gamma,
        model_arch,
        mode,
        dataset,
        beam_width,
        save_model,
        load_model,
        checkpoint_dir,
        model_config,
        saved_name,
        save_every,
        eval_every
    )
