import os
import json
import torch
import yaml
from tqdm import tqdm
import torch.optim as optim
from utils import load_model, transform
from get_loader import get_loader
from nlgmetricverse import NLGMetricverse, load_metric
from train import precompute_images, get_model, parse_args

def eval(
        num_workers,
        batch_size,
        val_ratio,
        test_ratio,
        model_arch,
        mode,
        dataset,
        beam_width,
        checkpoint_dir,
        model_config,
        saved_name,
):
    if os.path.exists(f'./eval/{model_arch}/{dataset}/{saved_name}'):
        exit(f"Model {model_arch}, {saved_name}, dataset {dataset} already evaluated")
    
    _, _, test_loader, train_dataset, _, _ = get_loader(
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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    load_model(torch.load(checkpoint_dir, weights_only=True), model)

    bleu = NLGMetricverse(metrics=load_metric("bleu"))
    meteor = NLGMetricverse(metrics=load_metric("meteor"))
    cider = NLGMetricverse(metrics=load_metric("cider"))

    print("Starting evaluation...")
    model.eval()

    # Accumulate predictions and references
    all_pred_tokens_greedy = []
    all_pred_tokens_beam = []
    all_caption_tokens = []

    with torch.no_grad():
        for idx, (img_ids, imgs, captions, ref_captions) in tqdm(
            enumerate(test_loader), total=len(test_loader), leave=False
        ):
            imgs = imgs.to(device)
            generated_captions_greedy = model.caption_images(imgs, train_dataset.vocab, mode=mode)
            # print("Images: ", imgs)
            print(f"Predicted (greedy): {generated_captions_greedy[0]}")
            print(f"Target: {ref_captions[0]}")
            
            all_pred_tokens_greedy.extend(generated_captions_greedy)
            all_caption_tokens.extend(ref_captions)

    test_bleu_score_greedy = bleu(
        predictions=all_pred_tokens_greedy,
        references=all_caption_tokens,
        reduce_fn='mean')['bleu']['score']

    test_meteor_score_greedy = meteor(
        predictions=all_pred_tokens_greedy,
        references=all_caption_tokens,
        reduce_fn='mean')['meteor']['score']

    test_cider_score_greedy = cider(
        predictions=all_pred_tokens_greedy,
        references=all_caption_tokens,
        reduce_fn='mean')['cider']['score']

    print("Greedy:")
    print(f"BLEU: {test_bleu_score_greedy:.4f} | METEOR: {test_meteor_score_greedy:.4f} | CIDEr: {test_cider_score_greedy:.4f}")
    
    all_caption_tokens = []
    with torch.no_grad():
        for idx, (img_ids, imgs, captions, ref_captions) in tqdm(
            enumerate(test_loader), total=len(test_loader), leave=False
        ):
            imgs = imgs.to(device)
            generated_captions_beam = model.caption_images_beam_search(imgs, train_dataset.vocab, beam_width, mode=mode)

            # print("Images: ", imgs)
            print(f"Predicted (beam): {generated_captions_beam[0]}")
            print(f"Target: {ref_captions[0]}")
            
            all_pred_tokens_beam.extend(generated_captions_beam)
            all_caption_tokens.extend(ref_captions)

    test_bleu_score_beam = bleu(
        predictions=all_pred_tokens_beam,
        references=all_caption_tokens,
        reduce_fn='mean')['bleu']['score']

    test_meteor_score_beam = meteor(
        predictions=all_pred_tokens_beam,
        references=all_caption_tokens,
        reduce_fn='mean')['meteor']['score']

    test_cider_score_beam = cider(
        predictions=all_pred_tokens_beam,
        references=all_caption_tokens,
        reduce_fn='mean')['cider']['score']
    
    print("beam:")
    print(f"BLEU: {test_bleu_score_beam:.4f} | METEOR: {test_meteor_score_beam:.4f} | CIDEr: {test_cider_score_beam:.4f}")
    
    # Save metrics
    metrics = {
        'val_greedy_bleus': test_bleu_score_greedy,
        'val_greedy_meteors': test_meteor_score_greedy,
        'val_greedy_ciders': test_cider_score_greedy,
        'val_beam_bleus': test_bleu_score_beam,
        'val_beam_meteors': test_meteor_score_beam,
        'val_beam_ciders': test_cider_score_beam
    }
    # Save metrics to a JSON file
    if not os.path.exists(f'./eval'):
        os.makedirs(f'./eval')
    eval_file_path = f'./eval/metrics.json'    
    os.makedirs(os.path.dirname(eval_file_path), exist_ok=True)
    with open(eval_file_path, 'r') as json_file:
        eval_data = json.load(metrics, json_file, indent=4)
    
    if model_arch not in eval_data:
        eval_data[model_arch] = {}
    if dataset not in eval_data[model_arch]:
        eval_data[model_arch][dataset] = {}
        
    eval_data[model_arch][dataset][saved_name] = metrics

    print(f"Metrics successfully saved to {eval_file_path}")
       
if __name__ == "__main__":
    args = parse_args()
    config_path = args.config_file
    print(f"Using config file: {config_path}")
    with open(f'./configs/{config_path}', 'r') as file:
        config = yaml.safe_load(file)

    learning_rate = float(config['training']['learning_rate'])
    num_epochs = int(config['training']['num_epochs'])
    num_workers = int(config['training']['num_workers'])
    batch_size = int(config['training']['batch_size'])
    val_ratio = float(config['training']['val_ratio'])
    test_ratio = float(config['training']['test_ratio'])
    model_arch = config['training']['model_arch']
    mode = config['training']['mode']
    dataset = config['training']['dataset']
    beam_width = int(config['training']['beam_width'])
    

    if "checkpoint_dir" in config['training']:
        checkpoint_dir = config['training']['checkpoint_dir']
        print(f"Using checkpoint directory: {checkpoint_dir}")
    else:
        raise ValueError("Checkpoint directory not found in config file")

    model_config = {}
    model_config['model_arch'] = model_arch
    
    if 'rnn_model' in config:
        embed_size = model_config['rnn_embed_size'] = int(config['rnn_model']['embed_size'])
        model_config['rnn_hidden_size'] = int(config['rnn_model']['hidden_size'])

    if 'attn_model' in config:
        embed_size = model_config['attn_embed_size'] = int(config['attn_model']['embed_size'])
        num_layers = model_config['attn_num_layers'] = int(config['attn_model']['num_layers'])
        model_config['attn_num_heads'] = int(config['attn_model']['num_heads'])

    if 'vitcnn_attn_model' in config:
        embed_size = model_config['vitcnn_embed_size'] = int(config['vitcnn_attn_model']['embed_size'])
        num_layers = model_config['vitcnn_num_layers'] = int(config['vitcnn_attn_model']['num_layers'])
        model_config['vitcnn_num_heads'] = int(config['vitcnn_attn_model']['num_heads'])
    
    if model_arch == "rnn_model":
        saved_name = f"bs{batch_size}_lr{learning_rate}_es{embed_size}"
    else:
        saved_name = f"bs{batch_size}_lr{learning_rate}_es{embed_size}_nl{num_layers}"

    print(f"Evaluating model {model_arch}, {saved_name}, dataset {dataset}")

    eval(
        num_workers,
        batch_size,
        val_ratio,
        test_ratio,
        model_arch,
        mode,
        dataset,
        beam_width,
        checkpoint_dir,
        model_config,
        saved_name,
    )
