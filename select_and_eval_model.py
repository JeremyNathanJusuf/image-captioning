import subprocess
import os
import json

num_epoch_eval = 50
# for model in ["cnn-rnn", "cnn-attn", "vitcnn-attn", "vit-attn", "yolo-attn", "yolocnn-attn"]:
for model in ["cnn-rnn", "cnn-attn", "vitcnn-attn", "vit-attn"]:
    for dataset in ["flickr"]:
    # for dataset in ["flickr", "coco"]:
        trained_models = os.listdir(f"./metric_logs/{model}/{dataset}")
        best_model_score = 0
        best_model = ""
        for trained_model in trained_models:
            train_model_path = f"./metric_logs/{model}/{dataset}/{trained_model}/train_val_to_epoch_{num_epoch_eval}.json"
            with open(train_model_path, 'r') as json_file:
                train_data = json.load(json_file)

            if train_data["val_beam_bleus"][-1] > best_model_score:
                best_model_score = train_data["val_beam_bleus"][-1]
                best_model = trained_model
                
        print(f"Best model for {model} on {dataset} is {best_model} with score {best_model_score}")
        
        'bs64_lr0.0005_es256'
        './checkpoints/vitcnn-attn/flickr/bs64_lr0.0005_es512_nl2/checkpoint_epoch_50.pth.tar'
        
        batch_size = trained_model.split("_")[0][2:]
        learning_rate = trained_model.split("_")[1][2:]
        embed_size = trained_model.split("_")[2][2:]
        if model != "cnn-rnn":
            num_layers = trained_model.split("_")[3][2:]
        
        if model == "cnn-rnn":
            cmd = f"python eval.py --batch_size={batch_size} --learning_rate={learning_rate} --embed_size={embed_size} \
            --model_arch={model} --dataset={dataset} \
            --checkpoint_dir=./checkpoints/{model}/{dataset}/{trained_model}/checkpoint_epoch_{num_epoch_eval}.pth.tar"
        else:
            cmd = f"python eval.py --batch_size={batch_size} --learning_rate={learning_rate} --embed_size={embed_size} \
                --num_layers={num_layers} --model_arch={model} --dataset={dataset} \
                --checkpoint_dir=./checkpoints/{model}/{dataset}/{trained_model}/checkpoint_epoch_{num_epoch_eval}.pth.tar"
        subprocess.call(cmd, shell=True)
                
                
# for config in config_file:
#     if "attn" in config:
#         for bs in batch_size:
#             for lr in learning_rate:
#                 for es in embed_size:
#                     for nl in num_layers:
#                         cmd = f"python train.py --config_file={config} --batch_size={bs} --learning_rate={lr} --num_layers={nl} --embed_size={es}"
#                         subprocess.call(cmd, shell=True)
#     else:
#         for bs in batch_size:
#             for lr in learning_rate:
#                 for es in embed_size:
#                     cmd = f"python train.py --config_file={config} --batch_size={bs} --learning_rate={lr} --embed_size={es}"
#                     subprocess.call(cmd, shell=True)