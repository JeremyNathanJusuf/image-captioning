import subprocess
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def create_metric_plots(best_model_metrics):
    metric_name_list = ["bleus", "ciders", "meteors"]
    for dataset in best_model_metrics:
        for metric_name in metric_name_list:
            f, ax = plt.subplots(1, 2, figsize=(15, 5))
            
            for model in best_model_metrics[dataset]:
                metrics = best_model_metrics[dataset][model]
                metrics = best_model_metrics[dataset][model]
                
                for i, inference_type in enumerate(["greedy", "beam"]):
                    # Generate x-tick labels as multiples of 5, but use normal indexing for plotting
                    x_indices = list(range(len(metrics[f"val_{inference_type}_{metric_name}"])))
                    x_labels = [(i + 1) * 5 for i in x_indices]
                
                    ax[i].plot(x_indices, metrics[f"val_{inference_type}_{metric_name}"], label=model)
                    ax[i].set_title(f"{inference_type} {metric_name.upper()} for {dataset}")
                    ax[i].set_xlabel("Epoch")
                    
                    ax[i].legend()
                    ax[i].set_xticks(x_indices)
                    ax[i].set_xticklabels(x_labels)  
            
            # save the dataset plot
            Path(f"./eval/metric_plots/").mkdir(parents=True, exist_ok=True)
            plt.savefig(f"./eval/metric_plots/{dataset}_{metric_name}_plot.png")
            
model_list = ["vitcnn-attn","cnn-rnn", "cnn-attn",  "vit-attn", "yolocnn-attn"]
dataset_list = ["flickr", "coco"]
num_epoch_eval = 50
best_model_metrics = {}

for model in model_list:
    for dataset in dataset_list:
        metric_folder = f"./metric_logs/{model}/{dataset}"
        if not os.path.exists(metric_folder):
            continue
        trained_models = os.listdir(metric_folder)
        best_model_score = 0
        best_model = ""
        best_metrics = {}
        
        for trained_model in trained_models:
            train_model_path = f"./metric_logs/{model}/{dataset}/{trained_model}/train_val_to_epoch_{num_epoch_eval}.json"
            with open(train_model_path, 'r') as json_file:
                train_data = json.load(json_file)

            if train_data["val_beam_bleus"][-1] > best_model_score:
                best_model_score = train_data["val_beam_bleus"][-1]
                best_model = trained_model
                best_metrics = train_data
                
                
        print(f"Best model for {model} on {dataset} is {best_model} with score {best_model_score}")
        
        batch_size = best_model.split("_")[0][2:]
        learning_rate = best_model.split("_")[1][2:]
        embed_size = best_model.split("_")[2][2:]
        
        best_metrics['saved_name'] = f"bs{batch_size}_lr{learning_rate}_es{embed_size}"
        if dataset not in best_model_metrics:
            best_model_metrics[dataset] = {}
        best_model_metrics[dataset][model] = best_metrics
        
        if model != "cnn-rnn":
            num_layers = best_model.split("_")[3][2:]
        
        if model == "cnn-rnn":
            cmd = f"python eval.py --batch_size={batch_size} --learning_rate={learning_rate} --embed_size={embed_size} \
            --model_arch={model} --dataset={dataset} \
            --checkpoint_dir=./checkpoints/{model}/{dataset}/{best_model}/checkpoint_epoch_{num_epoch_eval}.pth.tar"
        else:
            cmd = f"python eval.py --batch_size={batch_size} --learning_rate={learning_rate} --embed_size={embed_size} \
                --num_layers={num_layers} --model_arch={model} --dataset={dataset} \
                --checkpoint_dir=./checkpoints/{model}/{dataset}/{best_model}/checkpoint_epoch_{num_epoch_eval}.pth.tar"
        subprocess.call(cmd, shell=True)
                
create_metric_plots(best_model_metrics)