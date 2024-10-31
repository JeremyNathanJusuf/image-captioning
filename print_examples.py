from utils import load_checkpoint, transform
from get_loader import get_loader
import torch
from torch import nn, optim
from model import CNNtoRNN
from PIL import Image
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


def print_examples(model, device, dataset):
    model.eval()
    test_img1 = transform(Image.open("test_examples/dog.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 1 CORRECT: Dog on a beach by the ocean")
    print(
        "Example 1 OUTPUT: "
        + " ".join(model.caption_image(test_img1.to(device), dataset.vocab))
    )
    test_img2 = transform(
        Image.open("test_examples/child.jpg").convert("RGB")
    ).unsqueeze(0)
    print("Example 2 CORRECT: Child holding red frisbee outdoors")
    print(
        "Example 2 OUTPUT: "
        + " ".join(model.caption_image(test_img2.to(device), dataset.vocab))
    )
    test_img3 = transform(Image.open("test_examples/bus.png").convert("RGB")).unsqueeze(
        0
    )
    print("Example 3 CORRECT: Bus driving by parked cars")
    print(
        "Example 3 OUTPUT: "
        + " ".join(model.caption_image(test_img3.to(device), dataset.vocab))
    )
    test_img4 = transform(
        Image.open("test_examples/boat.png").convert("RGB")
    ).unsqueeze(0)
    print("Example 4 CORRECT: A small boat in the ocean")
    print(
        "Example 4 OUTPUT: "
        + " ".join(model.caption_image(test_img4.to(device), dataset.vocab))
    )
    test_img5 = transform(
        Image.open("test_examples/horse.png").convert("RGB")
    ).unsqueeze(0)
    print("Example 5 CORRECT: A cowboy riding a horse in the desert")
    print(
        "Example 5 OUTPUT: "
        + " ".join(model.caption_image(test_img5.to(device), dataset.vocab))
    )
    model.train()
    
if __name__ == '__main__':
    train_loader, val_loader, _, train_dataset, _, _ = get_loader(
        root_folder="./flickr8k/images",
        annotation_file="./flickr8k/captions.txt",
        transform=transform,
        num_workers=num_workers,
        batch_size=batch_size
    )
    vocab_size = len(train_dataset.vocab)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    load_checkpoint(torch.load("./checkpoints/checkpoint_epoch_100.pth.tar"), model, optimizer)
        
    print_examples(model, device, train_dataset)