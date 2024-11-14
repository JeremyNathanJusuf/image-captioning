import torch
import torchvision.transforms as transforms

transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    step = load_model(checkpoint, model)
    load_optimizer(checkpoint, optimizer)
    return step

def load_model(checkpoint, model):
    state_dict = checkpoint["state_dict"]
    # Filter out unexpected keys
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
    model.load_state_dict(filtered_state_dict)
    step = checkpoint["step"]
    return step

def load_optimizer(checkpoint, optimizer):
    print("=> Loading optimizer checkpoint")
    optimizer.load_state_dict(checkpoint["optimizer"])