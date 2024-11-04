import os  # when loading file paths
import pandas as pd  # for lookup in annotation file
import spacy  # for tokenizer
import torch
from torch.nn.utils.rnn import pad_sequence  # pad batch
from torch.utils.data import DataLoader, Dataset
from PIL import Image  # Load img
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import yaml
import nltk
from nltk.tokenize import word_tokenize
import string

# Configurations
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

test_ratio = float(config['training']['test_ratio'])
val_ratio = float(config['training']['val_ratio'])
seed = 42

# We want to convert text -> numerical values
# 1. We need a Vocabulary mapping each word to a index
# 2. We need to setup a Pytorch dataset to load the data
# 3. Setup padding of every batch (all examples should be
#    of same seq_len and setup dataloader)
# Note that loading the image is very easy compared to the text!

# Download with: python -m spacy download en
# spacy_eng = spacy.load("en_core_web_sm")

class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold
        self.tokenizer = word_tokenize

    def __len__(self):
        return len(self.itos)

    def tokenizer_eng(self, text):
        text = text.lower().strip().strip("\n")
        text = "".join([char for char in text if char not in string.punctuation])
        return [tok for tok in self.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1

                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]

class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5, max_length=40):
        self.root_dir = root_dir
        self.df = captions_file
        self.transform = transform

        # Get img, caption columns
        self.imgs = self.df["image"].tolist()
        raw_captions = self.df["caption"].astype(str).tolist()

        # Initialize vocabulary and build vocab
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(raw_captions)

        # Preprocess captions: tokenize and numericalize
        self.tokenized_captions = []
        self.numericalized_captions = []
        for caption in raw_captions:
            
            # Tokenize the caption
            tokens = self.vocab.tokenizer_eng(caption)
            self.tokenized_captions.append(tokens)
            
            # Numericalize the caption with <SOS> and <EOS> tokens
            numericalized = [self.vocab.stoi["<SOS>"]]
            numericalized += self.vocab.numericalize(caption)
            numericalized.append(self.vocab.stoi["<EOS>"])
            self.numericalized_captions.append(torch.tensor(numericalized, dtype=torch.long))

        # Sort the dataset based on the length of numericalized_captions
        self._sort_dataset()

    def _sort_dataset(self):
        # Create a list of (index, length) tuples
        lengths = [(idx, len(caption)) for idx, caption in enumerate(self.numericalized_captions)]
        # Sort the list based on length
        sorted_lengths = sorted(lengths, key=lambda x: x[1])
        # Extract the sorted indices
        sorted_indices = [idx for idx, length in sorted_lengths]
        
        # Reorder the datasets
        self.imgs = [self.imgs[idx] for idx in sorted_indices]
        self.numericalized_captions = [self.numericalized_captions[idx] for idx in sorted_indices]
        self.tokenized_captions = [self.tokenized_captions[idx] for idx in sorted_indices]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Image loading
        img_id = self.imgs[index]
        img = Image.open(os.path.join(self.root_dir, img_id))

        # Apply image transformations if provided
        if self.transform is not None:
            img = self.transform(img)

        # Retrieve preprocessed numericalized caption and tokens
        numericalized_caption = self.numericalized_captions[index]
        caption_tokens = self.tokenized_captions[index]

        # Return the processed image, numericalized caption, and original caption tokens
        return img, numericalized_caption, caption_tokens, len(numericalized_caption)

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        caption_tokens = [item[2] for item in batch]  # Collect caption_tokens
        caption_lengths = [item[3] for item in batch]  # Collect caption lengths
        targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)
        return imgs, targets, caption_tokens, caption_lengths


def get_loader(
    root_folder,
    annotation_file,
    transform,
    batch_size=32,
    num_workers=8,
    shuffle=True,
    pin_memory=True,
):
    img_captions = pd.read_csv(annotation_file)
    train_val_img_captions, test_img_captions = train_test_split(
        img_captions, test_size=test_ratio, random_state=seed
    )
    train_img_captions, val_img_captions = train_test_split(
        train_val_img_captions, test_size=val_ratio, random_state=seed
    )
    train_dataset = FlickrDataset(root_folder, train_img_captions, transform=transform)
    val_dataset = FlickrDataset(root_folder, val_img_captions, transform=transform)
    test_dataset = FlickrDataset(root_folder, test_img_captions, transform=transform)

    pad_idx = train_dataset.vocab.stoi["<PAD>"]

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(),]
    )

    loader, dataset = get_loader(
        "./flickr30k/images/", "./flickr30k/captions.txt", transform=transform
    )

    for idx, (imgs, captions) in enumerate(loader):
        print(imgs.shape)
        print(captions.shape)