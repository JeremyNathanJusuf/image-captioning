import os  # when loading file paths
import pandas as pd  # for lookup in annotation file
import spacy  # for tokenizer
import json
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
        # check if text = nan
        if not text or pd.isnull(text):
            return []
        text = text.lower().strip().strip("\n")
        text = "".join([char for char in text if char not in string.punctuation])
        return [tok for tok in self.tokenizer(text)][:48]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4

        for sentences in sentence_list:
            for sentence in sentences:
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

class ImageCaptionDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5, max_length=70):
        self.root_dir = root_dir
        self.df = captions_file
        self.transform = transform

        # remove nan values
        self.df = self.df.dropna()

        # Get img, caption columns
        self.imgs = self.df["image"].tolist()
        raw_captions = self.df["caption"].tolist()

        for i in range(len(raw_captions)):
            for j in range(len(raw_captions[i])):
                if pd.isnull(raw_captions[i][j]):
                    raw_captions[i][j] = ""
                else:
                    raw_captions[i][j] = raw_captions[i][j].strip().strip("\n")
                    raw_captions[i][j] = "".join([char for char in raw_captions[i][j] if char not in string.punctuation])

        self.ref_captions = raw_captions
        
        # Initialize vocabulary and build vocab
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(raw_captions)

        # Preprocess captions: tokenize and numericalize
        # self.tokenized_captions = []
        self.numericalized_captions = []
        for captions in raw_captions:
            caption_list = []
            for caption in captions:
                # Tokenize the caption
                tokens = self.vocab.tokenizer_eng(caption)
                # self.tokenized_captions.append(tokens)
                
                # Numericalize the caption with <SOS> and <EOS> tokens
                numericalized = [self.vocab.stoi["<SOS>"]]
                numericalized += self.vocab.numericalize(caption)
                numericalized.append(self.vocab.stoi["<EOS>"])
                caption_list.append(numericalized)
            
            self.numericalized_captions.append(caption_list)

        # Sort the dataset based on the length of numericalized_captions
        # self._sort_dataset()

    # def _sort_dataset(self):
    #     # Create a list of (index, length) tuples
    #     lengths = [(idx, len(caption)) for idx, caption in enumerate(self.numericalized_captions)]
    #     # Sort the list based on length
    #     sorted_lengths = sorted(lengths, key=lambda x: x[1])
    #     # Extract the sorted indices
    #     sorted_indices = [idx for idx, length in sorted_lengths]
        
    #     # Reorder the datasets
    #     self.imgs = [self.imgs[idx] for idx in sorted_indices]
    #     self.numericalized_captions = [self.numericalized_captions[idx] for idx in sorted_indices]
    #     self.tokenized_captions = [self.tokenized_captions[idx] for idx in sorted_indices]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Image loading
        img_id = self.imgs[index]
        img = Image.open(os.path.join(self.root_dir, str(img_id)))
        # Ensure the image has 3 channels
        if img.mode != 'RGB':
            img = img.convert('RGB')
        ref_caption = self.ref_captions[index]
        
        # Apply image transformations if provided
        if self.transform is not None:
            img = self.transform(img)

        # Retrieve preprocessed numericalized caption and tokens
        numericalized_caption = self.numericalized_captions[index]
        # caption_tokens = self.tokenized_captions[index]
        
        length = len(numericalized_caption)
        # sample the caption
        caption_idx = torch.randint(0, length, (1,)).item()
        numericalized_caption = numericalized_caption[caption_idx]
        # caption_tokens = caption_tokens[caption_idx]

        # Return the processed image, numericalized caption, and original caption tokens
        return img, numericalized_caption, ref_caption

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [torch.tensor(item[1]) for item in batch]
        # caption_tokens = [item[2] for item in batch]  # Collect caption_tokens
        ref_caption = [item[2] for item in batch]  # Collect caption lengths
        targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)
        return imgs, targets, ref_caption


def get_loader(
    transform,
    batch_size=32,
    num_workers=8,
    dataset='mscoco',
    shuffle=True,
    pin_memory=True,
):
    if dataset =='flickr':
        root_folder = "./flickr30k/images/"
        captions_path = "./flickr30k/captions.txt"
        
        img_captions = pd.read_csv(captions_path)
        img_captions = img_captions.groupby("image").agg(list).reset_index()
        
        train_val_img_captions, test_img_captions = train_test_split(
            img_captions, test_size=test_ratio, random_state=seed
        )
        train_img_captions, val_img_captions = train_test_split(
            train_val_img_captions, test_size=val_ratio, random_state=seed
        )
        
        train_dataset = ImageCaptionDataset(root_folder, train_img_captions, transform=transform)
        val_dataset = ImageCaptionDataset(root_folder, val_img_captions, transform=transform)
        test_dataset = ImageCaptionDataset(root_folder, test_img_captions, transform=transform)
        
    elif dataset == 'mscoco':
        train_caption_path = './mscoco/annotations/captions_train2014.json'
        val_test_caption_path = './mscoco/annotations/captions_val2014.json'
        train_root_folder = './mscoco/train2014/'
        val_test_root_folder = './mscoco/val2014/'
        
        with open(train_caption_path) as f:
            train_captions = json.load(f)
             
        with open(val_test_caption_path) as f:
            val_test_captions = json.load(f)
            
        
        # Create a list of dictionaries with 'image_id' and 'caption' keys
        train_img_captions = [{'image': annotation['image_id'], 'caption': annotation['caption']} for annotation in train_captions['annotations']]
        val_test_img_captions = [{'image': annotation['image_id'], 'caption': annotation['caption']} for annotation in val_test_captions['annotations']]
        
        # Create a DataFrame from the list of dictionaries
        train_img_captions = pd.DataFrame(train_img_captions)
        val_test_img_captions = pd.DataFrame(val_test_img_captions)
        
        # zfill caption by 12 and and jpg
        train_img_captions['image'] = train_img_captions['image'].apply(lambda x: f"COCO_train2014_{str(x).zfill(12)}.jpg")
        val_test_img_captions['image'] = val_test_img_captions['image'].apply(lambda x: f"COCO_val2014_{str(x).zfill(12)}.jpg")
        
        train_img_captions = train_img_captions.groupby("image").agg(list).reset_index()
        val_test_img_captions = val_test_img_captions.groupby("image").agg(list).reset_index()

        val_img_captions, test_img_captions = train_test_split(
            val_test_img_captions, test_size=val_ratio, random_state=seed
        )
        
        train_dataset = ImageCaptionDataset(train_root_folder, train_img_captions, transform=transform)
        val_dataset = ImageCaptionDataset(val_test_root_folder, val_img_captions, transform=transform)
        test_dataset = ImageCaptionDataset(val_test_root_folder, test_img_captions, transform=transform)
        
    else:
        raise ValueError("Invalid dataset. Choose either 'mscoco' or 'flickr'")


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
    