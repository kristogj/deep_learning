# PyTorch
import torch
from torchvision import transforms
from torch import nn
from torch.optim import Adam

# Custom
from data_loader import get_loader, CocoDataset, collate_fn, ResizeWithRatio
from utils import get_ids, get_anns
from build_vocab import Vocabulary
from optimization import train, val, test
from models import Baseline
from utils import get_device, train_val_sampler

# Others
import logging
import numpy as np
from pycocotools.coco import COCO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.FileHandler("./app.log", mode="w"),
        logging.StreamHandler()
    ])

# These are standard std and mean preprocess for all pre-trained models in PyTorch
std = [0.229, 0.224, 0.225]
mean = [0.485, 0.456, 0.406]
CROP_SIZE = 256

train_transformer = transforms.Compose([
    ResizeWithRatio(CROP_SIZE),
    transforms.CenterCrop(CROP_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(std=std,
                         mean=mean)
])
test_transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(std=std,
                         mean=mean)
])

train_caption_path = "./data/annotations/captions_train2014.json"
test_caption_path = "./data/annotations/captions_val2014.json"
vocab_path = "./vocabulary.pkl"

# Load Data
train_ids = get_anns(get_ids("./TrainImageIds.csv"), train_caption_path)
test_ids = get_anns(get_ids("./TestImageIds.csv"), test_caption_path)

# Initialize the Vocabulary class
vocab = Vocabulary(train_caption_path, vocab_path)

RANDOM_SEED = 42
VALIDATION_SPLIT = .1
BATCH_SIZE = 128

# Load dataset
train_dataset = CocoDataset(root="./data/images/train/",
                            json=train_caption_path,
                            ids=train_ids,
                            vocab=vocab,
                            transform=train_transformer)

test_dataset = CocoDataset(root="./data/images/test/",
                           json=test_caption_path,
                           ids=test_ids,
                           vocab=vocab,
                           transform=test_transformer,
                           test=True)

# Use a random sampler to split into training and validation
train_sampler, valid_sampler = train_val_sampler(train_dataset,
                                                 random_seed=RANDOM_SEED,
                                                 validation_split=VALIDATION_SPLIT,
                                                 shuffle=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           num_workers=4,
                                           collate_fn=collate_fn,
                                           sampler=train_sampler)

valid_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           num_workers=4,
                                           collate_fn=collate_fn,
                                           sampler=valid_sampler)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=1,
                                          num_workers=4,
                                          shuffle=False)

config = {
    "epochs": 100,
    "device": get_device(),
    "sampling": True,
    "temperature": 1.0,
    "max_sentence_length": 18
}

embedding_dim = 256
hidden_dim = 512
vocab_size = len(vocab)
model = Baseline(embedding_dim, hidden_dim, vocab_size, vanilla=False)

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=5e-4)

model.cuda()
train(model, optimizer, criterion, train_loader, valid_loader, vocab, config)
test(model, criterion, test_loader, vocab, config)
