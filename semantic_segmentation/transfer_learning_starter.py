from dataloader import CityScapesDataset, n_class
from optimization import train, test, init_weights
from models import TransferNet

# PyTorch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch

# Other
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.FileHandler("./app.log", mode="w"),
        logging.StreamHandler()
    ])

train_dataset = CityScapesDataset(csv_file='train.csv', train=True)
val_dataset = CityScapesDataset(csv_file='val.csv')
test_dataset = CityScapesDataset(csv_file='test.csv')
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=5,
                          num_workers=4,
                          shuffle=True)
val_loader = DataLoader(dataset=val_dataset,
                        batch_size=1,
                        num_workers=4,
                        shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=1,
                         num_workers=4)

# Initialize mode, criterion and optimizer
config = {
    'epochs': 100,
    'use_gpu': torch.cuda.is_available(),
    'weighted_loss': False
}

# Set up model
model = TransferNet(n_class=n_class)
model.decoder.apply(init_weights)

# Optimizer
optimizer = optim.Adam(model.decoder.parameters(), lr=5e-3)
criterion = nn.CrossEntropyLoss()

logging.info("Use GPU: {}".format(config["use_gpu"]))
logging.info("Train data length: {}".format(len(train_loader)))
if config['use_gpu']:
    model = model.cuda()
torch.cuda.empty_cache()

# Train, validate and test
train(model, optimizer, criterion, train_loader, val_loader, config)
test(test_loader, config)
