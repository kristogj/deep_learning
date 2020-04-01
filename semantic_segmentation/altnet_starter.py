from optimization import init_weights, train, val, test
from dataloader import CityScapesDataset, n_class
from models import AltNet
from utils import iou_class, iou, pixel_acc, inverted_weights

# PyTorch
import torch.optim as optim
import torch
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn

# Other
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.FileHandler("./app.log", mode="w"),
        logging.StreamHandler()
    ])

# Load Data
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
                         num_workers=1)

# Initialize mode, criterion and optimizer
config = {
    'epochs': 100,
    'use_gpu': torch.cuda.is_available(),
    "weighted_loss": False
}

model = AltNet(n_class=n_class)  # Insert model here
model.apply(init_weights)
if config['weighted_loss']:
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(inverted_weights).float().cuda())
else:
    criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-3)

logging.info("Use GPU: {}".format(config["use_gpu"]))
logging.info("Train data length: {}".format(len(train_loader)))
if config['use_gpu']:
    model = model.cuda()
torch.cuda.empty_cache()

# Train, validate and test
train(model, optimizer, criterion, train_loader, val_loader, config)
test(test_loader, config)
