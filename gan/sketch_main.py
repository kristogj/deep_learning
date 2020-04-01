# PyTorch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms

# Other
import logging
import os

# Custom
from utils import get_device, weight_init
from models import Discriminator, CustomGenerator
from training import train_with_sketch
from dataloader import PokemonDataset

test_path = "./datasets/pix2pix/test/"
sketch_path = "./datasets/pokemon/trainA/"
real_path = "./datasets/pokemon/trainB/"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.FileHandler("./app.log", mode="w"),
        logging.StreamHandler()
    ])

config = {
    "batch_size": 16,  # Mini-batch size used during training
    "image_size": 128,  # Resize all input image to this size
    "nc": 3,  # Number of channels for input images (RGB)
    "nz": 64,  # Size of z latent vector drawn fro standard normal distribution by Generator
    "ngf": 128,  # Size of feature maps in generator
    "ndf": 32,  # Size of feature maps in discriminator
    "epochs": 200,
    "lr_d": 1e-3,  # Learning rate for Adam Optimizer
    "lr_g": 1e-3,  # Learning rate for Adam Optimizer
    "beta1": 0.5,  # Beta hyperparameter for Adam Optimizer,
    "lr_w": 5e-5, # Learning rate for RMSprop
    "wasserstein": False, # Whether to use Wasserstein loss
    "res_path": "./results/5/"
}

# Prepare dataset
transformer = transforms.Compose([
    transforms.Resize(config["image_size"]),
    transforms.CenterCrop(config["image_size"]),
    transforms.ToTensor()
])

dataset = PokemonDataset(sketch_path, real_path, config)
# Initialize dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=config["batch_size"],
                                         shuffle=True, num_workers=2)

test_dataset = dset.ImageFolder(root=test_path, transform=transformer)

# Initialize dataloader
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8,
                                              shuffle=False, num_workers=1)

device = get_device()

# Show some images from the trainingset
# show_images(dataloader)

# Initialize the model
generator = CustomGenerator(config, encoder="vgg19").to(device)
discriminator = Discriminator(config).to(device)

# Initialize custom weights to model
generator.apply(weight_init)
discriminator.apply(weight_init)


# BCELoss for Discriminator
criterion = nn.BCELoss()

try:
    os.makedirs(config['res_path'])
except FileExistsError:
    pass

test_imgs = None
for (data, _) in test_dataloader:
    test_imgs = data.to(device)

# Optimizer for both generator and discriminator
if config["wasserstein"]:
    optimizer_g = optim.RMSprop(generator.parameters(), lr=config["lr_w"])
    optimizer_d = optim.RMSprop(discriminator.parameters(), lr=config["lr_w"])
else:
    optimizer_g = optim.Adam(generator.parameters(), lr=config["lr_g"], betas=(config["beta1"], 0.999))
    # optimizer_d = optim.Adam(discriminator.parameters(), lr=config["lr_d"], betas=(config["beta1"], 0.999))
    optimizer_d = optim.SGD(discriminator.parameters(), lr=config["lr_d"])

# Train GAN
train_with_sketch(generator, discriminator, criterion, optimizer_g, optimizer_d, dataloader, test_imgs, config)
