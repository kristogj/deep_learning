# PyTorch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms

# Other
import logging

# Custom
from utils import get_device, weight_init
# from plotting import show_images
from models import Generator, Discriminator
from training import train

data_path = "./datasets"

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
    "lr_d": 2e-4,  # Learning rate for Adam Optimizer
    "lr_g": 2e-4,  # Learning rate for Adam Optimizer
    "beta1": 0.5,  # Beta hyperparameter for Adam Optimizer,
    "lr_w": 5e-5,  # Learning rate for RMSprop
    "wasserstein": False,  # Whether to use Wasserstein loss
    "res_path": "./results/"
}

# Prepare dataset
transformer = transforms.Compose([
    transforms.Resize(config["image_size"]),
    transforms.CenterCrop(config["image_size"]),
    transforms.ToTensor(),
    transforms.Normalize([0.5685, 0.5598, 0.5426], [0.2799, 0.2730, 0.2694])
])
dataset = dset.ImageFolder(root=data_path, transform=transformer)

# Initialize dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=config["batch_size"],
                                         shuffle=True, num_workers=2)

device = get_device()

# Show some images from the trainingset
# show_images(dataloader)

# Initialize the model
generator = Generator(config).to(device)
discriminator = Discriminator(config).to(device)

# Initialize custom weights to model
generator.apply(weight_init)
discriminator.apply(weight_init)

# BCELoss for Discriminator
criterion = nn.BCELoss()

# Generate one batch of latent vectors that we will use to visualize the progression of the generator
fixed_noise = torch.rand(8, config["nz"], 1, 1, device=device)

# Optimizer for both generator and discriminator
if config["wasserstein"]:
    optimizer_g = optim.RMSprop(generator.parameters(), lr=config["lr_w"])
    optimizer_d = optim.RMSprop(discriminator.parameters(), lr=config["lr_w"])
else:
    optimizer_g = optim.Adam(generator.parameters(), lr=config["lr_g"], betas=(config["beta1"], 0.999))
    # optimizer_d = optim.Adam(discriminator.parameters(), lr=config["lr_d"], betas=(config["beta1"], 0.999))
    optimizer_d = optim.SGD(discriminator.parameters(), lr=config["lr_d"])

# Train GAN
train(generator, discriminator, criterion, optimizer_g, optimizer_d, dataloader, fixed_noise, config)
