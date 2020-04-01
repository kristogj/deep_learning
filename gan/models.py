import torch.nn as nn
from torchvision import models


class Generator(nn.Module):

    def __init__(self, config):
        super(Generator, self).__init__()

        # nc: Number of channels for RGB image
        # nz: Size of z latent vector
        # ngf: Size of feature maps in generator
        nc, nz, ngf = config["nc"], config["nz"], config["ngf"]
        self.main = nn.Sequential(
            # Input is Z, going into a convolution
            nn.ConvTranspose2d(nz, 16 * ngf, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(16 * ngf),
            nn.ReLU(inplace=True),

            #             nn.ConvTranspose2d(32 * ngf, 16 * ngf, kernel_size=4, stride=2, padding=1, bias=False),
            #             nn.BatchNorm2d(16 * ngf),
            #             nn.ReLU(inplace=True),

            nn.ConvTranspose2d(16 * ngf, 8 * ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8 * ngf),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(8 * ngf, 4 * ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4 * ngf),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(4 * ngf, 2 * ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2 * ngf),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(2 * ngf, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # Output is now of size (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class CustomGenerator(nn.Module):
    """
    Use transfer learning with ResNet50 to encode the images to a feature vector of size nz.
    We can apply this on the sketches before passing them to the generator.
    """

    def __init__(self, config, encoder="resnet50"):
        super(CustomGenerator, self).__init__()

        if encoder == "resnet50":
            self.encoder = models.resnet50()
            self.encoder.fc = nn.Linear(self.encoder.fc.in_features, config["nz"])
        elif encoder == "vgg11":
            self.encoder = models.vgg11()
            self.encoder.classifier.add_module("8", nn.ReLU(inplace=True))
            self.encoder.classifier.add_module("9", nn.Linear(1000, config["nz"]))

            # Change max pool to avg
            for i in [2, 5, 10, 15, 20]:
                self.encoder.features[i] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
        elif encoder == "alexnet":
            self.encoder = models.alexnet()
            self.encoder.classifier[-1] = nn.Dropout(p=0.5, inplace=False)
            self.encoder.classifier.add_module("7", nn.Linear(in_features=4096, out_features=1000, bias=True))
            self.encoder.classifier.add_module("8", nn.ReLU(inplace=True))
            self.encoder.classifier.add_module("9", nn.Linear(1000, config["nz"]))
        else:
            raise ModuleNotFoundError("Encoder not defined.")

        # Substitute trainable linear layer
        self.bn = nn.BatchNorm1d(config["nz"], momentum=0.01)

        self.generator = Generator(config)

    def forward(self, image):
        feature_map = self.bn(self.encoder(image)).unsqueeze_(-1).unsqueeze_(-1)
        return self.generator(feature_map)


class Discriminator(nn.Module):

    def __init__(self, config):
        super(Discriminator, self).__init__()
        # nc: Number of channels for RGB image
        # nz: Size of z latent vector
        # ndf: Size of feature maps in discriminator
        nc, nz, ndf = config["nc"], config["nz"], config["ndf"]
        self.main = nn.Sequential(
            # Input is a nc x 64 x64 image
            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, 2 * ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2 * ndf),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(2 * ndf, 4 * ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4 * ndf),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(4 * ndf, 8 * ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8 * ndf),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(8 * ndf, 16 * ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16 * ndf),
            nn.LeakyReLU(0.2, inplace=True),

            #             nn.Conv2d(16 * ndf, 32 * ndf, kernel_size=4, stride=2, padding=1, bias=False),
            #             nn.BatchNorm2d(32 * ndf),
            #             nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(16 * ndf, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            # The Discriminator is a binary classifier who predicts if an image is Fake (generated by Generator) or real
        )

    def forward(self, input):
        return self.main(input)
