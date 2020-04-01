from tqdm import tqdm
import os

from torch.utils.data import Dataset  # For custom data-sets
import torchvision.transforms as transforms
from PIL import Image
import torchvision.transforms.functional as TF


def convert_img():
    data_path = "./dataset/pokemon/"

    try:
        os.makedirs("datasets/pokemon")
    except FileExistsError:
        pass

    ind = 0
    for filename in tqdm(os.listdir(data_path)):
        if filename.endswith(".png"):
            img = Image.open(data_path + filename).convert('RGBA')

            background = Image.new('RGBA', img.size, (255, 255, 255))
            alpha_composite = Image.alpha_composite(background, img).convert('RGB')
            alpha_composite.save('datasets/pokemon/' + str(ind) + '.jpg', 'JPEG', quality=80)
        else:
            img = Image.open(data_path + filename).convert('RGB')
            img.save('datasets/pokemon/' + str(ind) + '.jpg')

        ind += 1


class PokemonDataset(Dataset):

    def __init__(self, sketch_path, real_path, config):
        self.sketch_path = sketch_path
        self.real_path = real_path

        self.sketch_filenames = os.listdir(sketch_path)
        #         self.real_filenames = os.listdir(real_path)

        self.transformer = transforms.Compose([
            transforms.Resize(config["image_size"]),
            transforms.CenterCrop(config["image_size"]),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.sketch_filenames)

    def __getitem__(self, idx):
        sketch_img_name = str(idx) + "_A.jpg"
        real_img_name = str(idx) + "_B.jpg"

        sketch_img = Image.open(self.sketch_path + sketch_img_name).convert("RGB")
        real_img = Image.open(self.real_path + real_img_name).convert("RGB")

        sketch_img = self.transformer(sketch_img)
        real_img = self.transformer(real_img)
        real_img = TF.normalize(real_img, [0.5685, 0.5598, 0.5426], [0.2799, 0.2730, 0.2694])

        return sketch_img, real_img
