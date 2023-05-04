import os

from PIL import Image
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_list = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.image_list[index])
        image = Image.open(image_path).convert("L")

        if self.transform:
            image = self.transform(image)

        return image, 0  # Return a dummy label, as labels are not needed for autoencoders