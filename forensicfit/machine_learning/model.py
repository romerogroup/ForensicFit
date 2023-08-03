import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms

from forensicfit.utils import ROOT

class ConvModel(nn.Module):
    def __init__(self):
        super(ConvModel, self).__init__()



        # Encoder

        # # Encoder
        self.conv_layers = nn.Sequential(
                        nn.Conv2d(1, 32, 3, stride=1, padding='same'),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
                        nn.Conv2d(32, 64, 3, stride=1, padding='same'),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
                        nn.Conv2d(64, 128, 3, stride=1, padding='same'),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
                        nn.Conv2d(128, 256, 3, stride=1, padding='same'),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
                        nn.Conv2d(256, 512, 3, stride=1, padding='same'),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
                        nn.Conv2d(512, 1024, 3, stride=1, padding='same'),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
        )

        self.mlp  = nn.Sequential(
                        # nn.Dropout(0.5),
                        nn.Linear(in_features=1024 * 18 * 6, out_features=500),
                        nn.ReLU(),
                        # nn.Dropout(0.5),
                        nn.Linear(in_features=500, out_features=100),
                        nn.ReLU(),
                        nn.Linear(in_features=100, out_features=1),
                        nn.Sigmoid(),
                    )
        
    def forward(self, x):
        out = self.conv_layers(x)
        out = torch.flatten(out,start_dim=1)
        out = self.mlp(out)
        return out
    

# Set device
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.Grayscale(1),
                                    transforms.ToTensor()])

    processed_dir = ROOT + os.sep + "data" + os.sep + "processed" + os.sep + "cross_validation" + os.sep + "match_nonmatch_ratio_0.3" + os.sep + "0"

    train_dataset = datasets.ImageFolder(root = f"{processed_dir}{os.sep}back{os.sep}train", transform=transform)
    test_dataset = datasets.ImageFolder(root = f"{processed_dir}{os.sep}back{os.sep}test", transform=transform)
    print(test_dataset[0][0].shape)
    print("The length of train dataset : ", len(train_dataset))
    print("The length of test dataset : ", len(test_dataset))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2,shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2,shuffle=True, num_workers=2)

    # Initialize
    model = ConvModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print( model( next(iter(train_loader) )[0].to(device) ).shape)


if __name__ == '__main__':
    main()