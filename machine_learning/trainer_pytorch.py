import os

import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torchvision import datasets
from model import ConvModel
# Hyperparameters



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


num_epochs = 30
learning_rate = 1e-4
batch_size = 5

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
g = torch.Generator()
g.manual_seed(0)

# torch.manual_seed(0)
####################################################################
####################################################################
####################################################################
####################################################################

transform = transforms.Compose([transforms.Grayscale(1),
                                # transforms.RandomHorizontalFlip(p=0.5),
                                # transforms.RandomVerticalFlip(p=0.5),
                                transforms.ToTensor()])

PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
processed_dir = PROJECT_DIR + os.sep + "data" + os.sep + "processed" + os.sep + "cross_validation" + os.sep + "match_nonmatch_ratio_0.3" + os.sep + "0"

train_dataset = datasets.ImageFolder(root = f"{processed_dir}{os.sep}back{os.sep}train", transform=transform)
test_dataset = datasets.ImageFolder(root = f"{processed_dir}{os.sep}back{os.sep}test", transform=transform)

n_train = len(train_dataset)
print("The length of train dataset : ", len(train_dataset))
print("The length of test dataset : ", len(test_dataset))

# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True, num_workers=2)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=False, num_workers=2)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True, num_workers=2,worker_init_fn=seed_worker,generator=g)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=False, num_workers=2,worker_init_fn=seed_worker,generator=g)


# Initialize the autoencoder and optimizer
model = ConvModel().to(device)

criterion = nn.BCELoss()
# criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# print( model( next(iter(train_loader) )[0].to(device) ).shape)

# Defining learning scheduler
# initial_learning_rate = 0.0001
# end_learning_rate = 0.00001
# num_train_steps = 10000  # Replace this value with your number of training steps
# num_train_steps = (n_train/batch_size)*num_epochs

# power = 2.0

# optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate)

# def lr_lambda(step):
#     return (1 - step / num_train_steps) ** power

# lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train():
    model.train()
    train_loss = 0.0
    for i, (x_batch, y_batch) in enumerate(train_loader):
        if i % 10 == 0:
            print(f"Batch # {i}")

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1).float()

        # Forward pass
        outputs = model(x_batch)
        # Compute loss
        loss = criterion(outputs.squeeze(), y_batch.squeeze())

        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update learning rate
        # lr_scheduler.step()
        train_loss += loss.item()*x_batch.size(0)

    train_loss /= len(train_loader)
    return train_loss

@torch.no_grad()
def test():
    model.eval()
    test_loss = 0
    y_pred = []
    y_true = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1).float()
            # Forward pass
            outputs = model(x_batch)

            y_pred.extend(outputs.tolist())
            y_true.extend(y_batch.tolist())

            # Compute loss
            loss = criterion(outputs.squeeze(), y_batch.squeeze())
            test_loss += loss.item()*x_batch.size(0)

    test_loss /= len(test_loader)

    # Convert predictions and true labels to numpy arrays
    y_pred = torch.tensor(y_pred)
    y_true = torch.tensor(y_true)

    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred.round()).ravel()

    print('True Negatives:', tn)
    print('False Positives:', fp)
    print('False Negatives:', fn)
    print('True Positives:', tp)
    return test_loss


# Training loop
for epoch in range(num_epochs):

    # Training dataset loop
    loss = train()

    # Testing dataset loop
    test_loss = test()
    
    # Print the progress
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss:.4f}, Test Loss: {test_loss:.4f}")

print("Training complete.")

# # Save the trained model
# torch.save(model.state_dict(), PROJECT_DIR + os.sep + 'models' + os.sep + "model.pth")
# print("Model saved.")