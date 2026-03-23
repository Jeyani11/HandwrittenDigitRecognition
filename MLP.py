import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

import numpy as np

# Load up the data

data_path = "./data"

train_dataset = datasets.MNIST( #training dataset load
    root=data_path,
    train=True,                 #train
    transform=transforms.ToTensor(),
    download=False  
)

test_dataset = datasets.MNIST( #testing dataset load
    root=data_path,
    train=False,                 #train
    transform=transforms.ToTensor(),
    download=False
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset, batch_size=64,
    shuffle= False
)


# Visualize the data

# Plot that shows 9 images from the dataset taken randomly
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_dataset), size=(1,)).item()
    img, label = train_dataset[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title('title :' + str(label))
    plt.axis("off")
    plt.imshow(img.squeeze())
plt.show()

# Define the network
class MLP(nn.Module):
    def __init__(self):
       super().__init__() # que fait cette ligne c'est qoi deja super
       self.layers = nn.Sequential(
            nn.Linear(8,32),
            nn.ReLU(),
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64,10)
        )

    def forward(self,x):
        return self.network(x)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Training the model
model = MLP().to(device)
print(model)
# Evaluation


# Saving the model

