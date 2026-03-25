import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

import numpy as np

# Hyperparameters

data_path = "./data"
learning_rate = 0.01

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

print(f'Train Loader : {train_dataset}')

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
            nn.Flatten(), # 28 x 28  => 784 converting multi-dimensional tensors into lower-dimensional forms
            nn.Linear(784, 128),  # 784 inputs
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,10)
        )

    def forward(self,x):
        return self.layers(x)

# Initialisation
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")


model = MLP().to(device)
print(model)
loss_fn = nn.CrossEntropyLoss()

#Optimization is the process of adjusting model parameters to reduce model error in each training step.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
def train(dataloader,model, loss_fn, optimizer):
    
    size = len(dataloader.dataset)
    model.train()
    for batch, (X,y) in enumerate(dataloader):
        #Compute prediction and loss
        y_pred = model(X)
        loss = loss_fn(y_pred,y)

        #Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * 64 + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



# Evaluation


# Saving the model

if __name__=="__main__":
    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_loader, model, loss_fn, optimizer)
        test(test_loader, model, loss_fn)
    print("Done!")