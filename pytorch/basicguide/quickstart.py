# %%

# Dataset: stores the samples and their corresponding labels
# DataLoader: wraps an iterable around the Dataset

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# %%
# use a TorchVision dataset

# Download training data from open datasets
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)
# %%

# Pass the Dataset as an argument to DataLoader
# This will suport:
# - wraps an iterable over our dataset
# - auto-batching
# - auto-sampling
# - auto-shuffling
# - multiprocess data loading

# each element in the dataloader iterable will return a batch of 64 features and labels
batch_size = 64

# Create data loaders
train_dataloader = DataLoader(training_data, batch_size)
test_dataloader = DataLoader(test_data, batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break
# %%

# To define a neural network, we create a class taht inherits from nn.Module
# __init__: define the layers of the network here
# forward: specify how data will pass through the network here

device = (
    torch.accelerator.current_accelerator()
    if torch.accelerator.is_available()
    else "cpu"
)


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# to: transfer model to device for computing
model = NeuralNetwork().to(device)
print(model)
# %% Optimizing the model parameters

# To train a model, we need a loss function and an optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-3)


# In a single loop:
# 1. the model makes prediction on the training dataset
# 2. backpropagates the prediction error to adjust the model's parameters
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backprobagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


# Check the model's performance against the test dataset to ensure it is learning
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


# %%

# The training process (train & eval) is conducted over epochs

# We'd like to see the accuracy increase and the loss decrease

epochs = 5
for t in range(epochs):
    print(f"Epoch {t + 1}\n------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")
# %% Saving models

# To save a model, we serialize the internal state dictionary (including the mdoel params)

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
# %% Loading models

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth", weights_only=True))
# %%
# This loaded model can now be used to make predictions
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual "{actual}"')
