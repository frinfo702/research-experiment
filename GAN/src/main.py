# %% Inspect the current working directory
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

print(os.getcwd())


# %% PyTorch-based GAN setup

latent_dim = 20  # size of noise z (latent variable)
img_size = 28 * 28
batch_size = 128
lr = 0.0003
epochs = 30

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)
dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# %% Model definitions
class Generator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, img_size),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # noqa: D401 - standard forward
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),  # 正解である確率を出したい
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401 - standard forward
        return self.net(x)


G = Generator()
D = Discriminator()

criterion = nn.BCELoss()
optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))


# %% Training utilities for the PyTorch GAN
def learn(m: int, k: int, datas) -> None:
    for epoch in range(k):
        for idx, (imgs, _) in enumerate(datas):
            real_imgs = imgs.view(-1, img_size)
            batch_size = real_imgs.size(0)

            # Train Discriminator
            z = torch.randn(batch_size, latent_dim)
            fake_imgs = G(z).detach()

            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)

            D_real = D(real_imgs)
            D_fake = D(fake_imgs)

            loss_D_real = criterion(D_real, real_labels)
            loss_D_fake = criterion(D_fake, fake_labels)
            loss_D = (loss_D_real + loss_D_fake) / 2

            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            # Train Generator
            z = torch.randn(batch_size, latent_dim)
            fake_imgs = G(z)
            D_fake = D(fake_imgs)
            loss_G = criterion(D_fake, real_labels)  # want D(fake)=1

            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

        print(
            f"Epoch [{epoch + 1}/{k}] | D loss: {loss_D.item():.4f} | G loss: {loss_G.item():.4f}"
        )

        if (epoch + 1) % 5 == 0:
            show_generated(G)


def show_generated(generator: Generator) -> None:
    z = torch.randn(16, latent_dim)
    with torch.no_grad():
        gen_imgs = generator(z).view(-1, 1, 28, 28)
    grid = torch.cat([gen_imgs[i] for i in range(16)], dim=2)
    plt.imshow(grid.permute(1, 2, 0).squeeze().cpu().numpy(), cmap="gray")
    plt.axis("off")
    plt.show()


# %% Kick off the PyTorch GAN training
learn(m=batch_size, k=epochs, datas=loader)
