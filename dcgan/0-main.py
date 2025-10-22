#!/usr/bin/python3
"""
Test the trained DCGAN Generator by producing MNIST-style samples.

Author: NamesteSensei
Holberton School Machine Learning Project
"""

import torch
import torchvision
import matplotlib.pyplot as plt
from utils.train_dcgan import Generator


def main():
    """Load generator and visualize sample outputs."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z_dim = 100
    gen = Generator(z_dim=z_dim, img_channels=1, features_g=64).to(device)
    gen.load_state_dict(torch.load("models/generator_trained.pth",
                                   map_location=device))
    gen.eval()

    with torch.no_grad():
        noise = torch.randn(16, z_dim, 1, 1).to(device)
        fake_imgs = gen(noise).cpu()
        img_grid = torchvision.utils.make_grid(fake_imgs, normalize=True)

    plt.figure(figsize=(6, 6))
    plt.imshow(img_grid.permute(1, 2, 0))
    plt.axis("off")
    plt.title("Generated MNIST-like Digits")
    plt.show()


if __name__ == "__main__":
    main()
