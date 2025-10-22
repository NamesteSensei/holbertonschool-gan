import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import wandb
import os

# -------------------
# 1. Configuration
# -------------------
config = {
    "z_dim": 100,
    "batch_size": 128,
    "lr": 2e-4,
    "epochs": 5,
    "img_channels": 1,
    "img_size": 64,
    "features_g": 64,
}

# -------------------
# 2. Initialize WandB
# -------------------
wandb.login()
wandb.init(project="dcgan-mnist", name="train_dcgan_script", config=config)

# -------------------
# 3. Device setup
# -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------
# 4. Data setup
# -------------------
transform = transforms.Compose([
    transforms.Resize(config["img_size"]),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

# -------------------
# 5. Model definitions
# -------------------
class Generator(nn.Module):
    def __init__(self, z_dim, img_channels, features_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(z_dim, features_g * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features_g * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(features_g * 8, features_g * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(features_g * 4, features_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(features_g * 2, features_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g),
            nn.ReLU(True),
            nn.ConvTranspose2d(features_g, img_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.gen(x)


class Discriminator(nn.Module):
    def __init__(self, img_channels):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(img_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x).view(-1)

# -------------------
# 6. Instantiate models
# -------------------
gen = Generator(config["z_dim"], config["img_channels"], config["features_g"]).to(device)
disc = Discriminator(config["img_channels"]).to(device)

opt_gen = torch.optim.Adam(gen.parameters(), lr=config["lr"], betas=(0.5, 0.999))
opt_disc = torch.optim.Adam(disc.parameters(), lr=config["lr"], betas=(0.5, 0.999))
criterion = nn.BCELoss()
fixed_noise = torch.randn(64, config["z_dim"], 1, 1).to(device)

# -------------------
# 7. Training loop
# -------------------
for epoch in range(config["epochs"]):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        batch_size = real.size(0)

        real_labels = torch.ones(batch_size).to(device)
        fake_labels = torch.zeros(batch_size).to(device)

        # Train Discriminator
        noise = torch.randn(batch_size, config["z_dim"], 1, 1).to(device)
        fake = gen(noise)
        disc_real = disc(real)
        disc_fake = disc(fake.detach())
        loss_real = criterion(disc_real, real_labels)
        loss_fake = criterion(disc_fake, fake_labels)
        loss_disc = (loss_real + loss_fake) / 2

        disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        # Train Generator
        output = disc(fake)
        loss_gen = criterion(output, real_labels)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch}/{config['epochs']}] Batch {batch_idx}/{len(loader)} "
                  f"Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f}")
            with torch.no_grad():
                fake_images = gen(fixed_noise).detach().cpu()
                img_grid = torchvision.utils.make_grid(fake_images, normalize=True)
                wandb.log({
                    "Generator Loss": loss_gen.item(),
                    "Discriminator Loss": loss_disc.item(),
                    "Generated Images": [wandb.Image(img_grid, caption=f"Epoch {epoch}")]
                })

# -------------------
# 8. Save models
# -------------------
os.makedirs("models", exist_ok=True)
torch.save(gen.state_dict(), "models/generator_script.pth")
torch.save(disc.state_dict(), "models/discriminator_script.pth")
print("✅ Models trained and saved successfully!")

wandb.finish()
