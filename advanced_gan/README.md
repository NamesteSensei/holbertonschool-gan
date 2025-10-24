# Advanced GAN – CelebA Image Generation

## 🎯 Overview
This project builds upon the **DCGAN MNIST** project and explores a more complex challenge: generating realistic human face images using the **CelebA dataset**.  
The model was implemented using **PyTorch** and tracked via **Weights & Biases (W&B)** for full experiment visualization and metric tracking.

## 📊 Weights & Biases Project
- **W&B Project:** [Advanced GAN – CelebA](https://wandb.ai/namestesensei-self/advanced-gan-celeba)
- Tracked metrics: Generator and Discriminator loss, generated samples, and hyperparameter logs.

## 🧠 Dataset
- **Dataset:** [CelebA – Large-scale CelebFaces Attributes](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- **Images used:** 11,611 aligned and cropped RGB images  
- **Resolution:** 64×64  
- **Preprocessing:** Resizing, normalization to [-1, 1], and random horizontal flipping

## ⚙️ Model Architecture
The architecture is a **Deep Convolutional GAN (DCGAN)** with:
- **Generator:** Transposed convolutions, ReLU activations, BatchNorm layers  
- **Discriminator:** Strided convolutions, LeakyReLU activations, and Sigmoid output  
- **Loss:** Binary Cross-Entropy  
- **Optimizer:** Adam (lr = 0.0002, betas = (0.5, 0.999))

## 🧪 Experiments
### 1️⃣ Baseline GAN
Trained the standard DCGAN configuration for 3 epochs.  
The model began generating coherent facial structures and basic color patterns.

### 2️⃣ Architecture Variation
Increased generator feature maps and adjusted kernel sizes.  
Improved image sharpness and reduced discriminator overfitting.

### 3️⃣ Hyperparameter Tuning
Tuned batch size and learning rate; found **lr = 0.0002** and **batch_size = 64** achieved best stability.

## 💾 Directory Structure
advanced_gan/
├── data/ # CelebA dataset (subset or preprocessed images)
├── models/ # Trained weights (.pth files)
├── logs/ # Generated samples and W&B logs
├── configs/ # Hyperparameter configuration files
├── experiments/ # Notebooks for experiment tracking
├── utils/ # Helper functions and training scripts
└── README.md # Project overview and documentation


## 📈 Results
- Discriminator loss stabilized around **0.28**
- Generator loss converged near **3.26**
- Generated samples improved across epochs, showing sharper and more realistic features.

## 🧩 Key Takeaways
- Applying **lessons from MNIST DCGAN** (batch normalization, optimizer tuning) improved stability.
- Using **W&B** streamlined tracking and comparison between experiments.
- CelebA, being more complex, benefitted greatly from deeper architecture and regularization.

## 🎥 Video Presentation
Loom/OBS link: *(Insert your presentation video link here)*

---

### 🚀 How to Run
```bash
python3 train_advanced_gan.py

