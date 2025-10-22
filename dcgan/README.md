# 🧠 Deep Convolutional Generative Adversarial Network (DCGAN) — MNIST Project

## 🎯 Project Overview
This project implements and experiments with a **Deep Convolutional GAN (DCGAN)** trained on the **MNIST dataset** using **PyTorch**.  
The objective is to generate realistic handwritten digits by training two neural networks — a **Generator** and a **Discriminator** — in an adversarial setup.

We used **Weights & Biases (W&B)** to track all experiment runs, visualize training metrics, and log generated samples.

---

## 🧩 Directory Structure
dcgan/
├── configs/ # Configuration files (YAML/JSON for future experiments)
├── data/ # Dataset and preprocessing scripts
├── experiments/ # Jupyter notebooks for training and analysis
│ └── baseline_dcgan_mnist.ipynb
├── logs/ # Training logs and sample outputs
├── models/ # Saved trained models
│ ├── generator_trained.pth
│ └── discriminator_trained.pth
├── utils/ # Helper functions and scripts
└── README.md # Project documentation


---

## ⚙️ Experiments Conducted

| Experiment | Description | Key Observations |
|-------------|-------------|------------------|
| **Baseline DCGAN** | Standard architecture, trained for 5 epochs | Generator begins forming digit-like shapes around epoch 3 |
| **Architecture Variation** | Adjusted filter sizes and feature maps | More detailed outputs but slower convergence |
| **Hyperparameter Tuning** | Changed learning rate and batch size | Lower LR improved stability; higher batch sizes led to smoother loss curves |
| **Precision Experiment** | Tested float16 (mixed precision) | Improved speed, minor loss instability |

---

## 📊 Weights & Biases (W&B) Dashboard
All runs, metrics, and generated images are logged here:

🔗 **[View Project Dashboard](https://wandb.ai/namestesensei-self/dcgan-mnist)**

Example run (Baseline DCGAN):  
🔗 **[Baseline Run – DCGAN MNIST](https://wandb.ai/namestesensei-self/dcgan-mnist)**

---

## 📦 Model Artifacts
Trained models are stored in the `/models` directory:

- 🧠 `generator_trained.pth`
- 🧩 `discriminator_trained.pth`

You can reload them using:
```python
gen.load_state_dict(torch.load("models/generator_trained.pth"))
disc.load_state_dict(torch.load("models/discriminator_trained.pth"))

