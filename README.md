# cartoon-face-generator
# DCGAN Image Generation — Anime & Cartoon Faces

Two PyTorch implementations of a **Deep Convolutional GAN (DCGAN)** that learn to generate synthetic face images — one trained on an anime face dataset, the other on a cartoon face dataset.

## Overview

Both notebooks follow the DCGAN architecture (Radford et al., 2015): a **generator** that maps a random latent vector to a 64×64 RGB image via transposed convolutions, and a **discriminator** that learns to distinguish real images from generated ones via strided convolutions. The two networks are trained adversarially — the generator tries to fool the discriminator, and the discriminator tries not to be fooled — until the generator produces convincingly realistic faces.

## Project Structure

```
.
├── 06b-anime-dcgan.ipynb       # DCGAN trained on the Anime Face Dataset
├── cartoon_face_dcgan.ipynb    # DCGAN trained on a cartoon face dataset
└── README.md
```

## Notebooks

### `06b-anime-dcgan.ipynb`
Part of the ["Deep Learning with PyTorch: Zero to GANs"](https://jovian.ai) tutorial series (Part 7), adapted for anime face generation.

- **Dataset:** [Anime Face Dataset](https://www.kaggle.com/splcher/animefacedataset) (63,000+ images), downloaded via the `opendatasets` library using the Kaggle API
- **Pipeline:** loads images with `ImageFolder`, resizes/crops to 64×64, normalizes to `[-1, 1]`
- **Training:** custom `fit()` loop alternating discriminator and generator updates, Adam optimizer (`lr=0.0002`, `betas=(0.5, 0.999)`), 25 epochs
- **Extras:**
  - Saves generated sample grids after each epoch to a `generated/` directory
  - Compiles the saved samples into a training progress video (`gans_training.avi`) using OpenCV
  - Plots generator/discriminator loss curves and real/fake discriminator scores over training
  - Saves final model weights (`G.ckpt`, `D.ckpt`)
  - Logs hyperparameters, metrics, and artifacts to [Jovian](https://jovian.ai) for experiment tracking

### `cartoon_face_dcgan.ipynb`
A DCGAN adapted from the [PyTorch official DCGAN tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html) architecture, trained on a cartoon face dataset.

- **Dataset:** cartoon face images loaded from a local/Kaggle input directory (`dataroot = "/kaggle/input/"`) via `ImageFolder`
- **Pipeline:** resizes/crops images to 64×64, normalizes to `[-1, 1]`
- **Training:** standard DCGAN training loop, Adam optimizer (`lr=0.0002`, `beta1=0.5`), 50 epochs
- **Extras:**
  - Plots generator/discriminator loss curves after training
  - Animates the generator's output across training iterations
  - Displays a side-by-side comparison of real vs. generated ("fake") images

## Architecture (both notebooks)

| Component | Details |
|---|---|
| Input image size | 64 × 64 × 3 |
| Latent vector size (`nz` / `latent_size`) | 100–128 |
| Generator | `ConvTranspose2d` → `BatchNorm2d` → `ReLU`, output activation `Tanh` |
| Discriminator | `Conv2d` → `BatchNorm2d` → `LeakyReLU(0.2)`, output activation `Sigmoid` |
| Loss | Binary Cross-Entropy (real vs. fake) |
| Optimizer | Adam, `lr=0.0002`, `beta1=0.5` |

## Requirements

```
torch
torchvision
numpy
matplotlib
tqdm
opencv-python      # video export (anime notebook)
opendatasets        # Kaggle dataset download (anime notebook)
jovian               # experiment tracking, optional (anime notebook)
```

Install with:
```bash
pip install torch torchvision numpy matplotlib tqdm opencv-python opendatasets jovian
```

A CUDA-capable GPU is strongly recommended — training a DCGAN for tens of epochs on thousands of images is slow on CPU.

## Usage

**Anime DCGAN:**
1. Run the notebook; when prompted, provide your Kaggle API credentials to auto-download the Anime Face Dataset via `opendatasets`.
2. Run all cells to train for the configured number of epochs (default 25).
3. Generated samples are saved to `./generated/`, and a training video is compiled to `gans_training.avi`.

**Cartoon Face DCGAN:**
1. Place your cartoon face image dataset in a folder structured for `torchvision.datasets.ImageFolder` (images inside a subdirectory), and update `dataroot` in the notebook to point to it.
2. Run all cells to train for the configured number of epochs (default 50).
3. Loss curves and a real-vs-fake image comparison are plotted at the end.

## Notes

- Both notebooks use the classic DCGAN architecture but were configured independently — hyperparameters (epoch count, latent size, exact layer widths) differ slightly between the two.
- `06b-anime-dcgan.ipynb` retains tutorial markdown/commentary from its original source (Jovian's "Zero to GANs" series); attribution is preserved in the notebook and above.
- GAN training can be unstable — if generated images collapse to a single pattern (mode collapse) or the discriminator/generator loss diverges, consider adjusting the learning rate, batch size, or adding label smoothing.
