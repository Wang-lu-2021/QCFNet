# QCFNet: Quantum-Classical Feature Fusion for Speckle Noise Filtering
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)]()
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)]()
[![PennyLane 0.23](https://img.shields.io/badge/PennyLane-0.23-purple.svg)]()


## 1. Project Introduction
This repository contains the official implementation of the paper **"A speckle noise filtering method based on quantum-classical feature fusion neural networks with Monte Carlo Tree Search"** (submitted to *ISPRS Journal of Photogrammetry and Remote Sensing*).  

### Core Objective
Speckle noise (a multiplicative interference in active coherent imaging like SAR and medical ultrasound) degrades image quality and impairs downstream tasks (e.g., land cover classification, tumor boundary recognition). Traditional methods (e.g., Lee filter) and classical deep learning models struggle to balance **noise suppression** and **structure preservation** in complex texture regions—this project solves this challenge via a quantum-classical hybrid approach.

### Key Advantages
- **Lightweight Architecture**: 48k parameters + 4 qubits (vs. state-of-the-art (SOTA) classical model FRANet: 9.46M parameters).
- **Superior Structural Preservation**: Higher SSIM (structural similarity) than SOTA models (e.g., 0.91 on CIFAR-10, 3.4% higher than FRANet).
- **Noise Robustness**: Retains stable performance even under extreme speckle noise (e.g., PSNR=26.95 dB when Looks=1, a metric where smaller values indicate stronger noise).


## 2. Environment Setup
Install required dependencies (consistent with the paper’s experimental environment):
```bash
pip install torch>=2.0.0 torchvision>=0.15.0 pennylane==0.23.0 numpy>=1.24.0 scikit-image>=0.21.0
```


## 3. Data Preparation
The paper uses two datasets for experiments—prepare data as follows:

### 3.1 CIFAR-10 Dataset (Auto-Download)
No manual operation needed:
- The code automatically downloads the CIFAR-10 dataset (60,000 32×32 color images).
- Preprocessing steps (embedded in code): Convert to grayscale, normalize pixel values to [0, π] (compatible with quantum gate angles), and add gamma-distributed speckle noise.
- Final split: 800 training pairs (noisy-clean) and 200 validation pairs (randomly selected from 1000 samples).

### 3.2 Sentinel-1 Dataset (User-Prepared)
1. **Data Source**: Retrieve Sentinel-1 GRD images from https://github.com/alessandrosebastianelli/sentinel_1_GRD_dataset.git, following the paper’s processing workflow.
2. **Preprocessing**:
   - Generate clean ground truth (GT) via temporal averaging of speckle-free intensity data for the same area.
   - Add gamma-distributed speckle noise (multiplied with clean images) to simulate real-world noisy scenarios.
   - Crop images into 64×64 patches and split into 1600 training pairs + 400 validation pairs.
3. **Folder Structure**:
   ```
   ./data/sentinel1/
   ├── clean/  # 2000 clean .png images (GT)
   └── noisy/  # 2000 noisy .png images (clean × speckle noise)
   ```


## 4. Quick Start
Run the full training pipeline (pre-training classical module → MCTS quantum circuit search → end-to-end training), as designed in the paper:

### 4.1 Train on CIFAR-10
```bash
python qcfnet_main.py --dataset cifar10 --data_root ./data/cifar10 --device cuda
```

### 4.2 Train on Sentinel-1
```bash
python qcfnet_main.py --dataset sentinel1 --data_root ./data/sentinel1 --device cuda
```

### 4.3 Outputs
- Trained model: Saved as `qcfnet_best_[dataset].pth` (e.g., `qcfnet_best_cifar10.pth`).
- Training log: Auto-generated as `qcfnet_denoising.log` (records loss, epoch progress, and MCTS simulation details).


## 5. Experimental Results (from the Paper)
The method outperforms SOTA models in both PSNR (peak signal-to-noise ratio, higher = better) and SSIM (higher = better) on two datasets:

### 5.1 CIFAR-10 Dataset
| Model               | PSNR (dB) | SSIM  | Model Size       |
|---------------------|-----------|-------|------------------|
| Mean Filter         | 22.76     | 0.65  | -                |
| Lee Filter          | 24.57     | 0.74  | -                |
| DnCNN               | 24.10     | 0.85  | 556k             |
| FFDNet              | 26.03     | 0.86  | 487k             |
| G-MoNet             | 26.86     | 0.83  | 557k             |
| FRANet (SOTA)       | 29.99     | 0.88  | 9.46M            |
| **Proposed Method** | 29.93     | 0.91  | 48k + 4q         |

### 5.2 Sentinel-1 Dataset
| Model               | PSNR (dB) | SSIM  | Model Size       |
|---------------------|-----------|-------|------------------|
| Speckled (No Filter)| 15.70     | 0.58  | -                |
| CNNSpeckleFilter    | 19.21     | 0.75  | 186k             |
| G-MoNet             | 25.32     | 0.78  | 557k             |
| QSPeckleFilter      | 21.72     | 0.81  | 42k + 16q        |
| FRANet (SOTA)       | 29.02     | 0.87  | 9.46M            |
| **Proposed Method** | 28.99     | 0.89  | 48k + 4q         |


## 6. Citation
If you use this code or the method in your research, please cite the paper:
```bibtex
@article{wang2025qcfnet,
  title={A speckle noise filtering method based on quantum-classical feature fusion neural networks with Monte Carlo Tree Search},
  author={Wang, Lu and Liu, Yuxiang and Meng, Fanxu and Zhang, Zaichen and Yu, Xutao},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  year={2025},
  publisher={Elsevier}
}
```


