# COMG-EVT: Open-Set Encrypted Traffic Recognition

This repository contains the official implementation of **Open-Set Encrypted Traffic Recognition via Contrastively-Optimized Multi-Modal Gating and Extreme Value Theory (COMG-EVT)**. 

Our framework overcomes the challenge of detecting zero-day threats in fully encrypted environments (e.g., TLS 1.3) by utilizing Graph Contrastive Learning (InfoNCE) to eliminate metric distortion, and Extreme Value Theory (EVT) to construct rigorous statistical boundaries for unknown attacks.

## 📂 Project Structure
```text
COMG-EVT/
├── features.py        # PCAP parsing, protocol feature extraction, and GNN Data modeling
├── models.py          # Network architectures: GNN, 1D-CNN, GatedFusion, and InfoNCE loss
├── evt_engine.py      # Statistical Extreme Value Theory (Weibull) boundary modeling
├── utils.py           # Evaluation metrics, dataset splitting, scaling, and loggers
├── dataset.py         # Directory traversal logic for PCAP folders
├── trainer.py         # End-to-end 4-stage optimization and evaluation pipeline
├── main.py            # CLI entry point
├── requirements.txt   # Required Python dependencies
└── README.md          # Usage instructions
```
# Installation

We recommend using a virtual environment (such as Conda or venv).

# Clone the repository
git clone [https://github.com/your-username/COMG-EVT.git](https://github.com/your-username/COMG-EVT.git)
cd COMG-EVT

# Install dependencies
pip install -r requirements.txt

Note: Ensure your torch and torch-geometric installations are compatible with your system's CUDA version.
🚀 Usage

You can launch the training and evaluation pipeline from main.py using command-line arguments. No manual hardcoding of directory paths is required.
Bash

python main.py --pcap_dir /path/to/your/pcap/dataset --result_dir ./results --num_unknowns 5

Arguments:

    --pcap_dir (Required): The absolute or relative path to the directory containing your traffic dataset. The directory should contain sub-folders, with each sub-folder representing a specific traffic category.

    --result_dir (Optional): The output folder for EVT threshold numpy arrays (CCR.npy, FPR.npy) and log files. Defaults to ./results.

    --num_unknowns (Optional): How many of the tail directories to isolate as zero-day attacks during testing. Defaults to 5.

🧠 Pipeline Overview

    Pre-training (Manifold Compression): Augments graph views of traffic logic and uses contrastive learning to squeeze instances of the same known traffic class tightly together.

    Fine-Tuning: Normalizes class assignments on the compacted latent space.
    # Performance Showcase

The COMG-EVT framework has been rigorously evaluated on two mainstream encrypted traffic datasets: **USTC-TFC2016** (focused on malware classification) and **CipherSpectrum** (a state-of-the-art 2025 dataset focused on TLS 1.3 protocol analysis).

## 1. Fine-Grained Closed-Set Classification
In the known-class classification task, COMG-EVT demonstrates superior discriminative power compared to traditional sequential models (e.g., CNN-LSTM) and recent payload-heavy frameworks (e.g., BSTFNet):

| Dataset | Macro F1-score (%) | Accuracy (%) |
| :--- | :--- | :--- |
| **USTC-TFC2016** | **94.31 ± 0.25** | **94.50 ± 0.22** |
| **CipherSpectrum** | **91.65 ± 0.35** | **91.76 ± 0.31** |

*Note: On the CipherSpectrum dataset (fully TLS 1.3), traditional models suffer from feature blindness due to the lack of plaintext shortcuts (e.g., SNI). COMG-EVT maintains over 91% accuracy by leveraging multi-modal GatedFusion.*

## 2. Open-Set Detection (Zero-Day Attack Recognition)
The core advantage of COMG-EVT lies in its precise interception of unseen zero-day threats. By establishing rigorous statistical boundaries via Extreme Value Theory (EVT), the model significantly reduces leakage (FPR) while maintaining high recall:

| Dataset | AUROC (%) | CCR (Correct Class. Rate) | FPR (Zero-day Leakage) |
| :--- | :--- | :--- | :--- |
| **USTC-TFC2016** | **78.85 ± 0.35** | 92.91% | **68.06%** |
| **CipherSpectrum** | **79.65 ± 0.55** | 85.36% | **56.50%** |

- **Comparative Superiority**: COMG-EVT reduces the False Positive Rate (FPR) by over 30% compared to traditional baselines like IM-ZDD, which often exhibit leakage rates exceeding 90% under similar conditions.
- **Robustness to Openness**: As the number of unknown classes (L) increases, COMG-EVT exhibits remarkable stability, with AUROC fluctuating by only ~4%, demonstrating strong resilience in dynamic open-world environments.

## 3. Ablation Study Insights
Our ablation study confirms that each architectural component is vital for defending against open-space risks:
- **Without EVT**: AUROC plummets by **34.01 points**, confirming the inherent overconfidence of standard Softmax in open space.
- **Without Contrastive Pre-training**: AUOSCR drops by **18.32 points**, validating that geometric manifold compression is a prerequisite for accurate tail fitting.
- **Without GatedFusion**: FPR surges to **93.44%**, proving that adaptive gating is essential to prevent noisy macroscopic features from corrupting representations under full encryption.


    EVT Calibration: Identifies prototypes using K-means and dynamically fits Weibull parameters to the distance thresholds on a localized calibration set.

    Open-Set Testing: Ingests novel/zero-day application flows and establishes operational boundaries computing AUROC and AUOSCR curves.
