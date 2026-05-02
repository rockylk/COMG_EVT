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
⚙️ Installation

We recommend using a virtual environment (such as Conda or venv).
Bash

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

    EVT Calibration: Identifies prototypes using K-means and dynamically fits Weibull parameters to the distance thresholds on a localized calibration set.

    Open-Set Testing: Ingests novel/zero-day application flows and establishes operational boundaries computing AUROC and AUOSCR curves.