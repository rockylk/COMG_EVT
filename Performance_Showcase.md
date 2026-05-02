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
