import os
import time
import random
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

class Logger:
    def __init__(self, filepath):
        self.filepath = filepath
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        self.log(f"=== COMG-EVT-V2 (Open-Set) Training & Evaluation Log ===")
        self.log(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")

    def log(self, message):
        print(message)
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(message + '\n')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def split_known_dataset(known_dataset, logger, train_ratio=0.7, calib_ratio=0.1):
    random.shuffle(known_dataset)
    n = len(known_dataset)
    n_train = max(1, int(n * train_ratio))
    n_calib = max(1, int(n * calib_ratio))

    if n_train + n_calib >= n:
        n_calib = max(1, n - n_train - 1)
    if n_train + n_calib >= n:
        n_train = max(1, n - n_calib - 1)

    train_set = known_dataset[:n_train]
    calib_set = known_dataset[n_train:n_train + n_calib]
    test_known_set = known_dataset[n_train + n_calib:]

    logger.log(f"Dataset Split -> Train: {len(train_set)}, Calib: {len(calib_set)}, TestKnown: {len(test_known_set)}")
    return train_set, calib_set, test_known_set

def fit_and_apply_scalers(train_set, calib_set, test_known_set, unknown_set, logger):
    if len(train_set) == 0:
        logger.log("[Scaler] Error: train_set is empty.")
        return

    node_scaler = StandardScaler().fit(np.concatenate([d.x.numpy() for d in train_set], axis=0))
    stats_scaler = StandardScaler().fit(np.concatenate([d.stats_attr.numpy() for d in train_set], axis=0))

    for ds in [train_set, calib_set, test_known_set, unknown_set]:
        for data in ds:
            data.x = torch.tensor(node_scaler.transform(data.x.numpy()), dtype=torch.float)
            data.stats_attr = torch.tensor(stats_scaler.transform(data.stats_attr.numpy()), dtype=torch.float)

    logger.log("[Scaler] Fitted on train_set only and applied to calib/test/unknown.")

def collect_features_and_logits(model, loader, device):
    model.eval()
    all_features, all_logits, all_labels = [], [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            feats = model.extract_fused_features(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                batch.stats_attr, batch.finger_attr, batch.entropy_attr
            )
            logits = model.classifier_head(feats)
            all_features.extend(feats.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())
            all_labels.extend(batch.y.view(-1).cpu().numpy())

    return (
        np.asarray(all_features, dtype=np.float32),
        np.asarray(all_logits, dtype=np.float32),
        np.asarray(all_labels, dtype=np.int64)
    )