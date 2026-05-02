import os
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, auc
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from models import ContrastiveGNN_V13, info_nce_loss
from features import augment_graph_view
from evt_engine import EVT_Engine_V2
from utils import set_seed, split_known_dataset, fit_and_apply_scalers, collect_features_and_logits

def train_v13_evt_v2(known_dataset, unknown_dataset, num_known_classes, idx_to_class, logger, result_dir):
    set_seed(42)

    # ---------- Step 0: Data Splitting ----------
    train_set, calib_set, test_known_set = split_known_dataset(known_dataset, logger, train_ratio=0.7, calib_ratio=0.1)
    fit_and_apply_scalers(train_set, calib_set, test_known_set, unknown_dataset, logger)

    train_batch_size = min(64, len(train_set))
    eval_batch_size = 64

    if train_batch_size < 2:
        logger.log("[Error] Train set is too small for contrastive learning.")
        return

    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, drop_last=False)
    calib_loader = DataLoader(calib_set, batch_size=eval_batch_size, shuffle=False)
    test_loader = DataLoader(test_known_set + unknown_dataset, batch_size=eval_batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.log(f"Using device: {device}")

    model = ContrastiveGNN_V13(5, 9, 20, 5, 32, num_known_classes).to(device)

    # --- Stage 1: Contrastive Pre-training ---
    logger.log(f"\n>>> [Stage 1] Pre-training (Compressing Manifolds)...")
    optimizer_pre = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    model.train()
    for epoch in range(30):
        total_loss = 0.0
        for batch_data in train_loader:
            data_list = batch_data.to_data_list()
            view1_list, view2_list = [], []
            for data_item in data_list:
                data_item = data_item.to(device)
                x1, ei1, ea1, b1, _, _, _ = augment_graph_view(data_item)
                x2, ei2, ea2, b2, _, _, _ = augment_graph_view(data_item)
                view1_list.append(Data(x=x1, edge_index=ei1, edge_attr=ea1))
                view2_list.append(Data(x=x2, edge_index=ei2, edge_attr=ea2))

            batch_view1 = Batch.from_data_list(view1_list).to(device)
            batch_view2 = Batch.from_data_list(view2_list).to(device)

            optimizer_pre.zero_grad()
            z1 = model.forward_contrastive(batch_view1.x, batch_view1.edge_index, batch_view1.edge_attr, batch_view1.batch)
            z2 = model.forward_contrastive(batch_view2.x, batch_view2.edge_index, batch_view2.edge_attr, batch_view2.batch)

            loss = info_nce_loss(z1, z2, temperature=0.1)
            loss.backward()
            optimizer_pre.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            logger.log(f"  Epoch {epoch + 1} | InfoNCE Loss: {total_loss / max(len(train_loader), 1):.4f}")

    # --- Stage 2: Classifier Fine-Tuning ---
    logger.log(f"\n>>> [Stage 2] Classifier Fine-tuning...")
    optimizer_fine = torch.optim.AdamW(model.parameters(), lr=0.002)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(50):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer_fine.zero_grad()
            logits = model.forward_classifier(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                batch.stats_attr, batch.finger_attr, batch.entropy_attr
            )
            loss = criterion(logits, batch.y.view(-1))
            loss.backward()
            optimizer_fine.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            logger.log(f"  Epoch {epoch + 1} | CE Loss: {total_loss / max(len(train_loader), 1):.4f}")

    # --- Stage 3: Offline EVT-V2 Calibration ---
    logger.log(f"\n>>> [Stage 3] Offline EVT-V2 Calibration (using calib set)...")
    calib_features, calib_logits, calib_labels = collect_features_and_logits(model, calib_loader, device)

    evt_engine = EVT_Engine_V2()
    evt_engine.fit(calib_features, calib_logits, calib_labels, known_classes=list(range(num_known_classes)), logger=logger)

    # --- Stage 4: Open-Set Evaluation ---
    logger.log("\n>>> [Stage 4] Calculating Open-Set Metrics (AUROC, AUOSCR, CCR, FPR)...")
    test_features, test_logits, test_labels = collect_features_and_logits(model, test_loader, device)

    closed_set_preds, confidences, accept_flags, true_labels = [], [], [], []

    for feat, logit, true_y in zip(test_features, test_logits, test_labels):
        pred_cls, conf, accepted, aux = evt_engine.predict_open_set_score(feat, logit)
        closed_set_preds.append(pred_cls)
        confidences.append(conf)
        accept_flags.append(1 if accepted else 0)
        true_labels.append(int(true_y))

    closed_set_preds = np.asarray(closed_set_preds, dtype=np.int64)
    confidences = np.asarray(confidences, dtype=np.float32)
    accept_flags = np.asarray(accept_flags, dtype=np.int64)
    true_labels = np.asarray(true_labels, dtype=np.int64)

    known_mask = true_labels != -1
    unknown_mask = true_labels == -1
    total_known = int(np.sum(known_mask))
    total_unknown = int(np.sum(unknown_mask))

    binary_true = np.where(unknown_mask, 1, 0)
    anomaly_scores = 1.0 - confidences
    auroc = roc_auc_score(binary_true, anomaly_scores) * 100.0 if (total_known > 0 and total_unknown > 0) else 0.0

    thresholds = np.linspace(0.0, 1.0, 1000)
    ccr_list, fpr_list = [], []

    for t in thresholds:
        ccr = np.sum((closed_set_preds == true_labels) & known_mask & (confidences >= t)) / total_known if total_known > 0 else 0.0
        fpr = np.sum(unknown_mask & (confidences >= t)) / total_unknown if total_unknown > 0 else 0.0
        ccr_list.append(ccr)
        fpr_list.append(fpr)

    fpr_arr = np.array(fpr_list)[::-1]
    ccr_arr = np.array(ccr_list)[::-1]
    auoscr = auc(fpr_arr, ccr_arr) * 100.0 if (len(fpr_arr) > 1 and len(ccr_arr) > 1) else 0.0

    # Save NPY arrays using argparse dir
    np.save(os.path.join(result_dir, "COMG_EVT_V2_FPR.npy"), np.array(fpr_list))
    np.save(os.path.join(result_dir, "COMG_EVT_V2_CCR.npy"), np.array(ccr_list))

    correct_knowns_oper = np.sum((closed_set_preds == true_labels) & known_mask & (accept_flags == 1))
    false_accepts_oper = np.sum(unknown_mask & (accept_flags == 1))

    ccr_oper = (correct_knowns_oper / total_known) * 100.0 if total_known > 0 else 0.0
    fpr_oper = (false_accepts_oper / total_unknown) * 100.0 if total_unknown > 0 else 0.0

    # Output Logs
    logger.log("\n" + "=" * 60)
    logger.log("  Open-Set Detection Metrics (COMG-EVT-V2)")
    logger.log("=" * 60)
    logger.log(f"  * AUROC  : {auroc:.2f}%  (Known vs Unknown discrimination)")
    logger.log(f"  * AUOSCR : {auoscr:.2f}%  (Area under CCR-FPR Curve)")
    logger.log("\n  [Operational Metrics @ calibrated class-thresholds + margin gate]")
    logger.log(f"  * CCR    : {ccr_oper:.2f}%  (Correct Classification Rate of Knowns)")
    logger.log(f"  * FPR    : {fpr_oper:.2f}%  (Leakage Rate of Zero-day Unknowns)")
    logger.log("=" * 60)