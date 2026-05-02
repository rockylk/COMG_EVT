import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scapy.all import rdpcap, IP, TCP, UDP, Raw
from sklearn.metrics import classification_report, roc_auc_score, auc
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# 0. 全局配置与路径切换 (支持一行切换)
# ==========================================
# 【请手动注释/取消注释以切换数据集】
DATASET_PATH = r"C:\Desktop\GNN\aes-128-gcm\aes-128-gcm"  # CipherSpectrum
# DATASET_PATH = r"C:\Desktop\GNN\USTC-TFC2016-Split"       # USTC-TFC2016

RESULT_DIR = r"C:\GNN\RESULT"
LOG_FILE = os.path.join(RESULT_DIR, f"IIoT_EVT_Baseline_Log_{int(time.time())}.txt")

BATCH_SIZE = 64
EPOCHS = 50  # 纯交叉熵有监督训练
LEARNING_RATE = 0.001
MAX_SEQ_LEN = 400
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Logger:
    def __init__(self, filepath):
        self.filepath = filepath
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        self.log(f"=== IIoT-EVT Baseline (1D-CNN + Feature Space EVT) Log ===")
        self.log(f"Dataset Path: {DATASET_PATH}")
        self.log(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")

    def log(self, message):
        print(message)
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(message + '\n')

# ==========================================
# 1. 特征工程 V13 (严格对齐脱敏标准，抛弃明文)
# ==========================================
def get_direction(pkt, client_ip):
    try:
        return 0 if pkt[IP].src == client_ip else 1
    except:
        return 0

def extract_features_v13_seq(packets, client_ip, max_nodes=400):
    node_features = []
    sizes, times = [], []
    start_time = float(packets[0].time)
    prev_time = start_time
    
    for pkt in packets[:max_nodes]:
        curr_time = float(pkt.time)
        size = len(pkt)
        direction = get_direction(pkt, client_ip)
        iat = curr_time - prev_time
        
        sizes.append(size); times.append(iat)
        node_features.append([direction, size, iat, np.mean(sizes[-5:]), np.mean(times[-5:])])
        prev_time = curr_time

    return np.array(node_features)

def pcap_to_seq_v13(pcap_path, label_int):
    try:
        packets = rdpcap(pcap_path)
        packets = [p for p in packets if IP in p and (TCP in p or UDP in p)]
        if len(packets) < 5: return None
        client_ip = packets[0][IP].src
    except:
        return None

    node_feats = extract_features_v13_seq(packets, client_ip)
    return node_feats, label_int

# ==========================================
# 2. 数据加载与零日攻击物理隔离
# ==========================================
class SeqDataset(Dataset):
    def __init__(self, data_list, max_len=MAX_SEQ_LEN):
        self.samples = []
        self.labels = []
        for x, y in data_list:
            N, F = x.shape
            if N < max_len:
                pad_width = ((0, max_len - N), (0, 0))
                x_padded = np.pad(x, pad_width, mode='constant', constant_values=0.0)
            else:
                x_padded = x[:max_len, :]
            self.samples.append(x_padded)
            self.labels.append(y)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx], dtype=torch.float32), \
               torch.tensor(self.labels[idx], dtype=torch.long)

def load_dataset_openset_seq(root_dir, logger, num_unknown_classes=5):
    subdirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    
    known_classes_names = subdirs[:-num_unknown_classes]
    unknown_classes_names = subdirs[-num_unknown_classes:]
    
    class_to_idx = {name: i for i, name in enumerate(known_classes_names)}
    for name in unknown_classes_names:
        class_to_idx[name] = -1  
        
    logger.log(f"Loading Dataset: {len(known_classes_names)} Knowns, {len(unknown_classes_names)} Unknowns (Zero-days).")
    
    known_data_list = []
    unknown_data_list = []
    
    for class_name in subdirs:
        class_dir = os.path.join(root_dir, class_name)
        label_int = class_to_idx[class_name]
        files = [f for f in os.listdir(class_dir) if f.endswith(".pcap") or f.endswith(".pcapng")]
        for f in files:
            res = pcap_to_seq_v13(os.path.join(class_dir, f), label_int)
            if res is not None:
                x, y = res
                if y != -1: known_data_list.append((x, y))
                else: unknown_data_list.append((x, y))
                    
    # 标准化 (仅使用 Known 数据 fit)
    if len(known_data_list) > 0:
        all_feats = np.concatenate([d[0] for d in known_data_list], axis=0)
        scaler = StandardScaler().fit(all_feats)
        known_data_list = [(scaler.transform(x), y) for x, y in known_data_list]
        unknown_data_list = [(scaler.transform(x), y) for x, y in unknown_data_list]
            
    return known_data_list, unknown_data_list, len(known_classes_names), {i: n for n, i in class_to_idx.items() if i != -1}

# ==========================================
# 3. IIoT-EVT 骨干架构 (1D-CNN Feature Extractor)
# ==========================================
class IIoT_EVT_Backbone(nn.Module):
    def __init__(self, num_classes, in_channels=5):
        super(IIoT_EVT_Backbone, self).__init__()
        
        # IIoT 常见的轻量级时空卷积网络
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )
        
        # 提取深层特征的降维层 (EVT 拟合的目标空间)
        self.feature_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # 闭集分类头
        self.classifier = nn.Linear(64, num_classes)

    def extract_feature(self, x):
        """专门提取倒数第二层的深层特征供 EVT 使用"""
        x_perm = x.permute(0, 2, 1) # (Batch, 5, 400)
        cnn_out = self.cnn(x_perm)
        deep_feature = self.feature_layer(cnn_out)
        return deep_feature

    def forward(self, x):
        deep_feature = self.extract_feature(x)
        logits = self.classifier(deep_feature)
        return deep_feature, logits

# ==========================================
# 4. Feature EVT Engine (在特征空间内拟合分布)
# ==========================================
class Feature_EVT_Engine:
    def __init__(self, tail_size=20, distance_metric='euclidean'):
        self.tail_size = tail_size
        self.distance_metric = distance_metric
        self.class_centroids = {}
        self.weibull_models = {}

    def calc_distance(self, x, y):
        if self.distance_metric == 'euclidean':
            return np.linalg.norm(x - y, axis=-1)
        return np.linalg.norm(x - y, axis=-1)

    def fit(self, features, logits, labels, known_classes, logger):
        """仅利用正确分类的样本的特征拟合 Weibull 分布"""
        features = np.array(features)
        logits = np.array(logits)
        labels = np.array(labels)
        preds = np.argmax(logits, axis=1)
        
        for c in known_classes:
            # 筛选正确分类的特征
            mask = (labels == c) & (preds == c)
            class_features = features[mask]
            
            # 兜底机制
            if len(class_features) < self.tail_size:
                class_features = features[labels == c]
            if len(class_features) == 0: continue
            
            centroid = np.mean(class_features, axis=0)
            self.class_centroids[c] = centroid
            
            distances = [self.calc_distance(f, centroid) for f in class_features]
            distances.sort(reverse=True)
            
            tail_distances = distances[:self.tail_size]
            shape, loc, scale = stats.weibull_min.fit(tail_distances, floc=0)
            self.weibull_models[c] = {'shape': shape, 'scale': scale}
            
        logger.log(f"  [IIoT-EVT] Calibrated EVT boundaries on Deep Features for {len(self.weibull_models)} classes.")

    def predict_score(self, feature, logit):
        """在线阶段：通过特征到类中心的距离评估拒识置信度"""
        pred_cls = np.argmax(logit)
        centroid = self.class_centroids.get(pred_cls)
        
        if centroid is None:
            return pred_cls, 0.0 
            
        dist = self.calc_distance(feature, centroid)
        shape = self.weibull_models[pred_cls]['shape']
        scale = self.weibull_models[pred_cls]['scale']
        
        prob_outlier = stats.weibull_min.cdf(dist, shape, loc=0, scale=scale)
        confidence_inlier = 1.0 - prob_outlier
        
        return pred_cls, confidence_inlier

# ==========================================
# 5. 训练与测试流程
# ==========================================
def train_iiot_evt_baseline(known_data_list, unknown_data_list, num_known_classes, idx_to_class, logger):
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    random.shuffle(known_data_list)
    split = int(len(known_data_list) * 0.8)
    
    train_dataset = SeqDataset(known_data_list[:split])
    test_dataset = SeqDataset(known_data_list[split:] + unknown_data_list)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = IIoT_EVT_Backbone(num_classes=num_known_classes).to(DEVICE)
    
    # --- Stage 1: Standard Supervised Training ---
    logger.log(f"\n>>> [Stage 1] Standard Closed-Set Training (Cross Entropy)...")
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            _, logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward(); optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0: logger.log(f"  Epoch {epoch+1}/{EPOCHS} | CE Loss: {total_loss/len(train_loader):.4f}")

    # --- Stage 2: Extract Deep Features & Fit EVT ---
    logger.log(f"\n>>> [Stage 2] Offline EVT Tail-fitting on Deep Features...")
    model.eval()
    all_train_features = []
    all_train_logits = []
    all_train_labels = []
    
    with torch.no_grad():
        extractor_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
        for batch_x, batch_y in extractor_loader:
            batch_x = batch_x.to(DEVICE)
            features, logits = model(batch_x)
            all_train_features.extend(features.cpu().numpy())
            all_train_logits.extend(logits.cpu().numpy())
            all_train_labels.extend(batch_y.numpy())
            
    feat_evt_engine = Feature_EVT_Engine(tail_size=20)
    feat_evt_engine.fit(all_train_features, all_train_logits, all_train_labels, list(range(num_known_classes)), logger)

    # --- Stage 3: Online Detection & Table II Metrics ---
    logger.log("\n>>> [Stage 3] Calculating Table II Metrics (IIoT-EVT AUROC, AUOSCR, CCR, FPR)...")
    
    model.eval()
    closed_set_preds = []
    confidences = []
    true_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(DEVICE)
            features, logits = model(batch_x)
            
            features_np = features.cpu().numpy()
            logits_np = logits.cpu().numpy()
            
            for i in range(len(features_np)):
                pred_cls, conf = feat_evt_engine.predict_score(features_np[i], logits_np[i])
                closed_set_preds.append(pred_cls)
                confidences.append(conf)
                true_labels.append(batch_y[i].item())

    closed_set_preds = np.array(closed_set_preds)
    confidences = np.array(confidences)
    true_labels = np.array(true_labels)
    
    known_mask = true_labels != -1
    unknown_mask = true_labels == -1
    total_known = np.sum(known_mask)
    total_unknown = np.sum(unknown_mask)

    # 1. AUROC
    binary_true = np.where(unknown_mask, 1, 0)
    anomaly_scores = 1.0 - confidences 
    auroc = roc_auc_score(binary_true, anomaly_scores) * 100.0

    # 2. AUOSCR
    thresholds = np.linspace(0.0, 1.0, 1000)
    ccr_list, fpr_list = [], []
    for t in thresholds:
        correct_knowns = np.sum((closed_set_preds == true_labels) & known_mask & (confidences >= t))
        ccr = correct_knowns / total_known if total_known > 0 else 0
        false_accepts = np.sum(unknown_mask & (confidences >= t))
        fpr = false_accepts / total_unknown if total_unknown > 0 else 0
        ccr_list.append(ccr); fpr_list.append(fpr)
        
    auoscr = auc(np.array(fpr_list)[::-1], np.array(ccr_list)[::-1]) * 100.0

    # 3. Metrics @ τ = 0.5
    tau = 0.5
    ccr_tau = (np.sum((closed_set_preds == true_labels) & known_mask & (confidences >= tau)) / total_known) * 100.0
    fpr_tau = (np.sum(unknown_mask & (confidences >= tau)) / total_unknown) * 100.0

    # 输出表格结果
    logger.log("\n" + "="*50)
    logger.log("  Open-Set Metrics (Baseline: IIoT-EVT)")
    logger.log("="*50)
    logger.log(f"  * AUROC  : {auroc:.2f}%  (vs. Unknowns)")
    logger.log(f"  * AUOSCR : {auoscr:.2f}%  (Area under Curve)")
    logger.log(f"  * CCR    : {ccr_tau:.2f}%  (@ τ={tau})")
    logger.log(f"  * FPR    : {fpr_tau:.2f}%  (Leakage Rate @ τ={tau})")
    logger.log("="*50)
    logger.log(f"\nResults saved to: {LOG_FILE}")

if __name__ == "__main__":
    logger = Logger(LOG_FILE)
    if os.path.exists(DATASET_PATH):
        k_list, u_list, n_k, idx_cls = load_dataset_openset_seq(DATASET_PATH, logger, num_unknown_classes=5)
        if len(k_list) > 0:
            train_iiot_evt_baseline(k_list, u_list, n_k, idx_cls, logger)
    else:
        logger.log(f"Error: Dataset path '{DATASET_PATH}' not found.")