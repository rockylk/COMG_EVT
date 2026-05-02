import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
LOG_FILE = os.path.join(RESULT_DIR, f"OpenDetect_Baseline_Log_{int(time.time())}.txt")

BATCH_SIZE = 64
EPOCHS = 50  # 生成式+分类联合训练
LEARNING_RATE = 0.001
MAX_SEQ_LEN = 400
RECON_WEIGHT = 1.0  # 重建损失的权重
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Logger:
    def __init__(self, filepath):
        self.filepath = filepath
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        self.log(f"=== Open-Detect (Generative AE + EVT) Training & Evaluation Log ===")
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
        class_to_idx[name] = -1  # 未知类统统打上 -1 标签
        
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
                    
    # 标准化 (仅使用 Known 数据 fit，杜绝数据泄露)
    if len(known_data_list) > 0:
        all_feats = np.concatenate([d[0] for d in known_data_list], axis=0)
        scaler = StandardScaler().fit(all_feats)
        known_data_list = [(scaler.transform(x), y) for x, y in known_data_list]
        unknown_data_list = [(scaler.transform(x), y) for x, y in unknown_data_list]
            
    return known_data_list, unknown_data_list, len(known_classes_names), {i: n for n, i in class_to_idx.items() if i != -1}

# ==========================================
# 3. Open-Detect 骨干架构 (Autoencoder + Classifier)
# ==========================================
class OpenDetect_Model(nn.Module):
    def __init__(self, num_classes, in_channels=5, seq_len=400):
        super(OpenDetect_Model, self).__init__()
        self.seq_len = seq_len
        
        # --- Encoder (编码器) ---
        self.enc_conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.enc_pool1 = nn.MaxPool1d(2) # length -> 200
        self.enc_conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.enc_pool2 = nn.MaxPool1d(2) # length -> 100
        self.flatten = nn.Flatten()
        self.latent_dim = 64 * (seq_len // 4)
        
        # --- Classifier (闭集分类器) ---
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
        # --- Decoder (解码重建器，生成式模型的核心) ---
        self.dec_fc = nn.Linear(self.latent_dim, 64 * (seq_len // 4))
        self.dec_up1 = nn.Upsample(scale_factor=2) 
        self.dec_conv1 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.dec_up2 = nn.Upsample(scale_factor=2) 
        self.dec_conv2 = nn.Conv1d(32, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # x: (Batch, Seq_len, Features) -> permute -> (Batch, 5, 400)
        x_perm = x.permute(0, 2, 1)
        
        # 编码
        e = F.relu(self.enc_pool1(self.enc_conv1(x_perm)))
        e = F.relu(self.enc_pool2(self.enc_conv2(e)))
        latent = self.flatten(e)
        
        # 分类
        logits = self.classifier(latent)
        
        # 重建解码
        d = F.relu(self.dec_fc(latent))
        d = d.view(-1, 64, self.seq_len // 4)
        d = F.relu(self.dec_conv1(self.dec_up1(d)))
        recon_x_perm = self.dec_conv2(self.dec_up2(d))
        
        # 还原回原来的形状 (Batch, 400, 5) 以计算 MSE
        recon_x = recon_x_perm.permute(0, 2, 1)
        
        return logits, recon_x

# ==========================================
# 4. 基于重建误差的 EVT 引擎 (Reconstruction-EVT)
# ==========================================
class Recon_EVT_Engine:
    def __init__(self, tail_size=20):
        self.tail_size = tail_size
        self.weibull_models = {}

    def fit(self, recon_errors, logits, labels, known_classes, logger):
        """对已知类的'重建误差(MSE)'的右尾(大误差部分)拟合Weibull分布"""
        recon_errors = np.array(recon_errors)
        logits = np.array(logits)
        labels = np.array(labels)
        preds = np.argmax(logits, axis=1)
        
        for c in known_classes:
            # 筛选出被正确分类的该类样本的重建误差
            mask = (labels == c) & (preds == c)
            class_errors = recon_errors[mask]
            
            # 兜底：如果分对的太少，就用所有该类样本
            if len(class_errors) < self.tail_size:
                class_errors = recon_errors[labels == c]
            if len(class_errors) == 0: continue
            
            # 误差从大到小排序，提取最大的那部分 (尾部)
            class_errors.sort()
            tail_errors = class_errors[-self.tail_size:] 
            
            # 拟合 Weibull
            shape, loc, scale = stats.weibull_min.fit(tail_errors, floc=0)
            self.weibull_models[c] = {'shape': shape, 'scale': scale}
            
        logger.log(f"  [Open-Detect] Calibrated Recon-EVT boundaries for {len(self.weibull_models)} classes.")

    def predict_score(self, recon_error, logit):
        """在线阶段：重建误差越大，越可能是零日攻击(Unknown)"""
        pred_cls = np.argmax(logit)
        if pred_cls not in self.weibull_models:
            return pred_cls, 0.0 
            
        shape = self.weibull_models[pred_cls]['shape']
        scale = self.weibull_models[pred_cls]['scale']
        
        # 计算该重建误差在分布中的累积概率 CDF (误差极大的话 CDF 趋近于1)
        prob_outlier = stats.weibull_min.cdf(recon_error, shape, loc=0, scale=scale)
        confidence_inlier = 1.0 - prob_outlier # 重建得越完美(误差小)，置信度越高
        
        return pred_cls, confidence_inlier

# ==========================================
# 5. 训练与测试流程
# ==========================================
def train_opendetect_baseline(known_data_list, unknown_data_list, num_known_classes, idx_to_class, logger):
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    random.shuffle(known_data_list)
    split = int(len(known_data_list) * 0.8)
    
    train_dataset = SeqDataset(known_data_list[:split])
    test_dataset = SeqDataset(known_data_list[split:] + unknown_data_list)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = OpenDetect_Model(num_classes=num_known_classes).to(DEVICE)
    
    # --- Stage 1: 联合训练 (分类交叉熵 + 重建MSE) ---
    logger.log(f"\n>>> [Stage 1] Training Generative AE & Classifier...")
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_recon = nn.MSELoss()
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss, total_recon = 0, 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            
            logits, recon_x = model(batch_x)
            loss_cls = criterion_cls(logits, batch_y)
            loss_recon = criterion_recon(recon_x, batch_x)
            
            loss = loss_cls + RECON_WEIGHT * loss_recon
            loss.backward(); optimizer.step()
            
            total_loss += loss.item()
            total_recon += loss_recon.item()
            
        if (epoch + 1) % 10 == 0: 
            logger.log(f"  Epoch {epoch+1}/{EPOCHS} | Total Loss: {total_loss/len(train_loader):.4f} | Recon MSE: {total_recon/len(train_loader):.4f}")

    # --- Stage 2: 提取重建误差并拟合 EVT ---
    logger.log(f"\n>>> [Stage 2] Offline EVT Tail-fitting on Reconstruction Errors...")
    model.eval()
    all_recon_errors = []
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        extractor_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
        for batch_x, batch_y in extractor_loader:
            batch_x = batch_x.to(DEVICE)
            logits, recon_x = model(batch_x)
            
            # 计算每个样本的 MSE 重建误差 (Batch_size,)
            mse_errors = torch.mean((recon_x - batch_x) ** 2, dim=(1, 2)).cpu().numpy()
            
            all_recon_errors.extend(mse_errors)
            all_logits.extend(logits.cpu().numpy())
            all_labels.extend(batch_y.numpy())
            
    recon_evt_engine = Recon_EVT_Engine(tail_size=20)
    recon_evt_engine.fit(all_recon_errors, all_logits, all_labels, list(range(num_known_classes)), logger)

    # --- Stage 3: Online Detection & Table II Metrics ---
    logger.log("\n>>> [Stage 3] Calculating Table II Metrics (Open-Detect AUROC, AUOSCR, CCR, FPR)...")
    
    model.eval()
    closed_set_preds = []
    confidences = []
    true_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(DEVICE)
            logits, recon_x = model(batch_x)
            
            mse_errors = torch.mean((recon_x - batch_x) ** 2, dim=(1, 2)).cpu().numpy()
            logits_np = logits.cpu().numpy()
            
            for i in range(len(mse_errors)):
                # 重建误差越大，置信度越低
                pred_cls, conf = recon_evt_engine.predict_score(mse_errors[i], logits_np[i])
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
    logger.log("  Open-Set Metrics (Baseline: Open-Detect / Gen-AE)")
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
            train_opendetect_baseline(k_list, u_list, n_k, idx_cls, logger)
    else:
        logger.log(f"Error: Dataset path '{DATASET_PATH}' not found.")