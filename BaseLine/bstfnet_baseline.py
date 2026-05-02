import os
import time
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score, accuracy_score
import warnings
warnings.filterwarnings("ignore")

# 严格对齐 V13 的数据管道，杜绝明文特征泄露
from V13_all import load_dataset_v13

# ==========================================
# 1. 全局配置与路径切换
# ==========================================
# 请根据需要切换数据集
# DATASET_PATH = r"C:\Desktop\GNN\aes-128-gcm\aes-128-gcm"  # CipherSpectrum
DATASET_PATH = r"C:\Desktop\GNN\USTC-TFC2016-Split"       # USTC-TFC2016

RESULT_DIR = r"C:\GNN\RESULT"
LOG_FILE = os.path.join(RESULT_DIR, f"BSTFNet_Log_{int(time.time())}.txt")

BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
MAX_SEQ_LEN = 400
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. 日志记录模块
# ==========================================
class Logger:
    def __init__(self, filepath):
        self.filepath = filepath
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        self.log(f"=== BSTFNet (BERT + TextCNN + BiGRU) Training Log ===")
        self.log(f"Dataset Path: {DATASET_PATH}")
        self.log(f"Device: {DEVICE}\n")

    def log(self, message):
        print(message)
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(message + '\n')

# ==========================================
# 3. 数据适配器 (统一标准，绝对公平)
# ==========================================
class SeqAdapterDataset(Dataset):
    def __init__(self, pyg_dataset, max_len=MAX_SEQ_LEN):
        self.samples = []
        self.labels = []
        
        for data in pyg_dataset:
            x = data.x.numpy()  # 获取节点特征序列 (N x 5)
            y = data.y.item()
            
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

# ==========================================
# 4. BSTFNet 核心架构设计
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class BSTFNet_Baseline(nn.Module):
    def __init__(self, num_classes, in_channels=5, hidden_dim=64):
        super(BSTFNet_Baseline, self).__init__()
        
        # --- Branch 1: 全局语义分支 (模拟 BERT / Transformer Encoder) ---
        self.embed = nn.Linear(in_channels, hidden_dim)
        self.pos_encoder = PositionalEncoding(d_model=hidden_dim, max_len=MAX_SEQ_LEN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=128, dropout=0.1, batch_first=True
        )
        self.bert_branch = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # --- Branch 2: 时序状态分支 (BiGRU) ---
        self.bigru_branch = nn.GRU(
            input_size=in_channels, hidden_size=hidden_dim // 2, 
            num_layers=1, batch_first=True, bidirectional=True
        ) # 输出维度 = 32 * 2 = 64
        
        # --- Branch 3: 局部空间感受野分支 (多尺度 TextCNN) ---
        self.conv1 = nn.Conv1d(in_channels, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels, out_channels=32, kernel_size=4, padding=2)
        self.conv3 = nn.Conv1d(in_channels, out_channels=32, kernel_size=5, padding=2)
        
        # --- 多维特征融合层 (Fusion) ---
        # BERT (64) + BiGRU (64) + TextCNN (32 * 3 = 96) = 224
        fusion_dim = hidden_dim + hidden_dim + 96
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (Batch, Seq_len, Features) -> (Batch, 400, 5)
        
        # 1. 提取 BERT 全局特征
        x_emb = self.embed(x)
        x_pos = self.pos_encoder(x_emb)
        bert_out = self.bert_branch(x_pos)
        bert_feat = bert_out.mean(dim=1)  # 全局均值池化 (Batch, 64)
        
        # 2. 提取 BiGRU 时序特征
        gru_out, _ = self.bigru_branch(x)
        gru_feat = gru_out[:, -1, :]  # 取最后一个时间步 (Batch, 64)
        
        # 3. 提取 TextCNN 局部空间特征
        x_cnn = x.permute(0, 2, 1)  # 转换维度给 Conv1d: (Batch, 5, 400)
        c1 = torch.relu(self.conv1(x_cnn))
        c2 = torch.relu(self.conv2(x_cnn))
        c3 = torch.relu(self.conv3(x_cnn))
        
        # 1D-Max Pooling over time
        c1 = torch.max(c1, dim=2)[0]  # (Batch, 32)
        c2 = torch.max(c2, dim=2)[0]
        c3 = torch.max(c3, dim=2)[0]
        cnn_feat = torch.cat([c1, c2, c3], dim=1)  # (Batch, 96)
        
        # 4. 特征级联与分类
        fused_features = torch.cat([bert_feat, gru_feat, cnn_feat], dim=1)  # (Batch, 224)
        logits = self.classifier(fused_features)
        
        return logits

# ==========================================
# 5. 训练与评估流程 (自动落盘)
# ==========================================
def main():
    logger = Logger(LOG_FILE)
    start_time = time.time()
    
    # 强制控制随机变量
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    logger.log(">>> Loading dataset using V13 pipeline for BSTFNet...")
    pyg_dataset, num_classes, idx_to_class = load_dataset_v13(DATASET_PATH)
    
    if not pyg_dataset:
        logger.log("Error: No data loaded. Check the dataset path.")
        return

    random.shuffle(pyg_dataset)
    split = int(len(pyg_dataset) * 0.8)
    
    train_adapter = SeqAdapterDataset(pyg_dataset[:split])
    test_adapter = SeqAdapterDataset(pyg_dataset[split:])
    
    train_loader = DataLoader(train_adapter, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_adapter, batch_size=BATCH_SIZE, shuffle=False)
    
    # 初始化 BSTFNet
    model = BSTFNet_Baseline(num_classes=num_classes, in_channels=5, hidden_dim=64).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    logger.log(f"\n>>> Starting BSTFNet (Multi-Branch Fusion) Training on {DEVICE}...")
    best_macro_f1 = 0.0
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                outputs = model(batch_x)
                preds = outputs.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
                
        acc = accuracy_score(all_targets, all_preds)
        macro_f1 = f1_score(all_targets, all_preds, average='macro')
        
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            
        if epoch % 10 == 0:
            logger.log(f"Epoch [{epoch:03d}/{EPOCHS}] | Loss: {total_loss/len(train_loader):.4f} | "
                       f"Test Acc: {acc:.4f} | Test Macro-F1: {macro_f1:.4f}")

    end_time = time.time()
    elapsed_minutes = (end_time - start_time) / 60
    
    logger.log(f"\n>>> BSTFNet Training Completed in {elapsed_minutes:.2f} minutes.")
    logger.log(f"Best Macro-F1: {best_macro_f1:.4f}")
    
    target_names = [idx_to_class[i] for i in range(num_classes)]
    report = classification_report(all_targets, all_preds, target_names=target_names, digits=4)
    logger.log("\nClassification Report:\n" + report)
    logger.log(f"Log successfully saved to {LOG_FILE}")

if __name__ == "__main__":
    main()