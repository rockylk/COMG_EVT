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

# 严格对齐 V13 的数据管道
from V13_all import load_dataset_v13

# ==========================================
# 1. 全局配置与路径切换
# ==========================================
# 建议继续跑 CipherSpectrum，以验证我们在论文中承诺的“性能崩盘”
DATASET_PATH = r"C:\Desktop\GNN\aes-128-gcm\aes-128-gcm"  # CipherSpectrum
# DATASET_PATH = r"C:\Desktop\GNN\USTC-TFC2016-Split"       # USTC-TFC2016

RESULT_DIR = r"C:\GNN\RESULT"
LOG_FILE = os.path.join(RESULT_DIR, f"ET_BERT_Log_{int(time.time())}.txt")

BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.0005  # Transformer 往往需要稍微小一点的学习率防止梯度爆炸
MAX_SEQ_LEN = 400
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. 日志记录模块
# ==========================================
class Logger:
    def __init__(self, filepath):
        self.filepath = filepath
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        self.log(f"=== ET-BERT (Transformer Encoder) Training Log ===")
        self.log(f"Dataset Path: {DATASET_PATH}")
        self.log(f"Device: {DEVICE}\n")

    def log(self, message):
        print(message)
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(message + '\n')

# ==========================================
# 3. 数据适配器 (完全沿用，确保绝对公平)
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
# 4. ET-BERT 架构: Positional Encoding + Transformer Encoder
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 增加 batch 维度
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x 维度: (Batch, Seq_Len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x

class ET_BERT_Baseline(nn.Module):
    def __init__(self, num_classes, in_channels=5, d_model=128, nhead=4, num_layers=3):
        super(ET_BERT_Baseline, self).__init__()
        
        # 1. 初始特征映射 (类似 Embedding 层)
        self.input_projection = nn.Linear(in_channels, d_model)
        
        # 2. 类别标识符 [CLS] Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 3. 位置编码
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=MAX_SEQ_LEN + 1)
        
        # 4. BERT 核心: Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, 
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 5. 分类头 (基于 [CLS] token 的输出)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x 初始维度: (Batch, Seq_len, Features) -> (Batch, 400, 5)
        batch_size = x.size(0)
        
        # 投影到 d_model
        x = self.input_projection(x)  # (Batch, 400, 128)
        
        # 拼接 [CLS] token 到序列的最前面
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (Batch, 1, 128)
        x = torch.cat((cls_tokens, x), dim=1)  # (Batch, 401, 128)
        
        # 加入位置编码
        x = self.pos_encoder(x)
        
        # 通过 Transformer Encoder
        transformer_out = self.transformer_encoder(x)  # (Batch, 401, 128)
        
        # 提取 [CLS] token 的隐状态 (即第 0 个位置的输出) 作为全局序列表示
        cls_output = transformer_out[:, 0, :]  # (Batch, 128)
        
        # 分类
        logits = self.classifier(cls_output)
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
    
    logger.log(">>> Loading dataset using V13 pipeline for ET-BERT...")
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
    
    # 初始化 BERT 架构
    model = ET_BERT_Baseline(num_classes=num_classes, in_channels=5, 
                             d_model=128, nhead=4, num_layers=3).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    logger.log(f"\n>>> Starting ET-BERT (Transformer) Training on {DEVICE}...")
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
    
    logger.log(f"\n>>> ET-BERT Training Completed in {elapsed_minutes:.2f} minutes.")
    logger.log(f"Best Macro-F1: {best_macro_f1:.4f}")
    
    target_names = [idx_to_class[i] for i in range(num_classes)]
    report = classification_report(all_targets, all_preds, target_names=target_names, digits=4)
    logger.log("\nClassification Report:\n" + report)
    logger.log(f"Log successfully saved to {LOG_FILE}")

if __name__ == "__main__":
    main()