import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score, accuracy_score
import warnings
warnings.filterwarnings("ignore")

# 同样直接调用你 V13 的数据加载管道，保证特征严格对齐
from V13_all import load_dataset_v13

# ==========================================
# 1. 全局配置与路径切换
# ==========================================
# 建议继续保持 CipherSpectrum，方便横向对比
DATASET_PATH = r"C:\Desktop\GNN\aes-128-gcm\aes-128-gcm"  # CipherSpectrum
# DATASET_PATH = r"C:\Desktop\GNN\USTC-TFC2016-Split"       # USTC-TFC2016

RESULT_DIR = r"C:\GNN\RESULT"
LOG_FILE = os.path.join(RESULT_DIR, f"Attn_ETC_Log_{int(time.time())}.txt")

BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
MAX_SEQ_LEN = 400  # 对齐 V13 中的 max_nodes
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. 日志记录模块
# ==========================================
class Logger:
    def __init__(self, filepath):
        self.filepath = filepath
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        self.log(f"=== Attn-ETC Training Log ===")
        self.log(f"Dataset Path: {DATASET_PATH}")
        self.log(f"Device: {DEVICE}\n")

    def log(self, message):
        print(message)
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(message + '\n')

# ==========================================
# 3. 数据适配器 (完全沿用)
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
# 4. Attn-ETC 架构: CNN + BiGRU + Attention
# ==========================================
class SelfAttention(nn.Module):
    """
    经典的时间步级自注意力机制 (Temporal Self-Attention)
    """
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        # 计算注意力分数的投影网络
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False)
        )
        
    def forward(self, x):
        # x 维度: (Batch, Seq_Len, Hidden)
        attn_weights = self.attention(x)  # (Batch, Seq_Len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)  # 对 Seq_Len 维度进行归一化
        
        # 将注意力权重施加到原始特征上
        weighted_x = x * attn_weights  # (Batch, Seq_Len, Hidden)
        
        # 沿时间步聚合，得到定长的序列向量表示
        context_vector = weighted_x.sum(dim=1)  # (Batch, Hidden)
        return context_vector, attn_weights

class Attn_ETC_Baseline(nn.Module):
    def __init__(self, num_classes, in_channels=5):
        super(Attn_ETC_Baseline, self).__init__()
        
        # 1. 局部特征提取层 (CNN)
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # 2. 时序上下文层 (BiGRU)
        self.bigru = nn.GRU(
            input_size=64, hidden_size=64, 
            num_layers=1, batch_first=True, bidirectional=True
        )
        # 双向 GRU 输出维度为 64 * 2 = 128
        
        # 3. 注意力融合层
        self.attention_layer = SelfAttention(hidden_dim=128)
        
        # 4. 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x 初始维度: (Batch, Seq_len, Features) -> (Batch, 400, 5)
        
        # 转换给 CNN: (Batch, Features, Seq_len)
        x = x.permute(0, 2, 1)
        x = self.feature_extractor(x)
        
        # 转换给 BiGRU: (Batch, Seq_len, Features)
        x = x.permute(0, 2, 1)
        gru_out, _ = self.bigru(x)
        
        # 通过注意力机制聚合整个序列的特征
        # context_vector: (Batch, 128)
        context_vector, _ = self.attention_layer(gru_out)
        
        logits = self.classifier(context_vector)
        return logits

# ==========================================
# 5. 训练与评估流程 (自动落盘)
# ==========================================
def main():
    logger = Logger(LOG_FILE)
    start_time = time.time()
    
    # 强制控制随机变量，确保横向对比的绝对公平
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    logger.log(">>> Loading dataset using V13 pipeline for Attn-ETC...")
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
    
    model = Attn_ETC_Baseline(num_classes=num_classes, in_channels=5).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    logger.log(f"\n>>> Starting Attention-based Training on {DEVICE}...")
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
    
    logger.log(f"\n>>> Attn-ETC Training Completed in {elapsed_minutes:.2f} minutes.")
    logger.log(f"Best Macro-F1: {best_macro_f1:.4f}")
    
    target_names = [idx_to_class[i] for i in range(num_classes)]
    report = classification_report(all_targets, all_preds, target_names=target_names, digits=4)
    logger.log("\nClassification Report:\n" + report)
    logger.log(f"Log successfully saved to {LOG_FILE}")

if __name__ == "__main__":
    main()