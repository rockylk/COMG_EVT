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

# 直接从你的 V13 脚本中导入数据加载管道，保证脱敏环境完全一致
from V13_all import load_dataset_v13

# ==========================================
# 1. 全局配置与路径切换
# ==========================================
# 注释掉不需要的数据集
# DATASET_PATH = r"C:\Desktop\GNN\aes-128-gcm\aes-128-gcm"  # CipherSpectrum
DATASET_PATH = r"C:\Desktop\GNN\USTC-TFC2016-Split"       # USTC-TFC2016

RESULT_DIR = r"C:\GNN\RESULT"
LOG_FILE = os.path.join(RESULT_DIR, f"CNN_LSTM_Log_{int(time.time())}.txt")

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
        self.log(f"=== CNN-LSTM Training Log ===")
        self.log(f"Dataset Path: {DATASET_PATH}")
        self.log(f"Device: {DEVICE}\n")

    def log(self, message):
        print(message)
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(message + '\n')

# ==========================================
# 3. 数据适配器 (PyG Data -> CNN-LSTM Sequence)
# ==========================================
class SeqAdapterDataset(Dataset):
    """
    将 V13 提取的图节点特征 (N x 5) 转换为定长序列，供 CNN-LSTM 使用。
    如果节点数 < 400，则在序列末尾补零。
    """
    def __init__(self, pyg_dataset, max_len=MAX_SEQ_LEN):
        self.samples = []
        self.labels = []
        
        for data in pyg_dataset:
            x = data.x.numpy()  # 获取节点特征序列 (N x 5)
            y = data.y.item()
            
            N, F = x.shape
            if N < max_len:
                # 补零 (padding)
                pad_width = ((0, max_len - N), (0, 0))
                x_padded = np.pad(x, pad_width, mode='constant', constant_values=0.0)
            else:
                x_padded = x[:max_len, :]
                
            self.samples.append(x_padded)
            self.labels.append(y)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 转换并返回 Tensor
        return torch.tensor(self.samples[idx], dtype=torch.float32), \
               torch.tensor(self.labels[idx], dtype=torch.long)

# ==========================================
# 4. CNN-LSTM 网络架构
# ==========================================
class CNN_LSTM_Baseline(nn.Module):
    def __init__(self, num_classes, in_channels=5):
        super(CNN_LSTM_Baseline, self).__init__()
        
        # 1D-CNN: 提取局部空间特征
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # LSTM: 提取全局时序依赖
        # CNN 池化两次后，Seq_len 从 400 变为 100
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=1, batch_first=True)
        
        # 分类器
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (Batch, Seq_len, Features) -> (Batch, 400, 5)
        # CNN 需要 channels 在中间: (Batch, Features, Seq_len)
        x = x.permute(0, 2, 1) 
        
        x = self.cnn(x)  # 输出: (Batch, 64, 100)
        
        # 还原回 LSTM 所需形状: (Batch, Seq_len, Features)
        x = x.permute(0, 2, 1) 
        
        lstm_out, _ = self.lstm(x) 
        
        # 提取序列最后一个时间步的隐状态
        final_feature = lstm_out[:, -1, :] 
        
        logits = self.fc(final_feature)
        return logits

# ==========================================
# 5. 训练与评估流程
# ==========================================
def main():
    logger = Logger(LOG_FILE)
    start_time = time.time()
    
    # 设置随机种子保证复现
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # 1. 挂载 V13 数据加载逻辑
    logger.log(">>> Loading dataset using V13 pipeline...")
    pyg_dataset, num_classes, idx_to_class = load_dataset_v13(DATASET_PATH)
    
    if not pyg_dataset:
        logger.log("Error: No data loaded. Check the dataset path.")
        return

    # 2. 划分数据集并转换为 CNN-LSTM 所需序列格式
    random.shuffle(pyg_dataset)
    split = int(len(pyg_dataset) * 0.8)
    
    train_adapter = SeqAdapterDataset(pyg_dataset[:split])
    test_adapter = SeqAdapterDataset(pyg_dataset[split:])
    
    train_loader = DataLoader(train_adapter, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_adapter, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. 初始化模型与优化器
    model = CNN_LSTM_Baseline(num_classes=num_classes, in_channels=5).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    logger.log(f"\n>>> Starting Training on {DEVICE}...")
    best_macro_f1 = 0.0
    
    # 4. 迭代训练
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
            
        # 验证评估
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

    # 5. 打印最终性能与运行时长
    end_time = time.time()
    elapsed_minutes = (end_time - start_time) / 60
    
    logger.log(f"\n>>> Training Completed in {elapsed_minutes:.2f} minutes.")
    logger.log(f"Best Macro-F1: {best_macro_f1:.4f}")
    
    # 输出详细分类报告
    target_names = [idx_to_class[i] for i in range(num_classes)]
    report = classification_report(all_targets, all_preds, target_names=target_names, digits=4)
    logger.log("\nClassification Report:\n" + report)
    logger.log(f"Log successfully saved to {LOG_FILE}")

if __name__ == "__main__":
    main()