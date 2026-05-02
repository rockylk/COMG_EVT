import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout, Sigmoid, Conv1d, MaxPool1d
from torch_geometric.nn import GATv2Conv, AttentionalAggregation

class FingerprintCNN(torch.nn.Module):
    def __init__(self, seq_len=20, out_dim=32):
        super(FingerprintCNN, self).__init__()
        self.conv1 = Conv1d(1, 16, kernel_size=3, padding=1)
        self.bn1 = BatchNorm1d(16)
        self.pool1 = MaxPool1d(2)
        self.conv2 = Conv1d(16, 32, kernel_size=3, padding=1)
        self.bn2 = BatchNorm1d(32)
        self.pool2 = MaxPool1d(2)
        self.fc = Linear(32 * (seq_len // 4), out_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x).relu()
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x).relu()
        x = self.pool2(x)
        return self.fc(x.view(x.size(0), -1)).relu()

class EntropyCNN(torch.nn.Module):
    def __init__(self, seq_len=5, out_dim=16):
        super(EntropyCNN, self).__init__()
        self.conv1 = Conv1d(1, 8, kernel_size=3, padding=1)
        self.bn1 = BatchNorm1d(8)
        self.fc = Linear(8 * seq_len, out_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x).relu()
        return self.fc(x.view(x.size(0), -1)).relu()

class GatedFusionV13(torch.nn.Module):
    def __init__(self, graph_dim, stats_dim, finger_dim, entropy_dim):
        super(GatedFusionV13, self).__init__()
        global_ctx_dim = stats_dim + finger_dim + entropy_dim
        self.global_gate = Sequential(
            Linear(global_ctx_dim, graph_dim),
            ReLU(),
            Linear(graph_dim, graph_dim),
            Sigmoid()
        )
        self.global_proj = Linear(global_ctx_dim, 64)

    def forward(self, h_graph, h_stats, h_finger, h_entropy):
        h_global_ctx = torch.cat([h_stats, h_finger, h_entropy], dim=1)
        gate = self.global_gate(h_global_ctx)
        return torch.cat([h_graph * gate, self.global_proj(h_global_ctx)], dim=1)

def info_nce_loss(z1, z2, temperature=0.2):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    cos_sim = torch.mm(z1, z2.t()) / temperature
    labels = torch.arange(z1.shape[0]).to(z1.device)
    return F.cross_entropy(cos_sim, labels)

class ContrastiveGNN_V13(torch.nn.Module):
    def __init__(self, node_in_channels, stats_dim, finger_seq_len, entropy_seq_len, hidden_channels, out_channels):
        super(ContrastiveGNN_V13, self).__init__()
        self.conv1 = GATv2Conv(node_in_channels, hidden_channels, heads=4, edge_dim=1, concat=True)
        self.bn1 = BatchNorm1d(hidden_channels * 4)

        self.conv2 = GATv2Conv(hidden_channels * 4, hidden_channels, heads=4, edge_dim=1, concat=True)
        self.bn2 = BatchNorm1d(hidden_channels * 4)
        self.lin_skip2 = Linear(hidden_channels * 4, hidden_channels * 4)

        self.conv3 = GATv2Conv(hidden_channels * 4, hidden_channels, heads=4, edge_dim=1, concat=True)
        self.bn3 = BatchNorm1d(hidden_channels * 4)
        self.lin_skip3 = Linear(hidden_channels * 4, hidden_channels * 4)

        self.pool = AttentionalAggregation(gate_nn=Linear(hidden_channels * 4, 1))

        self.graph_dim = hidden_channels * 4
        self.finger_cnn = FingerprintCNN(seq_len=finger_seq_len, out_dim=32)
        self.entropy_cnn = EntropyCNN(seq_len=entropy_seq_len, out_dim=16)
        self.fusion_layer = GatedFusionV13(self.graph_dim, stats_dim, 32, 16)

        self.projection_head = Sequential(
            Linear(self.graph_dim, 256),
            ReLU(),
            Linear(256, 128)
        )

        self.final_dim = self.graph_dim + 64
        self.classifier_head = Sequential(
            Linear(self.final_dim, 128),
            ReLU(),
            Dropout(0.5),
            Linear(128, out_channels)
        )

    def encode_graph(self, x, edge_index, edge_attr, batch):
        x = self.bn1(self.conv1(x, edge_index, edge_attr=edge_attr)).relu()
        x = self.bn2(self.conv2(x, edge_index, edge_attr=edge_attr)).relu() + self.lin_skip2(x)
        x = self.bn3(self.conv3(x, edge_index, edge_attr=edge_attr)).relu() + self.lin_skip3(x)
        return self.pool(x, batch)

    def forward_contrastive(self, x, edge_index, edge_attr, batch):
        return self.projection_head(self.encode_graph(x, edge_index, edge_attr, batch))

    def extract_fused_features(self, x, edge_index, edge_attr, batch, stats_attr, finger_attr, entropy_attr):
        h_graph = self.encode_graph(x, edge_index, edge_attr, batch)
        h_finger = self.finger_cnn(finger_attr)
        h_entropy = self.entropy_cnn(entropy_attr)
        return self.fusion_layer(h_graph, stats_attr, h_finger, h_entropy)

    def forward_classifier(self, x, edge_index, edge_attr, batch, stats_attr, finger_attr, entropy_attr):
        h_fused = self.extract_fused_features(x, edge_index, edge_attr, batch, stats_attr, finger_attr, entropy_attr)
        return self.classifier_head(h_fused)