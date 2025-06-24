import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import config_new  as config


# --------------------------------------------
# Mosbeatnet Model
# --------------------------------------------
class AdditiveAttention(nn.Module):
    def __init__(self, dim):
        super(AdditiveAttention, self).__init__()
        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.V = nn.Linear(dim, 1)

    def forward(self, query, value):
        q_proj = self.W_q(query)  # [batch_size, seq_len, dim]
        k_proj = self.W_k(value)  # [batch_size, seq_len, dim]
        scores = self.V(torch.tanh(q_proj + k_proj)).squeeze(-1)  # [batch_size, seq_len]
        attn_weights = F.softmax(scores, dim=-1)  # [batch_size, seq_len]
        context = attn_weights.unsqueeze(-1) * value  # [batch_size, seq_len, dim]
        return context, attn_weights

class Mosbeatnet(nn.Module):
    def __init__(self, n_timesteps, n_outputs):
        super(Mosbeatnet, self).__init__()
        self.n_timesteps = n_timesteps  
        self.num_segments = 20 # 0.5s * 8000
        self.n_outputs = n_outputs
        self.total_length = n_timesteps * self.num_segments  # 40000

        # Conv1D layers
        self.conv1 = nn.Conv1d(1, 32, kernel_size=10, stride=5, padding=50)
        self.norm1 = nn.InstanceNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=5, dilation=2, padding=32)
        self.norm2 = nn.InstanceNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, dilation=2, padding=25)
        self.norm3 = nn.InstanceNorm1d(64)
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=3)

        # คำนวณ output size หลัง Conv1D
        self._compute_output_size()

        # LSTM layers
        self.lstm1 = nn.LSTM(64, 64, batch_first=True, bidirectional=True, dropout=0.3)  # ลบ dropout
        # self.lstm2 = nn.LSTM(128, 64, batch_first=True, bidirectional=True, dropout=0.3)  # ลบ dropout

        # Attention
        self.attn = AdditiveAttention(128)
        self.layer_norm = nn.LayerNorm(128)

        # Feature Selection
        self.feature_conv = nn.Conv1d(128, 128, kernel_size=1)

        # Fully Connected
        self.fc1 = nn.Linear(128, 256)
        self.dropout = nn.Dropout(0.3)
        self.output_layer = nn.Linear(256, n_outputs)

    def _compute_output_size(self):
       
        x = torch.zeros(1, 1, self.total_length)  
        x = self.pool1(F.leaky_relu(self.norm1(self.conv1(x))))
        x = self.pool2(F.leaky_relu(self.norm2(self.conv2(x))))
        x = self.pool3(F.leaky_relu(self.norm3(self.conv3(x))))
        self.conv_output_length = x.shape[2]  

    def forward(self, x):
        batch_size, num_segments, channels, segment_length = x.shape  # (32, 20, 1, 2000)
        x = x.view(batch_size, channels, num_segments * segment_length)  # (32, 1, 40000)

        # Conv1D
        x = self.pool1(F.leaky_relu(self.norm1(self.conv1(x))))  # (32, 32, N1)
        x = self.pool2(F.leaky_relu(self.norm2(self.conv2(x))))  # (32, 32, N2)
        x = self.pool3(F.leaky_relu(self.norm3(self.conv3(x))))  # (32, 64, N3) เช่น (32, 64, 148)

        # Reshape for LSTM
        x = x.permute(0, 2, 1)  # (32, N3, 64)
        x, _ = self.lstm1(x)  # (32, N3, 128)
        # x, _ = self.lstm2(x)  # (32, N3, 128)

        # Attention
        context, attn_weights = self.attn(x, x)  # (32, N3, 128)
        x = self.layer_norm(x + context)  # (32, N3, 128)

        # Feature Selection
        x = x.permute(0, 2, 1)  # (32, 128, N3)
        x = self.feature_conv(x)  # (32, 128, N3)

        # แบ่งตาม num_segments
        segment_length_after_conv = self.conv_output_length // self.num_segments  
        if self.conv_output_length % self.num_segments != 0:
            
            x = x[:, :, :segment_length_after_conv * self.num_segments]
        x = x.view(batch_size, 128, self.num_segments, segment_length_after_conv)  
        x = x.mean(dim=3)  

        # Fully Connected
        x = x.permute(0, 2, 1)  # (32, 20, 128)
        x = F.relu(self.fc1(x))  # (32, 20, 256)
        x = self.dropout(x)
        x = self.output_layer(x)  # (32, 20, 5)

        return F.log_softmax(x, dim=-1)

# --------------------------------------------
# MosquitoSong+ Model
# --------------------------------------------
class MosqPlusModel(nn.Module):
    def __init__(self, n_timesteps, n_outputs):
        super(MosqPlusModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=100, stride=4)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=64, stride=4)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=64, stride=3)
        self.pool = nn.MaxPool1d(kernel_size=3)
        x = torch.zeros(1, 1, n_timesteps)
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        flat_size = x.numel()
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(flat_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, n_outputs)

    def forward(self, x):
        batch_size, seq_len, _, _ = x.size()
        x = x.view(batch_size * seq_len, 1, -1)
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(batch_size, seq_len, -1)
        return x

from torchinfo import summary
n_timesteps = 2000  # segment_length
num_segments = 20
n_outputs = 5
model = Mosbeatnet(n_timesteps, n_outputs) ##
# model = MosqPlusModel(n_timesteps, n_outputs)
summary(model, input_size=(32, 20, 1, 2000))