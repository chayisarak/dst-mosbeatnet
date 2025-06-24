import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import config_new  as config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # --------------------------------------------
# # Mosbeatnet Model
# # --------------------------------------------
# class AdditiveAttention(nn.Module):
#     def __init__(self, dim):
#         super(AdditiveAttention, self).__init__()
#         self.W_q = nn.Linear(dim, dim)
#         self.W_k = nn.Linear(dim, dim)
#         self.V = nn.Linear(dim, 1)

#     def forward(self, query, value):
#         q_proj = self.W_q(query)  # [batch_size, seq_len, dim]
#         k_proj = self.W_k(value)  # [batch_size, seq_len, dim]
#         scores = self.V(torch.tanh(q_proj + k_proj)).squeeze(-1)  # [batch_size, seq_len]
#         attn_weights = F.softmax(scores, dim=-1)  # [batch_size, seq_len]
#         context = attn_weights.unsqueeze(-1) * value  # [batch_size, seq_len, dim]
#         return context, attn_weights

# class Mosbeatnet(nn.Module):
#     def __init__(self, n_timesteps, n_outputs):
#         super(Mosbeatnet, self).__init__()
#         self.n_timesteps = n_timesteps  
#         self.num_segments = 20 # 0.5s * 8000
#         self.n_outputs = n_outputs
#         self.total_length = n_timesteps * self.num_segments  # 40000

#         # Conv1D layers
#         self.conv1 = nn.Conv1d(1, 32, kernel_size=10, stride=5, padding=50)
#         self.norm1 = nn.InstanceNorm1d(32)
#         self.pool1 = nn.MaxPool1d(kernel_size=3, stride=3)
#         self.conv2 = nn.Conv1d(32, 32, kernel_size=5, dilation=2, padding=32)
#         self.norm2 = nn.InstanceNorm1d(32)
#         self.pool2 = nn.MaxPool1d(kernel_size=3, stride=3)
#         self.conv3 = nn.Conv1d(32, 64, kernel_size=5, dilation=2, padding=25)
#         self.norm3 = nn.InstanceNorm1d(64)
#         self.pool3 = nn.MaxPool1d(kernel_size=3, stride=3)

#         # คำนวณ output size หลัง Conv1D
#         self._compute_output_size()

#         # LSTM layers
#         self.lstm1 = nn.LSTM(64, 64, batch_first=True, bidirectional=True, dropout=0.3)  # ลบ dropout
#         # self.lstm2 = nn.LSTM(128, 64, batch_first=True, bidirectional=True, dropout=0.3)  # ลบ dropout

#         # Attention
#         self.attn = AdditiveAttention(128)
#         self.layer_norm = nn.LayerNorm(128)

#         # Feature Selection
#         self.feature_conv = nn.Conv1d(128, 128, kernel_size=1)

#         # Fully Connected
#         self.fc1 = nn.Linear(128, 256)
#         self.dropout = nn.Dropout(0.3)
#         self.output_layer = nn.Linear(256, n_outputs)

#     def _compute_output_size(self):
       
#         x = torch.zeros(1, 1, self.total_length)  
#         x = self.pool1(F.leaky_relu(self.norm1(self.conv1(x))))
#         x = self.pool2(F.leaky_relu(self.norm2(self.conv2(x))))
#         x = self.pool3(F.leaky_relu(self.norm3(self.conv3(x))))
#         self.conv_output_length = x.shape[2]  

#     def forward(self, x):
#         batch_size, num_segments, channels, segment_length = x.shape  # (32, 20, 1, 2000)
#         x = x.view(batch_size, channels, num_segments * segment_length)  # (32, 1, 40000)

#         # Conv1D
#         x = self.pool1(F.leaky_relu(self.norm1(self.conv1(x))))  # (32, 32, N1)
#         x = self.pool2(F.leaky_relu(self.norm2(self.conv2(x))))  # (32, 32, N2)
#         x = self.pool3(F.leaky_relu(self.norm3(self.conv3(x))))  # (32, 64, N3) เช่น (32, 64, 148)

#         # Reshape for LSTM
#         x = x.permute(0, 2, 1)  # (32, N3, 64)
#         x, _ = self.lstm1(x)  # (32, N3, 128)
#         # x, _ = self.lstm2(x)  # (32, N3, 128)

#         # Attention
#         context, attn_weights = self.attn(x, x)  # (32, N3, 128)
#         x = self.layer_norm(x + context)  # (32, N3, 128)

#         # Feature Selection
#         x = x.permute(0, 2, 1)  # (32, 128, N3)
#         x = self.feature_conv(x)  # (32, 128, N3)

#         # แบ่งตาม num_segments
#         segment_length_after_conv = self.conv_output_length // self.num_segments  
#         if self.conv_output_length % self.num_segments != 0:
            
#             x = x[:, :, :segment_length_after_conv * self.num_segments]
#         x = x.view(batch_size, 128, self.num_segments, segment_length_after_conv)  
#         x = x.mean(dim=3)  

#         # Fully Connected
#         x = x.permute(0, 2, 1)  # (32, 20, 128)
#         x = F.relu(self.fc1(x))  # (32, 20, 256)
#         x = self.dropout(x)
#         x = self.output_layer(x)  # (32, 20, 5)

#         return F.log_softmax(x, dim=-1)




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

# from torchinfo import summary
# n_timesteps = 2000  # segment_length
# num_segments = 20
# n_outputs = 5
# model = Mosbeatnet(n_timesteps, n_outputs) ##
# # model = MosqPlusModel(n_timesteps, n_outputs)
# summary(model, input_size=(32, 20, 1, 2000))


# --------------------------------------------
# Proposed Model : Mosbeatnet 
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


        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # LSTM layers
        self.lstm1 = nn.LSTM(64, 64, batch_first=True, bidirectional=True, dropout=0.3)  # ลบ dropout
        self.lstm2 = nn.LSTM(128, 64, batch_first=True, bidirectional=True, dropout=0.3)  # ลบ dropout

        # Attention
        self.attn = AdditiveAttention(128)
        self.layer_norm = nn.LayerNorm(128)

        # Feature Selection
        self.feature_conv = nn.Conv1d(128, 128, kernel_size=1)

        # Fully Connected
        self.fc1 = nn.Linear(128, 256)
        self.dropout = nn.Dropout(0.3)
        self.output_layer = nn.Linear(256, n_outputs)

    def forward(self, x):
        batch_size, num_segments, channels, segment_length = x.shape  # (B, T, 1, 2000)
        
        # Reshape to merge batch and segment
        x = x.view(batch_size * num_segments, channels, segment_length)  # (B*T, 1, 2000)

        # Conv1D stack
        x = self.pool1(F.leaky_relu(self.norm1(self.conv1(x))))
        x = self.pool2(F.leaky_relu(self.norm2(self.conv2(x))))
        x = self.pool3(F.leaky_relu(self.norm3(self.conv3(x))))  # (B*T, 64, N)

        # Global Average Pooling
        x = self.global_pool(x).squeeze(-1)  # (B*T, 64)

        # Reshape back to sequence
        x = x.view(batch_size, num_segments, 64)  # (B, T, 64)

        # LSTM
        x, _ = self.lstm1(x)  # (B, T, 128)
        x, _ = self.lstm2(x)

        # Attention
        context, attn_weights = self.attn(x, x)  # (B, T, 128)
        x = self.layer_norm(x + context)

        # Feature Selection
        x = x.permute(0, 2, 1)  # (B, 128, T)
        x = self.feature_conv(x)  # (B, 128, T)
        x = x.permute(0, 2, 1)  # (B, T, 128)

        # FC
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.output_layer(x)  # (B, T, n_outputs)

        return x


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchaudio
# from torchinfo import summary


# class CNN2DBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dropout_rate, maxpool_kernel_size):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)  # if padding = 'same', stride must be 1
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.maxpool = nn.MaxPool2d(maxpool_kernel_size)
#         self.dropout = nn.Dropout(dropout_rate)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn(x)
#         x = F.relu(x)
#         x = self.maxpool(x)
#         x = self.dropout(x)
#         return x


# class RealSEDNet(nn.Module):
#     def __init__(self, sr, hop_len, input_duration, n_classes, train_dataloader=None):
#         super().__init__()
#         segment_length = int(sr * input_duration / hop_len)
#         n_fft = min(max(64, hop_len * 2), segment_length)
#         dropout_rate = 0.2
#         print(f"Input tensor shape before MelSpectrogram: {x.shape}")
#         self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
#             sr, n_fft=n_fft, hop_length=hop_len, n_mels=40, center=False, power=1
#         )

#         self._cnn1 = CNN2DBlock(1, 128, (3, 3), 1, 'same', dropout_rate, (5, 1))
#         self._cnn2 = CNN2DBlock(128, 128, (3, 3), 1, 'same', dropout_rate, (2, 1))
#         self._cnn3 = CNN2DBlock(128, 128, (3, 3), 1, 'same', dropout_rate, (2, 1))

#         self.cnn = nn.Sequential(self._cnn1, self._cnn2, self._cnn3)
#         self.rnn = nn.GRU(256, 32, num_layers=2, batch_first=True, dropout=dropout_rate, bidirectional=True)
#         self.linear1 = nn.Linear(64, n_classes)

#         if train_dataloader is not None:
#             x_mean, x_std = self.compute_normalization_factor(train_dataloader)
#             self.x_mean = nn.parameter.Parameter(x_mean, requires_grad=False)
#             self.x_std = nn.parameter.Parameter(x_std, requires_grad=False)
#         else:
#             print("init without dataset, set mean and std to 0 and 1 respectively")
#             self.x_mean = nn.parameter.Parameter(torch.tensor(0.), requires_grad=False)
#             self.x_std = nn.parameter.Parameter(torch.tensor(1.), requires_grad=False)

#     def compute_normalization_factor(self, dataloader):
#         with torch.no_grad():
#             x_sum = 0.
#             n = 0
#             for x, y in dataloader:
#                 x = x.to(device)
#                 if x.dim() == 4:
#                     batch_size, segments, channels, segment_length = x.shape
#                     x = x.view(batch_size * segments, channels, segment_length)

#                 x = torch.permute(x, (0, 2, 1))
#                 x = self.mel_spectrogram(x)
#                 x_sum += x.sum(dim=(0, 1, 3))
#                 n += x.shape[0] * x.shape[1] * x.shape[3]
#             x_mean = x_sum / n
#             x_std = 0.
#             for x, y in dataloader:
#                 x = x.to(device)
#                 if x.dim() == 4:
#                     batch_size, segments, channels, segment_length = x.shape
#                     x = x.view(batch_size * segments, channels, segment_length)
#                 x = torch.permute(x, (0, 2, 1))
#                 x = self.mel_spectrogram(x)
#                 x_std += ((x.transpose(2, 3) - x_mean)**2).sum(dim=(0, 1, 2))
#             x_std = torch.sqrt(x_std / n)
#         return x_mean, x_std

#     def forward(self, x):
#         # x: (B, T, C, L) → reshape to (B*T, C, L)
#         B, T, C, L = x.shape
#         x = x.view(B * T, C, L)                  # (BT, C, L)
#         x = self.mel_spectrogram(x)              # (BT, 1, n_mels, time)
#         x = ((x.transpose(2, 3) - self.x_mean) / self.x_std).transpose(2, 3)
#         x = self.cnn(x)                           # (BT, C, mel', time)
#         b, c, pnmel, t = x.shape
#         x = x.view(b, c * pnmel, t)              # collapse c & nmel → (BT, feature_dim, time)
#         x = x.permute(0, 2, 1)                   # (BT, time, feature_dim)
#         x, _ = self.rnn(x)                        # GRU → (BT, time, 64)
#         x = self.linear1(x)                       # (BT, time, n_classes)
#         x = x.view(B, T, -1, x.shape[-1])         # reshape back: (B, T, ?, n_classes)
#         x = x.mean(dim=2)                         # remove mel-time axis (avg pooling) → (B, T, n_classes)
#         return x





# from torchinfo import summary
# n_timesteps = 4000  # segment_length
# num_segments = 20
# n_outputs = 5
# model = Mosbeatnet(n_timesteps, n_outputs) ##
# # model = MosqPlusModel(n_timesteps, n_outputs)
# model = RealSEDNet(sr=config.SAMPLING_RATE,hop_len=config.HOP,input_duration=config.AUDIO_DURATION,n_classes=5).to(device)
# summary(model, input_size=(32, 20, 1, 4000))
# # summary(model, input_size=(batch_size, num_segments, channels, n_timesteps))



# class CNN2DBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dropout_rate, maxpool_kernel_size):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding) # if padding = 'same', stride must be 1
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.maxpool = nn.MaxPool2d(maxpool_kernel_size)
#         self.dropout = nn.Dropout(dropout_rate)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn(x)
#         x = F.relu(x)
#         x = self.maxpool(x)
#         x = self.dropout(x)
#         return x

# class RealSEDNet(nn.Module):
#     '''
#     based on https://github.com/sharathadavanne/sed-crnn/blob/master/sed.py
#     '''
#     def __init__(self, sr, hop_len, input_duration, n_classes, train_dataloader):
#         super().__init__()
#         dropout_rate = 0.2
#         self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(sr, hop_len, hop_length=hop_len, n_mels=40, center=False, power=1)
#         self._cnn1 = CNN2DBlock(1, 128, (3, 3), 1, 'same', dropout_rate, (5, 1))
#         self._cnn2 = CNN2DBlock(128, 128, (3, 3), 1, 'same', dropout_rate, (2, 1))
#         self._cnn3 = CNN2DBlock(128, 128, (3, 3), 1, 'same', dropout_rate, (2, 1))
#         self.cnn = nn.Sequential(self._cnn1, self._cnn2, self._cnn3)
#         self.rnn = nn.GRU(256, 32, num_layers=2, batch_first=True, dropout=dropout_rate, bidirectional=True)
#         self.linear1 = nn.Linear(64, n_classes)
        
#         if train_dataloader is not None:
#             x_mean, x_std = self.compute_normalization_factor(train_dataloader)
#             self.x_mean = nn.parameter.Parameter(x_mean, requires_grad=False)
#             self.x_std = nn.parameter.Parameter(x_std, requires_grad=False)
#         else:
#             print("init without dataset, set mean and std to 0 and 1 respectively")
#             self.x_mean = nn.parameter.Parameter(torch.tensor(0), requires_grad=False)
#             self.x_std = nn.parameter.Parameter(torch.tensor(1), requires_grad=False)

#     def compute_normalization_factor(self, dataloader):
#         with torch.no_grad():
#             x_sum = 0.
#             n = 0
#             for x, y in dataloader:
#                 x = torch.permute(x, (0, 2, 1))
#                 x = self.mel_spectrogram(x)
#                 x_sum += x.sum(dim=(0, 1, 3))
#                 n += x.shape[0] * x.shape[1] * x.shape[3]
#             x_mean = x_sum / n
#             x_std = 0.
#             for x, y in dataloader:
#                 x = torch.permute(x, (0, 2, 1))
#                 x = self.mel_spectrogram(x)
#                 x_std += ((x.transpose(2, 3) - x_mean)**2).sum(dim=(0, 1, 2))
#             x_std = x_std / n
#         return x_mean, x_std

#     def forward(self, x):
#         x = torch.permute(x, (0, 2, 1)) # x should be (b, c, sr*t)
#         x = self.mel_spectrogram(x)     # x should be (b, c, nmel, t)
#         x = ((x.transpose(2, 3) - self.x_mean) / (self.x_std)).transpose(2, 3)
#         x = self.cnn(x)                 # x should be (b, c, processed nmel, t)
#         b, c, pnmel, t = x.shape
#         x = x.view(b, c*pnmel, t)            # collapse c and processed nmel
#         x = torch.permute(x, (0, 2, 1)) # x should be (b, t, c*pnmel)
#         x, h = self.rnn(x)
#         x = self.linear1(x)
#         return x


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchinfo import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

class CNN2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout_rate, maxpool_kernel_size):
        super().__init__()
        padding = (kernel_size[0] // 2, kernel_size[1] // 2) if isinstance(kernel_size, tuple) else kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d(maxpool_kernel_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        return x


class RealSEDNet(nn.Module):
    def __init__(self, sr, hop_len, input_duration, n_classes, train_dataloader=None):
        super().__init__()

        segment_duration = 0.5  # seconds
        segment_length = int(sr * segment_duration)  # e.g., 8000 * 0.5 = 4000
        n_fft = min(max(64, hop_len * 2), segment_length)
        dropout_rate = 0.2

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_len,
            n_mels=40,
            center=False,
            power=1.0
        )

        self._cnn1 = CNN2DBlock(1, 128, (3, 3), 1, dropout_rate, (5, 1))
        self._cnn2 = CNN2DBlock(128, 128, (3, 3), 1, dropout_rate, (2, 1))
        self._cnn3 = CNN2DBlock(128, 128, (3, 3), 1, dropout_rate, (2, 1))
        self.cnn = nn.Sequential(self._cnn1, self._cnn2, self._cnn3)

        self.rnn = nn.GRU(
            input_size=256,
            hidden_size=32,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=True
        )

        self.linear1 = nn.Linear(64, n_classes)

        # Normalization factor
        if train_dataloader is not None:
            x_mean, x_std = self.compute_normalization_factor(train_dataloader)
            self.x_mean = nn.Parameter(x_mean, requires_grad=False)
            self.x_std = nn.Parameter(x_std, requires_grad=False)
        else:
            print("init without dataset, set mean and std to 0 and 1 respectively")
            self.x_mean = nn.Parameter(torch.tensor(0.), requires_grad=False)
            self.x_std = nn.Parameter(torch.tensor(1.), requires_grad=False)

    def compute_normalization_factor(self, dataloader):
        with torch.no_grad():
            x_mean_sum = 0.
            x_std_sum = 0.
            count = 0

            for x, _ in dataloader:
                x = x.to(next(self.parameters()).device)
                if x.dim() == 4:
                    B, T, C, L = x.shape
                    x = x.view(B * T, C, L)

                x = x.squeeze(1) if x.shape[1] == 1 else x  # (BT, L)
                x = self.mel_spectrogram(x)  # (BT, n_mels, time)
                
                # Recalculate n_mels dynamically
                current_n_mels = x.shape[1]

                # Mean & std across batch and time
                x_mean_batch = x.mean(dim=(0, 2))  # (n_mels,)
                x_std_batch = x.std(dim=(0, 2))    # (n_mels,)

                x_mean_sum += x_mean_batch
                x_std_sum += x_std_batch
                count += 1

            x_mean = x_mean_sum / count
            x_std = x_std_sum / count
            return x_mean, x_std


    def forward(self, x):
        # x: (B, T, C, L)
        B, T, C, L = x.shape
        x = x.view(B * T, C, L)               # (BT, 1, L)
        x = x.squeeze(1)                      # (BT, L)
        x = self.mel_spectrogram(x)          # (BT, n_mels, time)
        x = ((x - self.x_mean.unsqueeze(0).unsqueeze(-1)) / self.x_std.unsqueeze(0).unsqueeze(-1))  # normalize
        x = x.unsqueeze(1)                   # (BT, 1, n_mels, time) for CNN

        x = self.cnn(x)                      # (BT, C, F, T')
        BxT, Cc, F, Tt = x.shape
        x = x.view(BxT, Cc * F, Tt)          # (BT, feature_dim, time)
        x = x.permute(0, 2, 1)               # (BT, time, feature_dim)

        x, _ = self.rnn(x)                   # (BT, time, 64)
        x = self.linear1(x)                  # (BT, time, n_classes)
        x = x.view(B, T, -1, x.shape[-1])    # (B, T, ?, n_classes)
        x = x.mean(dim=2)                    # (B, T, n_classes)
        return x




import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

class SEDNetSegmentLevel(nn.Module):
    def __init__(self, sr=8000, hop_len=512, n_classes=5, dropout_rate=0.3):
        super().__init__()

        self.hop_len = hop_len
        self.n_classes = n_classes

        segment_duration = 0.5  # seconds
        segment_length = int(sr * segment_duration)
        n_fft = min(max(64, hop_len * 2), segment_length)

        # Mel Spectrogram Transform
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_len,
            n_mels=40,
            center=False,
            power=1.0
        )

        self.spec_bn = nn.BatchNorm2d(1)  # (B*T, 1, mel, time)

        # CNN Blocks (as in SEDNet)
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),  # Reduce time dim only
            nn.Dropout(dropout_rate)
        )

        self.cnn2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(dropout_rate)
        )

        # BiGRU
        self.rnn = nn.GRU(
            input_size=64 * 40,  # mel=40 not reduced → 64x40
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0  # won't apply if num_layers=1
        )

        # Fully Connected
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        """
        x: (B, T, 1, L) — waveform segments
        return: (B, T, n_classes)
        """
        B, T, C, L = x.shape
        x = x.view(B * T, C, L)                      # (B*T, 1, L)
        x = self.mel_spectrogram(x)                  # (B*T, mel, time)

        if x.dim() == 3:
            x = x.unsqueeze(1)                       # (B*T, 1, mel, time)

        x = self.spec_bn(x)                          # (B*T, 1, mel, time)
        x = self.cnn1(x)                             # (B*T, 64, mel, time//2)
        x = self.cnn2(x)                             # (B*T, 64, mel, time//4)

        BxT, Cc, F, Tt = x.shape
        x = x.permute(0, 3, 1, 2)                    # (B*T, T', C, F)
        x = x.reshape(BxT, Tt, -1)                   # (B*T, T', C*F = 64*40)

        x, _ = self.rnn(x)                           # (B*T, T', 128)
        x = self.fc1(x)                              # (B*T, T', 128)
        x = self.dropout(x)
        x = self.fc2(x)                              # (B*T, T', n_classes)

        x = x.mean(dim=1)                            # Average over frame → (B*T, n_classes)
        x = x.view(B, T, self.n_classes)             # Reshape → (B, T, n_classes)
        return x





# สมมุติว่า config กำหนดค่าพวกนี้ไว้
sr = 8000
hop_len = 512
input_duration = 10.0  # วินาที
n_classes = 5

# model = RealSEDNet(sr=sr, hop_len=hop_len, input_duration=input_duration, n_classes=n_classes).to(device)

model = SEDNetSegmentLevel(sr=sr, hop_len=hop_len, n_classes=n_classes).to(device)


summary(model, input_size=(32, 20, 1, 4000))  # Batch=32, 20 segments, mono waveform 4000 samples
