# import os
# import torch
# from model import Mosbeatnet
# from torch.utils.data import DataLoader
# from Dataloader import AudioSequenceDataset, train_model, evaluate_model, get_consistent_label_map
# from evaluate import  get_model_predictions, evaluate_segment_level, evaluate_event_level,save_segment_predictions_csv
# from model import Mosbeatnet
# import os
# import torch
# import pandas as pd
# import config_new as config
# from simulation import process_simulation
# from Dataloader import AudioSequenceDataset, train_model, evaluate_model, get_consistent_label_map,load_model
# from evaluate import  get_model_predictions, evaluate_segment_level, evaluate_event_level
# from model import Mosbeatnet,MosqPlusModel
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader

# # 🔹 ตั้งค่าตัวแปร
# MODEL_NAME = "Mosbeatnet_final"  # ชื่อโมเดล
# MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output", "models", MODEL_NAME))
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# criterion = nn.CrossEntropyLoss()
# # optimizer = optim.Adam(model.parameters(), lr=0.0001)

# # # Define metadata file paths
# train_csv = os.path.join(config.METADATA_DIR, "simulation_metadata_train.csv")
# val_csv = os.path.join(config.METADATA_DIR, "simulation_metadata_val.csv")
# test_csv = os.path.join(config.METADATA_DIR, "simulation_metadata_test.csv")
# simulation_dir = config.SIMULATED_DIR

# label_map = get_consistent_label_map(train_csv, val_csv, test_csv)




# # 🔹 ตรวจสอบว่า directory มีอยู่จริง
# if not os.path.exists(MODEL_DIR):
#     raise FileNotFoundError(f"⚠️ Model directory not found: {MODEL_DIR}")

# # 🔹 หาไฟล์ที่ขึ้นต้นด้วย "Mosbeatnet_final_round" และลงท้ายด้วย ".pth"
# model_files = [f for f in os.listdir(MODEL_DIR) if f.startswith("Mosbeatnet_final_round") and f.endswith(".pth")]

# # 🔹 ตรวจสอบว่าไฟล์มีอยู่หรือไม่
# if not model_files:
#     raise FileNotFoundError(f"⚠️ No model files found in {MODEL_DIR}")

# # 🔹 หาไฟล์ที่มีรอบล่าสุด
# try:
#     latest_model = max(model_files, key=lambda x: int(x.split("_round")[-1].split(".pth")[0]))
# except ValueError:
#     raise ValueError(f"⚠️ Model filenames are not formatted correctly in {MODEL_DIR}")

# # 🔹 กำหนด path ของไฟล์โมเดล
# MODEL_PATH = os.path.join(MODEL_DIR, latest_model)

# # 🔹 แสดง path ของไฟล์โมเดล
# print(f"📂 Model directory: {MODEL_DIR}")
# print(f"📄 Model files found: {model_files}")
# print(f"✅ Latest model selected: {latest_model}")
# print(f"🔍 Model path: {MODEL_PATH}")

# # 🔹 โหลด checkpoint
# checkpoint = torch.load(MODEL_PATH, map_location="cpu")

# # 🔹 ตรวจสอบว่ามี model_state หรือไม่
# if "model_state" not in checkpoint:
#     raise KeyError(f"⚠️ 'model_state' not found in checkpoint {MODEL_PATH}")



# # 🔹 ตั้งค่าตัวแปรที่ใช้สร้างโมเดล
# SEGMENT_LENGTH = 20  # ตัวอย่างค่าจำนวน timestep
# NUM_CLASSES = 5       # จำนวนคลาสของ output

# # 🔹 โหลดโมเดลและ state_dict
# model = Mosbeatnet(n_timesteps=SEGMENT_LENGTH, n_outputs=NUM_CLASSES).to(DEVICE)
# model.load_state_dict(checkpoint["model_state"])

# # 🔹 ส่งโมเดลไปยัง device (GPU หรือ CPU)
# model.to(DEVICE)

# print(f"✅ Model loaded successfully from {MODEL_PATH} (Epoch {checkpoint.get('epoch', 'Unknown')})")


# print("\nEvaluating Model")
# test_dataset = AudioSequenceDataset(metadata_path=test_csv, audio_dir=config.SIMULATED_DIR, segment_duration=0.5)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# test_loss, test_acc = evaluate_model(model, criterion, test_loader, DEVICE)
# print(f"Test Completed: Loss = {test_loss:.4f} | Accuracy = {test_acc:.4f}")
# # Get predictions
# y_pred, y_true, file_names = get_model_predictions(model, test_loader, DEVICE)
# # 🔹 กำหนด path สำหรับบันทึก segment prediction
# segment_csv_path = os.path.join(config.output_dir, f"{MODEL_NAME}_segment_predictions.csv")

# # 🔹 Save Segment-Level Predictions
# save_segment_predictions_csv(
#     y_pred=y_pred,
#     y_true=y_true,
#     file_names=file_names,
#     label_dict=label_map,
#     output_path=segment_csv_path,
#     model_name=MODEL_NAME,
#     segment_duration=0.5,
#     overlap=0.0 
# )
# # Evaluate Segment-Level Performance
# output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output", "evaluation"))
# segment_results = evaluate_segment_level(y_pred, y_true, label_map, output_dir)

# # Evaluate Event-Level Performance
# event_acc, event_true, event_pred = evaluate_event_level(y_pred, y_true, file_names, label_map)

# print(f"\n Final Evaluation Completed. Segment Accuracy: {segment_results['Value'][0]:.4f}, Event Accuracy: {event_acc:.4f}")


# import pandas as pd

# urban = pd.read_csv("/media/mosdet_nas/mosdet/Mosqui-DST/data/simulated_dataset/segments/test/test_segment_urban_metadata.csv")
# forest = pd.read_csv("/media/mosdet_nas/mosdet/Mosqui-DST/data/simulated_dataset/segments/test/test_segment_forest_metadata.csv")

# print("Urban files:", urban['file_name'].nunique())
# print("Forest files:", forest['file_name'].nunique())

# print("Urban duration mean:", urban['duration'].mean())
# print("Forest duration mean:", forest['duration'].mean())


# # run_forest_main.py
ENV_SCOPE = "forest"  
# ส่งให้ config
import config_new as config
config.set_env_scope(ENV_SCOPE)

import torch
torch.set_num_threads(1)

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


import os
import torch
import pandas as pd
import config_new as config
from simulation import process_simulation
from Dataloader import AudioSequenceDataset, train_model, evaluate_model, get_consistent_label_map,get_latest_model_path,load_model,merge_metadata
from evaluate import get_model_predictions, evaluate_segment_level, evaluate_event_level,save_segment_predictions_csv
from model import Mosbeatnet,MosqPlusModel
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


os.makedirs(config.METADATA_DIR, exist_ok=True)
os.makedirs(config.SIMULATED_DIR, exist_ok=True)

print(config.MODEL_DIR)
print(f"Current Environment : {config.ENV_SCOPE}")


# print("--- Forest Environment ---")
# # ------------------------------------------
# # Generate Simulate Dataset (1st)
# # ------------------------------------------
# # Generate Train Dataset
# print("\n Generating Train Dataset...")
# process_simulation(
#         mosquito_dirs=config.MOSQUITO_DIRS,
#         noise_dir=config.NOISE_TRAIN_DIR,
#         output_dir=config.SIMULATED_DIR,
#         num_simulations=5,
#         env_target="forest",  # เปลี่ยนตาม environment ที่ต้องการ
#         dataset_type="train"
#     )

from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score
)
import torch
import torch.nn.functional as F
import numpy as np

eps = 1e-8

def f1_overall_framewise(y_pred, y_true):
    assert y_pred.shape == y_true.shape, f"Shape mismatch: y_pred={y_pred.shape}, y_true={y_true.shape}"
    TP = ((2 * y_true - y_pred) == 1).sum()
    Nref = y_true.sum()
    Nsys = y_pred.sum()

    prec = TP / (Nsys + eps)
    recall = TP / (Nref + eps)
    f1_score = 2 * prec * recall / (prec + recall + eps)
    return f1_score.item(), prec.item(), recall.item()

def er_overall_framewise(y_pred, y_true):
    if len(y_pred.shape) == 3:
        y_pred = y_pred.view(-1, y_pred.shape[-1])
        y_true = y_true.view(-1, y_true.shape[-1])
    zero = torch.tensor(0)

    FP = torch.logical_and(y_true == 0, y_pred == 1).sum(1)
    FN = torch.logical_and(y_true == 1, y_pred == 0).sum(1)
    S = torch.minimum(FP, FN).sum()
    D = torch.maximum(zero, FN - FP).sum()
    I = torch.maximum(zero, FP - FN).sum()

    Nref = y_true.sum()
    ER = (S + D + I) / (Nref + eps)
    return ER.item()

def debug_segment_level(y_pred, y_true, label_dict):
    print("\n========== [DEBUG] Segment-Level Evaluation ==========")

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=list(label_dict.keys()), output_dict=True)

    macro_precision = report["macro avg"]["precision"]
    macro_recall = report["macro avg"]["recall"]
    macro_f1 = report["macro avg"]["f1-score"]
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    # Segmentwise metric (F1, ER) inspired by framewise metrics from SED
    num_classes = len(label_dict)
    y_true_tensor = torch.tensor(y_true)
    y_pred_tensor = torch.tensor(y_pred)

    y_true_bin = F.one_hot(y_true_tensor, num_classes=num_classes).float()
    y_pred_bin = F.one_hot(y_pred_tensor, num_classes=num_classes).float()

    seg_f1, seg_prec, seg_rec = f1_overall_framewise(y_pred_bin, y_true_bin)
    seg_er = er_overall_framewise(y_pred_bin, y_true_bin)

    print(f"\nSegment-Level Metrics")
    print(f"   - Accuracy                : {acc:.4f}")
    print(f"   - Balanced Accuracy       : {balanced_acc:.4f}")
    print(f"   - Macro Precision         : {macro_precision:.4f}")
    print(f"   - Macro Recall            : {macro_recall:.4f}")
    print(f"   - Macro F1-score          : {macro_f1:.4f}")

    print(f"\nSegmentwise Detection Metrics (Adapted from SED)")
    print(f"   - Segmentwise Precision   : {seg_prec:.4f}")
    print(f"   - Segmentwise Recall      : {seg_rec:.4f}")
    print(f"   - Segmentwise F1 (Overall): {seg_f1:.4f}")
    print(f"   - Segmentwise ER (Overall): {seg_er:.4f}")

    print("\nClassification Report (Per Class):")
    class_df = pd.DataFrame(report).transpose()
    print(class_df[["precision", "recall", "f1-score", "support"]])

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=label_dict.keys(), columns=label_dict.keys())
    print(cm_df)

    return {
        "accuracy": acc,
        "balanced_acc": balanced_acc,
        "macro_f1": macro_f1,
        "segmentwise_f1": seg_f1,
        "segmentwise_er": seg_er
    }
































train_in = os.path.join(config.METADATA_DIR, "simulation_metadata_forest_train.csv")
train_out = os.path.join(config.METADATA_DIR, "simulation_metadata_forest_outdoor_train.csv")
train_csv = merge_metadata([train_in, train_out], os.path.join(config.METADATA_DIR, "simulation_metadata_forest_all_train.csv"))

val_in = os.path.join(config.METADATA_DIR, "simulation_metadata_forest_val.csv")
val_out = os.path.join(config.METADATA_DIR, "simulation_metadata_forest_outdoor_val.csv")
val_csv = merge_metadata([val_in, val_out], os.path.join(config.METADATA_DIR, "simulation_metadata_forest_all_val.csv"))

test_in = os.path.join(config.METADATA_DIR, "simulation_metadata_forest_test.csv")
test_out = os.path.join(config.METADATA_DIR, "simulation_metadata_forest_outdoor_test.csv")
test_csv = merge_metadata([test_in, test_out], os.path.join(config.METADATA_DIR, "simulation_metadata_forest_all_test.csv"))

# Update simulation directory
simulation_dir = config.SIMULATED_DIR  


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model parameters
segment_length = int(config.SAMPLING_RATE * 0.5) 
num_classes =  5 
# num_segments = 20
label_map = get_consistent_label_map(train_csv, val_csv, test_csv)
train_dataset = AudioSequenceDataset(metadata_path=train_csv, audio_dir=simulation_dir, segment_duration=0.5, label_map=label_map, augmentor=None)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=32, pin_memory=True)
for x, y in train_loader:
    print(f"X shape: {x.shape}")  # (B, T, 1, L)
    print(f"y shape: {y.shape}")  # (B, T)
# # # # Define model
# # # model_class = MosqPlusModel
# # # model = MosqPlusModel(n_timesteps=segment_length, n_outputs=num_classes).to(device)

# model_class = Mosbeatnet
# model = Mosbeatnet(n_timesteps=segment_length, n_outputs=num_classes).to(device)

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0001)

# # # # Training
# # print("\nTraining the model...")
# # train_model(model, optimizer, criterion, train_csv, val_csv, simulation_dir, device, label_map, epochs=1000, batch_size=32, regenerate_every=10)


# # # # Load model for evaluation
# # print("\n Model Evaluation Soon...")

# # # # # # === MosqsongPlus ===
# # model_name = "MosqPlusModel_final"
# # model_path = get_latest_model_path(model_name)
# # model = load_model(model_path, n_timesteps=segment_length, n_outputs=num_classes, device=device, model_name=model_name)

# # # === Mosbeatnet ===
# model_name = "Mosbeatnet_final"
# model_path = get_latest_model_path(model_name)
# model = load_model(model_path, n_timesteps=segment_length, n_outputs=num_classes, device=device, model_name=model_name)


# latest_model_filename = os.path.basename(model_path).replace(".pth", "")  

# # # # For Test manual
# # # latest_model_filename = 'Mosbeatnet_final_round1'  # ไม่ต้องใส่ .pth ตรงนี้
# # # model_path = os.path.join(config.MODEL_DIR, model_name, latest_model_filename + ".pth")
# # # model = load_model(model_path, n_timesteps=segment_length, n_outputs=num_classes, device=device, model_name=model_name)
# # # # latest_model_filename = 'Mosbeatnet_final_round1.pth'


# # # ========== Evaluate on Test Set ==========
# print(f"\nEvaluating {model_name} Model")
# test_dataset = AudioSequenceDataset(
#     metadata_path=test_csv,
#     audio_dir=simulation_dir,
#     segment_duration=0.5,
#     save_metadata=True
# )
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# test_loss, test_acc = evaluate_model(model, criterion, test_loader, device)
# print(f"Test Completed: Loss = {test_loss:.4f} | Accuracy = {test_acc:.4f}")


# # ========== Generate Predictions ==========
# y_pred, y_true, file_names = get_model_predictions(model, test_loader, device)


# # ========== Segment-Level Evaluation ==========
# output_dir = os.path.join(config.EVAL_DIR, latest_model_filename)
# os.makedirs(output_dir, exist_ok=True)
# segment_results = debug_segment_level(y_pred, y_true, label_map)


# # Save predictions
# segment_csv_path = os.path.join(output_dir, f"{latest_model_filename}_segment_predictions.csv")
# save_segment_predictions_csv(
#     y_pred=y_pred,
#     y_true=y_true,
#     file_names=file_names,
#     label_dict=label_map,
#     output_path=segment_csv_path,
#     model_name=latest_model_filename,
#     segment_duration=0.5,
#     overlap=0.0,
#     metadata_csv_path=test_csv
# )
# print(f"Segment-Level predictions saved to: {segment_csv_path}")


# ========== Event-Level Evaluation ==========
# event_acc, event_true, event_pred = evaluate_event_level(y_pred, y_true, file_names, label_map)

# print(f"\nFinal Evaluation Completed.")
# print(f"Segment Accuracy: {segment_results['Value'][0]:.4f}")
# print(f"Event Accuracy  : {event_acc:.4f}")


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
#         dropout_rate = 0.2
#         self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
#             sr, n_fft=hop_len, hop_length=hop_len, n_mels=40, center=False, power=1
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
#         self.conv3 = nn.Conv1d(32, 64, kernel_size=5, dilation=4, padding=25)
#         self.norm3 = nn.InstanceNorm1d(64)
#         self.pool3 = nn.MaxPool1d(kernel_size=3, stride=3)

#         # self.global_pool = nn.AdaptiveAvgPool1d(1)

#         # คำนวณ output size หลัง Conv1D
#         self._compute_output_size()

#         self.global_pool = nn.AdaptiveAvgPool1d(1)

#         # LSTM layers
#         self.lstm1 = nn.LSTM(64, 64, batch_first=True, bidirectional=True, dropout=0.3)  # ลบ dropout
#         self.lstm2 = nn.LSTM(128, 64, batch_first=True, bidirectional=True, dropout=0.3)  # ลบ dropout

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
#         batch_size, num_segments, channels, segment_length = x.shape  # (B, T, 1, 2000)
        
#         # Reshape to merge batch and segment
#         x = x.view(batch_size * num_segments, channels, segment_length)  # (B*T, 1, 2000)

#         # Conv1D stack
#         x = self.pool1(F.leaky_relu(self.norm1(self.conv1(x))))
#         x = self.pool2(F.leaky_relu(self.norm2(self.conv2(x))))
#         x = self.pool3(F.leaky_relu(self.norm3(self.conv3(x))))  # (B*T, 64, N)

#         # Global Average Pooling
#         x = self.global_pool(x).squeeze(-1)  # (B*T, 64)

#         # Reshape back to sequence
#         x = x.view(batch_size, num_segments, 64)  # (B, T, 64)

#         # LSTM
#         x, _ = self.lstm1(x)  # (B, T, 128)
#         x, _ = self.lstm2(x)
#         # Attention
#         context, attn_weights = self.attn(x, x)  # (B, T, 128)
#         x = self.layer_norm(x + context)

#         # Feature Selection
#         x = x.permute(0, 2, 1)  # (B, 128, T)
#         x = self.feature_conv(x)  # (B, 128, T)
#         x = x.permute(0, 2, 1)  # (B, T, 128)

#         # FC
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.output_layer(x)  # (B, T, n_outputs)

#         return x

# # ========== Instantiate and Summarize ==========
# # model = RealSEDNet(
# #     sr=8000,
# #     hop_len=256,
# #     input_duration=10,
# #     n_classes=5,
# #     train_dataloader=None
# # )


# n_timesteps = 2000  # segment_length
# num_segments = 20
# n_outputs = 5
# model = Mosbeatnet(n_timesteps, n_outputs) ##

# # # ตัวอย่าง shape: (B=32, T=20, C=1, L=4000) → 0.5s segment @ 16kHz
# summary(model, input_size=(32, 20, 1, 4000))