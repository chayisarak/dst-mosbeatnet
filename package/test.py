import os
import torch
from model import Mosbeatnet
from torch.utils.data import DataLoader
from Dataloader import AudioSequenceDataset, train_model, evaluate_model, get_consistent_label_map
from evaluate import  get_model_predictions, evaluate_segment_level, evaluate_event_level,save_segment_predictions_csv
from model import Mosbeatnet
import os
import torch
import pandas as pd
import config_new as config
from simulation import process_simulation
from Dataloader import AudioSequenceDataset, train_model, evaluate_model, get_consistent_label_map,load_model
from evaluate import  get_model_predictions, evaluate_segment_level, evaluate_event_level
from model import Mosbeatnet,MosqPlusModel
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 🔹 ตั้งค่าตัวแปร
MODEL_NAME = "Mosbeatnet_final"  # ชื่อโมเดล
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output", "models", MODEL_NAME))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0001)

# # Define metadata file paths
train_csv = os.path.join(config.METADATA_DIR, "simulation_metadata_train.csv")
val_csv = os.path.join(config.METADATA_DIR, "simulation_metadata_val.csv")
test_csv = os.path.join(config.METADATA_DIR, "simulation_metadata_test.csv")
simulation_dir = config.SIMULATED_DIR

label_map = get_consistent_label_map(train_csv, val_csv, test_csv)




# 🔹 ตรวจสอบว่า directory มีอยู่จริง
if not os.path.exists(MODEL_DIR):
    raise FileNotFoundError(f"⚠️ Model directory not found: {MODEL_DIR}")

# 🔹 หาไฟล์ที่ขึ้นต้นด้วย "Mosbeatnet_final_round" และลงท้ายด้วย ".pth"
model_files = [f for f in os.listdir(MODEL_DIR) if f.startswith("Mosbeatnet_final_round") and f.endswith(".pth")]

# 🔹 ตรวจสอบว่าไฟล์มีอยู่หรือไม่
if not model_files:
    raise FileNotFoundError(f"⚠️ No model files found in {MODEL_DIR}")

# 🔹 หาไฟล์ที่มีรอบล่าสุด
try:
    latest_model = max(model_files, key=lambda x: int(x.split("_round")[-1].split(".pth")[0]))
except ValueError:
    raise ValueError(f"⚠️ Model filenames are not formatted correctly in {MODEL_DIR}")

# 🔹 กำหนด path ของไฟล์โมเดล
MODEL_PATH = os.path.join(MODEL_DIR, latest_model)

# 🔹 แสดง path ของไฟล์โมเดล
print(f"📂 Model directory: {MODEL_DIR}")
print(f"📄 Model files found: {model_files}")
print(f"✅ Latest model selected: {latest_model}")
print(f"🔍 Model path: {MODEL_PATH}")

# 🔹 โหลด checkpoint
checkpoint = torch.load(MODEL_PATH, map_location="cpu")

# 🔹 ตรวจสอบว่ามี model_state หรือไม่
if "model_state" not in checkpoint:
    raise KeyError(f"⚠️ 'model_state' not found in checkpoint {MODEL_PATH}")



# 🔹 ตั้งค่าตัวแปรที่ใช้สร้างโมเดล
SEGMENT_LENGTH = 20  # ตัวอย่างค่าจำนวน timestep
NUM_CLASSES = 5       # จำนวนคลาสของ output

# 🔹 โหลดโมเดลและ state_dict
model = Mosbeatnet(n_timesteps=SEGMENT_LENGTH, n_outputs=NUM_CLASSES).to(DEVICE)
model.load_state_dict(checkpoint["model_state"])

# 🔹 ส่งโมเดลไปยัง device (GPU หรือ CPU)
model.to(DEVICE)

print(f"✅ Model loaded successfully from {MODEL_PATH} (Epoch {checkpoint.get('epoch', 'Unknown')})")


print("\nEvaluating Model")
test_dataset = AudioSequenceDataset(metadata_path=test_csv, audio_dir=config.SIMULATED_DIR, segment_duration=0.5)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
test_loss, test_acc = evaluate_model(model, criterion, test_loader, DEVICE)
print(f"Test Completed: Loss = {test_loss:.4f} | Accuracy = {test_acc:.4f}")
# Get predictions
y_pred, y_true, file_names = get_model_predictions(model, test_loader, DEVICE)
# 🔹 กำหนด path สำหรับบันทึก segment prediction
segment_csv_path = os.path.join(config.output_dir, f"{MODEL_NAME}_segment_predictions.csv")

# 🔹 Save Segment-Level Predictions
save_segment_predictions_csv(
    y_pred=y_pred,
    y_true=y_true,
    file_names=file_names,
    label_dict=label_map,
    output_path=segment_csv_path,
    model_name=MODEL_NAME,
    segment_duration=0.5,
    overlap=0.0 
)
# Evaluate Segment-Level Performance
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output", "evaluation"))
segment_results = evaluate_segment_level(y_pred, y_true, label_map, output_dir)

# Evaluate Event-Level Performance
event_acc, event_true, event_pred = evaluate_event_level(y_pred, y_true, file_names, label_map)

print(f"\n Final Evaluation Completed. Segment Accuracy: {segment_results['Value'][0]:.4f}, Event Accuracy: {event_acc:.4f}")
