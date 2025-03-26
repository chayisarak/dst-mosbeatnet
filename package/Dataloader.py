# # Dataloader.py

import os
import numpy as np
import random
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import config_new as config
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import time
from model import Mosbeatnet
from torch.optim.lr_scheduler import ReduceLROnPlateau
# --------------------------------------------
# Audio Augmentation Class
# --------------------------------------------
class AudioAugmentor:
    def __init__(self, noise_level_range=(0.001, 0.01), pitch_shift_range=(-3, 3), stretch_range=(0.8, 1.2)):
        self.noise_level_range = noise_level_range
        self.pitch_shift_range = pitch_shift_range
        self.stretch_range = stretch_range

    def augment(self, audio, sr):
        audio = audio * np.random.uniform(0.7, 1.3)
        audio = audio + np.random.normal(0, np.random.uniform(*self.noise_level_range), audio.shape)
        if random.random() > 0.5:
            audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=random.uniform(*self.pitch_shift_range))
        if random.random() > 0.5:
            audio = librosa.effects.time_stretch(audio, rate=random.uniform(*self.stretch_range))
        return np.clip(audio, -1, 1)

# --------------------------------------------
# Sequence Dataset Class
# --------------------------------------------
# class AudioSequenceDataset(Dataset):
#     def __init__(self, metadata_path, audio_dir, segment_duration=0.5, label_map=None, augmentor=None):
#         self.df = pd.read_csv(metadata_path)
#         self.audio_dir = audio_dir
#         self.segment_duration = segment_duration
#         self.sr = config.SAMPLING_RATE
#         self.augmentor = augmentor
#         self.file_names = self.df['file_name'].unique()
#         self.label_map = label_map if label_map is not None else self._create_label_map()
#         self.max_segments = int(config.AUDIO_DURATION / segment_duration)

#         # ตรวจสอบว่า dataset เป็น urban หรือ forest
#         self.env_type = "urban" if "urban" in metadata_path else "forest"
#         self.dataset_type = "train" if "train" in metadata_path else "val" if "val" in metadata_path else "test"

#         # เก็บ segment metadata ลงใน DataFrame
#         self.segment_metadata = []

#         # path ที่ใช้บันทึก segment metadata
#         self.segment_metadata_path = self.get_segment_metadata_path()

#         # ประมวลผล segmentation 
#         self.process_all_segments()

    # def _create_label_map(self):
    #     labels = set(self.df['label'].dropna())
    #     labels.add("Noise") 
    #     return {label: idx for idx, label in enumerate(sorted(labels))}

#     def _segment_audio(self, audio,file_name):

#         segment_length = int(self.sr * self.segment_duration)
#         num_segments = len(audio) // segment_length
#         segments = [audio[i * segment_length:(i + 1) * segment_length] for i in range(num_segments)]
#         if len(segments) < self.max_segments:
#             segments.extend([np.zeros(segment_length)] * (self.max_segments - len(segments)))

#         for i in range(self.max_segments):
#             start_time = i * self.segment_duration
#             end_time = start_time + self.segment_duration
#             matching_events = self.df[(self.df['file_name'] == file_name) & 
#                                       (self.df['start_time'] <= start_time) & 
#                                       (self.df['end_time'] > start_time)]
#             label = matching_events['label'].iloc[0] if not matching_events.empty else "Noise"
#             snr = matching_events['snr'].iloc[0] if not matching_events.empty else None

#             self.segment_metadata.append({
#                 "file_name": file_name,
#                 "segment_index": i,
#                 "start_time": round(start_time, 2),
#                 "end_time": round(end_time, 2),
#                 "label": label,
#                 "snr": snr
#             })


#             segment_df = pd.DataFrame(self.segment_metadata)

#         segment_df.to_csv(self.segment_metadata_path,"se", index=False)


#         return torch.stack([torch.tensor(seg, dtype=torch.float32) for seg in segments[:self.max_segments]])
    

    
#     # def process_all_segments(self):
        
#     #     for file_name in self.file_names:
#     #         audio_path = os.path.join(self.audio_dir, file_name)
#     #         audio, _ = librosa.load(audio_path, sr=self.sr)
#     #         if self.augmentor:
#     #             audio = self.augmentor.augment(audio, self.sr)
#     #         _ = self._segment_audio(audio, file_name)  

#     #     # บันทึก segment metadata ลงไฟล์ CSV
#     #     self.save_segment_metadata()

#     # def save_segment_metadata(self):

#     #     segment_df = pd.DataFrame(self.segment_metadata)

#     #     segment_df.to_csv(self.segment_metadata_path, index=False)
#     #     print(f"Segment metadata saved: {self.segment_metadata_path}")

#     # def get_segment_metadata_path(self):

#     #     if self.dataset_type == "train":
#     #         return config.SEGMENT_METADATA_TRAIN_URBAN if self.env_type == "urban" else config.SEGMENT_METADATA_TRAIN_FOREST
#     #     elif self.dataset_type == "val":
#     #         return config.SEGMENT_METADATA_VAL_URBAN if self.env_type == "urban" else config.SEGMENT_METADATA_VAL_FOREST
#     #     elif self.dataset_type == "test":
#     #         return config.SEGMENT_METADATA_TEST_URBAN if self.env_type == "urban" else config.SEGMENT_METADATA_TEST_FOREST
#     #     else:
#     #         raise ValueError(f"Unknown dataset type: {self.dataset_type}")
    
    

#     def __len__(self):
#         return len(self.file_names)

#     def __getitem__(self, idx):
#         file_name = self.file_names[idx]
#         file_meta = self.df[self.df['file_name'] == file_name]
#         audio_path = os.path.join(self.audio_dir, file_name)
#         audio, _ = librosa.load(audio_path, sr=self.sr)
#         if self.augmentor:
#             audio = self.augmentor.augment(audio, self.sr)
#         segments = self._segment_audio(audio,file_name)

#         # print(f"Raw Audio Shape: {audio.shape}")  
#         # print(f"Segments Shape : {segments.shape}")  
        
#         labels = []
#         for i in range(self.max_segments):
#             start_time = i * self.segment_duration
#             matching_events = file_meta[(file_meta['start_time'] <= start_time) & (file_meta['end_time'] > start_time)]
#             label = matching_events['label'].iloc[0] if not matching_events.empty else "Noise" 
#             if label not in self.label_map:
#                 raise ValueError(f"Label '{label}' not found in label_map: {self.label_map}")
#             labels.append(self.label_map[label])
#         segments = segments.unsqueeze(1)  # (max_segments, 1, seg_length)
#         # print(f"Final Input Shape to Model: {segments.shape}")  

#         return segments, torch.tensor(labels, dtype=torch.long)

class AudioSequenceDataset(Dataset):
    def __init__(self, metadata_path, audio_dir, segment_duration=0.5, label_map=None, augmentor=None,save_metadata=False):
        self.df = pd.read_csv(metadata_path)
        self.audio_dir = audio_dir
        self.segment_duration = segment_duration
        self.sr = config.SAMPLING_RATE
        self.augmentor = augmentor
        self.file_names = self.df['file_name'].unique()
        self.label_map = label_map if label_map is not None else self._create_label_map()
        self.max_segments = int(config.AUDIO_DURATION / segment_duration)

        
        self.dataset_type = "train" if "train" in metadata_path else "val" if "val" in metadata_path else "test"
        self.env_type = "urban" if "urban" in metadata_path else "forest"

        
        self.segment_metadata_path = self.get_segment_metadata_path()

        
        self.segment_metadata = []

        
        self.save_metadata = save_metadata
        if self.save_metadata:
            self.process_all_segments()


    def _create_label_map(self):
        labels = set(self.df['label'].dropna())
        labels.add("Noise") 
        return {label: idx for idx, label in enumerate(sorted(labels))}

    def get_segment_metadata_path(self):
        
        if "mos_dataset" in self.df.columns:
                mos_dataset = self.df["mos_dataset"].iloc[0].lower()
                if "outdoor" in mos_dataset:
                    prefix = f"{self.dataset_type}_outdoor_segment_{self.env_type}_metadata.csv"
                else:
                    prefix = f"{self.dataset_type}_segment_{self.env_type}_metadata.csv"
        else:
            prefix = f"{self.dataset_type}_segment_{self.env_type}_metadata.csv"

        return os.path.join(config.SEGMENT_DIR, self.dataset_type, prefix)

    def _segment_audio(self, audio, file_name):
        
        segment_length = int(self.sr * self.segment_duration)
        num_segments = max(1, len(audio) // segment_length)
        segments = [audio[i * segment_length:(i + 1) * segment_length] for i in range(num_segments)]

        # Pad if necessary
        if len(segments) < self.max_segments:
            segments.extend([np.zeros(segment_length)] * (self.max_segments - len(segments)))

        
        for i in range(self.max_segments):
            start_time = i * self.segment_duration
            end_time = start_time + self.segment_duration
            matching_events = self.df[(self.df['file_name'] == file_name) & 
                                      (self.df['start_time'] <= start_time) & 
                                      (self.df['end_time'] > start_time)]
            label = matching_events['label'].iloc[0] if not matching_events.empty else "Noise"
            snr = matching_events['snr'].iloc[0] if not matching_events.empty else None

            self.segment_metadata.append({
                "file_name": file_name,
                "segment_index": i,
                "start_time": round(start_time, 2),
                "end_time": round(end_time, 2),
                "label": label,
                "snr": snr
            })

        return torch.stack([torch.tensor(seg, dtype=torch.float32) for seg in segments[:self.max_segments]])

    def process_all_segments(self):
        
        for file_name in self.file_names:
            audio_path = os.path.join(self.audio_dir, file_name)
            audio, _ = librosa.load(audio_path, sr=self.sr)
            if self.augmentor:
                audio = self.augmentor.augment(audio, self.sr)
            _ = self._segment_audio(audio, file_name)

        
        self.save_segment_metadata()

    def save_segment_metadata(self):
       
        segment_df = pd.DataFrame(self.segment_metadata)
        segment_df.to_csv(self.segment_metadata_path, index=False)
        print(f"Segment metadata saved: {self.segment_metadata_path}")

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_meta = self.df[self.df['file_name'] == file_name]
        audio_path = os.path.join(self.audio_dir, file_name)
        audio, _ = librosa.load(audio_path, sr=self.sr)
        if self.augmentor:
            audio = self.augmentor.augment(audio, self.sr)
        segments = self._segment_audio(audio, file_name)

        labels = []
        for i in range(self.max_segments):
            start_time = i * self.segment_duration
            matching_events = file_meta[(file_meta['start_time'] <= start_time) & (file_meta['end_time'] > start_time)]
            label = matching_events['label'].iloc[0] if not matching_events.empty else "Noise"
            if label not in self.label_map:
                raise ValueError(f"Label '{label}' not found in label_map: {self.label_map}")
            labels.append(self.label_map[label])

        segments = segments.unsqueeze(1)  # (max_segments, 1, seg_length)
        return segments, torch.tensor(labels, dtype=torch.long)


# --------------------------------------------
# Functions 
# --------------------------------------------

def evaluate_model(model, criterion, data_loader, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.shape[-1]), targets.view(-1))
            total_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(dim=-1) == targets).sum().item()
            total += targets.numel()
    total_loss /= len(data_loader.dataset)
    accuracy = correct / total
    return total_loss, accuracy

def get_next_model_path(model_name):
    model_dir = os.path.join(config.MODEL_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)  # สร้างโฟลเดอร์ถ้ายังไม่มี
    
    existing_models = [f for f in os.listdir(model_dir) if f.startswith(model_name) and f.endswith(".pth")]
    existing_rounds = [int(f.split("_round")[-1].split(".pth")[0]) for f in existing_models if f.split("_round")[-1].split(".pth")[0].isdigit()]
    
    next_round = max(existing_rounds, default=0) + 1
    return os.path.join(model_dir, f"{model_name}_round{next_round}.pth"), next_round

def save_model(model, optimizer, epoch, loss, model_name):
    model_path, round_num = get_next_model_path(model_name)
    
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "loss": loss,
    }
    torch.save(state, model_path)
    print(f"Model {model_name} saved as {model_path} (Epoch {epoch}, Loss: {loss:.4f})")
    return model, round_num

def load_model(model_path, n_timesteps, n_outputs, device, model_name=None):
    """โหลดโมเดลจากไฟล์ .pth"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    print(f"Loading model from {model_path}...")

    # เลือกโมเดลตามชื่อ
    if model_name and "Mosbeatnet" in model_name:
        from model import Mosbeatnet  # อ้างอิงจากไฟล์ model.py
        model = Mosbeatnet(n_timesteps=n_timesteps, n_outputs=n_outputs).to(device)
    elif model_name and "MosqPlusModel" in model_name:
        from Dataloader import MosqPlusModel  # อ้างอิงจาก Dataloader.py
        model = MosqPlusModel(n_timesteps=n_timesteps, n_outputs=n_outputs).to(device)
    else:
        raise ValueError(f"Unknown or unspecified model name: {model_name}")

    checkpoint = torch.load(model_path, map_location=device)
    if "model_state" not in checkpoint:
        raise KeyError(f"'model_state' not found in {model_path}")

    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.to(device)
    print(f"Loaded {model_name} from {model_path} (Epoch {checkpoint['epoch']})")
    return model

def get_latest_model_path(model_name):
    file_prefix_map = {
        "MosqPlusModel_final": "MosqPlusModel_final",
        "Mosbeatnet_final": "Mosbeatnet_final"
    }
    file_prefix = file_prefix_map.get(model_name, model_name)

    possible_dirs = [
        # os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output", "model"))),
        os.path.join(config.MODEL_DIR, model_name)
    ]
    
    for model_dir in possible_dirs:
        if os.path.exists(model_dir):
            model_files = [f for f in os.listdir(model_dir) if f.startswith(file_prefix) and f.endswith(".pth")]
            if model_files:
                latest_model = max(model_files, key=lambda x: int(x.split("_round")[-1].split(".pth")[0]))
                return os.path.join(model_dir, latest_model)





def get_consistent_label_map(train_csv, val_csv, test_csv):
    train_labels = set(pd.read_csv(train_csv)["label"].fillna("Noise").unique()) 
    val_labels = set(pd.read_csv(val_csv)["label"].fillna("Noise").unique())
    test_labels = set(pd.read_csv(test_csv)["label"].fillna("Noise").unique())
    all_labels = sorted(train_labels | val_labels | test_labels)
    label_map = {label: idx for idx, label in enumerate(all_labels)}
    print(f"Consistent Label Map: {label_map}")
    return label_map

# --------------------------------------------
# Training Function 
# --------------------------------------------
def train_model(model, optimizer, criterion, train_csv, val_csv, simulation_dir, device, label_map, epochs=100, batch_size=32, regenerate_every=20, patience=None):
    
    # TensorBoard
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=f"runs/{model.__class__.__name__}_mosquito_{timestamp}")
    
    
    train_dataset = AudioSequenceDataset(metadata_path=train_csv, audio_dir=simulation_dir, segment_duration=0.5, label_map=label_map,save_metadata=True)
    val_dataset = AudioSequenceDataset(metadata_path=val_csv, audio_dir=simulation_dir, segment_duration=0.5, label_map=label_map,save_metadata=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    best_val_loss = float('inf') if patience is not None else None
    epochs_no_improve = 0
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    total_start_time = time.time()

    for epoch in range(epochs):

        epoch_start = time.time()

        if epoch < 5:
            print(f"Epoch {epoch+1}: Using Original Dataset (No Augmentation)")
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        elif (epoch >= 5) and ((epoch - 5) % regenerate_every == 0):
        # elif epoch % regenerate_every == 0:
            print(f"Epoch {epoch+1}: Regenerating Dataset with Augmentation")
            augmentor = AudioAugmentor()
            train_dataset = AudioSequenceDataset(metadata_path=train_csv, audio_dir=simulation_dir, segment_duration=0.5, label_map=label_map, augmentor=augmentor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Training", leave=False)):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.shape[-1]), targets.view(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(dim=-1) == targets).sum().item()
            total += targets.numel()

        train_loss /= len(train_loader.dataset)
        train_acc = correct / total
        val_loss, val_acc = evaluate_model(model, criterion, val_loader, device)

        epoch_duration = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{epochs} {epoch_duration:.2f} sec | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")


        # scheduler.step(val_loss)




        #  Log results to TensorBoard
        writer.add_scalar("Loss/Train", train_loss, epoch + 1)
        writer.add_scalar("Loss/Validation", val_loss, epoch + 1)
        writer.add_scalar("Accuracy/Train", train_acc, epoch + 1)
        writer.add_scalar("Accuracy/Validation", val_acc, epoch + 1)


        if patience is not None:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                model_name = f"{model.__class__.__name__}_best"
                model, round_num = save_model(model, optimizer, epoch + 1, val_loss, model_name)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs (no improvement for {patience} epochs)")
                    break
        else:
            if epoch == epochs - 1:
                model_name = f"{model.__class__.__name__}_final"
                model, round_num = save_model(model, optimizer, epochs, val_loss, model_name)


    return model, round_num


def merge_metadata(files, output_path):
    dfs = [pd.read_csv(f) for f in files if os.path.exists(f)]
    if not dfs:
        raise ValueError("No metadata files found to merge.")
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_csv(output_path, index=False)
    print(f"Merged metadata saved to: {output_path}")
    return output_path