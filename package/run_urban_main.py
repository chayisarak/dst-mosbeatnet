# run_urban_main.py
ENV_SCOPE = "urban"  
# ส่งให้ config
import config_new as config
config.set_env_scope(ENV_SCOPE)
import os
import torch
import pandas as pd
import config_new as config
from simulation import process_simulation,generate_noise_only_simulations
from Dataloader import AudioSequenceDataset, train_model, evaluate_model, get_consistent_label_map,get_latest_model_path,load_model,merge_metadata
from evaluate import get_model_predictions, evaluate_segment_level, evaluate_event_level,save_segment_predictions_csv,compute_temporal_precision_recall_from_csv
from model import Mosbeatnet,MosqPlusModel
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


os.makedirs(config.METADATA_DIR, exist_ok=True)
os.makedirs(config.SIMULATED_DIR, exist_ok=True)
os.makedirs(config.EVAL_DIR, exist_ok=True)
print(config.MODEL_DIR)

print(f"Current Environment : {config.ENV_SCOPE}")

print("--- Urban Environment ---")
# ------------------------------------------
# Generate Simulate Dataset (1st) 
# Uncomment the following lines to generate the dataset for the first time.
# ------------------------------------------


## -------------- Start -----------------
# Generate Train Dataset 
# print("\n Generating Train Dataset...")
# process_simulation(
#         mosquito_dirs=config.MOSQUITO_DIRS,
#         noise_dir=config.NOISE_TRAIN_DIR,
#         output_dir=config.SIMULATED_DIR,
#         num_simulations=config.NUM_SIMULATIONS["train"]//2,
#         env_target="urban",  # เปลี่ยนตาม environment ที่ต้องการ
#         dataset_type="train"
#     )

## Generate Validation Dataset
# print("\n Generating Validation Dataset...")
# process_simulation(
#         mosquito_dirs=config.MOSQUITO_DIRS,
#         noise_dir=config.NOISE_VALTEST_DIR,
#         output_dir=config.SIMULATED_DIR,
#         num_simulations=config.NUM_SIMULATIONS["val"]//2,
#         env_target="urban",
#         dataset_type="val"
#     )

## Generate Test Dataset
# print("\n Generating Test Dataset...")
# process_simulation(
#         mosquito_dirs=config.MOSQUITO_DIRS,
#         noise_dir=config.NOISE_VALTEST_DIR,
#         output_dir=config.SIMULATED_DIR,
#         num_simulations=config.NUM_SIMULATIONS["test"]//2,
#         env_target="urban",
#         dataset_type="test"
#     )



# # --- Generate OUTDOOR ONLY Dataset ---

# print("\nGenerating Outdoor Train Dataset...")
# process_simulation(
#     mosquito_dirs=config.OUTDOOR_MOSQUITO_DIRS,  
#     noise_dir=config.NOISE_TRAIN_DIR,
#     output_dir=config.SIMULATED_DIR,
#     num_simulations=config.NUM_SIMULATIONS["train"]//2,
#     env_target="urban",
#     dataset_type="train"
# )

# print("\nGenerating Outdoor Validation Dataset...")
# process_simulation(
#     mosquito_dirs=config.OUTDOOR_MOSQUITO_DIRS,
#     noise_dir=config.NOISE_VALTEST_DIR,
#     output_dir=config.SIMULATED_DIR,
#     num_simulations=config.NUM_SIMULATIONS["val"]//2,
#     env_target="urban",
#     dataset_type="val"
# )

# print("\nGenerating Outdoor Test Dataset...")
# process_simulation(
#     mosquito_dirs=config.OUTDOOR_MOSQUITO_DIRS,
#     noise_dir=config.NOISE_VALTEST_DIR,
#     output_dir=config.SIMULATED_DIR,
#     num_simulations=config.NUM_SIMULATIONS["test"]//2,
#     env_target="urban",
#     dataset_type="test"
# )

## ------------------------------------------------

## -------------- Noise-Only Samples ----------------- ##
# generate_noise_only_simulations(
#     noise_dir=config.NOISE_VALTEST_DIR,
#     output_dir=config.SIMULATED_DIR,
#     num_simulations=125,
#     env_target="urban",          
#     dataset_type="test"
# )

## -------------- End -----------------




## ----- Merge Metadata ----
# Merge metadata (indoor + outdoor)

train_in = os.path.join(config.METADATA_DIR, "simulation_metadata_urban_train.csv")
train_out = os.path.join(config.METADATA_DIR, "simulation_metadata_urban_outdoor_train.csv")
test_noise = os.path.join(config.METADATA_DIR, "simulation_metadata_urban_noiseonly_test.csv")
train_csv = merge_metadata([train_in, train_out,test_noise], os.path.join(config.METADATA_DIR, "simulation_metadata_urban_all_train.csv"))

val_in = os.path.join(config.METADATA_DIR, "simulation_metadata_urban_val.csv")
val_out = os.path.join(config.METADATA_DIR, "simulation_metadata_urban_outdoor_val.csv")
val_csv = merge_metadata([val_in, val_out], os.path.join(config.METADATA_DIR, "simulation_metadata_urban_all_val.csv"))

test_in = os.path.join(config.METADATA_DIR, "simulation_metadata_urban_test.csv")
test_out = os.path.join(config.METADATA_DIR, "simulation_metadata_urban_outdoor_test.csv")
test_noise = os.path.join(config.METADATA_DIR, "simulation_metadata_urban_noiseonly_test.csv")
test_csv = merge_metadata([test_in, test_out,test_noise], os.path.join(config.METADATA_DIR, "simulation_metadata_urban_all_test.csv"))


# simulation directory
simulation_dir = config.SIMULATED_DIR

# define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model parameters
segment_length = int(config.SAMPLING_RATE * 0.5) 
num_classes =  5 
# num_segments = 20
label_map = get_consistent_label_map(train_csv, val_csv, test_csv)


## -------- Model Selection ----------

# Choose one of the model classes below by uncommenting:

# model_class = MosqPlusModel
model_class = Mosbeatnet
# model_class = RealSEDNet
# model_class = SEDNetSegmentLevel

## -------- Build Model ----------

if model_class == MosqPlusModel:
    model = model_class(n_timesteps=segment_length, n_outputs=num_classes).to(device)

elif model_class == Mosbeatnet:
    model = model_class(n_timesteps=segment_length, n_outputs=num_classes).to(device)

elif model_class == RealSEDNet:
    model = model_class(
        sr=config.SAMPLING_RATE,
        hop_len=config.HOP,
        input_duration=0.5,
        n_classes=num_classes
    ).to(device)

elif model_class == SEDNetSegmentLevel:
    model = model_class(
        sr=config.SAMPLING_RATE,
        hop_len=config.HOP,
        n_classes=num_classes
    ).to(device)

else:
    raise ValueError("No matching model class")


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)





## Training
# Uncomment the following lines to train the model

# print("\nTraining the model...")
# train_model(model, optimizer, criterion, train_csv, val_csv, simulation_dir, device, label_map, epochs=1000, batch_size=64, regenerate_every=10)


# # Load model for evaluation
print("\nLoading trained model for evaluation...")


# # Load model for evaluation
print("\n Model Evaluation \n")

# Uncomment your model name
# model_name = "MosqPlusModel_final"
model_name = "Mosbeatnet_final"
# model_name = "RealSEDNet_final"
# model_name = "SEDNetSegmentLevel_final"

model_path = get_latest_model_path(model_name)
model = load_model(model_path, n_timesteps=segment_length, n_outputs=num_classes, device=device, model_name=model_name)
latest_model_filename = os.path.basename(model_path).replace(".pth", "")





latest_model_filename = os.path.basename(model_path).replace(".pth", "")  

# # ========== Evaluate on Test Set ==========
print(f"\nEvaluating {model_name} Model")
test_dataset = AudioSequenceDataset(
    metadata_path=test_csv,
    audio_dir=simulation_dir,
    segment_duration=0.5,
    save_metadata=True
)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

test_loss, test_acc = evaluate_model(model, criterion, test_loader, device)
print(f"Test Completed: Loss = {test_loss:.4f} | Accuracy = {test_acc:.4f}")


# ========== Generate Predictions ==========
y_pred, y_true, file_names = get_model_predictions(model, test_loader, device)


# ========== Segment-Level Evaluation ==========
output_dir = os.path.join(config.EVAL_DIR, latest_model_filename)
os.makedirs(output_dir, exist_ok=True)
segment_results = evaluate_segment_level(y_pred, y_true, label_map, output_dir,
                                         model_name=latest_model_filename,
                                        environment=config.ENV_SCOPE,save=True)
# Save predictions
segment_csv_path = os.path.join(output_dir, f"{latest_model_filename}_{ENV_SCOPE}_segment_predictions.csv")
save_segment_predictions_csv(
    y_pred=y_pred,
    y_true=y_true,
    file_names=file_names,
    label_dict=label_map,
    output_dir=output_dir,  # ส่งเป็นโฟลเดอร์
    model_name=latest_model_filename,
    environment=config.ENV_SCOPE,
    segment_duration=0.5,
    overlap=0.0,
    metadata_csv_path=test_csv
)

print(f"Segment-Level predictions saved to: {segment_csv_path}")


# ========== Event-Level Evaluation ==========
event_acc, event_true, event_pred = evaluate_event_level(y_pred, y_true, file_names, label_map)

print("\nEvaluating Temporal Precision/Recall (per-event)")
temporal_precision, temporal_recall, temporal_f1 = compute_temporal_precision_recall_from_csv(
    csv_path=segment_csv_path,
    iou_threshold=0.5  
)

print(f"\nFinal Evaluation Completed.")
print(f"Segment Accuracy: {segment_results['Value'][0]:.4f}")
print(f"Event Accuracy  : {event_acc:.4f}")


from evaluate import analyze_from_csv

onset_deltas, offset_deltas, duration_errors, per_class_errors, plots = analyze_from_csv(
    model_name=latest_model_filename,
    csv_path=segment_csv_path,
    env=config.ENV_SCOPE,
    iou_threshold=0.5,
    output_dir=output_dir,
    save=True
)


from evaluate import save_event_level_results,match_events,extract_events_from_segments,balanced_accuracy_score  # อย่าลืมเพิ่มไว้ใน evaluate.py

# เก็บ TP/FP/FN จาก temporal evaluation
tp, fp, fn = match_events(
    extract_events_from_segments(pd.read_csv(segment_csv_path), label_col="true_label"),
    extract_events_from_segments(pd.read_csv(segment_csv_path), label_col="predicted_label"),
    iou_threshold=0.5
)

save_event_level_results(
    output_dir=output_dir,
    model_name=latest_model_filename,
    environment=config.ENV_SCOPE,
    event_acc=event_acc,
    event_balanced_acc=balanced_accuracy_score(event_true, event_pred),
    temporal_precision=temporal_precision,
    temporal_recall=temporal_recall,
    temporal_f1=temporal_f1,
    TP=tp,
    FP=fp,
    FN=fn
)
