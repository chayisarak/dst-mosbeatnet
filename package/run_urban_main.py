# run_urban_main.py
ENV_SCOPE = "urban"  
# ส่งให้ config
import config_new as config
config.set_env_scope(ENV_SCOPE)
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

print("--- Urban Environment ---")
## ------------------------------------------
## Generate Simulate Dataset (1st)
## ------------------------------------------
## Generate Train Dataset 
# print("\n Generating Train Dataset...")
# process_simulation(
#         mosquito_dirs=config.MOSQUITO_DIRS,
#         noise_dir=config.NOISE_TRAIN_DIR,
#         output_dir=config.SIMULATED_DIR,
#         num_simulations=config.NUM_SIMULATIONS["train"]//2,
#         env_target="urban",  # เปลี่ยนตาม environment ที่ต้องการ
#         dataset_type="train"
#     )

#     # Generate Validation Dataset
# print("\n Generating Validation Dataset...")
# process_simulation(
#         mosquito_dirs=config.MOSQUITO_DIRS,
#         noise_dir=config.NOISE_VALTEST_DIR,
#         output_dir=config.SIMULATED_DIR,
#         num_simulations=config.NUM_SIMULATIONS["val"]//2,
#         env_target="urban",
#         dataset_type="val"
#     )

#     # Generate Test Dataset
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



# # Define metadata file paths
# Merge metadata (indoor + outdoor)

train_in = os.path.join(config.METADATA_DIR, "simulation_metadata_urban_train.csv")
train_out = os.path.join(config.METADATA_DIR, "simulation_metadata_urban_outdoor_train.csv")
train_csv = merge_metadata([train_in, train_out], os.path.join(config.METADATA_DIR, "simulation_metadata_urban_all_train.csv"))

val_in = os.path.join(config.METADATA_DIR, "simulation_metadata_urban_val.csv")
val_out = os.path.join(config.METADATA_DIR, "simulation_metadata_urban_outdoor_val.csv")
val_csv = merge_metadata([val_in, val_out], os.path.join(config.METADATA_DIR, "simulation_metadata_urban_all_val.csv"))

test_in = os.path.join(config.METADATA_DIR, "simulation_metadata_urban_test.csv")
test_out = os.path.join(config.METADATA_DIR, "simulation_metadata_urban_outdoor_test.csv")
test_csv = merge_metadata([test_in, test_out], os.path.join(config.METADATA_DIR, "simulation_metadata_urban_all_test.csv"))

simulation_dir = config.SIMULATED_DIR


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model parameters
segment_length = int(config.SAMPLING_RATE * 0.5) 
num_classes =  5 
# num_segments = 20
label_map = get_consistent_label_map(train_csv, val_csv, test_csv)


# # Define model
# model_class = MosqPlusModel
# model = MosqPlusModel(n_timesteps=segment_length, n_outputs=num_classes).to(device)
model_class = Mosbeatnet
model = Mosbeatnet(n_timesteps=segment_length, n_outputs=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# # Training
print("\nTraining the model...")
train_model(model, optimizer, criterion, train_csv, val_csv, simulation_dir, device, label_map, epochs=500, batch_size=32, regenerate_every=10)


# # Load model for evaluation
# print("\nLoading trained model for evaluation...")

# # # === MosqsongPlus ===
# # model_name = "MosqPlusModel_final"
# # model_path = get_latest_model_path(model_name)
# # model = load_model(model_path, n_timesteps=segment_length, n_outputs=num_classes, device=device, model_name=model_name)

# === Mosbeatnet ===
model_name = "Mosbeatnet_final"
model_path = get_latest_model_path(model_name)
model = load_model(model_path, n_timesteps=segment_length, n_outputs=num_classes, device=device, model_name=model_name)
latest_model_filename = os.path.basename(model_path).replace(".pth", "")  # เช่น 'Mosbeatnet_final_round7'

# ========== Evaluate on Test Set ==========
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
output_dir = config.EVAL_DIR
segment_results = evaluate_segment_level(y_pred, y_true, label_map, output_dir)

# Save predictions
segment_csv_path = os.path.join(output_dir, f"{latest_model_filename}_segment_predictions.csv")
save_segment_predictions_csv(
    y_pred=y_pred,
    y_true=y_true,
    file_names=file_names,
    label_dict=label_map,
    output_path=segment_csv_path,
    model_name=latest_model_filename,
    segment_duration=0.5,
    overlap=0.0
)
print(f"Segment-Level predictions saved to: {segment_csv_path}")


# ========== Event-Level Evaluation ==========
event_acc, event_true, event_pred = evaluate_event_level(y_pred, y_true, file_names, label_map)

print(f"\nFinal Evaluation Completed.")
print(f"Segment Accuracy: {segment_results['Value'][0]:.4f}")
print(f"Event Accuracy  : {event_acc:.4f}")