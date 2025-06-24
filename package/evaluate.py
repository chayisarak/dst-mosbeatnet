import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             balanced_accuracy_score)
import torch
from torch.utils.data import DataLoader
from Dataloader import AudioSequenceDataset  # Adjust import path
import config_new as config

def get_model_predictions(model, data_loader, device):
    all_preds = []
    all_labels = []
    all_file_names = []
    dataset = data_loader.dataset

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            preds = outputs.argmax(dim=-1)

            all_preds.append(preds.cpu().numpy().flatten())
            all_labels.append(targets.cpu().numpy().flatten())

            batch_file_names = [dataset.file_names[i] for i in range(batch_idx * inputs.shape[0], batch_idx * inputs.shape[0] + inputs.shape[0])]
            repeated_file_names = []
            for fname in batch_file_names:
                repeated_file_names.extend([fname] * targets.shape[1])
            all_file_names.extend(repeated_file_names)

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    print(f"y_pred shape: {y_pred.shape}, y_true shape: {y_true.shape}")
    print(f"Unique y_pred: {np.unique(y_pred)}, Unique y_true: {np.unique(y_true)}")

    return y_pred, y_true, all_file_names  




def evaluate_segment_level(y_pred, y_true, label_dict, output_dir):
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=list(label_dict.keys()), output_dict=True)

    macro_precision = report["macro avg"]["precision"]
    macro_recall = report["macro avg"]["recall"]
    macro_f1 = report["macro avg"]["f1-score"]
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    print(f"Segment-Level Accuracy: {acc:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro F1-score: {macro_f1:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")

    results_df = pd.DataFrame({
        "Metric": ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1-score", "Balanced Accuracy"],
        "Value": [acc, macro_precision, macro_recall, macro_f1, balanced_acc]
    })
    results_path = os.path.join(output_dir, "segment_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Saved segment results: {results_path}")

    class_report_df = pd.DataFrame(report).transpose()
    class_report_path = os.path.join(output_dir, "segment_class_report.csv")
    class_report_df.to_csv(class_report_path, index=True)
    print(f"Saved per class report: {class_report_path}")

    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, list(label_dict.keys()), cm_path)

    return results_df

def plot_confusion_matrix(y_true, y_pred, labels, filename):
    cm = confusion_matrix(y_true, y_pred)
    cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_perc, annot=True, fmt=".1f", cmap="Blues", xticklabels=labels, yticklabels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Segment-Level Confusion Matrix (%)")

    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

    print(f"Saved confusion matrix: {filename}")

def evaluate_event_level(y_pred, y_true, file_names, label_dict, segment_duration=0.5):
    unique_files = sorted(set(file_names))
    event_true = []
    event_pred = []

    # step_size = segment_duration - overlap
    # segments_per_file = int(config.AUDIO_DURATION / step_size)

    for file in unique_files:
        file_indices = [i for i, fname in enumerate(file_names) if fname == file]
        file_preds = y_pred[file_indices]
        file_true = y_true[file_indices]

        noise_idx = label_dict.get("noise", 0)

        true_has_mosquito = any(label != noise_idx for label in file_true)
        pred_has_mosquito = any(pred != noise_idx for pred in file_preds)

        event_true.append(1 if true_has_mosquito else 0)
        event_pred.append(1 if pred_has_mosquito else 0)

    event_acc = accuracy_score(event_true, event_pred)
    event_balanced_acc = balanced_accuracy_score(event_true, event_pred)

    print(f"Event-Level Accuracy (Mosquito Presence): {event_acc:.4f}")
    print(f"Event-Level Balanced Accuracy: {event_balanced_acc:.4f}")

    return event_acc, event_true, event_pred

def save_segment_predictions_csv(y_pred, y_true, file_names, label_dict, output_path,
                                 model_name="unknown_model", segment_duration=0.5, overlap=0.25):
    inv_label_dict = {v: k for k, v in label_dict.items()}
    step_size = segment_duration - overlap

    file_segment_counters = {}  # <== ใช้นับ segment per file
    rows = []

    for i, (pred, true, file) in enumerate(zip(y_pred, y_true, file_names)):
        if file not in file_segment_counters:
            file_segment_counters[file] = 0
        segment_idx = file_segment_counters[file]

        start_time = round(segment_idx * step_size, 2)
        end_time = round(start_time + segment_duration, 2)

        rows.append({
            "model_name": model_name,
            "file_name": file,
            "start_time": start_time,
            "end_time": end_time,
            "true_label": inv_label_dict[true],
            "predicted_label": inv_label_dict[pred],
        })

        file_segment_counters[file] += 1 

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved segment-level predictions to: {output_path}")
