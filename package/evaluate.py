import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             balanced_accuracy_score)
import torch
from torch.utils.data import DataLoader
from Dataloader import AudioSequenceDataset  # import path
import config_new as config
import torch.nn.functional as F
from typing import List, Tuple
import numpy as np

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

from sklearn.metrics import confusion_matrix

def get_classwise_metrics(y_true, y_pred, label_dict):
    labels = list(label_dict.values())
    label_names = list(label_dict.keys())

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=label_names,
        output_dict=True
    )

    metrics_per_class = {}
    for idx, class_id in enumerate(labels):
        TP = cm[idx, idx]
        FP = cm[:, idx].sum() - TP
        FN = cm[idx, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)

        specificity = TN / (TN + FP + 1e-8)
        class_accuracy = TP / (TP + FP + FN + 1e-8)
        fpr = FP / (FP + TN + 1e-8)

        class_name = label_names[idx]
        precision = report[class_name]["precision"]
        recall = report[class_name]["recall"]
        f1 = report[class_name]["f1-score"]
        support = report[class_name]["support"]

        metrics_per_class[class_name] = {
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "Support": support,
            "Specificity": specificity,
            "Class Accuracy": class_accuracy,
            "FPR": fpr
        }

    return pd.DataFrame.from_dict(metrics_per_class, orient='index').reset_index().rename(columns={"index": "class"})



# For Multi-Label 
# eps = 1e-8
# # both functions accept inputs as torch tensor
# def f1_overall_framewise(y_pred, y_true):
#     assert y_pred.shape == y_true.shape, f"shape of y_pred is {y_pred.shape} but shape of y_true is {y_true.shape}. they need to be the same"
#     TP = ((2 * y_true - y_pred) == 1).sum()
#     Nref = y_true.sum() 
#     Nsys = y_pred.sum()

#     prec = (TP) / (Nsys + eps)
#     recall = (TP) / (Nref + eps)
#     f1_score = 2 * prec * recall / (prec + recall + eps)
#     return f1_score.item(), prec.item(), recall.item()

# def er_overall_framewise(y_pred, y_true):
#     if len(y_pred.shape) == 3:
#         y_pred, y_true = reshape_3Dto2D(y_pred), reshape_3Dto2D(y_true)
#     zero = torch.tensor(0)

#     FP = torch.logical_and(y_true == 0, y_pred == 1).sum(1)
#     FN = torch.logical_and(y_true == 1, y_pred == 0).sum(1)
#     S = torch.minimum(FP, FN).sum()
#     D = torch.maximum(zero, FN-FP).sum()
#     I = torch.maximum(zero, FP-FN).sum()

#     Nref = y_true.sum()
#     ER = (S+D+I) / (Nref + eps)
#     return ER.item()

# def reshape_3Dto2D(A):
#     return A.view(A.shape[0] * A.shape[1], A.shape[2]) # batch*temporal, classes

def evaluate_segment_level(y_pred, y_true, label_dict, output_dir, model_name=None, environment=None, save=None):
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=list(label_dict.keys()), output_dict=True)

    macro_precision = report["macro avg"]["precision"]
    macro_recall = report["macro avg"]["recall"]
    macro_f1 = report["macro avg"]["f1-score"]
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    weighted_precision = report["weighted avg"]["precision"]
    weighted_recall = report["weighted avg"]["recall"]
    weighted_f1 = report["weighted avg"]["f1-score"]

    print(f"\n=== Segment-Level Results ===")
    print(f"Segment-Level Accuracy       : {acc:.4f}")
    print(f"Macro Precision              : {macro_precision:.4f}")
    print(f"Macro Recall                 : {macro_recall:.4f}")
    print(f"Macro F1-score               : {macro_f1:.4f}")
    print(f"Balanced Accuracy            : {balanced_acc:.4f}")

    results_df = pd.DataFrame({
        "Metric": [
            "Accuracy", "Macro Precision", "Macro Recall",
            "Macro F1-score", "Balanced Accuracy",
            "Weighted Precision", "Weighted Recall", "Weighted F1-score"
        ],
        "Value": [
            acc, macro_precision, macro_recall, macro_f1, balanced_acc,
            weighted_precision, weighted_recall, weighted_f1
        ]
    })

    print("Preview segment-level metrics dataframe:")
    print(results_df.head())

    class_report_df = get_classwise_metrics(y_true, y_pred, label_dict)
    print("Preview per-class metrics dataframe:")
    print(class_report_df.head())

    if save:
        prefix = f"{model_name or 'model'}_{environment or 'env'}"
        results_path = os.path.join(output_dir, f"{prefix}_segment_results.csv")
        class_report_path = os.path.join(output_dir, f"{prefix}_segment_class_report.csv")
        cm_path = os.path.join(output_dir, f"{prefix}_confusion_matrix.png")

        results_df.to_csv(results_path, index=False)
        class_report_df.to_csv(class_report_path, index=False)
        plot_confusion_matrix(y_true, y_pred, list(label_dict.keys()), cm_path)

        print(f"\nSaved segment results: {results_path}")
        print(f"Saved per class report: {class_report_path}")

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


def save_segment_predictions_csv(y_pred, y_true, file_names, label_dict, output_dir,
                                 model_name=None, environment=None, segment_duration=0.5, overlap=0.25,metadata_csv_path=None):
    inv_label_dict = {v: k for k, v in label_dict.items()}
    step_size = segment_duration - overlap

    file_segment_counters = {}  # <== ใช้นับ segment per file
    rows = []



    if metadata_csv_path is not None:
        metadata_df = pd.read_csv(metadata_csv_path)
        # เลือกเฉพาะ column ที่ต้องการ
        metadata_df = metadata_df[["file_name", "mos_dataset", "subenv", "mos_path","noise_file"]]
        metadata_df = metadata_df.drop_duplicates("file_name")
    else:
        metadata_df = None


    for i, (pred, true, file) in enumerate(zip(y_pred, y_true, file_names)):
        if file not in file_segment_counters:
            file_segment_counters[file] = 0
        segment_idx = file_segment_counters[file]

        start_time = round(segment_idx * step_size, 2)
        end_time = round(start_time + segment_duration, 2)


        if metadata_df is not None and file in metadata_df["file_name"].values:
            meta_row = metadata_df[metadata_df["file_name"] == file].iloc[0]
            source_mos = meta_row["mos_dataset"]
            noise_file = os.path.basename(meta_row["noise_file"])  
            if noise_file == "guassian_noise":
                subenv = "guassian_noise"
            else:
                subenv = noise_file.split('_')[-1].replace('.wav', '')

            # subenv = meta_row["subenv"]
            mos_path = meta_row["mos_path"]
        else:
            source_mos = ""
            subenv = ""
            mos_path = ""

        rows.append({
            "model_name": model_name,
            "source_mos": source_mos,
            "subenv": subenv,
            # "mos_path": mos_path
            "file_name": file,
            "start_time": start_time,
            "end_time": end_time,
            "true_label": inv_label_dict[true],
            "predicted_label": inv_label_dict[pred],
        
        
        
        })

        file_segment_counters[file] += 1 

    df = pd.DataFrame(rows)

    model_prefix = model_name if model_name else "model"
    env_prefix = environment if environment else "env"
    filename = f"{model_prefix}_{env_prefix}_segment_predictions.csv"
    output_path = os.path.join(output_dir, filename)

    df.to_csv(output_path, index=False)
    print(f"Saved segment-level predictions to: {output_path}")



import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# =========================== Frame-level Mosquito Presence Evaluation ===========================
def evaluate_event_level(y_pred, y_true, file_names, label_dict, segment_duration=0.5):
    unique_files = sorted(set(file_names))
    event_true, event_pred = [], []

    # Fix noise index detection
    noise_idx = None
    for k, v in label_dict.items():
        if k.lower() == "noise":
            noise_idx = v
            break
    if noise_idx is None:
        raise ValueError("Could not find 'noise' class in label_dict")

    for file in unique_files:
        file_indices = [i for i, fname in enumerate(file_names) if fname == file]
        true_labels = [y_true[i] for i in file_indices]
        pred_labels = [y_pred[i] for i in file_indices]

        # หากมี class อื่นนอกเหนือจาก noise → ถือว่าเป็น mosquito event
        true_event = int(any(label != noise_idx for label in true_labels))
        pred_event = int(any(label != noise_idx for label in pred_labels))

        event_true.append(true_event)
        event_pred.append(pred_event)

    acc = accuracy_score(event_true, event_pred)
    balanced_acc = balanced_accuracy_score(event_true, event_pred)

    print(f"Event-Level Accuracy (Mosquito Presence): {acc:.4f}")
    print(f"Event-Level Balanced Accuracy: {balanced_acc:.4f}")
    return acc, event_true, event_pred

# =========================== Convert Segment Predictions to Event Sequences ===========================
def extract_events_from_segments(df: pd.DataFrame, label_col: str, time_cols=("start_time", "end_time")) -> List[Tuple[str, float, float, str]]:
    events = []
    for fname, group in df.groupby("file_name"):
        group = group.sort_values(time_cols[0])
        current_label, start_time = None, None

        for _, row in group.iterrows():
            label = row[label_col]
            if label != current_label:
                if current_label is not None:
                    events.append((current_label, start_time, prev_end, fname))
                current_label = label
                start_time = row[time_cols[0]]
            prev_end = row[time_cols[1]]
        if current_label is not None:
            events.append((current_label, start_time, prev_end, fname))
    return events

# =========================== IoU and Event Matching ===========================
def iou(event1: Tuple[float, float], event2: Tuple[float, float]) -> float:
    start = max(event1[0], event2[0])
    end = min(event1[1], event2[1])
    intersection = max(0.0, end - start)
    union = max(event1[1], event2[1]) - min(event1[0], event2[0])
    return intersection / union if union > 0 else 0.0

def match_events(gt_events, pred_events, iou_threshold=0.5):
    gt_matched = [False] * len(gt_events)
    pred_matched = [False] * len(pred_events)
    tp = 0

    for i, (gt_cls, gt_start, gt_end, _) in enumerate(gt_events):
        for j, (pred_cls, pred_start, pred_end, _) in enumerate(pred_events):
            if gt_cls == pred_cls and not pred_matched[j]:
                if iou((gt_start, gt_end), (pred_start, pred_end)) >= iou_threshold:
                    tp += 1
                    gt_matched[i] = True
                    pred_matched[j] = True
                    break

    fp = sum(not m for m in pred_matched)
    fn = sum(not m for m in gt_matched)
    return tp, fp, fn

def compute_temporal_precision_recall_from_csv(csv_path: str, iou_threshold=0.5):
    df = pd.read_csv(csv_path)
    gt_events = extract_events_from_segments(df, label_col="true_label")
    pred_events = extract_events_from_segments(df, label_col="predicted_label")
    tp, fp, fn = match_events(gt_events, pred_events, iou_threshold)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    print(f"Temporal Precision: {precision:.4f}")
    print(f"Temporal Recall   : {recall:.4f}")
    print(f"Temporal F1-score : {f1:.4f}")
    print(f"TP={tp}, FP={fp}, FN={fn}")
    return precision, recall, f1

# =========================== Temporal Error Analysis ===========================
def analyze_event_alignment(gt_events, pred_events, iou_threshold=0.5):
    onset_deltas, offset_deltas, duration_errors = [], [], []
    per_class_errors = {}

    for gt_cls, gt_start, gt_end, fname in gt_events:
        best_iou, best_pred = 0, None
        for pred_cls, pred_start, pred_end, _ in pred_events:
            if gt_cls != pred_cls:
                continue
            iou_val = iou((gt_start, gt_end), (pred_start, pred_end))
            if iou_val > best_iou:
                best_iou = iou_val
                best_pred = (pred_start, pred_end)

        if best_pred and best_iou >= iou_threshold:
            onset_error = best_pred[0] - gt_start
            offset_error = best_pred[1] - gt_end
            duration_error = (best_pred[1] - best_pred[0]) - (gt_end - gt_start)
            onset_deltas.append(onset_error)
            offset_deltas.append(offset_error)
            duration_errors.append(duration_error)
            per_class_errors.setdefault(gt_cls, []).append(duration_error)

    return onset_deltas, offset_deltas, duration_errors, per_class_errors

def plot_error_histogram(errors, title, xlabel):
    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=30, alpha=0.75, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    return plt

def analyze_from_csv(model_name:str,csv_path: str,env, iou_threshold=0.5, output_dir=".", save=True):
    df = pd.read_csv(csv_path)
    gt_events = extract_events_from_segments(df, label_col="true_label")
    pred_events = extract_events_from_segments(df, label_col="predicted_label")
    onset_deltas, offset_deltas, duration_errors, per_class_errors = analyze_event_alignment(gt_events, pred_events, iou_threshold)

    # ===== Plot histograms =====
    plt1 = plot_error_histogram(onset_deltas, "Onset Delay Distribution", "Onset Delay (s)")
    plt2 = plot_error_histogram(offset_deltas, "Offset Delay Distribution", "Offset Delay (s)")
    plt3 = plot_error_histogram(duration_errors, "Duration Error Distribution", "Duration Error (s)")

    if save:

        os.makedirs(output_dir, exist_ok=True)  
        plt1.savefig(os.path.join(output_dir, f"{model_name}_{env}_onset_delay_hist.png"))
        plt2.savefig(os.path.join(output_dir, f"{model_name}_{env}_offset_delay_hist.png"))
        plt3.savefig(os.path.join(output_dir, f"{model_name}_{env}_duration_error_hist.png"))

        
        df_out = pd.DataFrame({
            "onset_error": onset_deltas,
            "offset_error": offset_deltas,
            "duration_error": duration_errors
        })
        df_out.to_csv(os.path.join(output_dir, f"{model_name}_{env}_temporal_error_raw.csv"), index=False)

        
        summary = {
            "mean_onset_error": np.mean(onset_deltas),
            "std_onset_error": np.std(onset_deltas),
            "mean_offset_error": np.mean(offset_deltas),
            "std_offset_error": np.std(offset_deltas),
            "mean_duration_error": np.mean(duration_errors),
            "std_duration_error": np.std(duration_errors),
            "num_matched_events": len(onset_deltas)
        }
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(os.path.join(output_dir, f"{model_name}__{env}_temporal_error_summary.csv"), index=False)

        print(f"Saved plots and stats to: {output_dir}")

    return onset_deltas, offset_deltas, duration_errors, per_class_errors, [plt1, plt2, plt3]







def save_event_level_results(output_dir, model_name, environment,
                             event_acc, event_balanced_acc,
                             temporal_precision, temporal_recall, temporal_f1,
                             TP, FP, FN):
    results = {
        "Event-Level Accuracy": event_acc,
        "Event-Level Balanced Accuracy": event_balanced_acc,
        "Temporal Precision (IoU≥0.5)": temporal_precision,
        "Temporal Recall (IoU≥0.5)": temporal_recall,
        "Temporal F1-score (IoU≥0.5)": temporal_f1,
        "TP": TP,
        "FP": FP,
        "FN": FN
    }

    df = pd.DataFrame([results])
    filename = f"{model_name}_{environment}_event_results.csv"
    path = os.path.join(output_dir, filename)
    df.to_csv(path, index=False)
    print(f"Saved event-level evaluation to: {path}")

def save_segment_predictions_csv(y_pred, y_true, file_names, label_dict, output_dir,
                                 model_name=None, environment=None, segment_duration=0.5, overlap=0.25,metadata_csv_path=None):
    inv_label_dict = {v: k for k, v in label_dict.items()}
    step_size = segment_duration - overlap

    file_segment_counters = {}  # <== ใช้นับ segment per file
    rows = []



    if metadata_csv_path is not None:
        metadata_df = pd.read_csv(metadata_csv_path)
        # เลือกเฉพาะ column ที่ต้องการ
        metadata_df = metadata_df[["file_name", "mos_dataset", "subenv", "mos_path","noise_file"]]
        metadata_df = metadata_df.drop_duplicates("file_name")
    else:
        metadata_df = None


    for i, (pred, true, file) in enumerate(zip(y_pred, y_true, file_names)):
        if file not in file_segment_counters:
            file_segment_counters[file] = 0
        segment_idx = file_segment_counters[file]

        start_time = round(segment_idx * step_size, 2)
        end_time = round(start_time + segment_duration, 2)


        if metadata_df is not None and file in metadata_df["file_name"].values:
            meta_row = metadata_df[metadata_df["file_name"] == file].iloc[0]
            source_mos = meta_row["mos_dataset"]
            noise_file = os.path.basename(meta_row["noise_file"])  
            if noise_file == "guassian_noise":
                subenv = "guassian_noise"
            else:
                subenv = noise_file.split('_')[-1].replace('.wav', '')

            # subenv = meta_row["subenv"]
            mos_path = meta_row["mos_path"]
        else:
            source_mos = ""
            subenv = ""
            mos_path = ""

        rows.append({
            "model_name": model_name,
            "source_mos": source_mos,
            "subenv": subenv,
            # "mos_path": mos_path
            "file_name": file,
            "start_time": start_time,
            "end_time": end_time,
            "true_label": inv_label_dict[true],
            "predicted_label": inv_label_dict[pred],
        
        
        
        })

        file_segment_counters[file] += 1 

    df = pd.DataFrame(rows)

    model_prefix = model_name if model_name else "model"
    env_prefix = environment if environment else "env"
    filename = f"{model_prefix}_{env_prefix}_segment_predictions.csv"
    output_path = os.path.join(output_dir, filename)

    df.to_csv(output_path, index=False)
    print(f"Saved segment-level predictions to: {output_path}")
