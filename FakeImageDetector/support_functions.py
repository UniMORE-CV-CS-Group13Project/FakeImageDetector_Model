import os
import re
import requests
from io import BytesIO
from sklearn.metrics import accuracy_score, classification_report, multilabel_confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LABELS = ["0", "1", "2", "3", "4"]

# General Functions
## Create folder
def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        pass

# Dataset Creation Functions
## Clean URL
def clean_url(url):
    return re.sub(r"(\.(jpg|jpeg|png|gif)).*", r"\1", url, flags=re.IGNORECASE)

## URL Validity
def is_valid_url(url):
    try:
        url = clean_url(url)
        response = requests.head(
            url,
            timeout=2
        )
        contentType = response.headers.get('content-type')
        return response.status_code == 200 and contentType and "image" in contentType
    except requests.RequestException:
        return False

## Get image bytes
def image_to_bytes(image, format='PNG'):
    with BytesIO() as output:
        image.save(output, format=format)
        return output.getvalue()

# Cross Validation
## Create results files
def create_results_file(filename, split_name):
    if not os.path.exists(filename):
        targets = ["class_0", "class_1", "class_2", "class_3", "class_4", "micro", "macro", "weighted"]
        target_metrics = ["precision", "recall", "f1", "support"]
        metrics_to_score = f"{split_name}_loss,{split_name}_acc"
        for target in targets:
            for metric in target_metrics:
                metrics_to_score += f",{split_name}_{target}_{metric}"
        with open(filename, "w") as f:
            if split_name == "test":
                f.write(f"{metrics_to_score}\n")
            else:
                f.write(f"fold,epoch,{metrics_to_score}\n")

## Model evaluation
def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    single_class_metrics = {}
    for key in report.keys():
        if key in LABELS:
            single_class_metrics[key] = report[key]
    micro_averages = report["micro avg"]
    macro_averages = report["macro avg"]
    weighted_averages = report["weighted avg"]
    confusion_matrix = multilabel_confusion_matrix(y_true, y_pred, labels=LABELS)
    return acc, single_class_metrics, micro_averages, macro_averages, weighted_averages, confusion_matrix

## Log Results
def log_results(results_csv_file, confusion_matrices_folder, split_name, y_true, y_pred, loss, epoch=None, fold_idx=None):
    acc, single_class_metrics, micro_averages, macro_averages, weighted_averages, confusion_matrix = compute_metrics(y_true, y_pred)
    metrics_to_log = f"{loss:.6f},{acc:.6f}"
    # Metrics
    ## Single classes
    for label, metrics in single_class_metrics.items():
        for metric, value in metrics.items():
            metrics_to_log += f",{value:.6f}"
    ## Micro averages
    for metric, value in micro_averages.items():
        metrics_to_log += f",{value:.6f}"
    ## Macro averages
    for metric, value in macro_averages.items():
        metrics_to_log += f",{value:.6f}"
    ## Weighted averages
    for metric, value in weighted_averages.items():
        metrics_to_log += f",{value:.6f}"
    ## Save
    with open(results_csv_file, "a") as f:
        if split_name == "test":
            f.write(f"{metrics_to_log}\n")
        else:
            f.write(f"{fold_idx},{epoch},{metrics_to_log}\n")
    # Confusion Matrices
    for i, label in enumerate(LABELS):
        tn, fp, fn, tp = confusion_matrix[i].ravel()
        matrix_flattened = {}
        if not split_name == "test":
            matrix_flattened["fold"] = fold_idx
            matrix_flattened["epoch"] = epoch
        matrix_flattened["tp"] = tp
        matrix_flattened["tn"] = tn
        matrix_flattened["fp"] = fp
        matrix_flattened["fn"] = fn
        matrix_flattened_df = pd.DataFrame([matrix_flattened])
        filename = f"{confusion_matrices_folder}/confusion_matrix_class_{label}.csv"
        matrix_flattened_df.to_csv(
            filename,
            mode="a",
            header=not os.path.exists(filename),
            index=False,
        )
    return {
        "loss": loss,
        "accuracy": acc,
        "single_class_metrics": single_class_metrics,
        "micro_averages": micro_averages,
        "macro_averages": macro_averages,
        "weighted_averages": weighted_averages
    }