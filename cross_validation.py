import logging
import configparser
import os
import gc
import glob
import math
import time
import statistics
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from FakeImageDetector import support_functions, model

# Configuration Parser
config = configparser.ConfigParser()
config.read("parameters.ini")

# Get the necessary parameters from the config file
BASE_FOLDER = f"{config['General']['base_folder']}"
LOGS_FOLDER = f"{BASE_FOLDER}/{config['General']['logs_folder']}"
PROCESSED_DATASET_FOLDER = f"{BASE_FOLDER}/{config['Dataset']['processed_ds_folder']}"
SAMPLES = int(config['Dataset']['wanted_rows']) * 5 # Each row of the dataset contains 5 elements
SAMPLES_PER_CHUNK = int(config['Dataset']['chunk_size']) * 5 # (4 generated images + 1 real)
TEST_SIZE = float(config['Dataset']['test_size'])
RANDOM_STATE = int(config['Model']['random_state'])
IN_CHANNELS = int(config['Model']['in_channels'])
NUM_CLASSES = int(config['Model']['num_classes'])
BATCH_SIZE = int(config['Model']['batch_size'])
LEARNING_RATE = float(config['Model']['learning_rate'])
EPOCHS = int(config['CrossValidation']['epochs'])
PATIENCE = int(config['CrossValidation']['patience'])
NUM_WORKERS = int(config['CrossValidation']['num_workers'])
FOLDS = int(config['CrossValidation']['folds'])
CHECKPOINTS_FOLDER = f"{BASE_FOLDER}/{config['CrossValidation']['checkpoints_folder']}"
BASE_RESULTS_FOLDER = f"{BASE_FOLDER}/{config['CrossValidation']['base_results_folder']}"
TRAIN_RESULTS_FOLDER = f"{BASE_RESULTS_FOLDER}/{config['CrossValidation']['train_results_folder']}"
VALIDATION_RESULTS_FOLDER = f"{BASE_RESULTS_FOLDER}/{config['CrossValidation']['validation_results_folder']}"
TEST_RESULTS_FOLDER = f"{BASE_RESULTS_FOLDER}/{config['CrossValidation']['test_results_folder']}"
TRAIN_CONFUSION_MATRICES_FOLDER = f"{TRAIN_RESULTS_FOLDER}/{config['CrossValidation']['confusion_matrices_folder']}"
VALIDATION_CONFUSION_MATRICES_FOLDER = f"{VALIDATION_RESULTS_FOLDER}/{config['CrossValidation']['confusion_matrices_folder']}"
TEST_CONFUSION_MATRICES_FOLDER = f"{TEST_RESULTS_FOLDER}/{config['CrossValidation']['confusion_matrices_folder']}"
TRAIN_RESULTS_FILE = f"{TRAIN_RESULTS_FOLDER}/{config['CrossValidation']['train_results_file']}"
VALIDATION_RESULTS_FILE = f"{VALIDATION_RESULTS_FOLDER}/{config['CrossValidation']['validation_results_file']}"
TEST_RESULTS_FILE = f"{TEST_RESULTS_FOLDER}/{config['CrossValidation']['test_results_file']}"

# Logger initialization
logger = logging.getLogger(
    name="Cross Validation"
)
formatter = logging.Formatter(
    fmt="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M"
)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.setLevel("INFO")
logger.addHandler(console_handler)

## Set up logging to file
support_functions.create_folder(LOGS_FOLDER)
logging.basicConfig(
    filename=f"{LOGS_FOLDER}/cross_validation.log",
    encoding="utf-8",
    level=logging.INFO
)

class ModelEstimator():
    def __init__(self, fold_idx, device_type, training):
        super().__init__()
        self.fold_idx = fold_idx
        self.fold_checkpoint_folder = f"{CHECKPOINTS_FOLDER}/fold_{self.fold_idx}"
        support_functions.create_folder(self.fold_checkpoint_folder) # Ensure the checkpoint folder exists
        self.best_checkpoint_path = f"{self.fold_checkpoint_folder}/best_checkpoint.pt"
        self.last_checkpoint_path = f"{self.fold_checkpoint_folder}/last_checkpoint.pt"
        self.random_state = RANDOM_STATE
        self.in_channels = IN_CHANNELS
        self.num_classes = NUM_CLASSES
        self.batch_size = BATCH_SIZE
        self.learning_rate = LEARNING_RATE
        self.epochs = EPOCHS
        self.patience = PATIENCE
        self.device_type = device_type
        self.device = torch.device(self.device_type)
        self.criterion = nn.CrossEntropyLoss()
        self.model = None
        self.optimizer = None
        self.scaler = torch.amp.GradScaler(device=self.device_type, enabled=(self.device_type == "cuda"))
        self.best_loss = math.inf
        self.last_epoch = -1
        self.training = training

    def save_checkpoint(self, is_best):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_loss": self.best_loss,
            "last_epoch": self.last_epoch,
            "patience": self.patience
        }
        torch.save(checkpoint, self.last_checkpoint_path)
        logger.info(f"Fold {self.fold_idx} - Saved last checkpoint")
        if is_best:
            torch.save(checkpoint, self.best_checkpoint_path)
            logger.info(f"Fold {self.fold_idx} - Saved best checkpoint")

    def load_checkpoint(self):
        # Loads the best or last checkpoint on necessity
        if self.training:
            logger.info(f"Fold {self.fold_idx} - Model in training mode")
            if os.path.exists(self.last_checkpoint_path):
                logger.info(f"Fold {self.fold_idx} - Restoring latest available checkpoint")
                checkpoint = torch.load(self.last_checkpoint_path)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
                self.best_loss = checkpoint["best_loss"]
                self.last_epoch = checkpoint["last_epoch"]
                self.patience = checkpoint["patience"]
                logger.info(f"Fold {self.fold_idx} - Now restarting from epoch {self.last_epoch}. Best Loss value so far: {self.best_loss}")
            else:
                logger.info(f"Fold {self.fold_idx} - No available checkpoint, the training process will start from the beginning")
        else:
            logger.info(f"Fold {self.fold_idx} - Model in evaluation mode")
            if os.path.exists(self.best_checkpoint_path):
                checkpoint = torch.load(self.best_checkpoint_path)
                logger.info(f"Fold {self.fold_idx} - Restoring best checkpoint")
                self.model.load_state_dict(checkpoint["model_state_dict"])
                logger.info(f"Fold {self.fold_idx} - Model correctly loaded")
            else:
                logger.error(f"Fold {self.fold_idx} - Couldn't find best checkpoint")

    def train_model(self, X_train, X_val, y_train, y_val):
        # Initialize Model and move it to the device
        self.model = model.CNN_Model(self.in_channels, self.num_classes).to(self.device)
        self.model = torch.compile(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # Check if the training has to be resumed from an interrupted run
        self.load_checkpoint()
        self.last_epoch += 1
        # Create DataLoaders
        train_loader = DataLoader(
            dataset=TensorDataset(X_train, y_train),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=(self.device_type == "cuda")
        )
        validation_loader = DataLoader(
            dataset=TensorDataset(X_val, y_val),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=(self.device_type == "cuda")
        )
        for epoch in range(self.last_epoch, self.epochs):
            start_time = time.time()
            logger.info(f"Fold {self.fold_idx} - Epoch {self.last_epoch} started")
            # Train
            self.model.train()
            epoch_losses_train = []
            y_true_train = []
            y_pred_train = []
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                self.optimizer.zero_grad()
                # Training step
                ## With AMP
                with torch.amp.autocast(
                    device_type=self.device_type,
                    dtype=torch.bfloat16
                ):
                    preds = self.model(X_batch)
                    loss = self.criterion(preds, y_batch)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                # ## No AMP
                # preds = self.model(X_batch)
                # loss = self.criterion(preds, y_batch)
                # loss.backward()
                # self.optimizer.step()
                # Store results
                epoch_losses_train.append(loss.item())
                y_true_train.append(y_batch.detach().cpu())
                y_pred_train.append(preds.detach().argmax(dim=1).cpu())
            train_loss = statistics.mean(epoch_losses_train)
            y_true_train_np = torch.cat(y_true_train).numpy()
            y_pred_train_np = torch.cat(y_pred_train).numpy()
            train_metrics = support_functions.log_results(TRAIN_RESULTS_FILE, TRAIN_CONFUSION_MATRICES_FOLDER, "train", y_true_train_np, y_pred_train_np, train_loss, epoch, self.fold_idx)
            # Validation
            self.model.eval()
            epoch_losses_val = []
            y_true_val = []
            y_pred_val = []
            with torch.no_grad():
                for X_batch, y_batch in validation_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    preds = self.model(X_batch)
                    loss = self.criterion(preds, y_batch)
                    # Store results
                    epoch_losses_val.append(loss.item())
                    y_true_val.append(y_batch.detach().cpu())
                    y_pred_val.append(preds.detach().argmax(dim=1).cpu())
            val_loss = statistics.mean(epoch_losses_val)
            y_true_val_np = torch.cat(y_true_val).numpy()
            y_pred_val_np = torch.cat(y_pred_val).numpy()
            val_metrics = support_functions.log_results(VALIDATION_RESULTS_FILE, VALIDATION_CONFUSION_MATRICES_FOLDER, "validation", y_true_val_np, y_pred_val_np, val_loss, epoch, self.fold_idx)
            # Check for best results
            is_best = False
            if val_loss >= self.best_loss - 1e-12:
                self.patience -= 1
                logger.info(f"Fold {self.fold_idx} - Deterioration or no tangible improvement in validation loss value at epoch {self.last_epoch}\nPatience left before early stopping: {self.patience}")
            else:
                logger.info(f"Fold {self.fold_idx} - Improvement in validation loss value at epoch {self.last_epoch}")
                self.best_loss = val_loss
                while self.patience < PATIENCE:
                    self.patience += 1
                is_best = True
            # Print results
            logger.info(f"Fold {self.fold_idx} - Results for epoch {self.last_epoch}:\nTraining: {train_metrics}\nValidation: {val_metrics}")
            end_time = time.time()
            logger.info(f"Fold {self.fold_idx} - Epoch {self.last_epoch} completed in {(end_time - start_time):.3f}s")
            # Save Checkpoint
            self.save_checkpoint(is_best)
            # Increment last epoch counter
            self.last_epoch += 1
            # Early stop check
            if self.patience <= 0:
                logger.info(f"Fold {self.fold_idx} - Training stopped due to early stopping")
                break
        del train_loader, validation_loader, X_train, X_val, y_train, y_val
        gc.collect()
        if self.device_type == "cuda":
            torch.cuda.empty_cache()

    def predict_logit(self, X):
        # If the model is not loaded, initialize it and restore the best checkpoint
        if self.model is None:
            self.model = model.CNN_Model(self.in_channels, self.num_classes).to(self.device)
            self.model = torch.compile(self.model)
            self.load_checkpoint()
        self.model.eval()
        loader = DataLoader(
            dataset=TensorDataset(
                X,
                torch.zeros(X.shape[0], dtype=torch.uint8) # Dummy labels - not needed
            ),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=(self.device_type == "cuda")
        )
        logits = []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device)
                logit = self.model(X_batch)
                logits.append(logit)
        return torch.cat(logits).cpu().numpy()

if __name__ == "__main__":
    # Create necessary folders
    support_functions.create_folder(CHECKPOINTS_FOLDER)
    support_functions.create_folder(BASE_RESULTS_FOLDER)
    support_functions.create_folder(TRAIN_RESULTS_FOLDER)
    support_functions.create_folder(VALIDATION_RESULTS_FOLDER)
    support_functions.create_folder(TEST_RESULTS_FOLDER)
    support_functions.create_folder(TRAIN_CONFUSION_MATRICES_FOLDER)
    support_functions.create_folder(VALIDATION_CONFUSION_MATRICES_FOLDER)
    support_functions.create_folder(TEST_CONFUSION_MATRICES_FOLDER)
    # Create log files
    support_functions.create_results_file(f"{TRAIN_RESULTS_FILE}", "train")
    support_functions.create_results_file(f"{VALIDATION_RESULTS_FILE}", "validation")
    support_functions.create_results_file(f"{TEST_RESULTS_FILE}", "test")
    # Use CUDA if available
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device used: {device_type}")
    # Set random seeds to a deterministic value
    torch.manual_seed(RANDOM_STATE)
    # Load Data
    ## Locate files
    data_files = sorted(glob.glob(f"{PROCESSED_DATASET_FOLDER}/data/*.pt"))
    label_files = sorted(glob.glob(f"{PROCESSED_DATASET_FOLDER}/labels/*.pt"))
    if len(data_files) != len(label_files):
        error = "Mismatch in file counts! Quitting"
        logger.error(error)
        raise ValueError(error)
    ## Preallocate empty tensors
    X = torch.empty((SAMPLES, 4, 256, 256), dtype=torch.float32)
    y = torch.empty((SAMPLES), dtype=torch.uint8)
    ## Populate the tensors
    offset = 0
    for data_file, label_file in zip(data_files, label_files):
        x = torch.load(data_file)
        labels = torch.load(label_file)
        X[offset:offset+SAMPLES_PER_CHUNK].copy_(x)
        y[offset:offset+SAMPLES_PER_CHUNK].copy_(labels)
        offset += SAMPLES_PER_CHUNK
        del x, labels
        logger.info(f"Samples loaded: {offset}/{SAMPLES}")
    logger.info(f"All samples loaded:\nImages: {X.shape}\nLabels: {y.shape}")
    # Split data
    ## Split data into development + test
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    logger.info(f"Dataset split into:\nDevelopment (Train + Validation): {len(y_dev)} elements\nTest: {len(y_test)} elements")
    ## Split dev data using Stratified K-Fold into Train + Validation
    dev_skf = StratifiedKFold(
        n_splits=FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )
    # Training on the folds
    logger.info("Train and Validation started")
    for fold_idx, (train_idx, val_idx) in enumerate(dev_skf.split(X_dev, y_dev)):
        # Get the actual data
        X_train = X_dev[train_idx]
        X_val = X_dev[val_idx]
        y_train = y_dev[train_idx]
        y_val = y_dev[val_idx]
        # Model Estimator
        model_estimator_train = ModelEstimator(fold_idx, device_type, True)
        # Training
        logger.info(f"Fold {fold_idx} - Training and Validation started")
        model_estimator_train.train_model(X_train, X_val, y_train, y_val)
        logger.info(f"Fold {fold_idx} - Training and Validation finished")
    logger.info("Train and Validation finished")
    del model_estimator_train
    gc.collect()
    if device_type == "cuda":
        torch.cuda.empty_cache()
    # Ensemble soft voting
    ## To normalize classifications
    all_probs = np.zeros((X_test.shape[0], NUM_CLASSES), dtype=np.float32)
    ## To compute Losses in a coherent way wrt training and validation
    all_logits = np.zeros((X_test.shape[0], NUM_CLASSES), dtype=np.float32)
    logger.info("Test started")
    for fold_idx in range(FOLDS):
        logger.info(f"Fold {fold_idx} - Test started")
        model_estimator_test = ModelEstimator(fold_idx, device_type, False)
        logits = model_estimator_test.predict_logit(X_test)
        all_logits += logits
        logits_tensor = torch.from_numpy(logits)
        probs_tensor = nn.functional.softmax(logits_tensor, dim=1)
        all_probs += probs_tensor.numpy()
        logger.info(f"Fold {fold_idx} - Test fininshed")
    ensemble_logits = all_logits / FOLDS
    ensemble_probs = all_probs / FOLDS
    ensemble_pred = np.argmax(ensemble_probs, axis=1)
    logger.info("Test finished")
    criterion = nn.CrossEntropyLoss()
    loss = criterion(
        torch.tensor(ensemble_logits, dtype=torch.float32),
        y_test
    )
    support_functions.log_results(TEST_RESULTS_FILE, TEST_CONFUSION_MATRICES_FOLDER, "test", y_test, ensemble_pred, loss=loss.item())
    logger.info(f"Process finished")
