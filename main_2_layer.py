
import json
from pathlib import Path

import torch
import torch.nn as nn

from utils.model import BloodCellCNN
from utils.dataset_loader import create_dataloaders
from utils.trainer import train_model, set_seed
from utils.evaluator import (
    evaluate_model,
    save_evaluation_results,
)


# =====================================
# HYPERPARAMETERS
# =====================================

NUM_CONV_LAYERS = 2
MODEL_NAME = "cnn_2_layer"

DATA_DIR = "bloodcells_processed"

BATCH_SIZE = 16
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

NUM_EPOCHS = 100
PATIENCE = 10

OPTIMIZER = "adam"
SEED = 42

NUM_WORKERS = 0

# 0 = disable Grad-CAM during training
# 5 = generate snapshots every 5 epochs
GRADCAM_EVERY = 5
GRADCAM_INDICES = [0, 10, 20, 30]


# =====================================
# MAIN
# =====================================

def main():

    # Set seed BEFORE creating model and loaders
    set_seed(SEED)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # -------------------------
    # Load dataset
    # -------------------------

    train_loader, val_loader, test_loader = (
        create_dataloaders(
            root_dir=DATA_DIR,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
        )
    )

    print(f"Training images: {len(train_loader.dataset)}")
    print(f"Validation images: {len(val_loader.dataset)}")
    print(f"Testing images: {len(test_loader.dataset)}")

    # -------------------------
    # Create CNN
    # -------------------------

    model = BloodCellCNN(
        num_conv_layers=NUM_CONV_LAYERS,
        num_classes=8,
    )

    # -------------------------
    # Train and restore best model
    # -------------------------

    model, history, summary = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        model_name=MODEL_NAME,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        patience=PATIENCE,
        optimizer_name=OPTIMIZER,
        gradcam_every=GRADCAM_EVERY,
        gradcam_indices=GRADCAM_INDICES,
        seed=SEED,
    )

    # -------------------------
    # Evaluate on test set
    # -------------------------

    criterion = nn.CrossEntropyLoss()

    test_metrics, y_true, y_pred = evaluate_model(
        model=model,
        data_loader=test_loader,
        device=device,
        criterion=criterion,
    )

    results_dir = Path("results") / MODEL_NAME

    save_evaluation_results(
        metrics=test_metrics,
        y_true=y_true,
        y_pred=y_pred,
        save_dir=results_dir,
    )

    print("\nTEST RESULTS")
    print(f"Accuracy: {test_metrics['accuracy'] * 100:.2f}%")
    print(f"Macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"Test loss: {test_metrics['loss']:.4f}")


if __name__ == "__main__":
    main()
