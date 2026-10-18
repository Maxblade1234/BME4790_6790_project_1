
"""
inference.py

Run inference on a blood cell image using the
existing BloodCellDataset class.

The dataset loader handles:
- RGB conversion
- Center cropping to 363 x 360 (height x width)
- Tensor conversion
- Training-set normalization

Run:
    python inference.py
"""

import csv
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from utils.model import BloodCellCNN
from utils.dataset_loader import BloodCellDataset
from utils.gradcam import (
    GradCAM,
    denormalize_image,
)


# ============================================================
# SETTINGS
# ============================================================

NUM_CONV_LAYERS = 4

IMAGE_PATH = "bloodcells_processed/test/ERB_001517.jpg"

DATA_DIR = "bloodcells_processed"

CHECKPOINT_DIR = "saved_model"

OUTPUT_DIR = "results/inference"

GENERATE_GRADCAM = True


CLASS_NAMES = [
    "Basophil",
    "Eosinophil",
    "Erythroblast",
    "Immature Granulocyte",
    "Lymphocyte",
    "Monocyte",
    "Neutrophil",
    "Platelet",
]


# ============================================================
# FIND IMAGE IN DATASET MANIFEST
# ============================================================

def find_image_in_dataset(image_path, data_dir):
    """
    Locate the image in train.csv, val.csv, or test.csv.

    Returns:
        dataset
        image_tensor [3, 363, 360]
        true_label
        split
    """

    image_path = Path(image_path).resolve()
    data_dir = Path(data_dir)

    for split in ("test", "val", "train"):

        csv_path = data_dir / f"{split}.csv"

        if not csv_path.exists():
            continue

        dataset = BloodCellDataset(
            root_dir=data_dir,
            split=split,
            augment=False,
        )

        for idx, record in enumerate(dataset.records):

            candidate = (
                data_dir
                / record["folder"]
                / record["filename"]
            ).resolve()

            if candidate == image_path:

                image_tensor, true_label = dataset[idx]

                return (
                    dataset,
                    image_tensor,
                    true_label,
                    split,
                )

    raise FileNotFoundError(
        f"Image not found in train.csv, val.csv, "
        f"or test.csv:\n{image_path}\n\n"
        "The image must be listed in one of the "
        "dataset CSV manifests to use BloodCellDataset."
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(device):

    checkpoint_path = (
        Path(CHECKPOINT_DIR)
        / f"cnn_{NUM_CONV_LAYERS}_layer_best.pth"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model = BloodCellCNN(
        num_conv_layers=NUM_CONV_LAYERS,
        num_classes=8,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    return model, checkpoint, checkpoint_path


# ============================================================
# PREDICT
# ============================================================

@torch.no_grad()
def predict(model, input_tensor):

    if input_tensor.ndim != 4:
        raise ValueError(
            f"Expected 4D tensor, got "
            f"{tuple(input_tensor.shape)}"
        )

    logits = model(input_tensor)

    probabilities = torch.softmax(
        logits,
        dim=1,
    )[0]

    predicted_idx = int(
        torch.argmax(probabilities).item()
    )

    confidence = float(
        probabilities[predicted_idx].item()
    )

    return (
        predicted_idx,
        confidence,
        probabilities.cpu().numpy(),
    )


# ============================================================
# SAVE PROBABILITY PLOT
# ============================================================

def save_probability_plot(
    probabilities,
    predicted_idx,
    save_path,
):

    fig, ax = plt.subplots(figsize=(10, 6))

    values = np.asarray(probabilities) * 100

    colors = [
        "tab:orange" if idx == predicted_idx
        else "tab:blue"
        for idx in range(len(CLASS_NAMES))
    ]

    bars = ax.barh(
        CLASS_NAMES,
        values,
        color=colors,
    )

    ax.invert_yaxis()
    ax.set_xlabel("Probability (%)")
    ax.set_title("Blood Cell Classification Probabilities")
    ax.set_xlim(0, 105)

    for bar, value in zip(bars, values):
        ax.text(
            min(value + 1, 101),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}%",
            va="center",
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# SAVE GRAD-CAM
# ============================================================

def save_inference_gradcam(
    model,
    input_tensor,
    image_tensor,
    predicted_idx,
    true_label,
    mean,
    std,
    save_path,
):

    gradcam = GradCAM(model)

    try:
        heatmap, _ = gradcam.generate(
            input_tensor,
            target_class=predicted_idx,
        )
    finally:
        gradcam.remove_hooks()

    image_rgb = denormalize_image(
        image_tensor,
        mean,
        std,
    )

    heatmap = np.asarray(heatmap)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Image")

    axes[1].imshow(
        heatmap,
        cmap="jet",
        vmin=0,
        vmax=1,
    )
    axes[1].set_title("Grad-CAM Heatmap")

    axes[2].imshow(image_rgb)
    axes[2].imshow(
        heatmap,
        cmap="jet",
        alpha=0.4,
        vmin=0,
        vmax=1,
    )

    axes[2].set_title(
        f"True: {CLASS_NAMES[true_label]}\n"
        f"Predicted: {CLASS_NAMES[predicted_idx]}"
    )

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 55)
    print("BLOOD CELL INFERENCE")
    print("=" * 55)
    print(f"Device: {device}")
    print(f"Model depth: {NUM_CONV_LAYERS} layers")
    print(f"Image: {IMAGE_PATH}")

    # Load trained model
    model, checkpoint, checkpoint_path = load_model(
        device
    )

    print(
        f"Checkpoint epoch: "
        f"{checkpoint.get('epoch', 'Unknown')}"
    )

    best_loss = checkpoint.get("best_val_loss")

    if best_loss is not None:
        print(f"Best validation loss: {best_loss:.4f}")

    # --------------------------------------------------------
    # Load image using the EXISTING dataset loader
    # --------------------------------------------------------

    (
        dataset,
        image_tensor,
        true_label,
        split,
    ) = find_image_in_dataset(
        IMAGE_PATH,
        DATA_DIR,
    )

    # Dataset returns [3, 363, 360].
    # Add the batch dimension exactly ONCE.
    input_tensor = image_tensor.unsqueeze(0).to(device)

    print(f"Dataset split: {split}")
    print(f"Input tensor shape: {tuple(input_tensor.shape)}")
    print(f"True class: {CLASS_NAMES[true_label]}")

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    predicted_idx, confidence, probabilities = predict(
        model,
        input_tensor,
    )

    print("\n" + "=" * 55)
    print("PREDICTION RESULTS")
    print("=" * 55)

    print(f"Predicted class: {CLASS_NAMES[predicted_idx]}")
    print(f"Confidence: {confidence * 100:.2f}%")

    print("\nClass probabilities:")

    for name, probability in zip(
        CLASS_NAMES,
        probabilities,
    ):
        print(f"  {name:<24} {probability * 100:>7.2f}%")

    print(
        f"\nCorrect prediction: "
        f"{predicted_idx == true_label}"
    )

    # --------------------------------------------------------
    # Output filenames
    # --------------------------------------------------------

    prefix = (
        f"cnn_{NUM_CONV_LAYERS}_layer_"
        f"{Path(IMAGE_PATH).stem}"
    )

    json_path = output_dir / f"{prefix}_prediction.json"
    plot_path = output_dir / f"{prefix}_probabilities.png"
    gradcam_path = output_dir / f"{prefix}_gradcam.png"

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    result = {
        "image_path": IMAGE_PATH,
        "dataset_split": split,
        "model_depth": NUM_CONV_LAYERS,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "best_validation_loss": best_loss,
        "true_class": CLASS_NAMES[true_label],
        "predicted_class": CLASS_NAMES[predicted_idx],
        "confidence": confidence,
        "correct": predicted_idx == true_label,
        "class_probabilities": {
            name: float(probability)
            for name, probability in zip(
                CLASS_NAMES,
                probabilities,
            )
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    # --------------------------------------------------------
    # Save probability chart
    # --------------------------------------------------------

    save_probability_plot(
        probabilities,
        predicted_idx,
        plot_path,
    )

    # --------------------------------------------------------
    # Save Grad-CAM
    # --------------------------------------------------------

    if GENERATE_GRADCAM:

        save_inference_gradcam(
            model=model,
            input_tensor=input_tensor,
            image_tensor=image_tensor,
            predicted_idx=predicted_idx,
            true_label=true_label,
            mean=dataset.mean,
            std=dataset.std,
            save_path=gradcam_path,
        )

    print("\nSaved files:")
    print(f"  {json_path}")
    print(f"  {plot_path}")

    if GENERATE_GRADCAM:
        print(f"  {gradcam_path}")

    print("\nInference completed successfully.")
    print("=" * 55)


if __name__ == "__main__":
    main()
