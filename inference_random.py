
"""
inference_random.py

Randomly select one image from each of the 8 blood cell
classes in the TEST dataset, then run inference.

Uses a fixed random seed for reproducible selection.

Reuses:
    - BloodCellDataset from utils/dataset_loader.py
    - BloodCellCNN from utils/model.py
    - Prediction and visualization functions from inference.py

Run:
    python inference_random.py
"""

import csv
import json
import random
from pathlib import Path

import torch

from utils.dataset_loader import BloodCellDataset

from inference import (
    CLASS_NAMES,
    load_model,
    predict,
    save_probability_plot,
    save_inference_gradcam,
)


# ============================================================
# SETTINGS
# ============================================================

# Must match the NUM_CONV_LAYERS setting in inference.py
# because load_model() uses that module's configuration.
NUM_CONV_LAYERS = 3

SEED = 42

DATA_DIR = "bloodcells_processed"

OUTPUT_DIR = "results/inference_random"

GENERATE_GRADCAM = True


# ============================================================
# SELECT ONE TEST IMAGE PER CLASS
# ============================================================

def select_random_images(dataset, seed=42):
    """
    Select exactly one image from each class.

    Uses the dataset's CSV records and a fixed seed.
    """

    rng = random.Random(seed)

    class_to_indices = {
        class_idx: []
        for class_idx in range(len(CLASS_NAMES))
    }

    for idx, record in enumerate(dataset.records):
        class_idx = int(record["class_idx"])

        if class_idx in class_to_indices:
            class_to_indices[class_idx].append(idx)

    selected_indices = []

    for class_idx in range(len(CLASS_NAMES)):

        indices = class_to_indices[class_idx]

        if not indices:
            raise ValueError(
                f"No test images found for class "
                f"{CLASS_NAMES[class_idx]}"
            )

        # Sorting makes selection reproducible even if
        # the CSV order changes.
        indices = sorted(
            indices,
            key=lambda i: dataset.records[i]["filename"],
        )

        selected_idx = rng.choice(indices)

        selected_indices.append(selected_idx)

    return selected_indices


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

    # --------------------------------------------------------
    # Load test dataset
    # --------------------------------------------------------

    test_dataset = BloodCellDataset(
        root_dir=DATA_DIR,
        split="test",
        augment=False,
    )

    selected_indices = select_random_images(
        test_dataset,
        seed=SEED,
    )

    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------

    # IMPORTANT:
    # inference.load_model() reads the model depth
    # from inference.py, not this file.
    #
    # To ensure the correct depth is used, temporarily
    # synchronize that setting here.
    import inference
    inference.NUM_CONV_LAYERS = NUM_CONV_LAYERS

    model, checkpoint, checkpoint_path = load_model(
        device
    )

    print("\n" + "=" * 60)
    print("RANDOM TEST IMAGE INFERENCE")
    print("=" * 60)

    print(f"Device: {device}")
    print(f"Model depth: {NUM_CONV_LAYERS} layers")
    print(f"Random seed: {SEED}")
    print(f"Selected images: {len(selected_indices)}")
    print(f"Output directory: {output_dir}")

    summary_rows = []

    # --------------------------------------------------------
    # Run inference for each selected image
    # --------------------------------------------------------

    for number, dataset_idx in enumerate(
        selected_indices,
        start=1,
    ):

        record = test_dataset.records[dataset_idx]

        filename = record["filename"]
        true_class_idx = int(record["class_idx"])

        image_path = (
            Path(DATA_DIR)
            / record["folder"]
            / filename
        )

        # Reuse dataset_loader preprocessing.
        image_tensor, label = test_dataset[dataset_idx]

        # Dataset returns [3, 363, 360].
        # Add batch dimension exactly once.
        input_tensor = image_tensor.unsqueeze(0).to(device)

        predicted_idx, confidence, probabilities = predict(
            model,
            input_tensor,
        )

        true_class = CLASS_NAMES[true_class_idx]
        predicted_class = CLASS_NAMES[predicted_idx]

        correct = predicted_idx == true_class_idx

        print("\n" + "-" * 60)
        print(
            f"[{number}/{len(selected_indices)}] "
            f"{filename}"
        )
        print(f"True class: {true_class}")
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {confidence * 100:.2f}%")
        print(f"Correct: {correct}")

        # ----------------------------------------------------
        # Create per-class output folder
        # ----------------------------------------------------

        class_output_dir = (
            output_dir
            / true_class.lower().replace(" ", "_")
        )

        class_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        prefix = (
            f"cnn_{NUM_CONV_LAYERS}_layer_"
            f"{Path(filename).stem}"
        )

        json_path = (
            class_output_dir
            / f"{prefix}_prediction.json"
        )

        probability_path = (
            class_output_dir
            / f"{prefix}_probabilities.png"
        )

        gradcam_path = (
            class_output_dir
            / f"{prefix}_gradcam.png"
        )

        # ----------------------------------------------------
        # Save prediction JSON
        # ----------------------------------------------------

        result = {
            "image_path": str(image_path),
            "filename": filename,
            "dataset_split": "test",
            "random_seed": SEED,
            "model_depth": NUM_CONV_LAYERS,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "best_validation_loss": checkpoint.get(
                "best_val_loss"
            ),
            "true_class": true_class,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "correct": correct,
            "class_probabilities": {
                name: float(probability)
                for name, probability in zip(
                    CLASS_NAMES,
                    probabilities,
                )
            },
        }

        with open(
            json_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                result,
                file,
                indent=4,
            )

        # ----------------------------------------------------
        # Save probability plot
        # ----------------------------------------------------

        save_probability_plot(
            probabilities,
            predicted_idx,
            probability_path,
        )

        # ----------------------------------------------------
        # Save Grad-CAM
        # ----------------------------------------------------

        if GENERATE_GRADCAM:

            save_inference_gradcam(
                model=model,
                input_tensor=input_tensor,
                image_tensor=image_tensor,
                predicted_idx=predicted_idx,
                true_label=label,
                mean=test_dataset.mean,
                std=test_dataset.std,
                save_path=gradcam_path,
            )

        # ----------------------------------------------------
        # Summary row
        # ----------------------------------------------------

        summary_rows.append({
            "filename": filename,
            "true_class": true_class,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "correct": correct,
            "model_depth": NUM_CONV_LAYERS,
            "seed": SEED,
        })

    # --------------------------------------------------------
    # Save overall summary CSV
    # --------------------------------------------------------

    summary_csv = output_dir / "summary.csv"

    with open(
        summary_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "filename",
                "true_class",
                "predicted_class",
                "confidence",
                "correct",
                "model_depth",
                "seed",
            ],
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    # --------------------------------------------------------
    # Overall results
    # --------------------------------------------------------

    num_correct = sum(
        row["correct"]
        for row in summary_rows
    )

    accuracy = (
        num_correct / len(summary_rows)
    ) * 100

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    print(f"Images tested: {len(summary_rows)}")
    print(f"Correct predictions: {num_correct}")
    print(f"Accuracy on selected images: {accuracy:.2f}%")
    print(f"Summary saved to: {summary_csv}")

    print("\nInference completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
