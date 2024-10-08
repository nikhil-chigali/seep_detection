"""
Script to preprocess the dataset for training YOLO-based models::
    1. splits the data into train, validation, and test sets.
    2. saves the train, validation, and test splits into directories
    3. creates and  a YAML file for training YOLO-based models.
    4. converts the masks to YOLO polygon format.
Arguments::
    --data_dir: Data directory name
    --test_size: Test size fraction
    --seed: Random seed
"""

from argparse import ArgumentParser
from typing import Tuple

import os
import shutil
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from loguru import logger
from tqdm import tqdm

from src.consts import DATA_DIR, YAML_TEMPLATE


def masks_to_polygons(data_points: Tuple, masks_dir: str, polygons_dir: str):
    for data_point in tqdm(data_points, desc="Converting masks to polygons"):
        # Load the mask
        mask_path = masks_dir / f"{data_point}"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # Get the size of the mask and create a dictionary to store the polygons
        size = np.array(mask.shape[::-1])
        polygons = {}

        # Find contours and convert them to polygons
        # For each class, find the contours
        for cls in np.unique(mask):
            if cls == 0:
                continue
            contours, _ = cv2.findContours(
                (mask == cls).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            polygons[cls.item()] = []

            # For each contour, convert it to a polygon
            for contour in contours:
                # Skip small contours
                if len(contour) < 3:
                    continue

                # Normalize the contour
                contour = contour.squeeze() / size
                polygon = contour.flatten().tolist()
                polygons[cls].append(" ".join(map(str, polygon)))

        # Save the polygons to a txt file
        polygon_file = os.path.join(polygons_dir, data_point.replace(".tif", ".txt"))
        with open(polygon_file, "w", encoding="utf-8") as f:
            for cls, cls_polygons in polygons.items():
                for polygon in cls_polygons:
                    f.write(f"{cls-1} {polygon}\n")


def split_data(data_dir_name: str, test_size: float, seed: int):
    logger.info("Making dataset splits")
    logger.debug(f"Test size: {test_size}")
    logger.debug(f"Random seed: {seed}")

    # Define the paths
    data_path = DATA_DIR / data_dir_name

    # Original data directories
    images_dir = data_path / "train_images_256"
    masks_dir = data_path / "train_masks_256"

    # Create the directories
    inputs_dir = data_path / "images"
    labels_dir = data_path / "labels"
    for split in ["train", "val", "test"]:
        os.makedirs(inputs_dir / split, exist_ok=True)
        os.makedirs(labels_dir / split, exist_ok=True)

    # Get the data points
    data_points = os.listdir(images_dir)

    # Split the data
    logger.debug(f"Number of data points: {len(data_points)}")
    train_data, test_data = train_test_split(
        data_points, test_size=test_size, random_state=seed
    )
    train_data, val_data = train_test_split(
        train_data, test_size=0.15, random_state=seed
    )
    logger.debug(f"Train size: {len(train_data)}")
    logger.debug(f"Val size: {len(val_data)}")
    logger.debug(f"Test size: {len(test_data)}")

    # Save the data splits
    splits = {
        "train": train_data,
        "val": val_data,
        "test": test_data,
    }

    # Copy the data points to the respective directories
    for split in ["train", "val", "test"]:
        for data_point in splits[split]:
            # Copy the images
            image_path = images_dir / data_point
            shutil.copyfile(image_path, inputs_dir / split / data_point)
        logger.debug(f"Saved {split} data to {inputs_dir / split}")

        # Convert the masks to labels
        masks_to_polygons(splits[split], masks_dir, labels_dir / split)
        logger.debug(f"Saved {split} labels to {labels_dir / split}")


def main(data_dir_name: str, test_size: float, seed: int):
    global YAML

    data_path = DATA_DIR / data_dir_name

    split_data(data_dir_name, test_size, seed)

    # Create YAML file
    yaml_file = data_path / f"{data_dir_name}.yaml"
    yaml = YAML_TEMPLATE.replace("$data_dir$", str(data_path)).replace("\\", "/")

    with open(yaml_file, "w", encoding="utf-8") as f:
        f.write(yaml)
    logger.info(f"Saved the YAML file to {yaml_file}")
    logger.info("Preprocessing completed")


if __name__ == "__main__":
    argument_parser = ArgumentParser()
    argument_parser.add_argument(
        "--data_dir", type=str, default="seep_detection", help="Data directory name"
    )
    argument_parser.add_argument(
        "--test_size", type=float, default=0.2, help="Test size fraction"
    )
    argument_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = argument_parser.parse_args()

    main(args.data_dir, args.test_size, args.seed)
