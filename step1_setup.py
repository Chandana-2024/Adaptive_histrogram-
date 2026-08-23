# STEP 1: Project setup and input-image checking

import os
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Path(__file__) means: the folder where this Python file is saved.
PROJECT_FOLDER = Path(__file__).parent

# Main project folders
INPUT_FOLDER = PROJECT_FOLDER / "input"
OUTPUT_FOLDER = PROJECT_FOLDER / "output"
RESULTS_FOLDER = PROJECT_FOLDER / "results"

# Names of the required original images
REQUIRED_IMAGES = ["medical.jpg", "dark.jpg", "bright.jpg"]


def create_project_folders():
    """
    Create required folders if they do not already exist.
    exist_ok=True prevents an error if a folder is already present.
    """
    folders = [
        INPUT_FOLDER,
        OUTPUT_FOLDER / "medical",
        OUTPUT_FOLDER / "dark",
        OUTPUT_FOLDER / "bright",
        RESULTS_FOLDER,
    ]

    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
        print(f"Folder ready: {folder}")


def check_input_images():
    """
    Check whether all three required input images exist.
    """
    missing_images = []

    for image_name in REQUIRED_IMAGES:
        image_path = INPUT_FOLDER / image_name

        if image_path.exists():
            print(f"Found image: {image_path}")
        else:
            missing_images.append(image_name)

    # Stop the program with a clear error if an image is missing.
    if missing_images:
        print("\nERROR: The following required image file(s) are missing:")

        for image_name in missing_images:
            print(f" - input/{image_name}")

        print("\nPlace the missing images inside the input folder.")
        raise FileNotFoundError("Required input image(s) not found.")

    print("\nSuccess: All required input images are available.")


def show_library_versions():
    """
    Print versions to confirm that the libraries imported successfully.
    """
    print("\nLibrary versions:")
    print("OpenCV:", cv2.__version__)
    print("NumPy:", np.__version__)
    print("Matplotlib:", plt.matplotlib.__version__)
    print("Pandas:", pd.__version__)


# Program starts here
if __name__ == "__main__":
    print("Current working folder:", os.getcwd())

    create_project_folders()
    check_input_images()
    show_library_versions()