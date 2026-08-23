# STEP 3: Convert the three original images to grayscale

from pathlib import Path

import cv2
import matplotlib.pyplot as plt


# Project and input folder paths
PROJECT_FOLDER = Path(__file__).parent
INPUT_FOLDER = PROJECT_FOLDER / "input"


def load_color_image(image_name):
    """
    Load a color image using OpenCV.
    OpenCV loads color images in BGR order.
    """
    image_path = INPUT_FOLDER / image_name
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(
            f"Could not load: {image_path}\n"
            "Check whether the file is valid and not corrupted."
        )

    return image


def convert_to_grayscale(color_image):
    """
    Convert a BGR color image into a single-channel grayscale image.
    """
    grayscale_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    return grayscale_image


def display_original_and_grayscale(images):
    """
    Display Original vs Grayscale for all three images.
    """
    figure, axes = plt.subplots(3, 2, figsize=(10, 12))

    for row, (image_name, color_image) in enumerate(images.items()):
        # Convert color image from BGR to RGB for Matplotlib display
        color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        # Convert the same image to grayscale
        grayscale_image = convert_to_grayscale(color_image)

        # Left column: original color image
        axes[row, 0].imshow(color_image_rgb)
        axes[row, 0].set_title(f"Original: {image_name}")
        axes[row, 0].axis("off")

        # Right column: grayscale image
        axes[row, 1].imshow(grayscale_image, cmap="gray")
        axes[row, 1].set_title(f"Grayscale: {image_name}")
        axes[row, 1].axis("off")

        # Print grayscale shape information
        print(f"{image_name} grayscale shape: {grayscale_image.shape}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Load the original color images
    images = {
        "medical.jpg": load_color_image("medical.jpg"),
        "dark.jpg": load_color_image("dark.jpg"),
        "bright.jpg": load_color_image("bright.jpg"),
    }

    # Display all original and grayscale image pairs
    display_original_and_grayscale(images)