# STEP 4: Display each grayscale image with its intensity histogram

from pathlib import Path

import cv2
import matplotlib.pyplot as plt


PROJECT_FOLDER = Path(__file__).parent
INPUT_FOLDER = PROJECT_FOLDER / "input"


def load_grayscale_image(image_name):
    """
    Load an image directly as grayscale.

    Each pixel will have one intensity value from 0 to 255.
    """
    image_path = INPUT_FOLDER / image_name

    grayscale_image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if grayscale_image is None:
        raise ValueError(
            f"Could not load image: {image_path}\n"
            "Check whether the image exists and is not corrupted."
        )

    return grayscale_image


def calculate_histogram(grayscale_image):
    """
    Calculate the grayscale intensity histogram.

    There are 256 bins because pixel values range from 0 to 255.
    """
    histogram = cv2.calcHist(
        [grayscale_image],  # Image to analyze
        [0],                # Grayscale channel index
        None,               # No mask; use the full image
        [256],              # 256 intensity bins
        [0, 256]            # Intensity range: 0 to 255
    )

    return histogram


def display_image_and_histogram(image_name, grayscale_image):
    """
    Show the original grayscale image beside its histogram.
    """
    histogram = calculate_histogram(grayscale_image)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: grayscale image
    axes[0].imshow(grayscale_image, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(f"Grayscale Image: {image_name}")
    axes[0].axis("off")

    # Right: histogram
    axes[1].plot(histogram, color="black")
    axes[1].set_title(f"Intensity Histogram: {image_name}")
    axes[1].set_xlabel("Pixel Intensity (0 = black, 255 = white)")
    axes[1].set_ylabel("Number of Pixels")
    axes[1].set_xlim([0, 256])
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_names = ["medical.jpg", "dark.jpg", "bright.jpg"]

    for image_name in image_names:
        grayscale_image = load_grayscale_image(image_name)

        print(f"\nProcessing histogram for: {image_name}")
        print(f"Grayscale image shape: {grayscale_image.shape}")

        display_image_and_histogram(image_name, grayscale_image)