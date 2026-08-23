# STEP 5: Global Histogram Equalization (HE)

from pathlib import Path

import cv2
import matplotlib.pyplot as plt


PROJECT_FOLDER = Path(__file__).parent
INPUT_FOLDER = PROJECT_FOLDER / "input"
OUTPUT_FOLDER = PROJECT_FOLDER / "output"


def load_grayscale_image(image_name):
    """
    Load one image as grayscale.
    """
    image_path = INPUT_FOLDER / image_name
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(
            f"Could not load image: {image_path}\n"
            "Check whether the file is valid and not corrupted."
        )

    return image


def apply_global_histogram_equalization(grayscale_image):
    """
    Apply global histogram equalization to a grayscale image.

    cv2.equalizeHist() uses the histogram of the full image.
    """
    equalized_image = cv2.equalizeHist(grayscale_image)
    return equalized_image


def save_he_image(image_name, equalized_image):
    """
    Save the HE result in its required output folder.
    """
    image_label = Path(image_name).stem
    output_path = OUTPUT_FOLDER / image_label / "HE.jpg"

    success = cv2.imwrite(str(output_path), equalized_image)

    if not success:
        raise IOError(f"Could not save image: {output_path}")

    print(f"Saved: {output_path}")


def plot_histogram(axis, image, title):
    """
    Calculate and plot one grayscale histogram.
    """
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])

    axis.plot(histogram, color="black")
    axis.set_title(title)
    axis.set_xlabel("Pixel Intensity")
    axis.set_ylabel("Number of Pixels")
    axis.set_xlim([0, 256])
    axis.grid(alpha=0.3)


def display_comparison(image_name, original_image, he_image):
    """
    Display Original | HE and their histograms.
    """
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Row 1: image comparison
    axes[0, 0].imshow(original_image, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title(f"Original: {image_name}")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(he_image, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title(f"Global HE: {image_name}")
    axes[0, 1].axis("off")

    # Row 2: histogram comparison
    plot_histogram(axes[1, 0], original_image, "Histogram: Original")
    plot_histogram(axes[1, 1], he_image, "Histogram: HE")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_names = ["medical.jpg", "dark.jpg", "bright.jpg"]

    for image_name in image_names:
        print(f"\nProcessing Global HE for: {image_name}")

        # 1. Load original grayscale image
        original_image = load_grayscale_image(image_name)

        # 2. Apply global histogram equalization
        he_image = apply_global_histogram_equalization(original_image)

        # 3. Save the enhanced image
        save_he_image(image_name, he_image)

        # 4. Display images and histograms
        display_comparison(image_name, original_image, he_image)