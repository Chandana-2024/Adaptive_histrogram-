# STEP 8: Create and save organized comparison figures

from pathlib import Path

import cv2
import matplotlib.pyplot as plt


PROJECT_FOLDER = Path(__file__).parent
INPUT_FOLDER = PROJECT_FOLDER / "input"
OUTPUT_FOLDER = PROJECT_FOLDER / "output"
RESULTS_FOLDER = PROJECT_FOLDER / "results"


def load_grayscale_image(image_path):
    """
    Load an image in grayscale format.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(
            f"Could not load image: {image_path}\n"
            "Check whether the required file exists and is valid."
        )

    return image


def plot_histogram(axis, image, title):
    """
    Calculate and draw the grayscale intensity histogram.
    """
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])

    axis.plot(histogram, color="black", linewidth=1)
    axis.set_title(title)
    axis.set_xlabel("Intensity")
    axis.set_ylabel("Pixels")
    axis.set_xlim([0, 256])
    axis.grid(alpha=0.3)


def create_comparison_figure(image_name):
    """
    Create and save a 2-row, 4-column comparison figure
    for one original image.
    """
    image_label = Path(image_name).stem

    # Load original and all enhancement results.
    original_image = load_grayscale_image(
        INPUT_FOLDER / image_name
    )

    he_image = load_grayscale_image(
        OUTPUT_FOLDER / image_label / "HE.jpg"
    )

    ahe_image = load_grayscale_image(
        OUTPUT_FOLDER / image_label / "AHE.jpg"
    )

    clahe_image = load_grayscale_image(
        OUTPUT_FOLDER / image_label / "CLAHE.jpg"
    )

    methods = {
        "Original": original_image,
        "HE": he_image,
        "AHE": ahe_image,
        "CLAHE": clahe_image,
    }

    # Create two rows and four columns.
    figure, axes = plt.subplots(2, 4, figsize=(18, 9))

    for column, (method_name, image) in enumerate(methods.items()):
        # Row 1: image
        axes[0, column].imshow(image, cmap="gray", vmin=0, vmax=255)
        axes[0, column].set_title(method_name, fontsize=14)
        axes[0, column].axis("off")

        # Row 2: histogram
        plot_histogram(
            axes[1, column],
            image,
            f"Histogram: {method_name}"
        )

    # Add one title for the complete figure.
    figure.suptitle(
        f"Histogram Equalization Comparison: {image_label}",
        fontsize=18
    )

    # Leave space for the main title.
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the figure in the results folder.
    output_path = RESULTS_FOLDER / f"{image_label}_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    print(f"Saved comparison figure: {output_path}")

    # Display the same comparison figure.
    plt.show()

    # Release figure memory after closing its window.
    plt.close()


if __name__ == "__main__":
    # Create the results folder if it does not already exist.
    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

    image_names = ["medical.jpg", "dark.jpg", "bright.jpg"]

    for image_name in image_names:
        print(f"\nCreating comparison figure for: {image_name}")
        create_comparison_figure(image_name)