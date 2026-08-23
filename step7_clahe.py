# STEP 7: Contrast Limited Adaptive Histogram Equalization (CLAHE)

from pathlib import Path

import cv2
import matplotlib.pyplot as plt


PROJECT_FOLDER = Path(__file__).parent
INPUT_FOLDER = PROJECT_FOLDER / "input"
OUTPUT_FOLDER = PROJECT_FOLDER / "output"


def load_grayscale_image(image_path):
    """
    Load an image as grayscale and stop with a clear error if it cannot be read.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(
            f"Could not load image: {image_path}\n"
            "Check that the file exists and is not corrupted."
        )

    return image


def apply_clahe(grayscale_image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE to a grayscale image.

    clip_limit:
        Controls how strongly local contrast is enhanced.
        A larger value can give stronger enhancement but may amplify noise.

    tile_grid_size:
        Number of local tiles across and down the image.
    """
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    clahe_image = clahe.apply(grayscale_image)

    return clahe_image


def save_clahe_image(image_name, clahe_image):
    """
    Save CLAHE output as output/<image_name>/CLAHE.jpg
    """
    image_label = Path(image_name).stem
    output_path = OUTPUT_FOLDER / image_label / "CLAHE.jpg"

    success = cv2.imwrite(str(output_path), clahe_image)

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


def display_all_methods(image_name, original, he, ahe, clahe):
    """
    Display Original | HE | AHE | CLAHE
    and their histograms.
    """
    methods = {
        "Original": original,
        "HE": he,
        "AHE": ahe,
        "CLAHE": clahe,
    }

    figure, axes = plt.subplots(2, 4, figsize=(18, 9))

    for column, (method_name, image) in enumerate(methods.items()):
        # First row: processed images
        axes[0, column].imshow(image, cmap="gray", vmin=0, vmax=255)
        axes[0, column].set_title(f"{method_name}: {image_name}")
        axes[0, column].axis("off")

        # Second row: corresponding histograms
        plot_histogram(
            axes[1, column],
            image,
            f"Histogram: {method_name}"
        )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_names = ["medical.jpg", "dark.jpg", "bright.jpg"]

    # CLAHE starting parameters
    CLIP_LIMIT = 2.0
    TILE_GRID_SIZE = (8, 8)

    for image_name in image_names:
        image_label = Path(image_name).stem

        print(f"\nProcessing CLAHE for: {image_name}")

        # Load original grayscale input image.
        original_image = load_grayscale_image(
            INPUT_FOLDER / image_name
        )

        # Load images created in previous steps.
        he_image = load_grayscale_image(
            OUTPUT_FOLDER / image_label / "HE.jpg"
        )

        ahe_image = load_grayscale_image(
            OUTPUT_FOLDER / image_label / "AHE.jpg"
        )

        # Apply CLAHE to the original grayscale image.
        clahe_image = apply_clahe(
            original_image,
            clip_limit=CLIP_LIMIT,
            tile_grid_size=TILE_GRID_SIZE
        )

        # Save CLAHE result.
        save_clahe_image(image_name, clahe_image)

        # Show all methods and their histograms.
        display_all_methods(
            image_name,
            original_image,
            he_image,
            ahe_image,
            clahe_image
        )