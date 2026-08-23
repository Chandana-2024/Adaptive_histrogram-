# STEP 6: Basic Adaptive Histogram Equalization (AHE)
# This is a genuine local-tile AHE implementation, not CLAHE.

from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


PROJECT_FOLDER = Path(__file__).parent
INPUT_FOLDER = PROJECT_FOLDER / "input"
OUTPUT_FOLDER = PROJECT_FOLDER / "output"


def load_grayscale_image(image_name):
    """
    Load one image as a grayscale image.
    """
    image_path = INPUT_FOLDER / image_name
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(
            f"Could not load image: {image_path}\n"
            "Check whether the image exists and is valid."
        )

    return image


def equalize_one_tile(tile):
    """
    Apply histogram equalization to one small grayscale tile.

    Steps:
    1. Calculate the tile histogram.
    2. Calculate its cumulative distribution function (CDF).
    3. Create a mapping from old intensity values to new values.
    4. Apply that mapping to every pixel in the tile.
    """

    # Count how many pixels have each intensity from 0 to 255.
    histogram = np.bincount(tile.flatten(), minlength=256)

    # CDF is the cumulative sum of histogram values.
    cdf = histogram.cumsum()

    # Find the first non-zero CDF value.
    # This prevents the darkest existing intensity from being mapped incorrectly.
    non_zero_cdf = cdf[cdf > 0]

    # If the tile has no valid variation, return it unchanged.
    if len(non_zero_cdf) == 0:
        return tile.copy()

    cdf_min = non_zero_cdf[0]
    total_pixels = tile.size

    # If every pixel has the same intensity, equalization is not meaningful.
    if total_pixels == cdf_min:
        return tile.copy()

    # Create a lookup table (LUT) for intensities 0 to 255.
    lookup_table = (
        (cdf - cdf_min) * 255 / (total_pixels - cdf_min)
    )

    # Keep values within the valid grayscale range and convert to uint8.
    lookup_table = np.clip(lookup_table, 0, 255).astype(np.uint8)

    # Replace every tile pixel by its new intensity from the lookup table.
    equalized_tile = lookup_table[tile]

    return equalized_tile


def apply_basic_ahe(grayscale_image, tile_height=8, tile_width=8):
    """
    Apply basic AHE by equalizing each local tile independently.

    Border handling:
    If image height or width is not divisible by 8, the final tile
    is simply smaller and is still processed correctly.
    """

    height, width = grayscale_image.shape

    # Create an empty output image with the same size as the input image.
    ahe_image = np.zeros_like(grayscale_image)

    # Move through the image in 8-pixel vertical steps.
    for start_y in range(0, height, tile_height):

        # Move through the image in 8-pixel horizontal steps.
        for start_x in range(0, width, tile_width):

            # min() safely handles the bottom and right image borders.
            end_y = min(start_y + tile_height, height)
            end_x = min(start_x + tile_width, width)

            # Extract one local tile.
            tile = grayscale_image[start_y:end_y, start_x:end_x]

            # Equalize this tile using its own local histogram and CDF.
            equalized_tile = equalize_one_tile(tile)

            # Put the equalized tile back in the correct output location.
            ahe_image[start_y:end_y, start_x:end_x] = equalized_tile

    return ahe_image


def save_ahe_image(image_name, ahe_image):
    """
    Save the AHE result in the required output folder.
    """
    image_label = Path(image_name).stem
    output_path = OUTPUT_FOLDER / image_label / "AHE.jpg"

    success = cv2.imwrite(str(output_path), ahe_image)

    if not success:
        raise IOError(f"Could not save image: {output_path}")

    print(f"Saved: {output_path}")


def plot_histogram(axis, image, title):
    """
    Plot a grayscale intensity histogram.
    """
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])

    axis.plot(histogram, color="black")
    axis.set_title(title)
    axis.set_xlabel("Pixel Intensity")
    axis.set_ylabel("Number of Pixels")
    axis.set_xlim([0, 256])
    axis.grid(alpha=0.3)


def display_comparison(image_name, original_image, ahe_image):
    """
    Display Original | AHE and the corresponding histograms.
    """
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Row 1: image comparison
    axes[0, 0].imshow(original_image, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title(f"Original: {image_name}")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(ahe_image, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title(f"Basic AHE: {image_name}")
    axes[0, 1].axis("off")

    # Row 2: histogram comparison
    plot_histogram(axes[1, 0], original_image, "Histogram: Original")
    plot_histogram(axes[1, 1], ahe_image, "Histogram: AHE")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_names = ["medical.jpg", "dark.jpg", "bright.jpg"]

    for image_name in image_names:
        print(f"\nProcessing AHE for: {image_name}")

        # Load the original grayscale image.
        original_image = load_grayscale_image(image_name)

        # Apply basic local AHE with 8 x 8 pixel tiles.
        ahe_image = apply_basic_ahe(
            original_image,
            tile_height=8,
            tile_width=8
        )

        # Save the AHE result.
        save_ahe_image(image_name, ahe_image)

        # Display original vs AHE and their histograms.
        display_comparison(image_name, original_image, ahe_image)