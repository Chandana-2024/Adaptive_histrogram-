# STEP 9: Calculate image enhancement metrics

from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd


PROJECT_FOLDER = Path(__file__).parent
INPUT_FOLDER = PROJECT_FOLDER / "input"


def load_grayscale_image(image_name):
    """
    Load one original image in grayscale.
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
    Apply histogram equalization to one local AHE tile.
    """
    histogram = np.bincount(tile.flatten(), minlength=256)
    cdf = histogram.cumsum()
    non_zero_cdf = cdf[cdf > 0]

    # Keep a completely uniform tile unchanged.
    if len(non_zero_cdf) == 0:
        return tile.copy()

    cdf_min = non_zero_cdf[0]
    total_pixels = tile.size

    if total_pixels == cdf_min:
        return tile.copy()

    lookup_table = (cdf - cdf_min) * 255 / (total_pixels - cdf_min)
    lookup_table = np.clip(lookup_table, 0, 255).astype(np.uint8)

    return lookup_table[tile]


def apply_basic_ahe(grayscale_image, tile_height=8, tile_width=8):
    """
    Apply genuine basic AHE using independently equalized local tiles.
    """
    height, width = grayscale_image.shape
    ahe_image = np.zeros_like(grayscale_image)

    for start_y in range(0, height, tile_height):
        for start_x in range(0, width, tile_width):

            end_y = min(start_y + tile_height, height)
            end_x = min(start_x + tile_width, width)

            tile = grayscale_image[start_y:end_y, start_x:end_x]

            ahe_image[start_y:end_y, start_x:end_x] = equalize_one_tile(tile)

    return ahe_image


def calculate_entropy(grayscale_image):
    """
    Calculate entropy from the normalized grayscale histogram.
    """
    histogram = cv2.calcHist([grayscale_image], [0], None, [256], [0, 256])

    # Convert histogram counts into probabilities.
    probabilities = histogram.flatten() / grayscale_image.size

    # Ignore zero probabilities because log2(0) is undefined.
    probabilities = probabilities[probabilities > 0]

    entropy = -np.sum(probabilities * np.log2(probabilities))

    return entropy


def calculate_robust_contrast(grayscale_image):
    """
    Calculate robust intensity-range contrast.

    Formula:
    (95th percentile - 5th percentile) / 255

    Using percentiles avoids letting a few extreme pixels dominate
    the contrast measurement.
    """
    intensity_5 = np.percentile(grayscale_image, 5)
    intensity_95 = np.percentile(grayscale_image, 95)

    contrast = (intensity_95 - intensity_5) / 255

    return contrast


def calculate_metrics(grayscale_image, processing_time):
    """
    Calculate all metrics for one grayscale image.
    """
    mean_intensity = np.mean(grayscale_image)
    standard_deviation = np.std(grayscale_image)
    contrast = calculate_robust_contrast(grayscale_image)
    entropy = calculate_entropy(grayscale_image)

    return {
        "Mean_Intensity": mean_intensity,
        "Std_Dev": standard_deviation,
        "Contrast": contrast,
        "Entropy": entropy,
        "Processing_Time_Seconds": processing_time,
    }


def time_method(method_function, image):
    """
    Run one enhancement method and measure its processing time.
    """
    start_time = time.perf_counter()

    enhanced_image = method_function(image)

    end_time = time.perf_counter()
    processing_time = end_time - start_time

    return enhanced_image, processing_time


if __name__ == "__main__":
    image_names = ["medical.jpg", "dark.jpg", "bright.jpg"]

    all_results = []

    for image_name in image_names:
        image_label = Path(image_name).stem
        print(f"\nCalculating metrics for: {image_name}")

        # Load original grayscale image.
        original_image = load_grayscale_image(image_name)

        # Original image has no enhancement processing time.
        original_metrics = calculate_metrics(
            original_image,
            processing_time=0.0
        )

        all_results.append({
            "Image": image_label,
            "Method": "Original",
            **original_metrics
        })

        # Global Histogram Equalization (HE)
        he_image, he_time = time_method(cv2.equalizeHist, original_image)
        he_metrics = calculate_metrics(he_image, he_time)

        all_results.append({
            "Image": image_label,
            "Method": "HE",
            **he_metrics
        })

        # Basic Adaptive Histogram Equalization (AHE)
        ahe_image, ahe_time = time_method(apply_basic_ahe, original_image)
        ahe_metrics = calculate_metrics(ahe_image, ahe_time)

        all_results.append({
            "Image": image_label,
            "Method": "AHE",
            **ahe_metrics
        })

        # Contrast Limited AHE (CLAHE)
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        clahe_image, clahe_time = time_method(clahe.apply, original_image)
        clahe_metrics = calculate_metrics(clahe_image, clahe_time)

        all_results.append({
            "Image": image_label,
            "Method": "CLAHE",
            **clahe_metrics
        })

    # Create a clear table using Pandas.
    metrics_table = pd.DataFrame(all_results)

    # Round values only for easier terminal reading.
    display_table = metrics_table.copy()
    display_table["Mean_Intensity"] = display_table["Mean_Intensity"].round(2)
    display_table["Std_Dev"] = display_table["Std_Dev"].round(2)
    display_table["Contrast"] = display_table["Contrast"].round(4)
    display_table["Entropy"] = display_table["Entropy"].round(4)
    display_table["Processing_Time_Seconds"] = (
        display_table["Processing_Time_Seconds"].round(6)
    )

    print("\nIMAGE ENHANCEMENT METRICS")
    print(display_table.to_string(index=False))