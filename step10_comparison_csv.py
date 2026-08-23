# STEP 10: Create results/comparison.csv

from pathlib import Path

import cv2
import pandas as pd

# Import reusable functions from Step 9.
from step9_metrics import (
    load_grayscale_image,
    apply_basic_ahe,
    calculate_metrics,
    time_method
)


PROJECT_FOLDER = Path(__file__).parent
RESULTS_FOLDER = PROJECT_FOLDER / "results"


if __name__ == "__main__":
    # Ensure the results folder exists.
    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

    image_names = ["medical.jpg", "dark.jpg", "bright.jpg"]

    all_results = []

    for image_name in image_names:
        image_label = Path(image_name).stem

        print(f"\nProcessing: {image_name}")

        # Load the original grayscale image.
        original_image = load_grayscale_image(image_name)

        # ---------- Original ----------
        original_metrics = calculate_metrics(
            original_image,
            processing_time=0.0
        )

        all_results.append({
            "Image": image_label,
            "Method": "Original",
            **original_metrics
        })

        # ---------- Global HE ----------
        he_image, he_time = time_method(
            cv2.equalizeHist,
            original_image
        )

        he_metrics = calculate_metrics(he_image, he_time)

        all_results.append({
            "Image": image_label,
            "Method": "HE",
            **he_metrics
        })

        # ---------- Basic AHE ----------
        ahe_image, ahe_time = time_method(
            apply_basic_ahe,
            original_image
        )

        ahe_metrics = calculate_metrics(ahe_image, ahe_time)

        all_results.append({
            "Image": image_label,
            "Method": "AHE",
            **ahe_metrics
        })

        # ---------- CLAHE ----------
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        clahe_image, clahe_time = time_method(
            clahe.apply,
            original_image
        )

        clahe_metrics = calculate_metrics(clahe_image, clahe_time)

        all_results.append({
            "Image": image_label,
            "Method": "CLAHE",
            **clahe_metrics
        })

    # Convert the list of results into a Pandas table.
    comparison_table = pd.DataFrame(all_results)

    # Put columns in a clear and consistent order.
    comparison_table = comparison_table[
        [
            "Image",
            "Method",
            "Mean_Intensity",
            "Std_Dev",
            "Contrast",
            "Entropy",
            "Processing_Time_Seconds",
        ]
    ]

    # Save the table as CSV.
    csv_path = RESULTS_FOLDER / "comparison.csv"

    comparison_table.to_csv(
        csv_path,
        index=False,
        float_format="%.6f"
    )

    print(f"\nCSV file saved successfully: {csv_path}")
    print("\nCSV preview:")
    print(comparison_table.round(4).to_string(index=False))