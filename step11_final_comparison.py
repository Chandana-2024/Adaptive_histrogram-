# STEP 11: Create final comparison tables and graphs

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_FOLDER = Path(__file__).parent
RESULTS_FOLDER = PROJECT_FOLDER / "results"
CSV_PATH = RESULTS_FOLDER / "comparison.csv"


def create_pivot_table(data, metric_name):
    """
    Create a readable table for one metric.

    Rows: image types
    Columns: enhancement methods
    """
    return data.pivot(
        index="Image",
        columns="Method",
        values=metric_name
    )


def create_metric_graphs(data):
    """
    Create grouped bar charts for important metrics.
    """
    metrics = {
        "Mean_Intensity": "Mean Intensity",
        "Std_Dev": "Standard Deviation",
        "Contrast": "Robust Contrast",
        "Entropy": "Entropy",
    }

    figure, axes = plt.subplots(2, 2, figsize=(16, 11))
    axes = axes.flatten()

    for axis, (column_name, graph_title) in zip(axes, metrics.items()):
        table = create_pivot_table(data, column_name)

        # Plot each method as a separate colored bar.
        table.plot(
            kind="bar",
            ax=axis,
            width=0.8
        )

        axis.set_title(graph_title, fontsize=14)
        axis.set_xlabel("Image Type")
        axis.set_ylabel(graph_title)
        axis.tick_params(axis="x", rotation=0)
        axis.grid(axis="y", alpha=0.3)
        axis.legend(title="Method")

    figure.suptitle(
        "HE vs AHE vs CLAHE: Metric Comparison",
        fontsize=18
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    graph_path = RESULTS_FOLDER / "metric_comparison_graphs.png"
    plt.savefig(graph_path, dpi=300, bbox_inches="tight")

    print(f"\nSaved graph: {graph_path}")
    plt.show()
    plt.close()


def create_processing_time_graph(data):
    """
    Create a separate graph for processing time.
    """
    time_table = create_pivot_table(
        data,
        "Processing_Time_Seconds"
    )

    figure, axis = plt.subplots(figsize=(11, 6))

    time_table.plot(
        kind="bar",
        ax=axis,
        width=0.8
    )

    axis.set_title("Processing Time Comparison", fontsize=15)
    axis.set_xlabel("Image Type")
    axis.set_ylabel("Processing Time (seconds)")
    axis.tick_params(axis="x", rotation=0)
    axis.grid(axis="y", alpha=0.3)
    axis.legend(title="Method")

    plt.tight_layout()

    graph_path = RESULTS_FOLDER / "processing_time_comparison.png"
    plt.savefig(graph_path, dpi=300, bbox_inches="tight")

    print(f"Saved graph: {graph_path}")
    plt.show()
    plt.close()


if __name__ == "__main__":
    # Confirm that comparison.csv exists.
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find: {CSV_PATH}\n"
            "Run step10_comparison_csv.py first."
        )

    # Read the metric data from CSV.
    data = pd.read_csv(CSV_PATH)

    print("\nFINAL COMPARISON TABLES")

    # Print one table for each useful metric.
    for metric in [
        "Mean_Intensity",
        "Std_Dev",
        "Contrast",
        "Entropy",
        "Processing_Time_Seconds",
    ]:
        print(f"\n--- {metric} ---")
        table = create_pivot_table(data, metric)
        print(table.round(4).to_string())

    # Save a complete pivot-table version for later analysis.
    summary_path = RESULTS_FOLDER / "comparison_summary.txt"

    with open(summary_path, "w", encoding="utf-8") as file:
        file.write("HE vs AHE vs CLAHE: Final Comparison Tables\n")

        for metric in [
            "Mean_Intensity",
            "Std_Dev",
            "Contrast",
            "Entropy",
            "Processing_Time_Seconds",
        ]:
            file.write(f"\n--- {metric} ---\n")
            table = create_pivot_table(data, metric)
            file.write(table.round(6).to_string())
            file.write("\n")

    print(f"\nSaved table summary: {summary_path}")

    # Create and save graphs.
    create_metric_graphs(data)
    create_processing_time_graph(data)