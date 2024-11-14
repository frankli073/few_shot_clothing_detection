import numpy as np
import matplotlib.pyplot as plt
import os


def load_metrics(output_folder, dataset_name):
    """
    Load saved metrics for a specific dataset.

    Args:
        output_folder (str): Path to the folder containing metric files.
        dataset_name (str): Dataset name (e.g., 'top_10_train_top_10_val').

    Returns:
        dict: A dictionary containing train/val loss and accuracy.
    """
    metrics = {}
    try:
        metrics['train_loss'] = np.load(os.path.join(output_folder, f"{dataset_name}_train_loss.npy"))
        metrics['train_acc'] = np.load(os.path.join(output_folder, f"{dataset_name}_train_acc.npy"))
        metrics['val_loss'] = np.load(os.path.join(output_folder, f"{dataset_name}_val_loss.npy"))
        metrics['val_acc'] = np.load(os.path.join(output_folder, f"{dataset_name}_val_acc.npy"))
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Ensure the metric files exist in the specified output folder.")
        return None

    return metrics


def plot_comparison(output_folder, datasets, metric_type, ylabel, title, output_path=None):
    """
    Plot comparison of a specific metric across multiple datasets.

    Args:
        output_folder (str): Path to the folder containing metric files.
        datasets (list): List of dataset names to compare.
        metric_type (str): Metric to plot ('train_loss', 'val_loss', 'train_acc', 'val_acc').
        ylabel (str): Label for the y-axis.
        title (str): Title of the plot.
        output_path (str): Path to save the plot (optional).
    """
    plt.figure(figsize=(12, 8))
    for dataset in datasets:
        metrics = load_metrics(output_folder, dataset)
        if metrics is None:
            continue
        plt.plot(metrics[metric_type], label=dataset)

    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()

    if output_path:
        plt.savefig(os.path.join(output_path, f"{metric_type}_comparison.png"))
    plt.show()


if __name__ == "__main__":
    # Path to the output folder containing metric files
    output_folder = "C:\\work\\few_shot_clothing_detection\\models\\prototypical\\output"

    # Define datasets to compare
    datasets = [
        "top_10_train_top_10_val",
        "top_100_train_top_100_val",
        "all_data_train_all_data_val"
    ]

    # Create plots for comparison
    plot_comparison(output_folder, datasets, 'train_loss', 'Loss', 'Training Loss Comparison')
    plot_comparison(output_folder, datasets, 'val_loss', 'Loss', 'Validation Loss Comparison')
    plot_comparison(output_folder, datasets, 'train_acc', 'Accuracy', 'Training Accuracy Comparison')
    plot_comparison(output_folder, datasets, 'val_acc', 'Accuracy', 'Validation Accuracy Comparison')
