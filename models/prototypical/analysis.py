import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from scipy.stats import gaussian_kde
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns


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
        metrics['train_loss'] = np.load(os.path.join(output_folder, f"{dataset_name}\\train_loss.npy"))
        metrics['train_acc'] = np.load(os.path.join(output_folder, f"{dataset_name}\\train_acc.npy"))
        metrics['val_loss'] = np.load(os.path.join(output_folder, f"{dataset_name}\\val_loss.npy"))
        metrics['val_acc'] = np.load(os.path.join(output_folder, f"{dataset_name}\\val_acc.npy"))
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


def visualize_distance_distribution(model, dataloader, num_support, class_names, device):
    """
    Enhanced function to visualize the distribution of distances between query samples and prototypes.

    Args:
        model: Trained Prototypical Network model.
        dataloader: DataLoader for evaluation.
        num_support: Number of support samples per class.
        class_names: List of class names.
    """
    model.eval()
    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    distances = []
    class_distances = {class_name: [] for class_name in class_names}

    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            embeddings = model(x)
            unique_classes = torch.unique(y)
            prototypes = []
            for c in unique_classes:
                class_indices = torch.nonzero(y == c).squeeze()[:num_support]
                class_embeddings = embeddings[class_indices]
                prototype = class_embeddings.mean(dim=0)
                prototypes.append(prototype)
            prototypes = torch.stack(prototypes)

            # Compute distances between query samples and prototypes
            query_indices = torch.arange(len(y))[(num_support * len(unique_classes)) :]
            query_samples = embeddings[query_indices]
            for query, query_label in zip(query_samples, y[query_indices]):
                for prototype, prototype_class in zip(prototypes, unique_classes):
                    dist = torch.norm(query - prototype).item()
                    distances.append(dist)
                    class_distances[class_names[prototype_class.item()]].append(dist)

    # Plot overall distance distribution
    plt.figure(figsize=(15, 8))
    
    # Plot overall distribution
    distances = np.array(distances)
    density = gaussian_kde(distances)
    x = np.linspace(distances.min(), distances.max(), 1000)
    plt.plot(x, density(x), label="Overall", linewidth=3, color='black', alpha=0.5)
    
    # Plot class-specific distributions with different colors
    colors = plt.cm.rainbow(np.linspace(0, 1, len(class_names)))
    for class_name, dist_list, color in zip(class_distances.keys(), class_distances.values(), colors):
        dist_list = np.array(dist_list)
        density = gaussian_kde(dist_list)
        x = np.linspace(dist_list.min(), dist_list.max(), 1000)
        plt.plot(x, density(x), label=class_name, alpha=0.7, color=color)

    plt.title("Distribution of Distances Between Query Samples and Prototypes")
    plt.xlabel("Distance")
    plt.ylabel("Density")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(model, dataloader, class_names, device, normalize=False):
    """
    Enhanced function to generate and visualize a confusion matrix for the model.

    Args:
        model: Trained Prototypical Network model.
        dataloader: DataLoader for evaluation.
        class_names: List of class names for labeling.
        normalize: Whether to normalize the confusion matrix (row-wise).
    """
    model.eval()
    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            embeddings = model(x)
            unique_classes = torch.unique(y)
            prototypes = []
            for c in unique_classes:
                class_indices = torch.nonzero(y == c).squeeze()
                class_embeddings = embeddings[class_indices]
                prototype = class_embeddings.mean(dim=0)
                prototypes.append(prototype)
            prototypes = torch.stack(prototypes)

            # Predict class for each query sample
            for query, label in zip(embeddings, y):
                distances = torch.norm(query - prototypes, dim=1)
                pred_class = torch.argmin(distances)
                y_pred.append(pred_class.item())
                y_true.append(label.item())

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    # Normalize if required
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title(f" {mode} Confusion Matrix")
    plt.show()


if __name__ == "__main__":
    from protonet import ProtoNet
    from ClothingDataset import ClothingDataset
    from prototypical_batch_sampler import PrototypicalBatchSampler
    from torch.utils.data import DataLoader

    # Load the trained ProtoNet model
    model_path = "C:\\work\\few_shot_clothing_detection\\models\\prototypical\\output\\all_data_train_all_data_val\\best_model.pth"
    trained_model = ProtoNet(x_dim=3, hid_dim=64, z_dim=64)
    trained_model.load_state_dict(torch.load(model_path))
    trained_model.eval()

    # Initialize the test DataLoader
    test_dataset = ClothingDataset(
        mode="all_data", # "top_10", "top_100", "all_data"
        root="C:\\work\\few_shot_clothing_detection\\data\\clean_data"
    )
    test_sampler = PrototypicalBatchSampler(
        labels=test_dataset.targets,
        classes_per_it=5,
        num_samples=10,
        iterations=100
    )
    test_dataloader = DataLoader(test_dataset, batch_sampler=test_sampler)

    # Define class names
    class_names = [
        "Blazer", "Dress", "Hat", "Hoodie", "Longsleeve", "Outwear", 
        "Pants", "Polo", "Shirt", "Shoes", "Shorts", "Skirt", 
        "T-shirt", "Undershirt"
    ]

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    trained_model = trained_model.to(device)
    # Run the analysis functions
    # visualize_distance_distribution(trained_model, test_dataloader, num_support=5, class_names=class_names, device=device)
    plot_confusion_matrix(trained_model, test_dataloader, class_names=class_names, device=device, normalize=True)
