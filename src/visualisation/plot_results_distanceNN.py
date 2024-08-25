import os
import logging
import numpy as np
import matplotlib.pyplot as plt


def plot_train_val_loss(trn_val_loss_npz_file_path: str) -> None:
    """Saves the plot of the training and validation losses from the saved npz file.

    Args:
        trn_val_loss_npz_file_path (str): path to the saved npz file containing the training and validation losses
    """
    logging.info("Plotting training and validation losses...")
    data = np.load(trn_val_loss_npz_file_path)
    train_losses = data["train_losses"]
    val_losses = data["val_losses"]
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Losses")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/distanceNN/01_Train_val_loss_plot.jpeg")
    if os.path.exists("results/distanceNN/01_Train_val_loss_plot.jpeg"):
        logging.info(
            "Plot saved successfully to results/distanceNN/01_Train_val_loss_plot.jpeg."
        )
    else:
        logging.error("Error in saving the training and validation loss plot!")


def plot_evaluation(prediction_npz_file_path: str) -> None:
    """Save the plot of the actual distances vs predicted outputs in test set and the percentage error between them.

    Args:
        prediction_npz_file_path (str): path to the saved npz file containing the distances and output predictions
    """
    logging.info("Plotting evaluation results...")
    data = np.load(prediction_npz_file_path)
    test_distances = data["distances"]
    test_outputs = data["outputs"]
    percentage_error = (test_outputs - test_distances) * 100 / test_distances

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Scatter plot of actual distances vs predicted outputs
    ax1.scatter(test_distances, test_outputs, color="blue", alpha=0.5)
    ax1.plot(
        test_distances, test_distances, color="red", linestyle="--"
    )  # Plotting y = x line for reference
    ax1.set_title("Actual Distances vs Predicted Outputs")
    ax1.set_xlabel("Actual Distances")
    ax1.set_ylabel("Predicted Outputs")
    ax1.grid(False)

    # Plot 2: Scatter plot of differences between predicted outputs and actual distances
    ax2.bar(test_distances, percentage_error, color="blue")
    ax2.axhline(y=0, color="black", linestyle="--")  # Plotting y = 0 line for reference
    ax2.set_title("% Error between Predicted Outputs and Actual Distances")
    ax2.set_xlabel("Actual Distances")
    ax2.set_ylabel(" % Error(Predicted - Actual) ")
    ax2.grid(False)

    # Adjust layout and display plot
    # Improve plot aesthetics for both axes
    for ax in [ax1, ax2]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("black")
        ax.spines["bottom"].set_color("black")

    plt.tight_layout()
    fig.savefig("results/distanceNN/02_Evaluation_plot.jpeg")
    if os.path.exists("results/distanceNN/02_Evaluation_plot.jpeg"):
        logging.info(
            "Plot saved successfully to results/distanceNN/02_Evaluation_plot.jpeg."
        )
    else:
        logging.error("Error in saving the first evaluation plot!")


def plot_percentage_err(prediction_npz_file_path: str) -> None:
    """Loads test set prediction npz file and saves the plot of the mean percentage error in different distance ranges

    Args:
        prediction_npz_file_path (str): path to the saved npz file containing the distances and output predictions
    """
    logging.info("Plotting mean percentage error in different distance ranges...")
    # Load distances and outputs from the saved npz file
    data = np.load(prediction_npz_file_path)
    test_distances = data["distances"]
    test_outputs = data["outputs"]

    # Calculate differences  percentage
    percentage_error = (test_outputs - test_distances) * 100 / test_distances

    # Define distance ranges
    ranges = [(40, 100), (100, 200), (200, 300), (300, 400)]
    mean_errors = []

    # Calculate mean percentage error for each range
    for lower, upper in ranges:
        mask = (test_distances >= lower) & (test_distances <= upper)
        mean_error = np.round(np.mean(np.abs(percentage_error[mask])), 1)
        mean_errors.append(mean_error)

    # Range labels
    range_labels = ["40-100", "100-200", "200-300", "300-400"]
    colors = ["#ffbb78", "#aec7e8", "#ffbb78", "#aec7e8"]

    # Create the bar plot for mean errors in ranges
    plt.figure(figsize=(8, 5))
    bars = plt.bar(range_labels, mean_errors, color=colors, width=0.4)

    # Add rounded text labels inside the bars
    for bar, error in zip(bars, mean_errors):
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval - yval * 0.1,
            f"{error:.1f}",
            ha="center",
            va="center",
            color="black",
        )

    # Add a horizontal line at y=0
    plt.axhline(0, color="black", linewidth=1)

    # Set titles and labels
    plt.title("Mean % Absolute Error in Test Set")
    plt.xlabel("Distance Ranges")
    plt.ylabel("Mean % Absolute Error")

    # Improve plot aesthetics
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["left"].set_color("black")
    plt.gca().spines["bottom"].set_color("black")

    # Set the size of the figure if necessary
    plt.gcf().set_size_inches(6, 4)

    # save plot
    plt.savefig("results/distanceNN/03_Evaluation_plot_with_range.jpeg")
    if os.path.exists("results/distanceNN/03_Evaluation_plot_with_range.jpeg"):
        logging.info(
            "Plot saved successfully to results/distanceNN/03_Evaluation_plot_with_range.jpeg."
        )
    else:
        logging.error("Error in saving the second evaluation plot!")
