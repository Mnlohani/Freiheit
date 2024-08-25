import os
import logging
import numpy as np

import torch
import torch.nn as nn

from src.models.distanceNN.distanceNN import DistanceNN


def save_testset_results(
    trained_model: str,
    test_dataloader: torch.utils.data.DataLoader,
    prediction_npz_file_name: str,
    criterion: nn.Module,
) -> str:
    """
    Evaluate the trained model on the test set and save the results to a file.

    Args:
        trained_model (str): path to the saved trained model
        test_dataloader (torch.utils.data.DataLoader): test dataloader
        prediction_npz_file_name (str): name to the saved test results npz file with .npz extension (e.g. "test_results.npz")
        criterion (_type_, optional): loss function to calculate the loss (e.g. nn.MSELoss)
    Returns:
        test_results_npz_file_path (str): path to the saved test results npz file
    """
    # Lists to store distances and model outputs
    logging.info("Evaluating the model on the test set...")
    test_distances = []
    test_outputs = []

    best_model = DistanceNN()
    best_model.load_state_dict(torch.load(trained_model))
    best_model.eval()

    test_loss = 0.0
    with torch.no_grad():
        for embeddings, distances in test_dataloader:
            outputs = best_model(embeddings)
            loss = criterion(outputs.squeeze(), distances)
            test_loss += loss.item()

            # Append distances and outputs
            test_distances.extend(
                distances.cpu().numpy()
            )  # Convert tensor to numpy array and extend
            test_outputs.extend(
                outputs.squeeze().cpu().numpy()
            )  # Convert tensor to numpy array and extend

        test_loss /= len(test_dataloader)

    logging.info(f"Test set Loss: {test_loss:.4f}")

    # save distances and outputs to a file for analysing further
    prediction_npz_file_path = os.path.join(
        "data/03_models/distanceNN/", prediction_npz_file_name
    )
    np.savez(
        prediction_npz_file_path,
        distances=np.array(test_distances),
        outputs=np.array(test_outputs),
    )
    logging.info(f"Test results saved to {prediction_npz_file_path}")
    return prediction_npz_file_path
