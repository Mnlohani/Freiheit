import os
import logging
import torch
import torch.nn as nn
import numpy as np


def train_NN(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    trn_val_loss_file_name: str,
    model_file_name: str,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    patience: int,
) -> tuple:
    """Train the DistanceNN and evaluate in validation set.

    Args:
        model (torch.nn.Module): an instance of DistanceNN model
        train_dataloader (torch.utils.data.DataLoader): train dataloader
        val_dataloader (torch.utils.data.DataLoader): validation dataloader
        criterion (torch.nn.Module): loss function to calculate the loss (e.g. nn.MSELoss)
        optimizer (torch.optim.Optimizer): optimizer used to update the model's parameters (e.g. optim.Adam)
        num_epochs (int): number of epochs
        patience (int): patience for early stopping
        trn_val_loss_file_name (str): name of the train validation loss file with extension (e.g. "train_val_loss.npz")
        saved_model_name (str): name of the saved model with extension (e.g. "best_model_MSE.pt")

    Returns:
        trn_val_loss_npz_file_path (str): path to the saved train validation loss npz file
        saved_model_path (str): name of the saved model
    """
    logging.info("Training started...")

    logging.info(
        f"Training Parameters:\n"
        f"Number of epochs: {num_epochs}\n"
        f"Patience for Early stopping: {patience}\n"
        f"Loss function: {criterion}\n"
        f"Optimizer: {optimizer}\n"
        f"learning rate: {optimizer.param_groups[0]['lr']}"
    )

    assert model_file_name.endswith(
        ".pt"
    ), "Error: The model name to be saved should contain a .pt extension."

    assert trn_val_loss_file_name.endswith(
        ".npz"
    ), "Error: The train validation loss file should contain a .npz extension."

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    epochs_without_improvement = 0  # A counter for early stopping

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        for embeddings, distances in train_dataloader:
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs.squeeze(), distances)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        epoch_train_loss /= len(train_dataloader)
        train_losses.append(epoch_train_loss)

        # Validation step
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for embeddings, distances in val_dataloader:
                outputs = model(embeddings)
                loss = criterion(outputs.squeeze(), distances)
                epoch_val_loss += loss.item()
        epoch_val_loss /= len(val_dataloader)
        val_losses.append(epoch_val_loss)

        # Print epoch losses
        logging.info(
            f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}"
        )

        # Check for improvement abd save the best model
        saved_model_path = os.path.join("data/03_models/distanceNN/", model_file_name)
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), saved_model_path)
            epochs_without_improvement = 0  # Reset counter
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= patience:
            logging.info(f"Early stopping at epoch {epoch + 1}")
            break
        trn_val_loss_file_path = os.path.join(
            "data/03_models/distanceNN/", trn_val_loss_file_name
        )
    np.savez(
        trn_val_loss_file_path,
        train_losses=np.array(train_losses),
        val_losses=np.array(val_losses),
    )
    logging.info(f"Training complete. Best model saved to {saved_model_path}")
    logging.info(f"Training and validation losses saved to {trn_val_loss_file_path}")

    return trn_val_loss_file_path, saved_model_path
