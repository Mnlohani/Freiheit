import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoImageProcessor, Dinov2Model

from src.constants import DATASET_IMG_FILES_PATH, DINO_MODEL_PATH, IMAGE_PROCESSOR_PATH
from src.utils.distanceNN_utils import get_image_embedding_Dino

# -- for colab --
# image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
# dino_model = Dinov2Model.from_pretrained("facebook/dinov2-base")

# Initialize the image processor and model
image_processor = AutoImageProcessor.from_pretrained(IMAGE_PROCESSOR_PATH)
dino_model = Dinov2Model.from_pretrained(DINO_MODEL_PATH)


# Initialize lists to store embeddings and distances
embeddings_list = []
distances_list = []

# path to dataset
img_files_path = "../data/chair/data_chair/"
DATASET_CSV_FILE_PATH = "../data/chair/chairs_distances.csv"


class Preprocessor:
    """This Preprocessor class will create input embedding for the
    DistanceNN (A NN which predicts distances of the object from camera.
    The images has been generated by taking photo of a perticular object(which is chair)
    at interval of 5cm"""

    def __init__(self, data_dir: str, distances_file_path: str):
        self.data_dir = data_dir
        self.distances_file_path = distances_file_path
        self.embeddings_list = []
        self.distances_list = []

    def create_data(self):
        # Read csv file for lables
        df = pd.read_csv(self.distances_file_path)

        # Loop through each row in the CSV file
        for idx, row in df.iterrows():
            # Construct the full image path
            img_path = os.path.join(DATASET_IMG_FILES_PATH, row["fileName"])

            # Get the image embedding
            embedding = get_image_embedding_Dino(img_path)

            # Store the embedding and distance
            self.embeddings_list.append(embedding)
            self.distances_list.append(row["distance"])

            embeddings_np = np.array(self.embeddings_list)
            distances_np = np.array(self.distances_list)

            # Save embeddings and distances to an .npz file

            np.savez(
                "data/02_processed/embeddings_and_distances.npz",
                embeddings=embeddings_np,
                distances=distances_np,
            )

    def create_dataset(self, test_size=0.25):
        """Read embeddings and distances from dataset npz file and create
        PyTorch Tensor datasets for training and evaluation of DistanceNN

        Args:
            test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.25.

        Returns:
            train_embeddings (np.array): Training embeddings
            train_distances (np.array): Training distances
            val_embeddings (np.array): Validation embeddings
            val_distances (np.array): Validation distances
            test_embeddings (np.array): Test embeddings
            test_distances (np.array): Test distances
        """

        # Check if the embeddings and distances file exists
        if os.path.exists("data/02_processed/embeddings_and_distances.npz"):
            data = np.load("data/02_processed/embeddings_and_distances.npz")
        else:
            self.create_data()
            data = np.load("data/02_processed/embeddings_and_distances.npz")

        embeddings_np = data["embeddings"]
        distances_np = data["distances"]

        # Split the data into train+validation and test sets
        train_val_embeddings, test_embeddings, train_val_distances, test_distances = (
            train_test_split(
                embeddings_np, distances_np, test_size=test_size, random_state=42
            )
        )

        # Further split the training+validation set into training and validation sets
        train_embeddings, val_embeddings, train_distances, val_distances = (
            train_test_split(
                train_val_embeddings,
                train_val_distances,
                test_size=test_size,
                random_state=42,
            )
        )

        return [
            train_embeddings,
            train_distances,
            val_embeddings,
            val_distances,
            test_embeddings,
            test_distances,
        ]

    def create_dataloader(
        self,
        train_embeddings,
        train_distances,
        val_embeddings,
        val_distances,
        test_embeddings,
        test_distances,
    ):
        """Create DataLoader objects for training, validation and test datasets

        Args:
            train_embeddings (np.array): Training embeddings
            train_distances (np.array): Training distances
            val_embeddings (np.array): Validation embeddings
            val_distances (np.array): Validation distances
            test_embeddings (np.array): Test embeddings
            test_distances (np.array): Test distances

        Returns:
            train_dataloader (DataLoader): DataLoader for training dataset
            val_dataloader (DataLoader): DataLoader for validation dataset
            test_dataloader (DataLoader): DataLoader for test dataset
        """

        # Convert embeddings and distances into tensors
        tensors = map(
            torch.tensor,
            [
                train_embeddings,
                train_distances,
                val_embeddings,
                val_distances,
                test_embeddings,
                test_distances,
            ],
        )

        (
            train_embeddings,
            train_distances,
            val_embeddings,
            val_distances,
            test_embeddings,
            test_distances,
        ) = tensors

        # Create Datasets
        train_dataset = TensorDataset(train_embeddings, train_distances.float())
        val_dataset = TensorDataset(val_embeddings, val_distances.float())
        test_dataset = TensorDataset(test_embeddings, test_distances.float())

        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        return train_dataloader, val_dataloader, test_dataloader