import unittest
import os
import numpy as np

import torch.nn as nn
import torch.optim as optim


from src.constants import DATASET_CSV_FILE_PATH, DATASET_IMG_FILES_PATH
from src.data.preprocessor import Preprocessor
from src.models.distanceNN.distanceNN import DistanceNN
from src.models.distanceNN.train_NN import train_NN
from src.models.distanceNN.run_distanceNN_pipeline import run_distanceNN_pipeline


class TestDistanceNN(unittest.TestCase):

    def test_create_data(self):
        """To check if the embeddings are created from the data and is in correct directory"""
        preprocessor = Preprocessor(
            data_dir=DATASET_IMG_FILES_PATH,
            distances_file_path=DATASET_CSV_FILE_PATH,
        )
        preprocessor.create_data()
        self.assertIsNotNone("data/02_processed/embeddings_and_distances.npz")

    def test_data_read(self):
        """To check if data is embeddings are correctly created or not"""
        proccessed_file_path = "data/02_processed/embeddings_and_distances.npz"
        if os.path.exists(proccessed_file_path):
            data = np.load(proccessed_file_path)
        else:
            self.create_data()
            data = np.load(proccessed_file_path)

        embeddings_np = data["embeddings"]
        number_images = len(
            [f for f in os.listdir(DATASET_IMG_FILES_PATH) if f.endswith(".jpg")]
        )
        self.assertEqual(embeddings_np.shape[1], 768)
        self.assertEqual(embeddings_np.shape[0], number_images)

    def test_datasets_created(self):
        """To test that train, validation and test set are created correctly"""
        preprocessor = Preprocessor(
            data_dir=DATASET_IMG_FILES_PATH,
            distances_file_path=DATASET_CSV_FILE_PATH,
        )
        [trn_embeddings, _, val_embeddings, _, test_embeddings, _] = (
            preprocessor.create_dataset()
        )
        self.assertEqual(trn_embeddings.shape[1], val_embeddings.shape[1])
        self.assertEqual(val_embeddings.shape[1], test_embeddings.shape[1])

    def test_training(self):
        """This test case check the whole ML pipeline creates prediction on test test"""
        data_dir = DATASET_IMG_FILES_PATH
        distances_file_path = DATASET_CSV_FILE_PATH

        # preprocess images to embeddings and create dataloaders
        preprocessor = Preprocessor(data_dir, distances_file_path)
        datasets = preprocessor.create_dataset()
        train_dataloader, val_dataloader, _ = preprocessor.create_dataloader(*datasets)

        # initialise model
        model = DistanceNN()

        trn_val_loss_npz_file_path, saved_model_path = train_NN(
            model,
            train_dataloader,
            val_dataloader,
            trn_val_loss_file_name="trn_val_loss_test.npz",
            model_file_name="best_model_MSE_test.pt",
            criterion=nn.MSELoss(),
            optimizer=optim.Adam(model.parameters(), lr=0.001),
            num_epochs=5000,
            patience=50,
        )
        self.assertTrue(os.path.exists(saved_model_path))
        self.assertTrue(os.path.exists(trn_val_loss_npz_file_path))
        # remove the files created
        if os.path.exists(saved_model_path):
            os.remove(saved_model_path)
        if os.path.exists(trn_val_loss_npz_file_path):
            os.remove(trn_val_loss_npz_file_path)

    def test_ML_pipeline(self):
        """This test case check the whole ML pipeline creates prediction on test test"""
        data_dir = DATASET_IMG_FILES_PATH
        distances_file_path = DATASET_CSV_FILE_PATH

        run_distanceNN_pipeline(
            model=DistanceNN(),
            data_dir=data_dir,
            distances_file_path=distances_file_path,
            training_parameters={
                "criterion": nn.MSELoss(),
                "optimizer": optim.Adam(DistanceNN().parameters(), lr=0.001),
                "num_epochs": 5000,
                "patience": 50,
            },
            trn_val_loss_file_name="trn_val_loss_test.npz",
            model_file_name="best_model_MSE_test.pt",
            prediction_npz_file_name="pred_MLpipeline_test.npz",
        )

        self.assertTrue("data/03_models/distanceNN/pred_MLpipeline_test.npz")
        # remove the files created
        if os.path.exists("data/03_models/distanceNN/pred_MLpipeline_test.npz"):
            os.remove("data/03_models/distanceNN/pred_MLpipeline_test.npz")
        if os.path.exists("data/03_models/distanceNN/best_model_MSE_test.pt"):
            os.remove("data/03_models/distanceNN/best_model_MSE_test.pt")
        if os.path.exists("data/03_models/distanceNN/trn_val_loss_test.npz"):
            os.remove("data/03_models/distanceNN/trn_val_loss_test.npz")


if __name__ == "__main__":
    unittest.main()
