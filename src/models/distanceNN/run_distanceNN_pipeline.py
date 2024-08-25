import logging
import torch.nn as nn
import torch.optim as optim

from src.constants import DATASET_CSV_FILE_PATH, DATASET_IMG_FILES_PATH
from src.data.preprocessor import Preprocessor
from src.models.distanceNN.distanceNN import DistanceNN
from src.models.distanceNN.train_NN import train_NN
from src.models.distanceNN.evaluate_NN import save_testset_results
from src.visualisation.plot_results_distanceNN import (
    plot_evaluation,
    plot_percentage_err,
    plot_train_val_loss,
)

# configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/distanceNN_pipeline.log", mode="w"),
        logging.StreamHandler(),
    ],
)


def run_distanceNN_pipeline(
    model,
    data_dir,
    distances_file_path,
    training_parameters,
    trn_val_loss_file_name,
    model_file_name,
    prediction_npz_file_name,
):
    """Pipeline to process data, train, validate and test the DistanceNN model and
    store the visualisation results.

    Args:
        model (torch.nn.Module): an instance of DistanceNN model
        data_dir (str): path to the directory containing the images of the object at front.
        distances_file_path (str): path to the file containing the distances of
        the object at front in the images
        training_parameters (dict): dictionary containing the training parameters
        trn_val_loss_file_name (str): name of the train validation loss file with extension (e.g. "train_val_loss.npz")
        model_file_name (str): name of the model to be saved with extension (e.g. "best_model_MSE.pt")
        prediction_npz_file_name (str): name of the test set prediction file with extension (e.g. "pred_test_set.npz")
    """

    # preprocess images to embeddings and create dataloaders
    preprocessor = Preprocessor(data_dir, distances_file_path)

    datasets = preprocessor.create_dataset()
    train_dataloader, val_dataloader, test_dataloader = preprocessor.create_dataloader(
        *datasets
    )

    # training, evaluation in validation set, and save model as .pt
    trn_val_loss_npz_file_path, saved_model_path = train_NN(
        model,
        train_dataloader,
        val_dataloader,
        trn_val_loss_file_name,
        model_file_name,
        **training_parameters,
    )

    # plot train and validation results
    plot_train_val_loss(trn_val_loss_npz_file_path)

    # test set evaluation with saved model
    criterion = training_parameters["criterion"]
    prediction_npz_file_path = save_testset_results(
        saved_model_path,
        test_dataloader,
        prediction_npz_file_name,
        criterion,
    )

    # plot test set results
    plot_evaluation(prediction_npz_file_path)
    plot_percentage_err(prediction_npz_file_path)


if __name__ == "__main__":
    logging.info("Pipeline execution started.")

    data_dir = DATASET_IMG_FILES_PATH
    distances_file_path = DATASET_CSV_FILE_PATH

    # initialise model
    model = DistanceNN()
    # set training parameters
    training_parameters = {
        "criterion": nn.MSELoss(),
        "optimizer": optim.Adam(model.parameters(), lr=0.001),
        "num_epochs": 5000,
        "patience": 50,
    }

    run_distanceNN_pipeline(
        model,
        data_dir,
        distances_file_path,
        training_parameters,
        trn_val_loss_file_name="trn_val_loss.npz",
        model_file_name="best_model_MSE.pt",
        prediction_npz_file_name="pred_test_set.npz",
    )

    logging.info("Pipeline execution finished.")
