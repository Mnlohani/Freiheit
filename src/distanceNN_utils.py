import streamlit as st
import torch.nn as nn
import torch
import PIL
from transformers import AutoImageProcessor, Dinov2Model

from constants import EMBEDDING_DIM

image_processor = AutoImageProcessor.from_pretrained("./model/image_processor")
dino_model = Dinov2Model.from_pretrained("./model/dinov2_model")


class DistanceNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DistanceNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


@st.cache_resource
def load_DistanceNN(model_path: str) -> torch.nn.Module:
    """Load model for Distane Prediction

    Args:
        model_path (str): path of the saved model

    Returns:
        model: The distance NN model
    """
    model = DistanceNN(input_dim=EMBEDDING_DIM, output_dim=1)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set model to evaluation mode
    return model


def get_image_embedding_Dino(image):
    """Get embeddings of the image from DinoV2 model

    Args:
        image (PIL.Image.Image): Input image file

    Returns:
        return embedding_DINO: The embedding of image from DinoV2 model
    """
    # Preprocess a single image
    image = image.convert("RGB")
    inputs = image_processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]
    # get embeddings for a single image
    with torch.no_grad():
        outputs = dino_model(pixel_values=pixel_values)
        embedding_Dino = outputs.pooler_output.squeeze()
        return embedding_Dino


def predict_distance(model: torch.nn.Module, input_image: PIL.Image.Image) -> int:
    """Predict distance of the object in Front in the input image.

    Args:
        model (torch.nn.Module): The DistanceNN model loaded from the saved .pt file
        input_image (PIL.Image.Image): Input image

    Returns:
        distance(int): distance of the object
    """
    embedding = get_image_embedding_Dino(input_image)
    with torch.no_grad():
        output = model(embedding)
        distance = round(output.item())
    return distance
