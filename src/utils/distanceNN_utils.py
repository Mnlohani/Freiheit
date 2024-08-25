import streamlit as st
import torch.nn as nn
import torch
import PIL

from transformers import AutoImageProcessor, Dinov2Model

from src.models.distanceNN.distanceNN import DistanceNN
from src.constants import DINO_MODEL_PATH, EMBEDDING_DIM, IMAGE_PROCESSOR_PATH


# Initialize the image processor and model
image_processor = AutoImageProcessor.from_pretrained(IMAGE_PROCESSOR_PATH)
dino_model = Dinov2Model.from_pretrained(DINO_MODEL_PATH)


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
        image (str, PIL.Image.Image): The image file path or a PIL image

    Returns:
        return embedding_DINO: The embedding of image from DinoV2 model
    """
    # Preprocess a single image

    if isinstance(image, PIL.Image.Image):
        image = image.convert("RGB")
    elif isinstance(image, str):
        image = PIL.Image.open(image).convert("RGB")
    else:
        raise ValueError("The image must be a file path, or  a PIL image")

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
