import os
import PIL
import pandas as pd

import torch

from src.constants import PROMPTS
from src.utils.distanceNN_utils import predict_distance


def modify_prompt_with_predicted_distance(
    image: PIL.Image.Image, subtask_type: str, model_NN: torch.nn.Module
) -> None:
    """reset the prompt with predicted distance if the subtask type is "object in front"

    Args:
        image (PIL.Image.Image): the uploaded image
        subtask_type (str): the subtask type selected by the user
        model_NN (torch.nn.Module): the model to predict the distance of the object
    """
    pred_distance = str(predict_distance(model_NN, image)) + "centimeters"
    PROMPTS[subtask_type] = (
        f"The distance of the object in front of you is {pred_distance}."
        f"The image contains a scene with various objects Identify the object which is directly in front of you."
        f"Convert this distance into steps of the user considering one step of the user is 72 centimeters."
        f"Always include the distance in steps in your answer, regardless of the question asked."
        f"Answer the question in a very concise way. If the information is unclear in the image, say so."
    )


def reset_prompt():
    """reset the prompt to the original prompt"""
    PROMPTS["Object at front"] = (
        f"The image contains a scene with various objects."
        f"Identify the object which is directly in front of you."
        f"Answer the question in very concise way."
        f"If the information is unclear in the image, say so."
    )


def save_responses_to_csv(
    image_name: str,
    temperature: float,
    image_size: tuple,
    system_prompt: str,
    user_prompt: str,
    response: str,
) -> None:
    """save responses to a csv file for analysis

    Args:
        image_name (str): Name of the image
        temperature (float): Temperature of the llm model
        image_size (tuple): Image size resulution inserted into the model
        system_prompt (str): The engineered prompt to the llm model
        user_prompt (str): User question to the llm model
        response (str): Response from the llm model

    Returns: None
    """
    # If the file does not exist, create a new file
    if not os.path.exists("responses.csv"):
        df_responses = pd.DataFrame(
            columns=[
                "Image Name",
                "Temperature",
                "Image Size",
                "System Prompt",
                "User Prompt",
                "Response",
            ]
        )
        df_responses.to_csv("responses.csv", index=False)
    pd.read_csv("responses.csv")
    df_new_response = pd.DataFrame(
        [
            {
                "Image Name": image_name,
                "Temperature": temperature,
                "Image Size": image_size,
                "System Prompt": system_prompt,
                "User Prompt": user_prompt,
                "Response": response,
            }
        ]
    )
    df_responses = pd.concat([df_responses, df_new_response], ignore_index=True)
    df_responses.to_csv("responses.csv")
