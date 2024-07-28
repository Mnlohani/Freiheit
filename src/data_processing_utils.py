import base64
import io
import torch
import PIL
from PIL import Image, ExifTags
import streamlit as st

from constants import IMAGE_RESOLUTION, PROMPTS
from src.distanceNN_utils import predict_distance


def resize_and_getBase64(image, max_size=None):
    """corrects orientation of uploaded/ camera image, resize it
    to desired resolution and returns the base64 encoded image.

    Args:
        image (_type_): uploaded image
        max_size (tuple, optional): if max_size is provided, resize the image
                                    otherwise use the original size Defaults to None.

    Returns:
        tuple: (b64_image, resized_image, image.name) :
        Corresponding base64 image, resized image and the image name
    """
    original_image = Image.open(image)

    # Correct the orientation if necessary
    try:
        exif = original_image._getexif()
        if exif is not None:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == "Orientation":
                    break
            exif = dict(exif.items())
            orientation = exif.get(orientation)
            if orientation == 3:
                original_image = original_image.rotate(180, expand=True)
            elif orientation == 6:
                original_image = original_image.rotate(270, expand=True)
            elif orientation == 8:
                original_image = original_image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        # Cases: image doesn't have getexif
        pass

    resized_image = original_image.copy()
    if max_size:
        # if max_size is provided, resize the image otherwise use the original size
        resized_image.thumbnail(max_size, Image.LANCZOS)

    # Convert image to RGB if it's in RGBA mode
    if resized_image.mode == "RGBA":
        resized_image = resized_image.convert("RGB")

    buffered = io.BytesIO()
    resized_image.save(buffered, format="JPEG")
    b64_image = base64.b64encode(buffered.getvalue()).decode()
    return b64_image, resized_image, image.name


def preview_uploaded_image(image_input: any, image_resolution_type: tuple) -> tuple:
    """preview the uploaded/camera image

    Args:
        image_input (uploadedFile): the uploaded image file
        image_resolution_type (str): resolution type of the image to be feeded into LLM
                                    given by user input in selection box
                                    (High, Medium, Low, Very Low)

    Returns:
        b64_image (str): Base64 image
        resized_image (PIL.Image.Image): the resized image
        image_name (str): the name of the image
    """
    image = Image.open(image_input)
    b64_image, resized_image, image_name = resize_and_getBase64(
        image_input, IMAGE_RESOLUTION[image_resolution_type]
    )
    # preview Image
    st.session_state.uploaded_image = resized_image
    st.image(
        st.session_state.uploaded_image,
        caption="Uploaded image",
        use_column_width=True,
    )

    return b64_image, resized_image, image_name


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
