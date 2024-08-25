import base64
import io

from PIL import Image, ExifTags
import streamlit as st

from src.constants import IMAGE_RESOLUTION


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
