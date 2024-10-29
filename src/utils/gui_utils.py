import base64
import streamlit as st
from PIL import Image
from io import BytesIO
from gtts import gTTS


from src.constants import (
    INPUT_BG_IMAGE_PATH,
    LANGUAGE_OPTIONS,
    OUTPUT_BG_IMAGE_PATH,
)
from src.models.llm.llm import get_response


def set_background(file_path: str) -> None:
    """set a background image

    Args:
        file_path (str): The path to the background image

    Returns: None
    """

    # Read the file as bytes
    with open(file_path, "rb") as file:
        bytes_data = file.read()
    # Convert to base64
    encoded_image = base64.b64encode(bytes_data).decode("utf-8")
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_image}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
        height: 100vh;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)


def resize_background_image(
    size: tuple,
    input_path: str = INPUT_BG_IMAGE_PATH,
    output_path: str = OUTPUT_BG_IMAGE_PATH,
    quality: int = 85,
) -> None:
    """Resize the background image to increase app loading time

    Args:
        size (tuple): The size of the image
        input_path (str, optional): Path of background image to be resized. Defaults to INPUT_BG_IMAGE_PATH.
        output_path (str, optional): Path of resized background image. Defaults to OUTPUT_BG_IMAGE_PATH.
        quality (int, optional):
    """

    image = Image.open(input_path)
    image.thumbnail(size)
    image.save(output_path, quality=quality, optimize=True)


def set_background_image(size: tuple) -> None:
    """set the title and background of the app

    Args:
        size (tuple): The size of the output background image
    """
    # background image
    resize_background_image(size)
    set_background(OUTPUT_BG_IMAGE_PATH)


# CSS to change the font for a subtitle
st.markdown(
    """
    <style>
    .custom-gabriola {
        font-family: 'Gabriola', serif; 
        font-size: 16px; 
        margin-top: -20px;
        
    </style>
    """,
    unsafe_allow_html=True,
)


def set_title() -> None:
    """set title of the app"""
    # Add multiple line breaks for spacing
    st.markdown("<br>", unsafe_allow_html=True)
    st.title("Freiheit 🦋")
    st.markdown(
        '<p class="custom-gabriola">Your personal visual assistant',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)


def reset_inputs(session_state: any) -> None:
    """Reset session state of streamlit GUI

    Args:
        session_state (any): current session state

    Returns: None
    """
    session_state.uploaded_image = None
    session_state.user_prompt = ""
    # st.rerun()


def handle_submit_button(
    session_state: dict,
    user_prompt: str,
    llm: object,
    b64_image: str,
    subtask_type: str,
    language_of_response: str,
) -> str:
    """Handle the submit button action and returns
    the response from the AI model

    Args:
        session_state (dict): The session state of the streamlit app
        user_prompt (str): The user prompt
        llm (object): The language model
        b64_image (str): The base64 encoded image
        subtask_type (str): The subtask type
        language_of_response (str): The language of the response

    Returns:
        str: The response from the AI model
    """

    if session_state.uploaded_image is None:
        st.text("Please upload an image or take a picture")
    elif not user_prompt:
        st.warning("Please enter a prompt or comment before submitting")
    else:
        # language_code = LANGUAGE_OPTIONS[language_of_response]
        response = get_response(
            llm, b64_image, subtask_type, user_prompt, language_of_response
        )
    return response


def convert_text_to_speech(response: str, language_code: str) -> None:
    """Convert the text response to speech

    Args:
        response (str): The response from the AI model
        language_code (str): The language code of the response

    return: None
    """
    tts = gTTS(response, lang=language_code)
    audio_bytes = BytesIO()
    tts.write_to_fp(audio_bytes)
    st.audio(audio_bytes.getvalue(), format="audio/mp3")
    st.text(f"Response:{response}")


def save_mp3_for_options(text, sel):
    """Converts text to speech and saves it as an mp3 file

    Args:
        text (str): The text to be converted to speech
    """
    tts = gTTS(text=text, lang="en")
    filename = "task.mp3"
    tts.save(filename)
