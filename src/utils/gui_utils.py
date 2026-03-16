import base64
import streamlit as st
from PIL import Image

from src.constants import (
    INPUT_BG_IMAGE_PATH,
    OUTPUT_BG_IMAGE_PATH,
    WIDGET_KEYS,
)

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



def reset_inputs() -> None:
    """
    Reset all Streamlit widgets and session state back to their defaults.

    Streamlit widgets are controlled by their session state key.
    Deleting the key forces Streamlit to re-render the widget as fresh/empty.
    st.rerun() is called at the end to immediately reflect the reset in the UI.

    Returns
    -------
    None
    """
    for key in WIDGET_KEYS:
        if key in st.session_state:
            del st.session_state[key]   # deleting keys to reset the widget
        st.session_state.upload_counter = st.session_state.get("upload_counter", 0) + 1
        st.session_state.chat_history = []
        st.session_state.image_context = None
        st.session_state.text_mode = False
        st.session_state.tts_enabled = False
    

    
def render_chat_history() -> None:
    """
    Render scrollable chat history box showing all
    questions and responses in the current session.

    Returns
    -------
    None
    """
    if not st.session_state.chat_history:
        return

    st.markdown("### Conversation")

    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"""
                <div style="
                    background: #e8f4f8;
                    border-radius: 12px;
                    padding: 10px 14px;
                    margin: 6px 0;
                    text-align: right;
                ">
                    User: {message["content"]}
                </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown(f"""
                <div 
                    style="
                        background: #f0f0f0;
                        border-radius: 12px;
                        padding: 10px 14px;
                        margin: 6px 0;
                        text-align: left;
                    "
                    aria-live="polite"
                    tabindex="0"
                >
                    AI: {message["content"]}
                </div>
            """, unsafe_allow_html=True)