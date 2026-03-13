import streamlit as st
import requests
import warnings

from src.utils.gui_utils import (
    set_background_image,
    handle_submit_button,
    convert_text_to_speech,
    reset_inputs,
    set_title,
)

from src.utils.image_processing_utils import preview_uploaded_image

from src.constants import (
    TASK_TYPES,
    IMAGE_INPUT_TYPE,
    TASK_SUBTYPES_LLM,
    LANGUAGE_OPTIONS,
)

warnings.filterwarnings("ignore")  # Ignore warnings



# ------ GUI Design ------

# set background image and title
set_background_image((800, 600))
set_title()

# Initialise session variable
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

if "user_prompt" not in st.session_state:
    st.session_state.user_prompt = ""


# SelectionBox: Choose the task
task_type = st.selectbox(
    "Choose your type of task", options=TASK_TYPES, key="task_type"
)


# SelectionBox: Choose the subtask
subtask_type = st.selectbox(
    "Choose your subtask", options=TASK_SUBTYPES_LLM[task_type], key="subtask_type"
)


# Choose quality of image
image_resolution_type = st.selectbox(
    "Choose the quality of Image", ["High", "Medium", "Low", "Very Low"]
)


# SelectionBox: Tabs for image input
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image:
    files = {"file": (uploaded_image.getvalue())}
    requests.post("http://localhost:8000/upload", files=files)

# Text Input box: Question or user prompt
user_prompt = st.text_input(
    "How can I help you with this image?", st.session_state.user_prompt
)


# SelectionBox: Choose the language of the Answer
language_of_response = st.selectbox(
    "Choose the language of the answer", list(LANGUAGE_OPTIONS.keys())
)


# Button: Submit button
submit_button = st.button(label="Submit")

payload = {
            "subtask_type": subtask_type,
            "image_resolution_type": image_resolution_type,
            "user_prompt": user_prompt,
            "language_of_response": language_of_response
            }


# Handle form submission
if submit_button:
    if uploaded_image and user_prompt:
        # Prepare the files and data
        files = {"image": uploaded_image.getvalue()}
        

        payload = {
            "subtask_type": subtask_type,
            "image_resolution_type": image_resolution_type,
            "user_prompt": user_prompt,
            "language_of_response": language_of_response
            }
        
        
       
        
       


# Button: Reset session states
if st.button("Reset"):
    reset_inputs(st.session_state)