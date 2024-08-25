import streamlit as st
import PIL.Image as Image
import warnings

from src.utils.distanceNN_utils import load_DistanceNN
from src.models.llm.llm import load_llm_model
from src.utils.gui_utils import (
    set_title,
    set_background_image,
    handle_submit_button,
    convert_text_to_speech,
    reset_inputs,
)
from src.utils.gui_utils import convert_text_to_speech


from src.constants import (
    SAVED_MODEL_PATH,
    TASK_TYPES,
    IMAGE_INPUT_TYPE,
    TASK_SUBTYPES,
    LANGUAGE_OPTIONS,
)
from src.utils.image_processing_utils import preview_uploaded_image
from src.utils.llm_utils import modify_prompt_with_predicted_distance, reset_prompt


warnings.filterwarnings("ignore")  # Ignore warnings

# ------ Load model ------

llm = load_llm_model(model="gpt-4o")
model_NN = load_DistanceNN(SAVED_MODEL_PATH)
# predicted distance of object
pred_distance = 0


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
    "Choose your subtask", options=TASK_SUBTYPES[task_type], key="subtask_type"
)


# Choose quality of image
image_resolution_type = st.selectbox(
    "Choose the quality of Image", ["High", "Medium", "Low", "Very Low"]
)


# SelectionBox: Tabs for image input
tab1, tab2 = st.tabs(IMAGE_INPUT_TYPE)
with tab1:
    image_input = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if image_input:
        image = Image.open(image_input)
        b64_image, resized_image, image_name = preview_uploaded_image(
            image_input, image_resolution_type
        )
        if subtask_type == "Include distance with NN":
            modify_prompt_with_predicted_distance(image, subtask_type, model_NN)


with tab2:
    image_input = st.camera_input("Take a picture")
    if image_input:
        image = Image.open(image_input)
        b64_image, resized_image, image_name = preview_uploaded_image(
            image_input, image_resolution_type
        )
        if subtask_type == "Include distance with NN":
            modify_prompt_with_predicted_distance(image, subtask_type, model_NN)


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


# Handle form submission
if submit_button:
    response = handle_submit_button(
        st.session_state,
        user_prompt,
        llm,
        b64_image,
        subtask_type,
        language_of_response,
    )
    # Display the response and convert it to speech
    convert_text_to_speech(response, LANGUAGE_OPTIONS[language_of_response])
    # Reset the prompt if subtask type is "Include distance with NN"
    if subtask_type == "Include distance with NN":
        reset_prompt()


# Button: Reset session states
if st.button("Reset"):
    reset_inputs(st.session_state)
