from PIL import Image
import streamlit as st
import requests
import warnings
import os
from faster_whisper import WhisperModel
from dotenv import load_dotenv

from src.utils.gui_utils import (
    render_chat_history,
    set_background_image,
    reset_inputs,
    set_title,
)

from src.constants import (
    API_ENDPOINT,
    IMAGE_QUALITY_OPTIONS,
    LANGUAGE_DICT
)
from src.utils.voice_utils import autoplay_audio, infer_resolution_from_prompt, text_to_speech_gtts, transcribe_STT, translator


warnings.filterwarnings("ignore")  # Ignore warnings
load_dotenv() 


# ------ GUI Design ------

# set background image and title
set_background_image((800, 600))
set_title()

url = os.getenv("BACKEND_URL") + API_ENDPOINT


# ____________SESSION STATE INITIALISATION________________

# Initialise session variable


dict_init_session_var = {  
                         "uploaded_image": None, 
                         "audio_input": None, 
                         "upload_counter": 0, 
                         "chat_history":[],
                         "image_context": None}

for key, value in dict_init_session_var.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.toggle(
    "Toggle button for automatic voice responses or via Screen Reader ",
    key="tts_enabled",
    value=True,
)


# SelectionBox: image input
st.session_state.uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], key=f"uploaded_image_{st.session_state.upload_counter}")

# Preview image
if st.session_state.uploaded_image is not None:
    st.image(
            Image.open(st.session_state.uploaded_image),
            caption="Uploaded image",
            use_container_width=True
        )

# Record voice input
audio_input = st.audio_input("Record your question", key=f"audio_input_{st.session_state.upload_counter}")


# _____FASTER-WHISPER MODEL: Speech Detection: Speech to Text____
@st.cache_resource
def load_whisper_model():
    return WhisperModel(model_size_or_path="small", device="cpu", compute_type="int8")

whisper_model = load_whisper_model()
payload = {}

if audio_input:
    st.audio(audio_input)
    
    # Speech to Text___
    user_prompt, language_code, language_probability = transcribe_STT(whisper_model, audio_input.getvalue())
    st.write(f"**You asked**: {user_prompt}")
    language_of_response = LANGUAGE_DICT.get(language_code, "English")
    st.write(f"Detected language: {language_of_response}")
    
    # ___Infer resolution from keywords___
    if language_code != 'en':
        user_prompt = translator(user_prompt, 'language_code', 'en')
    
    image_resolution_type = infer_resolution_from_prompt(user_prompt)
    
    st.caption(f"📷 Image resolution set to: {image_resolution_type}")

    # ____Check image uploaded, speech detection___ 
    if not st.session_state.uploaded_image:
        st.markdown("""
            <div role="alert" aria-live="assertive">
                    ⚠️ Please upload an image first!
            </div>
            """, unsafe_allow_html=True)
        st.markdown("""
            <div role="alert" aria-live="assertive">
            ⚠️ Could not detect speech, please try again!
            </div>
            """, unsafe_allow_html=True)
    else:
        # Send image if first time
        if st.session_state.image_context is None:
            # First question + image
            payload = {
                "image_resolution_type": image_resolution_type,
                "user_prompt": user_prompt,
                "language_of_response": language_of_response,
                "chat_history": [],
                "send_image": True
                }
            files = {"file": st.session_state.uploaded_image.getvalue()}
        else:
            # Follow up question (No image is needed to send)
            payload = {
                "image_resolution_type": image_resolution_type,
                "user_prompt": user_prompt,
                "language_of_response": language_of_response,
                "chat_history" : st.session_state.chat_history,
                "send_image": False
            }
            files = {} # no image

        # ____Request to the FastAPI endpoint____
        
        st.markdown("""
            <div role="status" aria-live="polite" aria-label="Getting AI response, please wait">
            </div>
            """, unsafe_allow_html=True)
        
        with st.spinner("Getting AI response..."):
            try:
                response = requests.post(url, data=payload, files=files)
                
                if response.status_code == 200: # if successful
                    result = response.json()
                    final_response = result["ai_response"]
                    
                    # store image context after successful response
                    if st.session_state.image_context is None:
                        st.session_state.image_context = result.get("image_context")

                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": user_prompt
                    })

                    st.session_state.chat_history.append({
                    "role": "llm_ai",
                    "content": final_response
                    })

                    st.write("**Response:**")
                    st.markdown(f"""
                                <div 
                                    aria-live="polite"
                                    aria-atomic="true"
                                    tabindex="0"
                                    role="region"
                                    aria-label="AI Response"
                                    style="margin: 8px 0"
                                >
                                    {final_response}
                                </div>
                                """,unsafe_allow_html=True)
            except requests.exceptions.Timeout:
                if st.session_state.tts_enabled:
                    unexpected_response_bytes = text_to_speech_gtts("Please take photo again. Server is not reachable", lang_code=language_code
                    )
                    autoplay_audio(unexpected_response_bytes)

        if st.session_state.tts_enabled:
            audio_bytes = text_to_speech_gtts(final_response, lang_code=language_code)
            autoplay_audio(audio_bytes)

    render_chat_history()


# __________Reset Button____________
st.button(
    "📸 Take a New Photo",
    on_click=reset_inputs,
    use_container_width=True
)


# _______Accessibility________
# MutationObserver watches: every DOM change --> labels reapplied immediately 

st.markdown("""
    <script>
        function applyAriaLabels() {

            // Toggle button
            const toggle = document.querySelector('input[type="checkbox"][role="switch"]');
            if (toggle) {
                toggle.setAttribute('aria-label', 'Toggle auto-play voice response. Turn off if using a screen reader like VoiceOver or NVDA');
            }

            // File uploader
            const fileInput = document.querySelector('input[type="file"]');
            if (fileInput) {
                fileInput.setAttribute('aria-label', 'Upload an image in jpg, jpeg or png format');
            }

            // Audio record button
            const audioBtn = document.querySelector('button[data-testid="stAudioInputRecordButton"]');
            if (audioBtn) {
                audioBtn.setAttribute('aria-label', 'Press to start recording your question');
            }

            // Reset button
            const resetBtn = document.querySelector('div.stButton > button');
            if (resetBtn) {
                resetBtn.setAttribute('aria-label', 'Take a new photo');
            }
        }

        // Apply once immediately on load
        window.addEventListener('load', applyAriaLabels);

        // Then watch for ANY DOM change and reapply
        // This handles every Streamlit rerender
        const observer = new MutationObserver(function(mutations) {
            applyAriaLabels();
        });

        observer.observe(document.body, {
            childList: true,      // watch for added/removed elements
            subtree: true,        // watch all children deeply
        });
    </script>
""", unsafe_allow_html=True)