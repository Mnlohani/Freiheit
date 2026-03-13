import base64
import os
import tempfile
from deep_translator import GoogleTranslator
import streamlit.components.v1 as components
from gtts import gTTS
from io import BytesIO
import streamlit as st
from faster_whisper import WhisperModel

from src.constants import DEFAULT_RESOLUTION, RESOLUTION_KEYWORD_MAP

def transcribe_STT(whisper_model: WhisperModel, audio_bytes: bytes) -> tuple:
    """
    Save input audio bytes to a temp file, transcribe with Whisper,
    and clean up the temp file.

    Parameters
    ----------
    whisper_model : WhisperModel
        Loaded faster-whisper model instance.
    audio_bytes : bytes
        Raw audio bytes from st.audio_input().

    Returns
    -------
    transcribed_text : str
        The transcribed text from the audio.
    language_code : str
        Detected 2-letter language code e.g. 'en', 'fi'.
    language_probability : float
        Confidence score of the detected language (0.0 to 1.0).
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_path = tmp_file.name

    st.markdown("""
            <div role="status" aria-live="polite" aria-label="Transcribing your audio, please wait">
            </div>
            """, unsafe_allow_html=True)

    with st.spinner("Transcribing your audio... 🎙️ "):
        try:
            segments, info = whisper_model.transcribe(tmp_path, 
                                                      beam_size=1, 
                                                      vad_filter=True # skip silence
                                                      )
            segments = list(segments)
            transcribed_text = " ".join(segment.text for segment in segments)
            language_code = info.language
            language_probability = info.language_probability
        finally:
            os.unlink(tmp_path)   # always delete temp file even if error
    
    return transcribed_text.strip(), language_code, language_probability


def text_to_speech_gtts(text: str, lang_code: str = "en") -> bytes:
    """
    Convert a given text string into speech audio bytes using Google Text-to-Speech (gTTS).

    This function sends the text to Google's TTS engine and returns
    the resulting audio as raw bytes in MP3 format, stored in memory
    without saving any file to disk.

    Parameters
    ----------
    text : str
        The text you want to convert to speech.
        Example: "The bus comes in 2 minutes"

    lang_code : str, optional
        The language code for the speech output.
        This matches with what Fast-Whisper returns in our code (e.g. 'en', 'fi', 'ar', 'fr').
        Defaults to 'en' (English) if not provided.

    Returns
    -------
    bytes
        Raw MP3 audio bytes that can be passed to autoplay_audio()
        or saved to a .mp3 file.

    Raises
    ------
    gTTSError
        If Google's TTS service is unavailable or blocks the request.
    ValueError
        If an unsupported language code is passed.
    """
    tts = gTTS(text=text, lang=lang_code, slow=False)

    memory_buffer = BytesIO()           # create in-memory buffer instead of saving to disk
    tts.write_to_fp(memory_buffer)      # write mp3 audio into the buffer
    memory_buffer.seek(0)               # rewind to start so .read() gets all the bytes
    return memory_buffer.read()



def autoplay_audio(audio_bytes: bytes, format: str = "audio/mp3"):
            """Embed and autoplay audio in a Streamlit app directly in the browser.
            This function encodes the audio as
            Base64 and injects it into an HTML <audio> tag with autoplay enabled,
            so the browser plays it automatically when the response is ready.
            
            Parameters
            ----------
            audio_bytes : bytes
                    Raw audio bytes in MP3 format.
                    Typically the return value of text_to_speech_gtts() or
                    any other TTS function that returns bytes.

            height : int, optional
                    The height in pixels of the HTML audio player widget.
                    Defaults to 60px which fits a standard audio control bar.
                    Increase if the player appears clipped in your layout.

            Returns
            -------
            None
                This function renders directly into the Streamlit UI.
                It does not return a value
            """

            # encode raw bytes to base64 string so HTML can embed it inline
            audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
            audio_html = f"""
                <audio autoplay controls style="width:100%">
                    <source src="data:{format};base64,{audio_base64}" type="{format}">
            </audio>
            """
            components.html(audio_html, height=60)


def infer_resolution_from_prompt(user_prompt:str):
    """Infer the best image resolution based on keywords in the user prompt.
        Matches the prompt against predefined keyword lists to determine
        if the object is likely close (High) or far (Low).
        
        Parameters:
        -----------
        user_prompt: str
            Transcribed user question from Whisper model
        
        Return:
        -------
        resolution type: str
            Resolution type: 'High', 'Low', or 'Medium'
        """
     
    prompt_lower = user_prompt.lower()

    for resolution, keywords in RESOLUTION_KEYWORD_MAP.items():
         for keyword in keywords:
              if keyword in prompt_lower:
                   return resolution
    return DEFAULT_RESOLUTION
    
    
def translator(text:str, source_lang_code:str, dest_lang_code:str='en') -> str:
     """Transalte the text from one language to another using google translator
     Parameters
     ----------
     text : str
        User prompt in any language.
     source_lang : str
        2-letter language code from Whisper e.g. 'fi', 'ar', 'de'.

     Returns
     -------
     str
         Translated text (default english).
     """
     try:
        translated = GoogleTranslator(source=source_lang_code, target="en").translate(text)
        return translated
     except:
        return text

     