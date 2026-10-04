from deep_translator import GoogleTranslator
from src.constants import DEFAULT_RESOLUTION, RESOLUTION_KEYWORD_MAP


def infer_resolution_from_prompt(user_prompt: str):
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


def translator(text: str, source_lang_code: str, dest_lang_code: str = "en") -> str:
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
        translated = GoogleTranslator(
            source=source_lang_code, target=dest_lang_code
        ).translate(text)
        return translated
    except:
        return text
