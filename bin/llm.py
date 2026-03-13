import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from src.constants import HAUPT_PROMPT, LLM_MAX_TOKENS, LLM_TEMPERATURE, PROMPTS
from src.utils.voice_utils import play_audio, text_to_audio


def load_llm_model(model: str) -> object:
    """load the large language model

    Args:
        model_name (str): The name of the model to load. Supported values are "gpt-4o" and "llava".

    Raises:
        Exception: _description_

    Returns:
        object: An instance of the loaded language model
    """
    # load  environment variables
    load_dotenv()
    if model == "gpt-4o":
        llm = ChatOpenAI(
            model="gpt-4o",
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            streaming=True, 
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
    elif model == "gemini-2.5-flash":
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=LLM_TEMPERATURE,
            max_output_tokens=LLM_MAX_TOKENS,
            streaming=True,
            api_key=os.environ.get("Gemini_API_KEY")
        )
    elif model == "llava":
        llm = ChatOllama(model="llava")
    else:
        raise Exception("currently supported Models: Gemini, chatgpt and llava")
    return llm


def construct_message(
    b64image: str,
    human_message: str,
    language_of_response: str
) -> str:
    """
    Constructs a message to be sent to the AI model
    Args:
        b64image (str): the base64 encoded image
        human_message (str): Question asked by the user
        language_of_response (str): Language of the response
        image_format (str): format of the image eg. png, jpeg

    Returns:
        str: messages to be sent to llm model
    """
    prompt = (
        "The user wants the answer in "
        + str(language_of_response)
        + " language. "
        + HAUPT_PROMPT 
        + PROMPTS
    )
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(
            content=[
                {"type": "text", "text": human_message},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64image}",
                        "detail": "auto",
                    },
                },
            ]
        ),
    ]
    print(prompt)
    return messages


def get_response(
    model: object,
    b64image: str,
    user_prompt: str,
    language_of_response: str = "English",
    language_code: str = "en",
    speak_response: bool = True
):
    """Get the response from the AI model

    Args:
        model (object): The llm model
        b64image (str): the base64 encoded image
        prompt (str): system message to the AI model Aka Prompt
        user_prompt (str): Question asked by the user
        distance (int): distance of the object from the user
        language_of_response (str): Language of the response
        speak_response (bool): Whether to speak the response aloud

    Returns:
        str: Full Response from the AI model
    """
    message_to_model = construct_message(
        b64image,user_prompt, language_of_response
    )
    
    buffer = ""     # Accumulates chunk untill sentence is complete
    full_response = ""    # accumulates the entire response

    print("Assistant: ", end="", flush=True)

    for chunk in model.stream(message_to_model):
        text_of_chunk = chunk.content

        if not text_of_chunk:
            continue

        print(text_of_chunk, end="", flush=True)

        buffer += text_of_chunk
        full_response += text_of_chunk

        # when sentence is complete → speak it immediately
        if any(punct in buffer for punct in [".", "!", "?", ","]):
            if buffer.strip() and speak_response:
                try:
                    audio = text_to_audio(buffer.strip(), lang=language_code)
                    play_audio(audio)
                except Exception as e:
                    print(f"\nTTS error: {e}")
            buffer = ""     # reset buffer for next sentenc
            
    # speak any remaining text that didn't end with punctuation
    # example: If LLM responds "The image shows a busy street. There is a red bus"". The  
    #if buffer.strip() and speak_response:
    #    try:
    #        audio = text_to_audio(buffer.strip(), lang=language_code)
    #        play_audio(audio)
    #    except Exception as e:
    #        print(f"\nTTS error: {e}")

    #    print()  # new line after response

    return full_response
    
