import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from src.constants import HAUPT_PROMPT, LLM_MAX_TOKENS, LLM_TEMPERATURE, PROMPTS


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
    return messages


def get_response(
    model: object,
    b64image: str,
    user_prompt: str,
    language_of_response: str = "English",
):
    """Get the response from the AI model

    Args:
        model (object): The llm model
        b64image (str): the base64 encoded image
        user_prompt (str): Question asked by the user
        language_of_response (str): Language of the response

    Returns:
        str: Response from the AI model
    """
    message_to_model = construct_message(
        b64image, user_prompt, language_of_response
    )
    ai_msg = model.invoke(message_to_model)
    return ai_msg.content