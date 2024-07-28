import os
import pandas as pd
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, SystemMessage

from constants import LLM_MAX_TOKENS, LLM_TEMPERATURE, PROMPTS


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
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
    elif model == "llava":
        llm = Ollama(model="llava")
    else:
        raise Exception("currently supported Models: gpt-4o, and llava")
    return llm


def construct_message(
    b64image: str,
    subtask_type: str,
    human_message: str,
    language_of_response: str,
    image_format: str = "jpg",
) -> str:
    """
    Constructs a message to be sent to the AI model
    Args:
        b64image (str): the base64 encoded image
        subtask_type (str): Task sub type choosen by the user
        human_message (str): Question asked by the user
        language_of_response (str): Language of the response
        image_format (str): format of the image eg. png, jpeg

    Returns:
        str: messages to be sent to llm model
    """
    prompt = (
        PROMPTS[subtask_type] + "Answer in" + str(language_of_response) + "language"
    )
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(
            content=[
                {"type": "text", "text": human_message},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{image_format};base64,{b64image}",
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
    subtask_type: str,
    human_message: str,
    language_of_response: str = "English",
):
    """Get the response from the AI model

    Args:
        model (object): The llm model
        b64image (str): the base64 encoded image
        prompt (str): system message to the AI model Aka Prompt
        human_message (str): Question asked by the user
        distance (int): distance of the object from the user
        language_of_response (str): Language of the response

    Returns:
        str: Response from the AI model
    """
    message_to_model = construct_message(
        b64image, subtask_type, human_message, language_of_response
    )
    ai_msg = model.invoke(message_to_model)
    return ai_msg.content


def save_responses_to_csv(
    image_name: str,
    temperature: float,
    image_size: tuple,
    system_prompt: str,
    user_prompt: str,
    response: str,
) -> None:
    """save responses to a csv file for analysis

    Args:
        image_name (str): Name of the image
        temperature (float): Temperature of the llm model
        image_size (tuple): Image size resulution inserted into the model
        system_prompt (str): The engineered prompt to the llm model
        user_prompt (str): User question to the llm model
        response (str): Response from the llm model

    Returns: None
    """
    # If the file does not exist, create a new file
    if not os.path.exists("responses.csv"):
        df_responses = pd.DataFrame(
            columns=[
                "Image Name",
                "Temperature",
                "Image Size",
                "System Prompt",
                "User Prompt",
                "Response",
            ]
        )
        df_responses.to_csv("responses.csv", index=False)
    pd.read_csv("responses.csv")
    df_new_response = pd.DataFrame(
        [
            {
                "Image Name": image_name,
                "Temperature": temperature,
                "Image Size": image_size,
                "System Prompt": system_prompt,
                "User Prompt": user_prompt,
                "Response": response,
            }
        ]
    )
    df_responses = pd.concat([df_responses, df_new_response], ignore_index=True)
    df_responses.to_csv("responses.csv")
