import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from src.constants import HAUPT_PROMPT, LLM_MAX_TOKENS, LLM_TEMPERATURE, PROMPTS


def load_llm_model(model: str) -> object:
    """load the large language model

    Args:
        model (str): The name of the model to load.
            Supported values: "gpt-4o", "gemini-2.5-flash", "llava".

    Raises:
        Exception: if an unsupported model name is passed.

    Returns:
        object: An instance of the loaded language model
    """
    load_dotenv()
    if "gpt" in model:
        llm = ChatOpenAI(
            model=model,
            max_tokens=LLM_MAX_TOKENS,
            reasoning_effort="minimal",
            temperature=LLM_TEMPERATURE,
            verbosity="low",
            streaming=True,
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
    elif "gemini" in model:
        llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=LLM_TEMPERATURE,
            max_output_tokens=LLM_MAX_TOKENS,
            streaming=True,
            api_key=os.environ.get("Gemini_API_KEY"),
        )
    elif model == "llava":
        llm = ChatOllama(model=model)
    else:
        raise Exception("currently supported Models: Gemini, chatgpt and llava")
    return llm


def construct_message(
    b64image: str,
    human_message: str,
    language_of_response: str,
    chat_history: list[dict] | None = None,
):
    """
    Builds the full message list sent to the LLM for one turn.

    Structure:
      1. SystemMessage -- instructions (always first) makes "messages" variable
      2. The whole conversation, rebuilt on every call because the LLM API
         is stateless: chat_history (Human/AI messages) + the new question.
      3. The stored image is attached to the FIRST user message on every call,
         so the model can always see it. The frontend still uploads it only once;
         the backend keeps it in the session.

    Args:
        b64image (str): base64-encoded JPEG image (same image every turn).
        human_message (str): the current user prompt/question.
        language_of_response (str): language the answer should be given in.
        chat_history (list[dict] | None):
            {"role": "user" | "assistant", "content": str}, in order.

    Returns:
        list: LangChain message objects ready for model.invoke(...)

    Example:
        Human: "What color is the shirt?"  + [IMAGE]   <- only the first message
        AI:    "Blue."
        Human: "What does the tag say?"
        AI:    "Size M."
        Human: "How much is it?"
    """
    chat_history = chat_history or []

    prompt = (
        "The user wants the answer in "
        + str(language_of_response)
        + " language. "
        + HAUPT_PROMPT
        + PROMPTS
    )

    messages = [SystemMessage(content=prompt)]

    # Full conversation = previous turns + the new question
    turns = chat_history + [{"role": "user", "content": human_message}]

    for i, turn in enumerate(turns):
        if turn["role"] == "assistant":
            messages.append(AIMessage(content=turn["content"]))
        elif i == 0:
            # Very first message of the conversation: text + image together
            messages.append(
                HumanMessage(
                    content=[
                        {"type": "text", "text": turn["content"]},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64image}",
                                "detail": "auto",
                            },
                        },
                    ]
                )
            )
        else:
            # All later user messages: text only
            messages.append(HumanMessage(content=turn["content"]))

    return messages


def get_response(
    model: object,
    b64image: str,
    user_prompt: str,
    language_of_response: str = "English",
    chat_history: list[dict] | None = None,
) -> str:
    """Get the response from the AI model, with full multi-turn context.

    Args:
        model (object): The llm model
        b64image (str): the base64 encoded image (stored server-side once)
        user_prompt (str): Question asked by the user this turn
        language_of_response (str): Language of the response
        chat_history (list[dict] | None): prior turns of this session's
            conversation, used to reconstruct context for the model.

    Returns:
        str: Response from the AI model
    """
    message_to_model = construct_message(
        b64image, user_prompt, language_of_response, chat_history
    )
    ai_msg = model.invoke(message_to_model)
    return ai_msg.content
