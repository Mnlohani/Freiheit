import uvicorn
import base64
import io
import uuid
from typing import Optional

from PIL import Image, ExifTags
from fastapi import FastAPI, Form, UploadFile, File, HTTPException

from src.constants import IMAGE_RESOLUTION, LANGUAGE_DICT, LLM_MODEL_NAME
from src.models.llm.llm import get_response, load_llm_model
from src.utils.text_utils import infer_resolution_from_prompt, translator
from src.utils.stt import transcribe_STT

app = FastAPI()

# _____Load model______
llm = load_llm_model(model=LLM_MODEL_NAME)

# ___________In-memory session store______
# Keyed by session_id. Holds
# 1. base64-encoded image (sent by the client ONCE, on the first turn)
# 2. running chat_history for that session.

SESSIONS: dict[str, dict] = {}
# ---------------------------------------------------------------------------


def _process_image(image_bytes: bytes, resolution_type: str) -> str:
    """Fix EXIF orientation, convert to RGB, downscale, and base64-encode."""
    img = Image.open(io.BytesIO(image_bytes))

    # Correct the orientation if necessary
    try:
        exif = img._getexif()
        if exif is not None:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == "Orientation":
                    break
            exif = dict(exif.items())
            orientation = exif.get(orientation)
            if orientation == 3:
                img = img.rotate(180, expand=True)
            elif orientation == 6:
                img = img.rotate(270, expand=True)
            elif orientation == 8:
                img = img.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        # Cases: image doesn't have getexif
        pass

    # RGB conversion (also handles RGBA -> RGB) and resize.
    # .thumbnail keeps aspect ratio and won't upscale small images.
    img = img.convert("RGB")
    img.thumbnail(IMAGE_RESOLUTION[resolution_type], Image.Resampling.LANCZOS)

    resized_buffer = io.BytesIO()
    img.save(resized_buffer, format="JPEG")
    return base64.b64encode(resized_buffer.getvalue()).decode("utf-8")


@app.post("/transcribe")
def transcribe(audio: UploadFile = File(...)):
    """
    Returns text, language name, and image resolution from keywords using Whisper call: Speech to text
    """

    text, language_code = transcribe_STT(audio.file.read())
    language_of_response = LANGUAGE_DICT.get(language_code, "English")
    prompt_en = text if language_code == "en" else translator(text, language_code, "en")
    return {
        "text": text,
        "language_code": language_code,
        "language_of_response": language_of_response,
        "image_resolution_type": infer_resolution_from_prompt(prompt_en),
    }


@app.post("/get_ai_response")
def get_ai_response(
    session_id: Optional[str] = Form(None),
    image_resolution_type: str = Form(...),
    user_prompt: str = Form(...),
    language_of_response: str = Form(...),
    send_image: bool = Form(...),
    file: Optional[UploadFile] = File(None),
):
    # Session creation
    if session_id is None or session_id not in SESSIONS:
        session_id = str(uuid.uuid4())
        SESSIONS[session_id] = {"base64_image": None, "chat_history": []}

    session = SESSIONS[session_id]

    # First turn of a session must include an image
    if send_image:
        if file is None:
            raise HTTPException(
                status_code=400,
                detail="First message of a session must include an image.",
            )
        image_bytes = file.file.read()
        session["base64_image"] = _process_image(image_bytes, image_resolution_type)

    elif session["base64_image"] is None:
        # Follow-up request but we have nothing stored (e.g. server restarted,
        # or client sent a stale/unknown session_id)
        raise HTTPException(
            status_code=400,
            detail="No image on file for this session. Please start over with an image.",
        )

    # Call the LLM with full context: image (once) + running history
    ai_response = get_response(
        llm,
        b64image=session["base64_image"],
        user_prompt=user_prompt,
        language_of_response=language_of_response,
        chat_history=session["chat_history"],
    )

    # Update server-side history
    session["chat_history"].append({"role": "user", "content": user_prompt})
    session["chat_history"].append({"role": "assistant", "content": ai_response})

    return {"ai_response": ai_response, "session_id": session_id}


@app.post("/reset_session")
def reset_session(session_id: str = Form(...)):
    """Drop a session's stored image + history (called by 'Start over')."""
    SESSIONS.pop(session_id, None)
    return {"status": "reset"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
