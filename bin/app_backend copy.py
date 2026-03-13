import uvicorn
import base64
import io
from typing import Literal, Annotated
from pydantic import BaseModel
from PIL import Image, ExifTags
from fastapi import Depends, FastAPI, Form, UploadFile, File

from src.constants import IMAGE_RESOLUTION
from src.models.llm.llm import get_response, load_llm_model

import logging

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ------ Load model ------
llm = load_llm_model(model="gemini-2.5-flash")

class UserInput(BaseModel):
    image_resolution_type: Literal["High", "Medium", "Low", "Very Low"]
    user_prompt: str
    language_of_response: Literal["en", "de", "hi", "fr", "es", "it", "nl"]


@app.post("/get_ai_response")
async def get_ai_response(
    image_resolution_type = Form(...),
    user_prompt: str = Form(...),
    language_of_response = Form(...),
    file: UploadFile = File(...)
    ):
    # Read file bytes
    image_bytes = await file.read()

    # Turn into PIL image
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
    
    #  RGB conversion (Also handles RGBA to RGB conversion) and resize 
    # .thumbnail to maintain aspect ratio; it won't upscale small images
    img = img.convert("RGB")
    img.thumbnail(IMAGE_RESOLUTION[image_resolution_type], Image.Resampling.LANCZOS)

    # Save the resized image to a NEW buffer
    resized_buffer = io.BytesIO()
    img.save(resized_buffer, format="JPEG")
    resized_bytes = resized_buffer.getvalue()
    
    # Encode the NEW resized bytes to Base64 and Get response from LLM
    base64_image = base64.b64encode(resized_bytes).decode("utf-8")
    ai_response = get_response(llm, base64_image, user_prompt, language_of_response)
    
    return {"ai_response":ai_response}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)