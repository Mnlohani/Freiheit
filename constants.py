# GUI Constants
TASK_TYPES = [
    "Bus stop",
    "Metro station",
    "Clothes",
    "Laundry",
    "Products",
    "Object at front",
]

# Subtask types for each task type for prediction with Both DistanceNN and LLM
TASK_SUBTYPES = {
    "Object at front": ["Include distance with NN", "Include distance with LLM"],
    "Clothes": ["Cloth color", "Read lable tag", "Cloth pattern"],
    "Products": ["Read products"],
    "Bus stop": ["Bus number", "Departure time"],
    "Metro station": ["Street for exit", "Departure time"],
    "Laundry": ["Cloth color", "Cloth pattern"],
}

# Subtask types for each task type for prediction with LLM only
TASK_SUBTYPES_LLM = {
    "Object at front": [],
    "Clothes": ["Cloth color", "Read lable tag", "Cloth pattern"],
    "Products": ["Read products"],
    "Bus stop": ["Bus number", "Departure time"],
    "Metro station": ["Street for exit", "Departure time"],
    "Laundry": ["Cloth color", "Cloth pattern"],
}

IMAGE_INPUT_TYPE = ["Upload Image", "Camera"]

LANGUAGE_OPTIONS = [
    "English",
    "German",
    "Hindi",
    "French",
    "Spanish",
    "Italian",
    "Dutch",
]

LANGUAGE_OPTIONS = {
    "English": "en",
    "German": "de",
    "Hindi": "hi",
    "French": "fr",
    "Spanish": "es",
    "Italian": "it",
    "Dutch": "nl",
}

IMAGE_RESOLUTION = {
    "High": None,
    "Medium": (2048, 2048),
    "Low": (1024, 1024),
    "Very Low": (512, 512),
}

# To set background Image of the app
INPUT_BG_IMAGE_PATH = "./images/background_image_in.jpg"
OUTPUT_BG_IMAGE_PATH = "./images/background_image.jpg"


# Engineered Prompts
PROMPTS = {
    "Include distance with NN": "The image contains a scene with various objects. Identify the object which is directly in front of you. Always include distance of the object in your answer. Answer the question in very concise way. If the information is unclear in the image, say so.",
    "Include distance with LLM": "The image contains a scene with various objects. Identify the object which is directly in front of you. Measure its distance into steps of the user. Always include the distance in steps in your answer, regardless of the question asked. Answer the question in a very concise way. If the information is unclear in the image, say so.",
    "Cloth color": "The image contains a scene with various objects. Identify the cloths and color of cloths if there's a cloth present. If the information is not visible in the image, say so. Answer in a very concise way. Do not give information in bullet points",
    "Read lable tag": "The image contains a scene with various objects. Identify the size of the cloth from the label tag  of the cloth and tell very concisely the size of the cloth and the price if present? If the information is invisible in the image, say so.",
    "Cloth pattern": "The image contains a scene with various objects. Identify the pattern of cloths if there's a cloth present. Answer in a concise way. If the information is not visible in the image, say so",
    "Read products": "The image contains a scene with various objects. Identify the information accurately that the user has requested about the name, nutrition, and expiry date of the product from the image, and concisely convey the relevant information. The labels are primarily in German but may also be in other languages. If the relevant information requested is not visible in the image, state this. Accurately recognize the product the user is touching or holding in the image. The user needs very accurate information and does not want it in bullet points. Provide a very concise answer without unnecessary details.",
    "Read dial": """Explain in clear and short words. Be polite in explaining. The User want to ask about of dial of appliances like washing machine, oven, microwave, etc. The infomation are primarily in German, but do not ignore the possibility of other languages. Be precise in answering the question. Be polite in explaining.""",
    "Bus number": "The angles of view are simliar to a clock, from 10 o'clock to 2 o'clock. Always include its position in angles from the user in the answer. Convert distances to footsteps if needed. The image contains a scene with various objects. Can you tell if there's a bus present? If there is, can you identify accuretly any numbers on the bus and any text that might indicate its destination and its position with angles of the view as mentioned earlier? No guessing. If there is a yes/no question, answer with yes or no only. If the information is unlcear in the image, say so. Answer in a very concise manner without giving unneccsary information",
    "Departure time": "The image contains a digital display that might show bus or metro information. Read the display from top to bottom. Can you specifically tell me the departure time for the bus or metro? Be very accurate in reading the display regardung the bus number in the image. No yapping. If the information is not visible in the image, say so and do not give wrong information if you are not sure about it. Answer very concisely. Do not yap",
    "Street for exit": "The user is near or in a metro station. The names of streets and stations are in German. Be very precise about the presence of railway tracks or potential tracks, even if they are not visible in the photo. The user wants to know the direction for the street asked in question. Identify the correct direction towards the street for exit. If asked, provide the distance in footsteps. Answer concisely without giving any unnecessary information. Do not yap",
}


# DistanceNN model Parameters
EMBEDDING_DIM = 768  #  embedding dimension of DinoV2 model
SAVED_MODEL_PATH = "./model/best_model_MSE.pt"


# LLM Parameters
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1028
