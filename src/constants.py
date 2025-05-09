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
    "Object at front": ["Include distance with LLM"],
    "Clothes": ["Cloth color", "Read lable tag", "Cloth pattern"],
    "Products": ["Read products"],
    "Bus stop": ["Bus number", "Departure time"],
    "Metro station": ["Street for exit", "Departure time"],
    "Laundry": ["Cloth color", "Cloth pattern"],
}

# Image input ways
IMAGE_INPUT_TYPE = ["Upload Image", "Camera"]


# Language options to answer the user
LANGUAGE_OPTIONS = {
    "English": "en",
    "German": "de",
    "Hindi": "hi",
    "French": "fr",
    "Spanish": "es",
    "Italian": "it",
    "Dutch": "nl",
}


# The image resolutions for user to set so to lower the cost if paid models subscriptions are used.
IMAGE_RESOLUTION = {
    "High": None,
    "Medium": (2048, 2048),
    "Low": (1024, 1024),
    "Very Low": (512, 512),
}


# To set background Image of the app
INPUT_BG_IMAGE_PATH = "./assets/images/background_image_in.jpg"
OUTPUT_BG_IMAGE_PATH = "./assets/images/background_image.jpg"


# Engineered Prompts
HAUPT_PROMPT = """You are helping a person which has lower vision or blindness. The person will give you a video or an image. The image contains a scene with various objects. You need to analyse the image and identify the objects in the image accurately. Its very important to give the person the accurate information because incorrect information could lead user to a inadequate situations and it could also be dangerous the person. If you unconfident about the information the use has asked and unable to get the required the information correctly, say that you are not able to required information. Hallucination is strictly forbidden. Be polite in answering and do not yap. Explain in clear and short words with short sentences with at most 15 words. Provide a very concise answer without unnecessary details. No bullet points"""

PROMPTS = {
    "Include distance with NN": "Identify the object which is directly in front of you. Always include distance of the object in your answer.",
    "Include distance with LLM": "Identify the object which is directly in front of you. Measure its distance into steps of the user. Always include the distance in steps in your answer, regardless of the question asked.",
    "Cloth color": "Identify the cloths and color of cloths if there's a cloth present.",
    "Read lable tag": "Identify the size of the cloth from the label tag  of the cloth and tell very concisely the size of the cloth and the price if present?",
    "Cloth pattern": "Identify the pattern of cloths if there's a cloth present.",
    "Read products": "Identify the information accurately that the user has requested about the name, nutrition, and expiry date of the product from the image, and concisely convey the relevant information. The labels are primarily in German but may also be in other languages. If the relevant information requested is not visible in the image, state this. Accurately recognize the product the user is touching or holding in the image.",
    "Read dial": "The User want to ask about of dial of appliances like washing machine, oven, microwave, etc. The infomation are primarily in German, but do not ignore the possibility of other languages.",
    "Bus number": "The angles of view are simliar to a clock, from 10 o'clock to 2 o'clock. Always include its position in angles from the user in the answer. Convert distances to footsteps if needed. Can you tell if there's a bus present? If there is, can you identify accuretly any numbers on the bus and any text that might indicate its destination and its position with angles of the view as mentioned earlier?. If there is a yes/no question, answer with yes or no only.",
    "Departure time": "The image contains a digital display that might show bus or metro information. Read the display from top to bottom. Can you specifically tell me the departure time for the bus or metro? Be very accurate in reading the display regarding the bus number in the image.",
    "Street for exit": "The user is near or in a metro station. The names of streets and stations are in German. Be very precise about the presence of railway tracks or potential tracks, even if they are not visible in the photo. The user wants to know the direction for the street asked in question. Identify the correct direction towards the street for exit. If asked, provide the distance in footsteps.",
}


# DistanceNN model Parameters
EMBEDDING_DIM = 768  #  embedding dimension of DinoV2 model
SAVED_MODEL_PATH = "./data/03_models/distanceNN/best_model_MSE.pt"


# DINOv2 model directories
IMAGE_PROCESSOR_PATH = "data/03_models/DINOv2_HuggingFace/image_processor"
DINO_MODEL_PATH = "data/03_models/DINOv2_HuggingFace/dinov2_model"


# LLM Parameters
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1028


# Path to datasets for Training of DistanceNN
DATASET_IMG_FILES_PATH = "data/01_raw/chair/"
DATASET_CSV_FILE_PATH = "data/01_raw/chairs_distances.csv"
