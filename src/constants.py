API_ENDPOINT = "/get_ai_response"

# ── Widget Keys for UI elements to reset_inputs stays in sync easily
WIDGET_KEYS = [
    "uploaded_image",
    "audio_input",
    "image_resolution_type"
]

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
IMAGE_QUALITY_OPTIONS = ["Very Low", "Low", "Medium", "High"]

IMAGE_RESOLUTION = {
    "High": (256, 256),
    "Medium": (256, 256),
    "Low": (256, 256),
}

# To set background Image of the app
INPUT_BG_IMAGE_PATH = "./assets/images/background_image_in.jpg"
OUTPUT_BG_IMAGE_PATH = "./assets/images/background_image.jpg"

# Engineered Prompts

HAUPT_PROMPT = """You are helping a person which has lower vision or blindness. The person will give you a video or an image. The image contains a scene with various objects. You need to analyse the image and identify the objects in the image accurately. Its very important to give the person the accurate information because incorrect information could lead user to a inadequate situations and it could also be dangerous the person. If you unconfident about the information the use has asked and unable to get the required the information correctly, say that you are not able to required information. Hallucination is strictly forbidden. Be polite in answering and do not yap. Explain in clear and short words with short sentences with at most 15 words. Provide a very concise answer without unnecessary details. No bullet points"""

PROMPTS = """Based on the content of the image, the user wants to ask questions. The tasks are divided into the options below. Choose the correct option for the correct prompt.
If the image contains a cloth and the user asks about the cloth color, select Option 1.
If the image contains a cloth and the user asks about the cloth pattern, select Option 2.
If the image contains a bus and the user asks about the bus number, select Option 3.
If the image contains a label tag of clothing or shoes and the user asks about the bus number, select Option 4.
If the image shows a digital display that might contain bus, metro, or flight information and the user asks about the departure time, select Option 5.
If the image is near or inside a metro station and the user asks for directions to the street, select Option 6.
If the image contains a product and the user wants information about the product, select Option 7.
If the user ask about the object in front, select option 8.
Otherwise, provide the correct information to the user based on their question. Hallucination is not allowed and could be dangerous to the person. If you do not read the image clearly, say so rather than hallucinating.
    
option 1 : Cloth color : Identify the cloths and color of cloths if there's a cloth present.
option 2 : Cloth pattern: Identify the pattern of cloths if there's a cloth present.
option 3 : Read lable tag: Identify the size of the product such as cloth or shoes from the label tag  of the cloth or product and tell very concisely the size of the cloth and the price if present.
option 4 : Bus number : The angles of view are simliar to a clock, from 10 o'clock to 2 o'clock. Always include its position in angles from the user in the answer. Convert distances to footsteps if needed. Can you tell if there's a bus present? If there is, can you identify accuretly any numbers on the bus and any text that might indicate its destination and its position with angles of the view as mentioned earlier? If there is a yes/no question, answer with yes or no only.
option 5 : Departure time : The image contains a digital display that might show bus or metro or flight information. Read the display from top to bottom. Can you specifically tell me the departure time for the bus or metro? Be very accurate in reading the display regarding the bus number in the image.
option 6 : Street for exit : The user is near or in a metro station. The names of streets and stations are in German. Be very precise about the presence of railway tracks or potential tracks, even if they are not visible in the photo. The user wants to know the direction for the street asked in question. Identify the correct direction towards the street for exit. If asked, provide the distance in footsteps.
option 7 : Read products : Identify the information accurately that the user has requested about the name, nutrition, and expiry date of the product from the image, and concisely convey the relevant information. The labels are primarily in German but may also be in other languages. Accurately recognize the product the user is touching or holding in the image.
option 8 : Identify the object which is directly in front of you. Measure its distance into steps of the user. Include the distance in feetsteps in your answer, if asked.
"""

RESOLUTION_KEYWORD_MAP = {
    "Low": [
        # Cloth related — Option 1, 2
        "color", "colour", "pattern", "fabric", "material",
        "shirt", "dress", "jacket", "cloth", "clothes", "clothing",
        "trousers", "pants", "skirt", "coat", "sweater", "laundry", "socks"
        
        # Label/tag related — Option 3
        "label", "tag", "size", "price", "expiry", "nutrition",
        "ingredient", "barcode",
        
        # Product related — Option 7
        "product", "bottle", "can", "package", "brand",
        
        # Close object — Option 8
        "front", "holding", "touching", "this object", "what is this",  
    ],
    
    "High": [
        # Bus related — Option 4
        "bus", "number", "destination", "route",
        
        # Display related — Option 5
        "departure", "arrival", "schedule", "display", "screen",
        "timetable", "metro", "flight", "platform",
        
        # Navigation related — Option 6
        "exit", "street", "direction", "station", "track",
        "where", "navigate", "way to", "how to get",
    ],
}

DEFAULT_RESOLUTION = "Medium"   # fallback if no keywords match

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
DATASET_IMG_FILES_PATH = "data/01_raw/demo/object_front"
DATASET_CSV_FILE_PATH = "data/01_raw/demo/chairs_distances.csv"

LANGUAGE_DICT = {
    "af": "Afrikaans",
    "am": "Amharic",
    "ar": "Arabic",
    "as": "Assamese",
    "az": "Azerbaijani",
    "ba": "Bashkir",
    "be": "Belarusian",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "bo": "Tibetan",
    "br": "Breton",
    "bs": "Bosnian",
    "ca": "Catalan",
    "cs": "Czech",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "eu": "Basque",
    "fa": "Persian",
    "fi": "Finnish",
    "fo": "Faroese",
    "fr": "French",
    "gl": "Galician",
    "gu": "Gujarati",
    "ha": "Hausa",
    "haw": "Hawaiian",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "ht": "Haitian Creole",
    "hu": "Hungarian",
    "hy": "Armenian",
    "id": "Indonesian",
    "is": "Icelandic",
    "it": "Italian",
    "ja": "Japanese",
    "jw": "Javanese",
    "ka": "Georgian",
    "kk": "Kazakh",
    "km": "Khmer",
    "kn": "Kannada",
    "ko": "Korean",
    "la": "Latin",
    "lb": "Luxembourgish",
    "ln": "Lingala",
    "lo": "Lao",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mg": "Malagasy",
    "mi": "Maori",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mn": "Mongolian",
    "mr": "Marathi",
    "ms": "Malay",
    "mt": "Maltese",
    "my": "Myanmar",
    "ne": "Nepali",
    "nl": "Dutch",
    "nn": "Nynorsk",
    "no": "Norwegian",
    "oc": "Occitan",
    "pa": "Punjabi",
    "pl": "Polish",
    "ps": "Pashto",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sa": "Sanskrit",
    "sd": "Sindhi",
    "si": "Sinhala",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sn": "Shona",
    "so": "Somali",
    "sq": "Albanian",
    "sr": "Serbian",
    "su": "Sundanese",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "tg": "Tajik",
    "th": "Thai",
    "tk": "Turkmen",
    "tl": "Tagalog",
    "tr": "Turkish",
    "tt": "Tatar",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "uz": "Uzbek",
    "vi": "Vietnamese",
    "yi": "Yiddish",
    "yo": "Yoruba",
    "zh": "Chinese",
}


LANGUAGE_LABELS = {
    "en": {
        "name": "English",
        "aria": "Your language is set to English. Tap to confirm.",
        "display": "🌐 Your language is English\n\nTap anywhere to confirm",
    },
    "ur": {
        "name": "Urdu",
        "aria": "آپ کی زبان اردو ہے۔ تصدیق کے لیے تھپتھپائیں",
        "display": "🌐 آپ کی زبان اردو ہے\n\nتصدیق کے لیے یہاں تھپتھپائیں",
    },
    "ar": {
        "name": "Arabic",
        "aria": "لغتك هي العربية. انقر للتأكيد",
        "display": "🌐 لغتك هي العربية\n\nانقر في أي مكان للتأكيد",
    },
    "hi": {
        "name": "Hindi",
        "aria": "आपकी भाषा हिंदी है। पुष्टि करने के लिए टैप करें",
        "display": "🌐 आपकी भाषा हिंदी है\n\nपुष्टि के लिए यहाँ टैप करें",
    },
    "fr": {
        "name": "French",
        "aria": "Votre langue est le français. Appuyez pour confirmer",
        "display": "🌐 Votre langue est le français\n\nAppuyez n'importe où pour confirmer",
    },
    "de": {
        "name": "German",
        "aria": "Ihre Sprache ist Deutsch. Tippen Sie zum Bestätigen",
        "display": "🌐 Ihre Sprache ist Deutsch\n\nTippen Sie zum Bestätigen",
    },

    "es": {
    "name": "Spanish",
    "aria": "Tu idioma es español. Toca para confirmar.",
    "display": "🌐 Tu idioma es español\n\nToca en cualquier lugar para confirmar",
    },

"it": {
    "name": "Italian",
    "aria": "La tua lingua è l'italiano. Tocca per confermare.",
    "display": "🌐 La tua lingua è italiano\n\nTocca ovunque per confermare",
    },
    # fallback
    "default": {
        "name": "English",
        "aria": "Your language is set to English. Tap to confirm.",
        "display": "🌐 Your language is English\n\nTap anywhere to confirm",
    },
}