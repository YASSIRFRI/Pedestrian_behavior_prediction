"""Constants used throughout the behavior classifier package."""

# Behavior classes for classification
BEHAVIOR_CLASSES = [
    "walking",
    "running",
    "biking",
    "dog walking",
    "scootering",
    "standing",
    "sitting",
    "stroller",
]

# Model configurations
DEFAULT_BLIP2_MODEL = "Salesforce/blip2-opt-2.7b"
AVAILABLE_BLIP2_MODELS = [
    "Salesforce/blip2-opt-2.7b",
    "Salesforce/blip2-opt-6.7b", 
    "Salesforce/blip2-flan-t5-xl"
]

GROUNDING_DINO_MODEL = "IDEA-Research/grounding-dino-tiny"
ZERO_SHOT_MODEL = "facebook/bart-large-mnli"

# Detection parameters
DEFAULT_DETECTION_THRESHOLD = 0.5
MIN_AREA_RATIO = 0.001
MIN_RESOLUTION = 32
EXPANSION_FACTOR = 1.5
PROXIMITY_THRESHOLD = 0.3

# Video processing
DEFAULT_TARGET_FPS = 1
DEFAULT_OUTPUT_FPS = 5

# Behavior mapping for classification
BEHAVIOR_LABEL_MAPPING = {
    "biking": "riding a bicycle",
    "dog walking": "walking a dog", 
    "scootering": "riding a scooter",
    "stroller": "pushing a"
}

# File extensions
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

# Visualization colors (BGR format for OpenCV)
VISUALIZATION_COLORS = {
    'original_box': (0, 0, 255),    # Red
    'expanded_box': (0, 255, 0),    # Green
    'text_bg': (255, 255, 255),     # White
    'text_color': (0, 0, 0),        # Black
}