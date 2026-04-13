import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM Configuration ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
LLM_MODEL = "gemini/gemini-2.5-flash"

# --- File Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

APP_REVIEWS_CSV = os.path.join(DATA_DIR, "app_store_reviews.csv")
SUPPORT_EMAILS_CSV = os.path.join(DATA_DIR, "support_emails.csv")
EXPECTED_CSV = os.path.join(DATA_DIR, "expected_classifications.csv")

GENERATED_TICKETS_CSV = os.path.join(OUTPUT_DIR, "generated_tickets.csv")
PROCESSING_LOG_CSV = os.path.join(OUTPUT_DIR, "processing_log.csv")
METRICS_CSV = os.path.join(OUTPUT_DIR, "metrics.csv")

# --- Classification Thresholds ---
CLASSIFICATION_CONFIDENCE_THRESHOLD = 0.7

# --- Priority Rules ---
RATING_PRIORITY_MAP = {
    1: "High",
    2: "High",
    3: "Medium",
    4: "Low",
    5: "Low",
}

CATEGORY_PRIORITY_DEFAULTS = {
    "Bug": "High",
    "Feature Request": "Medium",
    "Complaint": "Medium",
    "Praise": "Low",
    "Spam": "Low",
}

VALID_CATEGORIES = ["Bug", "Feature Request", "Praise", "Complaint", "Spam"]
VALID_PRIORITIES = ["Critical", "High", "Medium", "Low"]
