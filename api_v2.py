import logging
import fasttext
import requests
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from deep_translator import GoogleTranslator
from mangum import Mangum
from dotenv import load_dotenv
import sys
import numpy as np
import joblib
import re

# Load environment variables
load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()]
)

app = FastAPI(title="Expense Categorizer API v2")

S3_BUCKET_URL = os.getenv("S3_BUCKET_URL")
if not S3_BUCKET_URL:
    raise ValueError("❌ Missing S3_BUCKET_URL environment variable!")

MODEL_URL = f"{S3_BUCKET_URL}/expense_categorizer_v2.ftz"
MODEL_PATH = "/tmp/expense_categorizer_v2.ftz"
CATEGORY_MAPPING_PATH = "/tmp/category_mapping_v2.pkl"

def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'[^\w\s]', ' ', str(text))
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def download_file(url, path):
    if not os.path.exists(path):
        logging.info(f"📥 Downloading {url} to {path}...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(path, "wb") as f:
                f.write(response.content)
            logging.info(f"✅ Downloaded {path}")
        else:
            logging.error(f"❌ Failed to download {url}")
            raise Exception(f"Failed to download {url}")

# Download model and mapping
try:
    download_file(MODEL_URL, MODEL_PATH)
    download_file(f"{S3_BUCKET_URL}/category_mapping_v2.pkl", CATEGORY_MAPPING_PATH)
except Exception as e:
    logging.error(f"❌ Failed to download v2 model or mapping: {e}")
    raise

try:
    fasttext_model = fasttext.load_model(MODEL_PATH)
    category_mapping = joblib.load(CATEGORY_MAPPING_PATH)
    logging.info("✅ FastText v2 model loaded successfully.")
except Exception as e:
    logging.error(f"❌ Error loading v2 model: {e}")
    raise

translator = GoogleTranslator(source="auto", target="en")

class ExpenseRequest(BaseModel):
    description: str

@app.post("/predict")
def predict_category(expense: ExpenseRequest):
    logging.info(f"🔍 Received prediction request: {expense.description}")
    try:
        cleaned_text = clean_text(expense.description)
        if not cleaned_text:
            raise HTTPException(status_code=400, detail="Empty or invalid description")
        translated_text = translator.translate(cleaned_text)
        logging.info(f"🌍 Translated description: {translated_text}")
        clean_translated = clean_text(translated_text)
        
        # Apply specific rules before ML prediction
        category = apply_categorization_rules(clean_translated)
        
        if category:
            logging.info(f"✅ Rule-based category: {category}")
            return {"category": category, "translated_description": translated_text}
        
        # Fallback to ML prediction
        predictions = fasttext_model.predict(clean_translated, k=3)
        labels = predictions[0]
        predicted_label = labels[0].replace("__label__", "")
        logging.info(f"✅ ML predicted category: {predicted_label}")
        return {"category": predicted_label, "translated_description": translated_text}
    except Exception as e:
        logging.error(f"❌ Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

def apply_categorization_rules(text: str) -> str | None:
    """Apply specific categorization rules before ML prediction"""
    text_lower = text.lower()
    
    # Rule 1: If translated description includes "salary" -> Salary category
    if "salary" in text_lower:
        return "Salary"
    
    # Rule 2: If description includes "bruno" -> Pets category
    if "bruno" in text_lower:
        return "Pets"
    
    # Rule 3: If description includes travel-related keywords -> Travels category
    if is_travel_related(text_lower):
        return "Travels"
    
    # Rule 4: If description includes "wolt" (food delivery) -> Eating out category
    if "wolt" in text_lower or "cibus" in text_lower or "volt" in text_lower:
        return "Eating out"
    
    # Rule 5: If description includes food-related keywords -> Food/Drinks category
    if is_food_related(text_lower):
        return "Food/Drinks"
    
    return None

def is_travel_related(text: str) -> bool:
    """Check if text contains travel-related keywords (countries, cities outside Israel)"""
    # Common countries and cities outside Israel
    travel_keywords = [
        # Countries
        "usa", "united states", "america", "canada", "mexico", "brazil", "argentina",
        "uk", "united kingdom", "england", "scotland", "wales", "ireland",
        "france", "germany", "italy", "spain", "portugal", "netherlands", "belgium",
        "switzerland", "austria", "greece", "turkey", "cyprus", "malta",
        "poland", "czech", "hungary", "romania", "bulgaria", "croatia", "serbia",
        "russia", "ukraine", "belarus", "latvia", "lithuania", "estonia",
        "sweden", "norway", "denmark", "finland", "iceland",
        "japan", "china", "korea", "thailand", "vietnam", "singapore", "malaysia",
        "india", "pakistan", "bangladesh", "sri lanka",
        "australia", "new zealand", "fiji",
        "south africa", "kenya", "morocco", "egypt", "tunisia",
        "uae", "dubai", "abu dhabi", "qatar", "kuwait", "bahrain", "oman",
        "jordan", "lebanon", "syria", "iraq", "iran",
        
        # Major cities
        "new york", "los angeles", "chicago", "miami", "san francisco", "boston",
        "london", "paris", "berlin", "madrid", "barcelona", "rome", "milan",
        "amsterdam", "brussels", "vienna", "prague", "budapest", "warsaw",
        "moscow", "st petersburg", "kyiv", "minsk",
        "stockholm", "oslo", "copenhagen", "helsinki", "reykjavik",
        "tokyo", "osaka", "kyoto", "beijing", "shanghai", "seoul", "bangkok",
        "singapore", "kuala lumpur", "mumbai", "delhi", "bangalore",
        "sydney", "melbourne", "auckland", "wellington",
        "cape town", "johannesburg", "nairobi", "casablanca", "cairo",
        "dubai", "abu dhabi", "doha", "kuwait city", "manama", "muscat",
        "amman", "beirut", "damascus", "baghdad", "tehran",
        
        # Travel-related words
        "vacation", "holiday", "trip", "travel", "flight", "hotel", "resort",
        "tourist", "tourism", "airport", "airline", "booking", "reservation"
    ]
    
    return any(keyword in text for keyword in travel_keywords)

def is_food_related(text: str) -> bool:
    """Check if text contains food-related keywords"""
    food_keywords = [
        # Fruits
        "apple", "banana", "orange", "grape", "strawberry", "blueberry", "raspberry",
        "peach", "pear", "plum", "cherry", "apricot", "mango", "pineapple", "kiwi",
        "lemon", "lime", "grapefruit", "pomegranate", "fig", "date", "prune",
        "coconut", "avocado", "tomato", "cucumber", "pepper", "eggplant",
        
        # Vegetables
        "carrot", "potato", "onion", "garlic", "lettuce", "spinach", "kale",
        "broccoli", "cauliflower", "cabbage", "celery", "asparagus", "mushroom",
        "zucchini", "squash", "pumpkin", "corn", "peas", "beans", "lentil",
        "chickpea", "soybean", "radish", "turnip", "beet", "artichoke",
        
        # Food categories
        "fruit", "vegetable", "produce", "fresh", "organic", "natural",
        "jade", "jade", "ירק", "פירות", "ירקות", "פרי", "ירק"
    ]
    
    return any(keyword in text for keyword in food_keywords)

@app.get("/ping")
def health_check():
    logging.info("🏓 Received ping request.")
    return {"status": "alive"}

logging.info(f"🐍 Python version: {sys.version}")
logging.info(f"📦 FastText v2 model loaded: {fasttext_model is not None}")

try:
    handler = Mangum(app, lifespan="off")
    logging.info("✅ Mangum initialized successfully")
except Exception as e:
    logging.error(f"❌ Mangum initialization failed: {e}")
    raise 