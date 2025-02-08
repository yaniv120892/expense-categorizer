import logging
import fasttext
import requests  # For downloading model from S3
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from deep_translator import GoogleTranslator
from mangum import Mangum  # Required for Vercel deployment
from dotenv import load_dotenv  # Load environment variables from .env

# Load environment variables from .env (only in local environment)
load_dotenv()

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()]
)

# Initialize FastAPI app 
app = FastAPI()

# 🛠️ Replace with your actual S3 URLs
S3_BUCKET_URL = os.getenv("S3_BUCKET_URL")
if not S3_BUCKET_URL:
    raise ValueError("❌ Missing S3_BUCKET_URL environment variable!")

MODEL_URL = f"{S3_BUCKET_URL}/expense_categorizer.ftz"

# Temporary storage paths (Vercel allows using `/tmp`)
MODEL_PATH = "/tmp/expense_categorizer.ftz"

# 🔽 Function to download files from S3
def download_file(url, path):
    if not os.path.exists(path):  # Avoid re-downloading
        logging.info(f"📥 Downloading {url} to {path}...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(path, "wb") as f:
                f.write(response.content)
            logging.info(f"✅ Downloaded {path}")
        else:
            logging.error(f"❌ Failed to download {url}")
            raise Exception(f"Failed to download {url}")

# 🔽 Download models if they don’t exist
download_file(MODEL_URL, MODEL_PATH)

# 🔽 Load models
try:
    fasttext_model = fasttext.load_model(MODEL_PATH)  # Load FastText model
    logging.info("✅ FastText model loaded successfully.")
except Exception as e:
    logging.error(f"❌ Error loading model: {e}")
    raise

# Initialize Translator
translator = GoogleTranslator(source="auto", target="en")

class ExpenseRequest(BaseModel):
    description: str

@app.post("/predict")
def predict_category(expense: ExpenseRequest):
    logging.info(f"🔍 Received prediction request: {expense.description}")

    try:
        # 🔹 Translate to English if needed
        translated_text = translator.translate(expense.description)
        logging.info(f"🌍 Translated description: {translated_text}")

        # 🔹 Predict category using FastText
        predicted_label = fasttext_model.predict(translated_text)[0][0]  # e.g., '__label__food'
        category = predicted_label.replace("__label__", "")  # Remove FastText label prefix

        logging.info(f"✅ Predicted category: {category}")
        return {"category": category, "translated_description": translated_text}

    except Exception as e:
        logging.error(f"❌ Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Health check endpoint
@app.get("/ping")
def health_check():
    logging.info("🏓 Received ping request.")
    return {"status": "alive"}

# Wrap FastAPI app with Mangum for Vercel serverless deployment
handler = Mangum(app, lifespan="off")  # Disable lifespan events for serverless
