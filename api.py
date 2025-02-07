import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from deep_translator import GoogleTranslator

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()]
)

# Load trained model and encoders
try:
    model = joblib.load("models/expense_categorizer.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    logging.info("✅ Model and encoders loaded successfully.")
except Exception as e:
    logging.error(f"❌ Error loading model: {e}")
    raise

# Initialize FastAPI app
app = FastAPI()

translator = GoogleTranslator(source="auto", target="en")

# Request model
class ExpenseRequest(BaseModel):
    description: str

@app.post("/predict")
def predict_category(expense: ExpenseRequest):
    logging.info(f"🔍 Received prediction request: {expense.description}")
    try:
        translated_text = translator.translate(expense.description)
        logging.info(f"🌍 Translated description: {translated_text}")

        # Transform text into vector
        text_vectorized = vectorizer.transform([translated_text])

        # Predict category
        category_encoded = model.predict(text_vectorized)[0]

        # Decode category
        category = label_encoder.inverse_transform([category_encoded])[0]

        logging.info(f"✅ Predicted category: {category}")
        return {"category": category}

    except Exception as e:
        logging.error(f"❌ Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Health check endpoint
@app.get("/ping")
def health_check():
    logging.info("🏓 Received ping request.")
    return {"status": "alive"}

# Run the server with: uvicorn api:app --reload
