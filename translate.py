from deep_translator import GoogleTranslator
import pandas as pd
from tqdm import tqdm  # To track progress

# Load dataset
data_path = "data/exported.csv"
df = pd.read_csv(data_path, encoding="utf-8-sig", delimiter=";")
df.dropna(subset=["Description"], inplace=True)

# Initialize Translator
translator = GoogleTranslator(source="auto", target="en")

# Translate Hebrew descriptions to English
tqdm.pandas()  # Adds a progress bar
df["Translated_Description"] = df["Description"].progress_apply(lambda x: translator.translate(x) if isinstance(x, str) else x)

# Save translated dataset
df.to_csv("data/exported_translated.csv", encoding="utf-8-sig", index=False)
print("✅ Hebrew descriptions translated to English and saved.")
