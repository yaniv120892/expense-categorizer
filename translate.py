from deep_translator import GoogleTranslator
import pandas as pd
from tqdm import tqdm
import time
import hashlib
import json
import os

def create_cache_key(text):
    """Create a hash key for caching"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def load_translation_cache():
    """Load existing translation cache"""
    cache_file = "data/translation_cache.json"
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_translation_cache(cache):
    """Save translation cache"""
    cache_file = "data/translation_cache.json"
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def batch_translate(texts, translator, cache, batch_size=50):
    """Translate texts in batches with caching"""
    results = []
    
    # Calculate how many need actual translation (not cached)
    uncached_texts = [text for text in texts if isinstance(text, str) and text.strip() and create_cache_key(text) not in cache]
    cached_count = len(texts) - len(uncached_texts)
    
    print(f"📊 {cached_count} translations found in cache")
    print(f"🌍 {len(uncached_texts)} translations needed")
    
    if len(uncached_texts) > 0:
        # Estimate time: ~2 seconds per translation (including delay)
        estimated_time = len(uncached_texts) * 2
        print(f"⏱️  Estimated time: {estimated_time//60} minutes {estimated_time%60} seconds")
    
    # Use tqdm for progress bar
    for i in tqdm(range(0, len(texts), batch_size), desc="Translating batches"):
        batch = texts[i:i + batch_size]
        batch_results = []
        
        for text in batch:
            if not isinstance(text, str) or not text.strip():
                batch_results.append(text)
                continue
                
            cache_key = create_cache_key(text)
            
            if cache_key in cache:
                batch_results.append(cache[cache_key])
            else:
                try:
                    translated = translator.translate(text)
                    cache[cache_key] = translated
                    batch_results.append(translated)
                    time.sleep(0.1)  # Small delay to avoid rate limiting
                except Exception as e:
                    print(f"\n❌ Error translating '{text}': {e}")
                    batch_results.append(text)  # Keep original if translation fails
        
        results.extend(batch_results)
        
        # Save cache periodically
        if i % (batch_size * 10) == 0:
            save_translation_cache(cache)
    
    return results

# Load dataset
print("📁 Loading dataset...")
data_path = "data/exported.csv"
df = pd.read_csv(data_path, encoding="utf-8-sig", delimiter=";")
df.dropna(subset=["Description"], inplace=True)

print(f"📊 Found {len(df)} descriptions to translate")

# Load existing cache
print("🔄 Loading translation cache...")
cache = load_translation_cache()
print(f"💾 Cache contains {len(cache)} translations")

# Initialize Translator
translator = GoogleTranslator(source="auto", target="en")

# Get unique descriptions to avoid translating duplicates
unique_descriptions = df["Description"].unique()
print(f"🎯 Found {len(unique_descriptions)} unique descriptions")

# Translate unique descriptions
print("🌍 Translating descriptions...")
translated_unique = batch_translate(unique_descriptions, translator, cache)

# Create mapping from original to translated
translation_mapping = dict(zip(unique_descriptions, translated_unique))

# Apply translations to the dataframe
print("📝 Applying translations to dataset...")
df["Translated_Description"] = df["Description"].map(translation_mapping)

# Save translated dataset
print("💾 Saving translated dataset...")
df.to_csv("data/exported_translated.csv", encoding="utf-8-sig", index=False)

# Save final cache
save_translation_cache(cache)

print(f"✅ Translation complete!")
print(f"📊 Original descriptions: {len(df)}")
print(f"📊 Unique descriptions translated: {len(unique_descriptions)}")
print(f"💾 Cache size: {len(cache)}")
