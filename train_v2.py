import pandas as pd
import fasttext
import joblib
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'[^\w\s]', ' ', str(text))
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def analyze_data_distribution(df):
    category_counts = df['Category'].value_counts()
    print("\n📊 Category Distribution:")
    print(category_counts)
    print(f"\n📈 Statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Number of categories: {len(category_counts)}")
    print(f"Most common category: {category_counts.index[0]} ({category_counts.iloc[0]} samples)")
    print(f"Least common category: {category_counts.index[-1]} ({category_counts.iloc[-1]} samples)")
    return category_counts

def create_balanced_dataset(df, min_samples_per_category=30, max_samples_per_category=200):
    balanced_data = []
    for category in df['Category'].unique():
        category_data = df[df['Category'] == category]
        n_samples = len(category_data)
        if n_samples < min_samples_per_category:
            print(f"⚠️  Skipping {category}: only {n_samples} samples")
            continue
        elif n_samples > max_samples_per_category:
            category_data = category_data.sample(max_samples_per_category, random_state=42)
        balanced_data.append(category_data)
    balanced_df = pd.concat(balanced_data, ignore_index=True)
    return balanced_df

def prepare_fasttext_data(df, output_file):
    df['clean_description'] = df['Translated_Description'].apply(clean_text)
    df = df[df['clean_description'].str.len() > 0]
    fasttext_data = []
    for _, row in df.iterrows():
        label = f"__label__{row['Category']}"
        text = row['clean_description']
        fasttext_data.append(f"{label} {text}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in fasttext_data:
            f.write(line + '\n')
    return len(fasttext_data)

def train_and_evaluate_model(train_file, test_file=None):
    model = fasttext.train_supervised(
        input=train_file,
        epoch=50,
        lr=0.1,
        wordNgrams=3,
        dim=200,
        minCount=2,
        minn=3,
        maxn=6,
        neg=10,
        loss='ova'
    )
    result = model.test(train_file)
    print(f"\n✅ Training Accuracy: {result[1]:.3f}")
    if len(result) > 2:
        print(f"📊 Training Precision: {result[2]:.3f}")
    if len(result) > 3:
        print(f"📊 Training Recall: {result[3]:.3f}")
    return model

def predict_and_analyze(model, df, sample_size=50):
    print(f"\n🧪 Testing predictions on {sample_size} random samples:")
    test_samples = df.sample(min(sample_size, len(df)), random_state=42)
    correct = 0
    predictions = []
    for _, row in test_samples.iterrows():
        text = clean_text(row['Translated_Description'])
        if len(text) == 0:
            continue
        prediction = model.predict(text)[0][0].replace("__label__", "")
        actual = row['Category']
        predictions.append({'text': text, 'actual': actual, 'predicted': prediction, 'correct': prediction == actual})
        if prediction == actual:
            correct += 1
    accuracy = correct / len(predictions) if predictions else 0
    print(f"🎯 Test Accuracy: {accuracy:.3f}")
    print(f"\n📝 Sample Predictions:")
    for i, pred in enumerate(predictions[:10]):
        status = "✅" if pred['correct'] else "❌"
        print(f"{status} '{pred['text'][:50]}...'")
        print(f"   Actual: {pred['actual']} | Predicted: {pred['predicted']}")
    return predictions

def main():
    print("🚀 Starting v2 Expense Categorizer Training")
    data_path = "data/exported_translated.csv"
    df = pd.read_csv(data_path, encoding="utf-8-sig")
    print(f"📁 Loaded {len(df)} samples from {data_path}")
    original_counts = analyze_data_distribution(df)
    print("\n⚖️  Creating balanced dataset...")
    balanced_df = create_balanced_dataset(df, min_samples_per_category=30, max_samples_per_category=150)
    balanced_counts = analyze_data_distribution(balanced_df)
    train_file = "data/fasttext_train_v2.txt"
    n_samples = prepare_fasttext_data(balanced_df, train_file)
    print(f"\n📝 Prepared {n_samples} training samples")
    print("\n🤖 Training FastText v2 model...")
    model = train_and_evaluate_model(train_file)
    predictions = predict_and_analyze(model, df)
    os.makedirs("models", exist_ok=True)
    model_path = "models/expense_categorizer_v2.ftz"
    model.quantize(qnorm=True, cutoff=50000)
    model.save_model(model_path)
    print(f"\n💾 Model saved to: {model_path}")
    categories = sorted(balanced_df['Category'].unique())
    category_mapping = {i: cat for i, cat in enumerate(categories)}
    joblib.dump(category_mapping, "models/category_mapping_v2.pkl")
    print(f"📋 Category mapping saved with {len(categories)} categories")
    return model, balanced_df

if __name__ == "__main__":
    model, df = main() 