import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os
from sklearn.utils import resample

min_samples = 50

def oversample(df, min_samples):
    categories = df["Category"].value_counts()
    for category, count in categories.items():
        if count < min_samples:
            additional_samples = resample(
                df[df["Category"] == category],
                replace=True,  # Duplicate
                n_samples=min_samples - count,  # Increase to min_samples
                random_state=42
            )
            df = pd.concat([df, additional_samples])
    return df

# Load dataset
data_path = "data/exported_translated.csv"
df = pd.read_csv(data_path, encoding="utf-8-sig", delimiter=",")
df.dropna(subset=["Translated_Description"], inplace=True)

df = oversample(df, min_samples)
df = df.groupby("Category").apply(lambda x: x.sample(min(len(x), 100), random_state=42))
df = df.reset_index(drop=True)

# Encode category labels into numerical values
label_encoder = LabelEncoder()
df["category_encoded"] = label_encoder.fit_transform(df["Category"])

# 🔍 Show category distribution before training
category_counts = df["Category"].value_counts()
print("\n🔹 Category Distribution:")
print(category_counts)

# Plot the category distribution as a histogram
plt.figure(figsize=(10, 5))
sns.barplot(x=category_counts.index, y=category_counts.values)
plt.xticks(rotation=45, ha="right")
plt.xlabel("Category")
plt.ylabel("Count")
plt.title("Expense Category Distribution")
plt.show()

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df["Translated_Description"], df["category_encoded"], test_size=0.2, random_state=42
)

# Convert text into numerical features using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression(max_iter=500)
model.fit(X_train_tfidf, y_train)

# Evaluate model accuracy
accuracy = model.score(X_test_tfidf, y_test)
print(f"\n✅ Model Accuracy: {accuracy:.2f}")

# Save the trained model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/expense_categorizer.pkl", protocol=4)
joblib.dump(vectorizer, "models/vectorizer.pkl", protocol=4)
joblib.dump(label_encoder, "models/label_encoder.pkl", protocol=4)

print("✅ Model training complete. Files saved in 'models' folder.")
