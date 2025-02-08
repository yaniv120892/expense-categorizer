import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import fasttext
import fasttext.util
import joblib
import os
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder

# Ensure fasttext embeddings are available
fasttext.util.download_model('en', if_exists='ignore')  # English model
ft = fasttext.load_model('cc.en.300.bin')  # Load FastText embeddings

# 🔹 Define minimum samples for balancing categories
min_samples = 50

def oversample(df, min_samples):
    categories = df["Category"].value_counts()
    for category, count in categories.items():
        if count < min_samples:
            additional_samples = resample(
                df[df["Category"] == category],
                replace=True,  
                n_samples=min_samples - count,  
                random_state=42
            )
            df = pd.concat([df, additional_samples])
    return df

# Load dataset
data_path = "data/exported_translated.csv"
df = pd.read_csv(data_path, encoding="utf-8-sig", delimiter=",")
df.dropna(subset=["Translated_Description"], inplace=True)

# Balance dataset & limit each category to 100 samples
df = oversample(df, min_samples)
df = df.groupby("Category").apply(lambda x: x.sample(min(len(x), 100), random_state=42))
df = df.reset_index(drop=True)

# Encode category labels into numerical values
label_encoder = LabelEncoder()
df["category_encoded"] = label_encoder.fit_transform(df["Category"])

# Show category distribution before training
category_counts = df["Category"].value_counts()
print("\n🔹 Category Distribution:")
print(category_counts)

# Plot the category distribution
plt.figure(figsize=(10, 5))
sns.barplot(x=category_counts.index, y=category_counts.values)
plt.xticks(rotation=45, ha="right")
plt.xlabel("Category")
plt.ylabel("Count")
plt.title("Expense Category Distribution")
plt.show()

# Prepare FastText training data format: "__label__category description"
df["fasttext_label"] = df["Category"].apply(lambda x: f"__label__{x}")
df["fasttext_text"] = df["Translated_Description"]

# Save training data in FastText format
train_file = "data/fasttext_train.txt"
df[["fasttext_label", "fasttext_text"]].to_csv(
    train_file,
    index=False,
    sep=" ",
    header=False,
    quoting=3,  # QUOTE_NONE (Prevents extra quotes)
    escapechar="\\",  # Adds a backslash to escape special characters
    encoding="utf-8"
)

# Train FastText model
fasttext_model = fasttext.train_supervised(input=train_file, epoch=25, lr=0.5, wordNgrams=2, dim=100)

# Evaluate model accuracy
correct = 0
total = 0
for _, row in df.iterrows():
    predicted_label = fasttext_model.predict(row["fasttext_text"])[0][0]
    actual_label = f"__label__{row['Category']}"
    if predicted_label == actual_label:
        correct += 1
    total += 1
accuracy = correct / total
print(f"\n✅ Model Accuracy: {accuracy:.2f}")

# Save the trained FastText model
os.makedirs("models", exist_ok=True)
fasttext_model.save_model("models/expense_categorizer.ftz")

# 🔹 Save Label Encoder for API
joblib.dump(label_encoder, "models/label_encoder.pkl", protocol=4)

print("✅ Model training complete. Files saved in 'models' folder.")
