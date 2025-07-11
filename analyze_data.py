import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

def analyze_translation_quality(df):
    """Analyze the quality of translations"""
    print("🔍 Analyzing translation quality...")
    
    # Check for untranslated Hebrew text
    hebrew_pattern = re.compile(r'[\u0590-\u05FF]')
    untranslated = []
    
    for _, row in df.iterrows():
        if hebrew_pattern.search(row['Translated_Description']):
            untranslated.append({
                'original': row['Description'],
                'translated': row['Translated_Description'],
                'category': row['Category']
            })
    
    print(f"⚠️  Found {len(untranslated)} entries with Hebrew text in translation")
    if untranslated:
        print("Examples of poor translations:")
        for i, item in enumerate(untranslated[:5]):
            print(f"  {i+1}. '{item['original']}' -> '{item['translated']}'")
    
    return untranslated

def analyze_category_distribution(df):
    """Analyze category distribution"""
    print("\n📊 Category Distribution Analysis:")
    
    category_counts = df['Category'].value_counts()
    
    print(f"Total categories: {len(category_counts)}")
    print(f"Total samples: {len(df)}")
    print(f"\nTop 10 categories:")
    for i, (category, count) in enumerate(category_counts.head(10).items()):
        percentage = (count / len(df)) * 100
        print(f"  {i+1}. {category}: {count} samples ({percentage:.1f}%)")
    
    print(f"\nBottom 10 categories:")
    for i, (category, count) in enumerate(category_counts.tail(10).items()):
        percentage = (count / len(df)) * 100
        print(f"  {i+1}. {category}: {count} samples ({percentage:.1f}%)")
    
    return category_counts

def analyze_text_lengths(df):
    """Analyze text length distribution"""
    print("\n📏 Text Length Analysis:")
    
    df['original_length'] = df['Description'].str.len()
    df['translated_length'] = df['Translated_Description'].str.len()
    
    print(f"Original text lengths:")
    print(f"  Mean: {df['original_length'].mean():.1f}")
    print(f"  Median: {df['original_length'].median():.1f}")
    print(f"  Min: {df['original_length'].min()}")
    print(f"  Max: {df['original_length'].max()}")
    
    print(f"\nTranslated text lengths:")
    print(f"  Mean: {df['translated_length'].mean():.1f}")
    print(f"  Median: {df['translated_length'].median():.1f}")
    print(f"  Min: {df['translated_length'].min()}")
    print(f"  Max: {df['translated_length'].max()}")
    
    # Find very short or very long texts
    short_texts = df[df['translated_length'] < 5]
    long_texts = df[df['translated_length'] > 100]
    
    print(f"\n⚠️  Found {len(short_texts)} very short texts (< 5 chars)")
    print(f"⚠️  Found {len(long_texts)} very long texts (> 100 chars)")
    
    return df

def find_similar_descriptions(df, category='Bar', top_n=10):
    """Find similar descriptions within a category"""
    print(f"\n🔍 Analyzing '{category}' category descriptions:")
    
    category_data = df[df['Category'] == category]
    
    if len(category_data) == 0:
        print(f"No data found for category '{category}'")
        return
    
    # Count word frequencies
    all_words = []
    for text in category_data['Translated_Description']:
        words = text.lower().split()
        all_words.extend(words)
    
    word_counts = Counter(all_words)
    
    print(f"Most common words in '{category}' category:")
    for word, count in word_counts.most_common(top_n):
        print(f"  '{word}': {count} times")
    
    # Show some examples
    print(f"\nSample descriptions from '{category}' category:")
    for i, (_, row) in enumerate(category_data.head(10).iterrows()):
        print(f"  {i+1}. '{row['Description']}' -> '{row['Translated_Description']}'")

def create_visualizations(df):
    """Create visualizations of the data"""
    print("\n📈 Creating visualizations...")
    
    # Category distribution
    plt.figure(figsize=(15, 8))
    category_counts = df['Category'].value_counts()
    
    # Show top 20 categories
    top_categories = category_counts.head(20)
    plt.subplot(1, 2, 1)
    top_categories.plot(kind='bar')
    plt.title('Top 20 Categories by Sample Count')
    plt.xlabel('Category')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Text length distribution
    plt.subplot(1, 2, 2)
    df['translated_length'] = df['Translated_Description'].str.len()
    plt.hist(df['translated_length'], bins=50, alpha=0.7)
    plt.title('Distribution of Translated Text Lengths')
    plt.xlabel('Text Length (characters)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    
    plt.savefig('data_analysis.png', dpi=300, bbox_inches='tight')
    print("📊 Visualization saved as 'data_analysis.png'")

def main():
    print("🚀 Starting Data Analysis")
    
    # Load data
    data_path = "data/exported_translated.csv"
    df = pd.read_csv(data_path, encoding="utf-8-sig")
    
    print(f"📁 Loaded {len(df)} samples from {data_path}")
    
    # Run analyses
    untranslated = analyze_translation_quality(df)
    category_counts = analyze_category_distribution(df)
    df = analyze_text_lengths(df)
    
    # Analyze Bar category specifically
    find_similar_descriptions(df, 'Bar')
    
    # Create visualizations
    create_visualizations(df)
    
    # Summary recommendations
    print("\n💡 Recommendations:")
    print("1. Fix poor translations - many Hebrew texts remain untranslated")
    print("2. Balance the dataset - some categories have too few samples")
    print("3. Consider merging similar categories (e.g., 'Bar' and 'Eating out')")
    print("4. Clean up very short or very long descriptions")
    print("5. Use better text preprocessing before training")
    
    return df, category_counts

if __name__ == "__main__":
    df, category_counts = main() 