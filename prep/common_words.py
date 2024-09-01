#To count most common words (for customising stopwords later)

import pandas as pd
from collections import Counter
import re
from nltk.corpus import stopwords
import nltk

# Ensure you have NLTK stopwords
nltk.download('stopwords')

file_path = 'data/translated_file_corrected.csv'
file_path_cat = 'data/wi_labels.csv'

df = pd.read_csv(file_path)
df_cat = pd.read_csv(file_path_cat)

column_name = 'title'

# Concatenates all text into a single string
all_text = ' '.join(df['title'].dropna().astype(str).tolist())


words = re.findall(r'\b\w+\b', all_text.lower())

# Remove stopwords
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word not in stop_words]

# Find the most common words
word_counts = Counter(filtered_words)
most_common_words = word_counts.most_common(50) 

for word, count in most_common_words:
    print(f"{word}: {count}")