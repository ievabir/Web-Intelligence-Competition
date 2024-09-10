# Test dataset was generated to later fine-tune the model. For all those
# occupations that generated similarity score bigger than .7 in the beginning, 
# 1 was assigned.

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import contractions
import nltk

nltk.download('stopwords')

job_file_path = 'data/translated_file_corrected.csv'
labels_file_path = 'data/wi_labels.csv'
model = SentenceTransformer('all-mpnet-base-v2')
stopwords = nltk.corpus.stopwords.words('english')

stopwords.extend(['junior','senior', 'advertisement','london','head','part',
                        'project','lead','job','intern','remote','trainee'])
if 'it' in stopwords:
    stopwords.remove('it')
if 'not' in stopwords:
    stopwords.remove('not')

df = pd.read_csv(job_file_path, usecols=['id','title'])
df_cat = pd.read_csv(labels_file_path, usecols=['code','description'])

def preprocess_text(text: str) -> str:
    """Preprocess text by expanding contractions and removing stopwords."""
    text = contractions.fix(text.lower())
    words = text.split()
    filtered_words = [word for word in words if word not in stopwords]
    return ' '.join(filtered_words).lower()

df['title'] = df['title'].apply(preprocess_text)

def find_excluded(desc: str) -> str:
    """Removes text after 'Excluded from this group are:'."""
    idx = desc.find("Excluded from this group are:")
    return desc[:idx] if idx != -1 else desc

def find_classified_elsewhere (desc: str) -> str:
    """Removes text after 'Some related occupations classified elsewhere:'"""
    idx = desc.find("Some related occupations classified elsewhere:")
    return desc[:idx] if idx != -1 else desc

def remove_notes(desc: str) -> str:
    """Removes 'Notes'"""
    idx = desc.find("Notes")
    return desc[:idx] if idx != -1 else desc

def remove_note(desc):
    """Removes 'Note'"""
    idx = desc.find("Note")
    return desc[:idx] if idx != -1 else desc

df_cat['description'] = df_cat['description'].apply(find_excluded)
df_cat['description'] = df_cat['description'].apply(find_classified_elsewhere)
df_cat['description'] = df_cat['description'].apply(remove_notes)
df_cat['description'] = df_cat['description'].apply(remove_note)


def preprocess_labels_descriptions(): 
    """Clean label descriptions."""
    df_cat['description'] = df_cat['description'].astype(str).fillna('')
    df_cat['description'] = df_cat['description'].str.lower()
    df_cat['description'] = df_cat['description'].apply(preprocess_text)

preprocess_labels_descriptions()


def compute_embeddings( texts) -> np.ndarray:
    return model.encode(texts)

def find_best_matching_code(job_embeddings: np.ndarray, category_embeddings: np.ndarray) -> np.ndarray:
    """Find best label for each job based on cosine similarity."""
    similarities = cosine_similarity(job_embeddings, category_embeddings)
    best_match_indices = np.argmax(similarities, axis=1)
    return df_cat.iloc[best_match_indices]['code'].values

job_embeddings = compute_embeddings(df['title'].tolist())
category_embeddings = compute_embeddings(df_cat['description'].tolist())

similarities = cosine_similarity(job_embeddings, category_embeddings)
best_match_indices = np.argmax(similarities, axis=1)
best_match_scores = np.max(similarities, axis=1)

df['isco_code'] = df_cat.iloc[best_match_indices]['code'].values
df['description'] = df_cat.iloc[best_match_indices]['description'].values
df['similarity_score'] = best_match_scores

df = df[df['similarity_score'] >= 0.70]

df = df[['title', 'description', 'similarity_score']]
df['similarity_score'] = df['similarity_score'].apply(lambda x: 1)

df.to_csv('data/training.csv')