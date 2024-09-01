import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import contractions
import nltk

nltk.download('stopwords')

# Paths to your CSV files
file_path = 'data/translated_file_corrected.csv'
file_path_cat = 'data/wi_labels.csv'

# Read CSV files
df = pd.read_csv(file_path)
df_cat = pd.read_csv(file_path_cat)


def find_included(desc):
    idx = desc.find("Examples of the occupations classified here:")
    if idx != -1:
        return desc[idx:-1]
    else:
        return desc


def find_excluded(desc):
    idx = desc.find("Excluded from this group are:")
    if idx != -1:
        return desc[:idx]
    else:
        return desc

def find_classified_elsewhere(desc):
    idx = desc.find("Some related occupations classified elsewhere:")
    if idx != -1:
        return desc[:idx]
    else:
        return desc
    
def remove_notes(desc):
    idx = desc.find("Notes")
    if idx != -1:
        return desc[:idx]
    else:
        return desc

def remove_redundant(desc):
    # After seeing that there's statistically more noise than meaningful info 
    # 4000 character mark, the rest of job description is cut.
    return desc[:4000]

def remove_stopwords(text):
    stopwords = [nltk.corpus.stopwords.words('english')]
    stopwords.extend(['junior','senior', 'advertisement','london','head','part',
                      'project','lead','job','intern','remote','trainee']) 
    if 'it' in stopwords:
        stopwords.remove('it')
    words = text.split() 
    filtered_words = [word for word in words if word.lower() not in stopwords]
    return ' '.join(filtered_words)


df['title'] = df['title'].astype(str).fillna('')
df['title'] = df['title'].str.lower()
df['title'] = df['title'].apply(lambda x: contractions.fix(x))
df['title'] = df['title'].apply(remove_stopwords)


df_cat['description'] = df_cat['description'].astype(str).fillna('')
df_cat['description'] = df_cat['label'] + '.' + df_cat['description']
df_cat['description'] = df_cat['description'].apply(find_included)
df_cat['description'] = df_cat['description'].apply(find_excluded)
df_cat['description'] = df_cat['description'].apply(find_classified_elsewhere)
df_cat['description'] = df_cat['description'].apply(remove_notes)
df_cat['description'] = df_cat['description'].apply(remove_redundant)
df_cat['description'] = df_cat['description'].str.lower()
df_cat['description'] = df_cat['description'].apply(lambda x: contractions.fix(x))    
df_cat['description'] = df_cat['description'].apply(remove_stopwords)


# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_sentence_embeddings(text):

    embeddings = model.encode(text)
    return embeddings

def find_best_matching_code(job_embeddings, category_embeddings, df_cat):

    similarities = cosine_similarity(job_embeddings, category_embeddings)

    best_match_indices = np.argmax(similarities, axis=1)
    
    isco_codes = df_cat.iloc[best_match_indices]['code'].values
    return isco_codes

# Compute embeddings for job titles and category descriptions
df['embedding'] = df['title'].apply(get_sentence_embeddings)
df_cat['embedding'] = df_cat['description'].apply(get_sentence_embeddings)

# Convert embeddings to numpy arrays for calculating similarity
job_vectors = np.vstack(df['embedding'].values)
category_vectors = np.vstack(df_cat['embedding'].values)

df['isco_code'] = find_best_matching_code(job_vectors, category_vectors, df_cat)

df[['id', 'isco_code']].to_csv('classification_final.csv', header=False, index=False)