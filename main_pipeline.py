import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import contractions
import nltk

nltk.download('stopwords')

class Classifier:
    def __init__(self, job_file_path: str, labels_file_path: str):
        self.job_file_path = job_file_path
        self.labels_file_path = labels_file_path
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.stopwords = nltk.corpus.stopwords.words('english')

        self.stopwords.extend(['junior','senior', 'advertisement','london','head','part',
                               'project','lead','job','intern','remote','trainee'])
        if 'it' in self.stopwords:
            self.stopwords.remove('it')
        if 'not' in self.stopwords:
            self.stopwords.remove('not')

    def load_data(self) -> None:
        """Load data."""
        self.df = pd.read_csv(self.job_file_path)
        self.df_cat = pd.read_csv(self.labels_file_path)

    def preprocess_text(self, text: str) -> str:
        """Preprocess text by expanding contractions and removing stopwords."""
        text = contractions.fix(text.lower())
        words = text.split()
        filtered_words = [word for word in words if word not in self.stopwords]
        return ' '.join(filtered_words).lower()

    def process_job_titles(self) -> None:
        """Casts everything as string, removes NA values and calls function 
        to preprocess."""
        self.df['title'] = self.df['title'].astype(str).fillna('')
        self.df['title'] = self.df['title'].apply(self.preprocess_text)

    def find_excluded(self, desc: str) -> str:
        """Removes text after 'Excluded from this group are:'."""
        idx = desc.find("Excluded from this group are:")
        return desc[:idx] if idx != -1 else desc

    def find_classified_elsewhere(self, desc: str) -> str:
        """Removes text after 'Some related occupations classified elsewhere:'"""
        idx = desc.find("Some related occupations classified elsewhere:")
        return desc[:idx] if idx != -1 else desc

    def remove_notes(self, desc: str) -> str:
        """Removes 'Notes'"""
        idx = desc.find("Notes")
        return desc[:idx] if idx != -1 else desc
    
    def remove_note(self, desc: str) -> str:
        """Turns out there's also few instances of 'Note' instead of 'Notes'
        Remove those"""
        idx = desc.find("Note")
        return desc[:idx] if idx != -1 else desc

    def preprocess_labels_descriptions(self) -> None:
        """Clean label descriptions."""
        self.df_cat['description'] = self.df_cat['description'].astype(str).fillna('')
        self.df_cat['description'] = self.df_cat['description'].apply(self.find_excluded)
        self.df_cat['description'] = self.df_cat['description'].apply(self.find_classified_elsewhere)
        self.df_cat['description'] = self.df_cat['description'].apply(self.remove_notes)
        self.df_cat['description'] = self.df_cat['description'].apply(self.remove_note)
        self.df_cat['description'] = self.df_cat['description'].str.lower()
        self.df_cat['description'] = self.df_cat['description'].apply(self.preprocess_text)
        self.df_cat['description'] = self.df_cat['description'].apply(lambda x: x[:4000]) 

    def compute_embeddings(self, texts) -> np.ndarray:
        """Compute sentence embeddings for a list of texts."""
        return self.model.encode(texts)
    
    def find_best_matching_code(self, job_embeddings: np.ndarray, category_embeddings: np.ndarray) -> np.ndarray:
        """Find best label for each job based on cosine similarity."""
        similarities = cosine_similarity(job_embeddings, category_embeddings)
        best_match_indices = np.argmax(similarities, axis=1)
        return self.df_cat.iloc[best_match_indices]['code'].values
    
    def classify_jobs(self) -> None:
        """Asigns labels to jobs."""
        self.load_data()
        self.process_job_titles()
        self.preprocess_labels_descriptions()

        # Compute embeddings
        job_embeddings = self.compute_embeddings(self.df['title'].tolist())
        category_embeddings = self.compute_embeddings(self.df_cat['description'].tolist())

        # Find best matching codes
        self.df['isco_code'] = self.find_best_matching_code(job_embeddings, category_embeddings)

    def save_results(self, output_file_path: str) -> None:
        """Save classification results to a CSV file."""
        self.df[['id', 'isco_code']].to_csv(output_file_path, header=False, index=False)

# Usage
if __name__ == "__main__":
    classifier = Classifier('data/translated_file_corrected.csv', 'data/wi_labels.csv')
    classifier.classify_jobs()
    classifier.save_results('data/results.csv')