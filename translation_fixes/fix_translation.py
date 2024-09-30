#Was only used to fix some issues with previous translation logic without 
#rerunning translation. No longer relevant.

from deep_translator import GoogleTranslator
import logging
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

df = pd.read_csv('translated_file.csv')
df_og = pd.read_csv('wi_dataset.csv')
df['description'] = df['description'].astype(str).fillna('')

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

def find_mistakes(phrase):
    ind = phrase.find("deep_translator.google.GoogleTranslator object ")
    if ind != -1:
        return 1

 
df['test'] = df['description'].apply(find_mistakes)
print(df['test'].sum())

subset = df[df['test']==1]

df_full = pd.merge(subset, df_og, on='id')


def translate(phrase):
    try:
        # Ensure the phrase is within the valid length range
        if 0 < len(phrase) <= 5000:
            return GoogleTranslator(source='auto', target='en').translate(phrase[:4000])
        else:
            phrase = phrase[:4000]
            return GoogleTranslator(source='auto', target='en').translate(phrase)
    except Exception as e:
        # Log the error and return an appropriate message
        logging.error(f"Translation error: {str(e)}")
        return f"Translation error: {str(e)}"
    
def parallel_apply(df, func, column_name, num_workers=5):
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        result = list(executor.map(func, df[column_name].astype(str)))  # Ensure column data is of type str
    return result

# Apply the translation function to 'description' and 'title' columns
df_full['description_t'] = parallel_apply(df_full, translate, 'description_y')
df_full['title_t'] = parallel_apply(df_full, translate, 'title_y')

# Save the DataFrame
df_full.to_csv('translation_mid.csv', index=False)
