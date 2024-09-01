# The script to translate job posting info to English


from deep_translator import GoogleTranslator
import logging
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

df = pd.read_csv('wi_dataset.csv')

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

def translate(phrase):
    try:
        # Ensure the phrase is within the valid length range
        if 0 < len(phrase) <= 4000:
            return GoogleTranslator(source='auto', target='en').translate(phrase)
        else:
            phrase = phrase[:4000]
            return GoogleTranslator(source='auto', target='en' ).translate(phrase)
    except Exception as e:
        return f"Translation error: {str(e)}"
    
def parallel_apply(df, func, column_name, num_workers=5):
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        result = list(executor.map(func, df[column_name]))
    return result

df['description'] = parallel_apply(df, translate, 'description')
df['title'] = parallel_apply(df, translate, 'title')

df.to_csv('translated_file.csv', index=False)