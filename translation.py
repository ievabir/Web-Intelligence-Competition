""" The script to translate job posting titles from different languages to 
English using GoogleTranslator. 
WARNING: This runs very slowly. time.sleep(0.3) in between each request
was added not to hit the limit of api calls to GoogleTranslator per second. 
More sophisticated solution could be added, but since this script only needs
to run once it will be sufficient for the purpose at the moment. 
"""

import time
from deep_translator import GoogleTranslator
import logging
import pandas as pd

df = pd.read_csv('data/wi_dataset.csv')
df = df[['id','title']]

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
        if 0 < len(phrase) <= 4000:
            translated = GoogleTranslator(source='auto', target='en').translate(phrase)
            time.sleep(0.3)  
            return translated
        else:
            phrase = phrase[:4000]
            translated = GoogleTranslator(source='auto', target='en').translate(phrase)
            time.sleep(0.3)
            return translated
    except Exception as e:
        return f"Translation error: {str(e)}"

df['title'] = df['title'].apply(translate)

df.to_csv('data/translated_file.csv', index=False)