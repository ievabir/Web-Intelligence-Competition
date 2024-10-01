# This repo contains scripts that were used to assign isco nomenclature for occupations and their descriptions scrapped from various job posting websites as a part of WEB INTELLIGENCE - CLASSIFICATION CHALLENGE (European Statistics Awards)

The data itself is under NDA. This is simple approach using 'all-mpnet-base-v2'
model from Hugging Face. 

__Python scrips:__

**translation.py** : since the data was scrapped from job posting websites
    in different languages, this script translates it to English using Google Translator

**main_pipeline.py** : this script takes the translated data, cleans it, 
    and uses 'all-mpnet-base-v2' to calculate semantic similarity between 
    job posting data and isco descriptions. it assign the most semantically similar 
    description for each job title

**tests.py** : this scrip checks if there's issues with the submission file.

files in  \prep and \translation_fixes directories were used at different phases
when troubleshooting.