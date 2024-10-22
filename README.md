# LLM-Based Occupation Classification Using ISCO Nomenclature

 This repo contains code that was used to assign isco nomenclature for 
 occupations and their descriptions scrapped from various job posting websites 
 as a part of WEB INTELLIGENCE - CLASSIFICATION CHALLENGE (European Statistics 
 Awards)

Note: The data is under NDA and cannot be shared.
The approach is using 'all-mpnet-base-v2' model from Hugging Face. 

__Python scrips:__

**translation.py** : 
    This script handles the translation of job postings that are in different 
    languages into English using the Google Translator API.

    - Inputs:
    A .csv file containing the scraped job postings with at least two columns: id and title.

    - Outputs:
    A translated .csv file where all job titles are in English.


**main_pipeline.py** :
    This is the main script that performs the occupation classification.

    - Inputs:
    A translated .csv file containing job postings (id and title columns).
    A .csv file containing ISCO labels (code, label, and description columns).

    - Functionality:
    Cleans the input data.
    Uses the all-mpnet-base-v2 model to compute the semantic similarity between 
    job titles and ISCO descriptions.
    Assigns the most semantically similar ISCO label to each job title.


**tests.py** : A script for testing the submission file before submission, 
    ensuring there are no formatting or content-related issues.


__Additional Directories__

/prep/ and /translation_fixes/: These directories contain additional scripts 
used during preprocessing and troubleshooting phases.

__Technologies Used__

    - Hugging Face Transformers for sentence embeddings and semantic similarity.
    - Google Translator API for translating non-English job postings to English.
    - Python Libraries: pandas, transformers, nltk.