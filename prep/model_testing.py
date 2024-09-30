#We choose few models from SentenceTransofrmers, and give them one job title 
#to classify. We also provide three descriptions: one correct one, one very 
#similar but incorrect, and one radically incorrect. 
#we then do some preprocessing, and check the cosine similarity for each model
#embedded values.

from sentence_transformers import SentenceTransformer
import re


# model = SentenceTransformer('all-mpnet-base-v2')
# 0.43, 0.65, 0.04
# model = SentenceTransformer('all-distilroberta-v1')
# 0.46, 0.74, 0.08
model = SentenceTransformer('all-MiniLM-L6-v2')
# 0.43, 0.73, -0.001

sentence1 = "Window Cleaners"
sentence2 = "Domestic cleaners and helpers sweep, vacuum clean, wash and polish, take care of household linen, purchase household supplies, prepare food, serve meals and perform various other domestic duties. Tasks include sweeping, vacuum cleaning, polishing and washing floors and furniture, or washing windows and other fixtures; washing, ironing and mending linen and other textiles;washing dishes; helping with preparation, cooking and serving of meals and refreshments;  purchasing food and various other household supplies;  cleaning, disinfecting and deodorizing kitchens, bathrooms and toilets;cleaning windows and other glass surfaces.Examples of the occupations classified here:Charworker (domestic),Domestic cleaner,  Domestic helper"
sentence3 = "Window cleaners wash and polish windows and other glass fittings. Tasks include - washing windows or other glass surfaces  with  water or various solutions, and drying and polishing them; using ladders, swinging scaffolds, bosun’s chairs, hydraulic bucket trucks and other equipment to reach and clean windows in multistorey buildings; selecting appropriate cleaning or polishing implements. Examples of the occupations classified here: Window cleaner"
sentence4 = "Commissioned armed forces officers provide leadership and management to organizational units in the armed forces and/or perform similar tasks to those performed in a variety of civilian occupations outside the armed forces. This group includes all members of the armed forces holding the rank of second lieutenant (or equivalent) or higher.Examples of the occupations classified here:  Admiral, Air commodore, Air marshal, Brigadier (army), Captain (air force), Captain (army), Captain (navy), Colonel (army), Field marshal, Flight lieutenant (air force), Flying officer (military), General (army), Group captain, (air force),Lieutenant (army), Major (army), Midshipman, Naval officer (military), Navy commander, Officer cadet (armed forces), Second lieutenant (army), Squadron leader, Sublieutenant (navy), Wing commander"


encoding1 = model.encode(sentence1)
encoding2 = model.encode(sentence2)
encoding3 = model.encode(sentence3)
encoding4 = model.encode(sentence4)


def preprocess_text(text):
    # Removes bullet points and numbering
    text = re.sub(r"^\s*[-*•]+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\(?[a-zA-Z0-9]+\)?\.\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]|[\d]', '', text)
    return text



def get_sentence_embeddings(text):
    # Function to compute sentence embeddings
    
    text = preprocess_text(text)
    embeddings = model.encode(text)

    return embeddings

def default_similarity(embedding1, embedding2):
    similarity = model.similarity(embedding1, embedding2)
    return similarity


# Get sentence embeddings
embedding1 = get_sentence_embeddings(sentence1)
embedding2 = get_sentence_embeddings(sentence2)
embedding3 = get_sentence_embeddings(sentence3)
embedding4 = get_sentence_embeddings(sentence4)


# Calculate cosine similarity
similarity = default_similarity(embedding1, embedding2)
print(f"Similarity Score (Similar): {similarity.item()}")
similarity = default_similarity(embedding1, embedding3)
print(f"Similarity Score (Correct): {similarity.item()}")
similarity = default_similarity(embedding1, embedding4)
print(f"Similarity Score (Wrong): {similarity.item()}")
