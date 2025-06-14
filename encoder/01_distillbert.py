from transformers import DistilBertTokenizer, DistilBertModel
from scipy.spatial.distance import cosine
import torch

def calculate_sentence_similarity(sentence1, sentence2):
    model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertModel.from_pretrained(model_name)

    tokens = tokenizer([sentence1, sentence2], padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**tokens)
        sentence_embeddings = outputs.last_hidden_state[:, 0, :].numpy()

    similarity = 1 - cosine(sentence_embeddings[0], sentence_embeddings[1])
    return similarity

#### data load ####
sentence1 = "I like cats"
sentence2 = "There are some boys playing football"

#### inference ####
similarity_score = calculate_sentence_similarity(sentence1, sentence2)
print(f"Similarity score: {similarity_score}")