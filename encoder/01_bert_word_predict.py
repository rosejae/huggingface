#########################
#### word prediction ####
#########################

import torch
from transformers import BertTokenizer, BertForMaskedLM

device = torch.device('cuda')

def predict_next_word(text):
    tokenized_text = tokenizer.tokenize(text)
    masked_index = tokenized_text.index('[MASK]')
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    tokens_tensor = torch.tensor([indexed_tokens]).to(device)

    with torch.no_grad():
        outputs = model.to('cuda')(tokens_tensor)

    predictions = outputs[0][0, masked_index].topk(k=5).indices.tolist()

    predicted_tokens = []
    for token_index in predictions:
        predicted_token = tokenizer.convert_ids_to_tokens([token_index])[0]
        predicted_tokens.append(predicted_token)

    return predicted_tokens

#### model load ####
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)

#### inference ####
text_with_mask = "I want to [MASK] a pizza for dinner."
predicted_tokens = predict_next_word(text_with_mask)
print(f"top 5 words: {predicted_tokens}")